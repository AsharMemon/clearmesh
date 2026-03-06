#!/usr/bin/env python3
"""Ctrl-Adapter — Normal-guided multi-view texture generation.

The only trainable component of the Easy3E editing pipeline.
A lightweight ControlNet-style adapter that conditions ERA3D's
multi-view diffusion model on surface normal maps for consistent
texture generation across 6 views.

Architecture (from Easy3E Section 3.4):
  - Input: 6-view normal maps (front/back/left/right/top/bottom)
  - Adapter: Lightweight residual blocks injected into ERA3D
  - Output: Textured 6-view RGB images conditioned on normals

Training:
  - Dataset: 5K-10K Objaverse meshes with 6-view RGB + normal pairs
  - Loss: MSE between predicted and ground-truth RGB views
  - Time: ~20 GPU-hours on A100

Usage:
    adapter = CtrlAdapter()
    adapter.load_state_dict(torch.load("ctrl_adapter.pt"))

    # Generate textured views from normals
    rgb_views = adapter.generate(
        normal_maps=normal_views,  # (6, 3, 512, 512)
        prompt="a detailed 3D model",
    )
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CtrlAdapterBlock(nn.Module):
    """Single adapter block — extracts control features from normal maps.

    A lightweight residual block that processes normal map features
    and produces control signals to inject into the base diffusion model.
    """

    def __init__(self, channels: int, mid_channels: int | None = None):
        super().__init__()
        mid = mid_channels or channels
        self.conv1 = nn.Conv2d(channels, mid, 3, padding=1)
        self.norm1 = nn.GroupNorm(32, mid)
        self.conv2 = nn.Conv2d(mid, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, channels)
        self.act = nn.SiLU()
        self.scale = nn.Parameter(torch.zeros(1))  # Zero-init for stable training

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with zero-initialized residual.

        Args:
            x: Input features (B, C, H, W).

        Returns:
            Control signal (B, C, H, W).
        """
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return x + h * self.scale


class CtrlAdapterEncoder(nn.Module):
    """Normal map encoder — extracts multi-scale features from normal maps.

    Processes 6-view normal maps into multi-scale feature maps that
    can be injected into a diffusion model's U-Net/DiT decoder.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_levels: int = 4,
        blocks_per_level: int = 2,
    ):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.levels = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        ch = base_channels

        for level in range(num_levels):
            blocks = nn.ModuleList()
            for _ in range(blocks_per_level):
                blocks.append(CtrlAdapterBlock(ch))
            self.levels.append(blocks)

            if level < num_levels - 1:
                next_ch = ch * 2
                self.downsamples.append(
                    nn.Conv2d(ch, next_ch, 3, stride=2, padding=1)
                )
                ch = next_ch
            else:
                self.downsamples.append(nn.Identity())

    def forward(self, normal_maps: torch.Tensor) -> list[torch.Tensor]:
        """Extract multi-scale features from normal maps.

        Args:
            normal_maps: Normal maps (B, 3, H, W) — values in [0, 1].

        Returns:
            List of feature maps at each scale level.
        """
        # Normalize from [0, 1] to [-1, 1]
        x = normal_maps * 2.0 - 1.0
        x = self.input_conv(x)

        features = []
        for level, (blocks, downsample) in enumerate(
            zip(self.levels, self.downsamples)
        ):
            for block in blocks:
                x = block(x)
            features.append(x)
            x = downsample(x)

        return features


class CtrlAdapter(nn.Module):
    """Full Ctrl-Adapter model for normal-guided texture generation.

    Combines:
      1. Normal map encoder (extracts control features)
      2. Control injection (adds features to base diffusion model)
      3. Multi-view consistency (processes all 6 views jointly)

    The adapter is trained while the base model (ERA3D) stays frozen.
    This makes training fast and data-efficient (~5K examples, 20 GPU-hrs).
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_levels: int = 4,
        blocks_per_level: int = 2,
        num_views: int = 6,
    ):
        super().__init__()
        self.num_views = num_views

        # Per-view normal encoder
        self.encoder = CtrlAdapterEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
            num_levels=num_levels,
            blocks_per_level=blocks_per_level,
        )

        # Cross-view attention for multi-view consistency
        # Channels at each level: base, base*2, base*4, base*8
        self.cross_view_attn = nn.ModuleList()
        ch = base_channels
        for level in range(num_levels):
            self.cross_view_attn.append(
                nn.MultiheadAttention(ch, num_heads=max(1, ch // 64), batch_first=True)
            )
            if level < num_levels - 1:
                ch *= 2

        # Zero convolutions for injecting control into base model
        self.zero_convs = nn.ModuleList()
        ch = base_channels
        for level in range(num_levels):
            self.zero_convs.append(
                nn.Conv2d(ch, ch, 1, bias=True)
            )
            # Initialize to zero
            nn.init.zeros_(self.zero_convs[-1].weight)
            nn.init.zeros_(self.zero_convs[-1].bias)
            if level < num_levels - 1:
                ch *= 2

    def forward(
        self,
        normal_maps: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Extract control signals from multi-view normal maps.

        Args:
            normal_maps: (B, num_views, 3, H, W) — 6-view normal maps.

        Returns:
            List of control feature maps at each scale level,
            each shaped (B * num_views, C, H', W').
        """
        B, V, C, H, W = normal_maps.shape
        assert V == self.num_views, f"Expected {self.num_views} views, got {V}"

        # Reshape to process all views at once
        x = normal_maps.view(B * V, C, H, W)

        # Extract per-view features
        per_view_features = self.encoder(x)  # List of (B*V, C_l, H_l, W_l)

        # Apply cross-view attention and zero convolutions
        control_signals = []
        for level, (feats, attn, zero_conv) in enumerate(
            zip(per_view_features, self.cross_view_attn, self.zero_convs)
        ):
            BV, C_l, H_l, W_l = feats.shape

            # Reshape for cross-view attention: (B, V, C*H*W)
            feats_flat = feats.view(B, V, C_l, H_l * W_l)
            # Pool spatial for attention (B, V, C_l)
            feats_pooled = feats_flat.mean(dim=-1)

            # Cross-view attention
            attn_out, _ = attn(feats_pooled, feats_pooled, feats_pooled)

            # Add attention output back (broadcast spatial)
            attn_out = attn_out.unsqueeze(-1).unsqueeze(-1)  # (B, V, C_l, 1, 1)
            feats_with_attn = feats.view(B, V, C_l, H_l, W_l) + attn_out
            feats_with_attn = feats_with_attn.view(BV, C_l, H_l, W_l)

            # Zero convolution
            control = zero_conv(feats_with_attn)
            control_signals.append(control)

        return control_signals

    def generate(
        self,
        normal_maps: torch.Tensor,
        base_model: nn.Module | None = None,
        prompt: str = "",
        num_steps: int = 50,
        guidance_scale: float = 7.5,
    ) -> torch.Tensor:
        """Generate textured multi-view images from normal maps.

        Args:
            normal_maps: (B, 6, 3, H, W) — 6-view normal maps in [0, 1].
            base_model: Base multi-view diffusion model (ERA3D or similar).
            prompt: Text prompt for texture generation.
            num_steps: Diffusion sampling steps.
            guidance_scale: CFG scale.

        Returns:
            Generated RGB views (B, 6, 3, H, W).
        """
        # TODO: Integrate with ERA3D or similar multi-view diffusion model
        # 1. Extract control signals from normal maps
        # 2. Run base model's denoising loop
        # 3. At each step, inject control signals via addition
        # 4. Return final denoised RGB views
        raise NotImplementedError(
            "Full generation requires ERA3D base model integration. "
            "The Ctrl-Adapter is trained independently first."
        )
