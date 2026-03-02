"""Stage 2 Refinement DiT: predicts local SDF corrections from coarse O-Voxels.

Architecture follows UltraShape's approach:
  - Input: Coarse O-Voxel grid from TRELLIS.2 Stage 1
  - Spatial anchoring via RoPE encoding tied to coarse geometry positions
  - Output: Refined SDF values at each voxel position
  - The model only learns LOCAL corrections, not global structure

The DiT (Diffusion Transformer) operates on voxel tokens with:
  - 3D RoPE positional encoding (anchored to coarse voxel positions)
  - AdaLN-Zero conditioning on diffusion timestep
  - Cross-attention to conditioning image features (DINO)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RoPE3D(nn.Module):
    """3D Rotary Position Embedding anchored to coarse geometry positions.

    Unlike standard positional encoding, positions come from the coarse
    voxel grid rather than a regular grid. This anchors the refinement
    to the coarse structure.
    """

    def __init__(self, dim: int, max_freq: float = 10.0):
        super().__init__()
        self.dim = dim
        assert dim % 6 == 0, "dim must be divisible by 6 for 3D RoPE (x,y,z pairs)"
        self.max_freq = max_freq

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Apply 3D RoPE.

        Args:
            x: (B, N, D) token features
            positions: (B, N, 3) xyz positions from coarse geometry
        """
        B, N, D = x.shape
        dim_per_axis = self.dim // 6  # sin/cos for each of x, y, z

        # Frequency bands
        freqs = torch.exp(
            torch.linspace(0, math.log(self.max_freq), dim_per_axis, device=x.device)
        )

        # Compute angles for each axis
        angles = []
        for axis in range(3):
            pos = positions[:, :, axis : axis + 1]  # (B, N, 1)
            axis_angles = pos * freqs.unsqueeze(0).unsqueeze(0)  # (B, N, dim_per_axis)
            angles.append(axis_angles)

        angles = torch.cat(angles, dim=-1)  # (B, N, dim_per_axis * 3)

        # Apply rotation
        cos = torch.cos(angles)
        sin = torch.sin(angles)

        # Split x into rotated and non-rotated parts
        rope_dim = dim_per_axis * 6
        x_rope = x[:, :, :rope_dim]
        x_pass = x[:, :, rope_dim:]

        # Rotate pairs
        x1, x2 = x_rope.chunk(2, dim=-1)
        x_rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

        return torch.cat([x_rotated, x_pass], dim=-1)


class AdaLNZero(nn.Module):
    """Adaptive Layer Norm Zero for timestep conditioning (from DiT paper)."""

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj = nn.Linear(cond_dim, 6 * dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        """Returns shift, scale, gate parameters for pre/post attention + MLP."""
        params = self.proj(cond).unsqueeze(1)  # (B, 1, 6*D)
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = params.chunk(6, dim=-1)
        return shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp


class DiTBlock(nn.Module):
    """Diffusion Transformer block with RoPE, AdaLN-Zero, and cross-attention."""

    def __init__(self, dim: int, num_heads: int, cond_dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads

        # Self-attention
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        # Cross-attention to conditioning features
        self.norm_cross = nn.LayerNorm(dim, elementwise_affine=False)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_cond = nn.LayerNorm(dim)

        # MLP
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim),
        )

        # AdaLN-Zero conditioning
        self.adaln = AdaLNZero(dim, cond_dim)

        # RoPE
        self.rope = RoPE3D(dim)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        timestep_emb: torch.Tensor,
        cond_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Get AdaLN parameters
        s_a, sc_a, g_a, s_m, sc_m, g_m = self.adaln(x, timestep_emb)

        # Self-attention with RoPE
        h = self.norm1(x) * (1 + sc_a) + s_a
        h = self.rope(h, positions)
        h = self.attn(h, h, h, need_weights=False)[0]
        x = x + g_a * h

        # Cross-attention to conditioning
        if cond_features is not None:
            h = self.norm_cross(x)
            cond = self.norm_cond(cond_features)
            h = self.cross_attn(h, cond, cond, need_weights=False)[0]
            x = x + h

        # MLP
        h = self.norm2(x) * (1 + sc_m) + s_m
        h = self.mlp(h)
        x = x + g_m * h

        return x


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        emb = t.float().unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.mlp(emb)


class RefinementDiT(nn.Module):
    """Stage 2 Refinement Diffusion Transformer.

    Takes coarse O-Voxel data and predicts refined SDF corrections.
    Trained with a diffusion process — at inference, iteratively denoises
    SDF corrections starting from noise.

    Args:
        voxel_dim: Dimension of input voxel features
        model_dim: Hidden dimension of the transformer
        num_heads: Number of attention heads
        num_layers: Number of DiT blocks
        cond_dim: Dimension of conditioning features (DINO)
        supervision_points: Number of SDF supervision points per object
    """

    def __init__(
        self,
        voxel_dim: int = 32,
        model_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 12,
        cond_dim: int = 768,  # DINO ViT-B feature dim
        supervision_points: int = 1_600_000,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.supervision_points = supervision_points

        # Input projection: voxel features -> model dim
        self.input_proj = nn.Linear(voxel_dim, model_dim)

        # Timestep embedding
        self.time_embed = TimestepEmbedding(model_dim)

        # Conditioning projection (DINO features -> model dim)
        self.cond_proj = nn.Linear(cond_dim, model_dim)

        # DiT blocks
        self.blocks = nn.ModuleList(
            [DiTBlock(model_dim, num_heads, model_dim) for _ in range(num_layers)]
        )

        # Output head: predict SDF correction
        self.output_norm = nn.LayerNorm(model_dim)
        self.output_proj = nn.Linear(model_dim, 1)  # SDF value

    def forward(
        self,
        coarse_voxels: torch.Tensor,
        positions: torch.Tensor,
        timestep: torch.Tensor,
        cond_features: torch.Tensor | None = None,
        noisy_sdf: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            coarse_voxels: (B, N, voxel_dim) coarse voxel features
            positions: (B, N, 3) spatial positions from coarse geometry
            timestep: (B,) diffusion timestep
            cond_features: (B, M, cond_dim) conditioning image features
            noisy_sdf: (B, N, 1) noisy SDF values (for diffusion training)

        Returns:
            (B, N, 1) predicted SDF corrections (noise prediction)
        """
        B, N, _ = coarse_voxels.shape

        # Project inputs
        x = self.input_proj(coarse_voxels)

        # Add noisy SDF if provided (for training)
        if noisy_sdf is not None:
            sdf_proj = nn.Linear(1, self.model_dim, device=x.device)
            x = x + sdf_proj(noisy_sdf)

        # Timestep conditioning
        t_emb = self.time_embed(timestep)

        # Project conditioning features
        if cond_features is not None:
            cond = self.cond_proj(cond_features)
        else:
            cond = None

        # DiT blocks
        for block in self.blocks:
            x = block(x, positions, t_emb, cond)

        # Output SDF correction
        x = self.output_norm(x)
        sdf_correction = self.output_proj(x)

        return sdf_correction

    def refine(
        self,
        coarse_voxels: torch.Tensor,
        positions: torch.Tensor,
        cond_features: torch.Tensor | None = None,
        num_steps: int = 50,
    ) -> torch.Tensor:
        """Run full diffusion inference to refine coarse voxels.

        Args:
            coarse_voxels: (B, N, voxel_dim) coarse voxel features
            positions: (B, N, 3) spatial positions
            cond_features: (B, M, cond_dim) conditioning features
            num_steps: Number of diffusion steps (50 default, 12 for fast)

        Returns:
            (B, N, 1) refined SDF values
        """
        B, N, _ = coarse_voxels.shape
        device = coarse_voxels.device

        # Start from noise
        sdf = torch.randn(B, N, 1, device=device)

        # Linear noise schedule
        timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

        for i in range(num_steps):
            t = timesteps[i].expand(B)
            t_next = timesteps[i + 1].expand(B)

            # Predict noise
            with torch.no_grad():
                noise_pred = self.forward(coarse_voxels, positions, t, cond_features, sdf)

            # DDPM step
            alpha_t = 1 - t.view(B, 1, 1)
            alpha_next = 1 - t_next.view(B, 1, 1)

            sdf = (sdf - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
            if i < num_steps - 1:
                noise = torch.randn_like(sdf)
                sdf = alpha_next.sqrt() * sdf + (1 - alpha_next).sqrt() * noise

        return sdf
