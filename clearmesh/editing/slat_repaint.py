#!/usr/bin/env python3
"""SLAT Repainting — Per-voxel feature repainting for edited regions.

After Voxel FlowEdit changes the structure, we need to update the
per-voxel features ({z_p}) to match. SLAT Repainting:
  - Edited voxels: regenerate features conditioned on target image
  - Unedited voxels: replay source trajectory to preserve identity

This is training-free — it uses TRELLIS.2's pretrained SLAT decoder
(the flow model for per-voxel features) with selective conditioning.

From Easy3E (arxiv 2602.21499v1), Section 3.3:
  "We design a repainting technique that ensures seamless integration
   of edited and unedited regions... edited voxels receive features
   generated from the target conditions, while unedited voxels replay
   their original trajectories."

Usage:
    repainter = SLATRepainter(flow_model=trellis2_slat_flow_model)
    new_features = repainter.repaint(
        edited_ss_latent=edited_structure,
        source_features=original_slat.shape_latent,
        edit_mask=mask,
        target_image=edit_image,
        source_image=source_render,
    )
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from PIL import Image


@dataclass
class RepaintConfig:
    """Configuration for SLAT Repainting."""

    num_steps: int = 25  # Flow ODE steps for feature generation
    guidance_scale: float = 7.5  # CFG scale
    blend_boundary: int = 2  # Voxel dilation for boundary blending
    boundary_noise_strength: float = 0.3  # Noise added at boundary for blending


class SLATRepainter:
    """Training-free per-voxel feature repainting.

    After structure editing, regenerates per-voxel features:
      - Edited region: run flow ODE conditioned on target image
      - Unedited region: replay source flow trajectory
      - Boundary: blend between edited and unedited with soft mask
    """

    def __init__(
        self,
        feature_flow_model: torch.nn.Module | None = None,
        device: str = "cuda",
        config: RepaintConfig | None = None,
    ):
        """Initialize SLATRepainter.

        Args:
            feature_flow_model: TRELLIS.2's SLAT feature flow model.
            device: Compute device.
            config: Repainting configuration.
        """
        self.feature_flow_model = feature_flow_model
        self.device = device
        self.config = config or RepaintConfig()

    def repaint(
        self,
        edited_ss_latent: torch.Tensor,
        source_features: torch.Tensor,
        edit_mask: torch.Tensor,
        target_image: Image.Image,
        source_image: Image.Image | None = None,
        voxel_indices: torch.Tensor | None = None,
        config: RepaintConfig | None = None,
    ) -> torch.Tensor:
        """Repaint per-voxel features after structure editing.

        Args:
            edited_ss_latent: Edited sparse structure latent (B, N_new, D_ss).
            source_features: Original per-voxel features (B, N_old, D_feat).
            edit_mask: Binary mask — 1=edited voxels, 0=preserved (N_new,).
            target_image: Target/edited image for conditioning.
            source_image: Original source image for unedited region conditioning.
            voxel_indices: Voxel positions (N_new, 3) for spatial operations.
            config: Override default config.

        Returns:
            Repainted features (B, N_new, D_feat).
        """
        cfg = config or self.config

        if self.feature_flow_model is None:
            raise RuntimeError(
                "Feature flow model not loaded. "
                "Pass feature_flow_model to constructor."
            )

        B, N_new, D_feat = source_features.shape[0], edited_ss_latent.shape[1], source_features.shape[2]
        device = edited_ss_latent.device

        # Create soft boundary mask for smooth blending
        soft_mask = self._create_soft_mask(edit_mask, voxel_indices, cfg.blend_boundary)

        # Generate features for edited region (target-conditioned)
        target_features = self._generate_features(
            edited_ss_latent,
            target_image,
            num_steps=cfg.num_steps,
            guidance_scale=cfg.guidance_scale,
        )

        # Replay source trajectory for unedited region
        source_replayed = self._replay_source_trajectory(
            edited_ss_latent,
            source_features,
            source_image,
            edit_mask,
            num_steps=cfg.num_steps,
        )

        # Blend: edited regions get target features, unedited get source
        soft_mask_expanded = soft_mask.unsqueeze(0).unsqueeze(-1)  # (1, N, 1)
        repainted = (
            soft_mask_expanded * target_features
            + (1 - soft_mask_expanded) * source_replayed
        )

        return repainted

    def _create_soft_mask(
        self,
        edit_mask: torch.Tensor,
        voxel_indices: torch.Tensor | None,
        boundary_width: int,
    ) -> torch.Tensor:
        """Create a soft mask with smooth boundary transition.

        Dilates the edit mask and creates a gradient at the boundary
        to avoid hard seams between edited and unedited regions.

        Args:
            edit_mask: Binary mask (N,).
            voxel_indices: Voxel positions (N, 3) for spatial dilation.
            boundary_width: Width of the boundary transition zone.

        Returns:
            Soft mask (N,) with values in [0, 1].
        """
        if voxel_indices is None or boundary_width == 0:
            return edit_mask.float()

        # Simple approach: for each unedited voxel near the boundary,
        # compute distance to nearest edited voxel and create gradient
        soft_mask = edit_mask.float().clone()

        edited_positions = voxel_indices[edit_mask > 0.5]  # (M, 3)
        unedited_positions = voxel_indices[edit_mask < 0.5]  # (K, 3)

        if len(edited_positions) == 0 or len(unedited_positions) == 0:
            return soft_mask

        # Compute pairwise distances (K, M)
        dists = torch.cdist(
            unedited_positions.float().unsqueeze(0),
            edited_positions.float().unsqueeze(0),
        )[0]  # (K, M)
        min_dists = dists.min(dim=1)[0]  # (K,)

        # Create gradient for nearby unedited voxels
        boundary_mask = min_dists < boundary_width
        gradient = 1.0 - min_dists[boundary_mask] / boundary_width
        gradient = gradient.clamp(0, 1)

        # Update soft mask
        unedited_indices = torch.where(edit_mask < 0.5)[0]
        soft_mask[unedited_indices[boundary_mask]] = gradient

        return soft_mask

    def _generate_features(
        self,
        ss_latent: torch.Tensor,
        target_image: Image.Image,
        num_steps: int,
        guidance_scale: float,
    ) -> torch.Tensor:
        """Generate per-voxel features conditioned on target image.

        Runs the feature flow model from noise to clean features,
        conditioned on the edited structure and target image.

        Args:
            ss_latent: Edited sparse structure latent (B, N, D_ss).
            target_image: Target image for conditioning.
            num_steps: ODE integration steps.
            guidance_scale: CFG scale.

        Returns:
            Generated features (B, N, D_feat).
        """
        # TODO: Implement feature generation using TRELLIS.2's flow model
        # 1. Encode target_image with DINOv2
        # 2. Start from noise z_0 ~ N(0, I)
        # 3. Integrate flow ODE: dz/dt = v(z_t, t, cond, ss_latent)
        # 4. Return z_1 (clean features)
        raise NotImplementedError(
            "Feature generation requires TRELLIS.2's SLAT feature flow model."
        )

    def _replay_source_trajectory(
        self,
        ss_latent: torch.Tensor,
        source_features: torch.Tensor,
        source_image: Image.Image | None,
        edit_mask: torch.Tensor,
        num_steps: int,
    ) -> torch.Tensor:
        """Replay source trajectory for unedited voxels.

        For identity preservation: unedited voxels follow their
        original flow trajectory to maintain consistent features.

        Args:
            ss_latent: Current structure latent.
            source_features: Original per-voxel features.
            source_image: Original source image.
            edit_mask: Binary mask (1=edited).
            num_steps: ODE steps.

        Returns:
            Replayed features (B, N, D_feat).
        """
        # TODO: Implement source trajectory replay
        # For now, just return source features (identity)
        # The full implementation would:
        # 1. Forward-diffuse source features to t_start
        # 2. Integrate reverse ODE with source conditioning
        # 3. This ensures consistent denoising path

        # Pad or truncate source features to match new structure size
        B, N_new, _ = ss_latent.shape
        N_old, D_feat = source_features.shape[1], source_features.shape[2]

        if N_new == N_old:
            return source_features
        elif N_new < N_old:
            return source_features[:, :N_new, :]
        else:
            # Pad with zeros for new voxels
            padding = torch.zeros(
                B, N_new - N_old, D_feat, device=ss_latent.device
            )
            return torch.cat([source_features, padding], dim=1)
