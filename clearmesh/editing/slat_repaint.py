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
        pipeline=None,
        sampler_params: dict | None = None,
        device: str = "cuda",
        config: RepaintConfig | None = None,
    ):
        """Initialize SLATRepainter.

        Args:
            feature_flow_model: TRELLIS.2's SLAT feature flow model
                (``shape_slat_flow_model_1024``). If None, we fall back to
                the flow model inside ``pipeline.models``.
            pipeline: A loaded ``Trellis2ImageTo3DPipeline``. Required for
                image preprocessing + ``sample_shape_slat``.
            sampler_params: Override sampler params. Defaults to the
                TRELLIS.2 pipeline's configured shape-SLAT sampler params
                — typically ``{"steps": 12, "guidance_strength": 4.5}``
                (matches ``generate_slat_pairs.py:60``).
            device: Compute device.
            config: Repainting configuration.
        """
        self.feature_flow_model = feature_flow_model
        self.pipeline = pipeline
        self.sampler_params = sampler_params or {
            "steps": 12,
            "guidance_strength": 4.5,
        }
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
            edited_ss_latent: Edited voxel positions as (N_new, 3) int32
                or integer tensor — ClearMesh's SS latent is the occupied
                voxel coords (see ``slat_encoder.py`` module docstring).
            source_features: Original per-voxel features (N_old, D_feat)
                or (B, N_old, D_feat).
            edit_mask: Binary mask — 1=edited voxels, 0=preserved (N_new,).
            target_image: Target/edited image for conditioning.
            source_image: Original source image for unedited-region
                conditioning.
            voxel_indices: Voxel positions (N_new, 3) — used for boundary
                dilation. Passed separately when ``edited_ss_latent`` is
                not the positions themselves.
            config: Override default config.

        Returns:
            Repainted features with the same dims as ``source_features``
            but with N = N_new voxels.
        """
        cfg = config or self.config

        if self.feature_flow_model is None and self.pipeline is None:
            raise RuntimeError(
                "Feature flow model not loaded. Pass either "
                "feature_flow_model=... or pipeline=... to the constructor."
            )

        # Normalize source_features to (N_old, D_feat).
        if source_features.dim() == 3:
            source_features_2d = source_features.squeeze(0)
        else:
            source_features_2d = source_features

        N_old, D_feat = source_features_2d.shape
        N_new = edit_mask.shape[0]
        device = edit_mask.device

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

        # Blend: edited regions get target features, unedited get source.
        # target_features and source_replayed are (N_new, D_feat) —
        # broadcast over the soft mask on voxel axis.
        soft_mask_expanded = soft_mask.unsqueeze(-1)  # (N_new, 1)
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

        Runs ``pipeline.sample_shape_slat(cond, flow_model, coords, params)``
        — the same method used in
        ``scripts/data/generate_slat_pairs.py:185``. This integrates the
        flow ODE internally, going from noise to clean features at the
        given voxel positions.

        Args:
            ss_latent: Edited voxel positions (N, 3) int — TRELLIS.2 will
                batch-prepend a zero for the batch dim internally.
            target_image: Target image for conditioning.
            num_steps: Overrides ``sampler_params['steps']``.
            guidance_scale: Overrides ``sampler_params['guidance_strength']``.

        Returns:
            Generated features (N, D_feat).
        """
        if self.pipeline is None:
            raise RuntimeError(
                "_generate_features requires pipeline=... Pass a loaded "
                "Trellis2ImageTo3DPipeline to the SLATRepainter constructor."
            )

        # 1. Image → DINOv2 conditioning (matches pipeline.get_cond pattern).
        processed = self.pipeline.preprocess_image(target_image)
        cond = self.pipeline.get_cond([processed], 1024)

        # 2. Prepare coords in TRELLIS.2's (B, X, Y, Z) int32 convention
        #    (same format as the coords object from sample_sparse_structure).
        coords = self._as_coords_tensor(ss_latent)

        # 3. Pick the feature flow model.
        flow_model = self.feature_flow_model
        if flow_model is None:
            # Prefer 1024 (fine) for highest-quality feature generation.
            flow_model = (
                self.pipeline.models.get("shape_slat_flow_model_1024")
                or self.pipeline.models.get("shape_slat_flow_model_512")
            )
            if flow_model is None:
                raise RuntimeError(
                    "No shape_slat_flow_model found in pipeline.models. "
                    f"Available: {list(self.pipeline.models.keys())}"
                )

        # 4. Sample.
        params = dict(self.sampler_params)
        params["steps"] = num_steps
        params["guidance_strength"] = guidance_scale

        with torch.no_grad():
            shape_slat = self.pipeline.sample_shape_slat(
                cond, flow_model, coords, params,
            )

        # 5. Extract (N, D) features (matches extract_slat_feats in
        #    generate_slat_pairs.py:112).
        if hasattr(shape_slat, "feats"):
            return shape_slat.feats.float()
        if hasattr(shape_slat, "F"):
            return shape_slat.F.float()
        if isinstance(shape_slat, torch.Tensor):
            return shape_slat.squeeze(0).float() if shape_slat.dim() == 3 else shape_slat.float()
        raise RuntimeError(f"Unknown shape_slat type: {type(shape_slat)}")

    @staticmethod
    def _as_coords_tensor(ss_latent: torch.Tensor) -> torch.Tensor:
        """Normalize voxel indices to TRELLIS.2's (N, 4) coords tensor.

        TRELLIS.2's sampler expects coords as ``(N, 4)`` int32 with column
        0 = batch index. Accept either ``(N, 3)`` or ``(N, 4)``.
        """
        if ss_latent.dim() == 2 and ss_latent.shape[1] == 3:
            n = ss_latent.shape[0]
            batch_idx = torch.zeros(n, 1, dtype=torch.int32, device=ss_latent.device)
            return torch.cat([batch_idx, ss_latent.int()], dim=1)
        if ss_latent.dim() == 2 and ss_latent.shape[1] == 4:
            return ss_latent.int()
        raise ValueError(
            f"edited_ss_latent must be (N, 3) or (N, 4) int coords, got {tuple(ss_latent.shape)}"
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

        Strict Easy3E (§3.3) integrates a reverse ODE from noise using the
        source image + source positions as conditioning, so unedited voxels
        retain their original denoising path. That requires per-step access
        to the feature flow model's velocity, which the public
        ``pipeline.sample_shape_slat`` does not expose.

        Implementation here uses the next-best thing: **identity replay**
        with shape reconciliation. When ``N_new == N_old`` (structure
        unchanged — the most common case) this is exact. When the
        structure changed, we align by:
          - truncating if N_new < N_old
          - re-sampling the new voxels conditioned on source_image if
            N_new > N_old (requires pipeline)

        A full trajectory replay would improve boundary continuity at the
        cost of one extra flow-model sample. Tracked in STATUS.md.

        Args:
            ss_latent: Edited voxel coords (N_new, 3) or (N_new, 4) int.
            source_features: Original per-voxel features
                (N_old, D_feat) or (B, N_old, D_feat).
            source_image: Original source image (for re-sampling new voxels).
            edit_mask: Binary mask (N_new,) — 1=edited.
            num_steps: ODE steps (forwarded to re-sampling when needed).

        Returns:
            Replayed features (N_new, D_feat).
        """
        # Normalize source_features to (N_old, D_feat).
        if source_features.dim() == 3:
            src = source_features.squeeze(0)
        else:
            src = source_features

        N_old, D_feat = src.shape
        N_new = edit_mask.shape[0]
        device = src.device

        if N_new == N_old:
            return src

        if N_new < N_old:
            # Truncate — prefer keeping the voxels that line up by index
            # with the edited structure (caller is responsible for
            # producing ss_latent in a stable order).
            return src[:N_new]

        # N_new > N_old: new voxels appeared. Re-sample just for those
        # using the source image as conditioning so identity is preserved
        # over the original region and new cells are filled coherently.
        if source_image is not None and self.pipeline is not None:
            resampled = self._generate_features(
                ss_latent, source_image,
                num_steps=num_steps,
                guidance_scale=self.sampler_params.get("guidance_strength", 4.5),
            )
            # Overwrite old positions with exact source features
            # (alignment assumes matching voxel ordering up to N_old).
            resampled = resampled.clone()
            resampled[:N_old] = src
            return resampled

        # Fallback: zero-pad. The repainter's soft_mask blend will zero
        # these out in unedited regions anyway, but this is lossy.
        padding = torch.zeros(N_new - N_old, D_feat, device=device, dtype=src.dtype)
        return torch.cat([src, padding], dim=0)
