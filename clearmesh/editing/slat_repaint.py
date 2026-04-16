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
        pipeline=None,
    ):
        """Initialize SLATRepainter.

        Args:
            feature_flow_model: TRELLIS.2's SLAT feature flow model
                (``shape_slat_flow_model_512`` under ``pipeline.models``).
                If None and ``pipeline`` is provided, resolved from
                the pipeline automatically.
            device: Compute device.
            config: Repainting configuration.
            pipeline: Optional ``Trellis2ImageTo3DPipeline`` for image
                conditioning and for resolving the flow model.
        """
        self.feature_flow_model = feature_flow_model
        self.device = device
        self.config = config or RepaintConfig()
        self._pipeline = pipeline

        if self.feature_flow_model is None and pipeline is not None:
            models = getattr(pipeline, "models", {})
            for key in ("shape_slat_flow_model_512", "shape_slat_flow_model", "slat_flow_model"):
                if key in models:
                    self.feature_flow_model = models[key]
                    break

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
        coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate per-voxel features via the feature flow ODE.

        Matches the pattern used by ``Trellis2ImageTo3DPipeline.sample_shape_slat``
        (seen at [generate_pairs.py:985-997](scripts/data/generate_pairs.py:985)):

            cond = pipeline.get_cond([img], 512)
            slat = pipeline.sample_shape_slat(
                cond,
                pipeline.models["shape_slat_flow_model_512"],
                coords,  # from edited SS latent, in (B, N, 3) or (B, N, 4) format
                params,  # {"steps": ..., "guidance_strength": ...}
            )

        We delegate to the pipeline's sampler because CFG + guidance-interval
        scheduling is baked in there. Trying to reproduce it by calling the
        flow model manually would require mirroring `FlowEulerGuidanceIntervalSampler`.

        Args:
            ss_latent: Edited sparse structure latent (B, N, D_ss).
            target_image: Target image for conditioning.
            num_steps: ODE integration steps.
            guidance_scale: CFG scale (``guidance_strength`` in TRELLIS.2 terms).
            coords: Optional (B, N, 3) voxel coords. If None, derived from
                ``ss_latent`` geometry (caller is responsible for ensuring
                the SS latent has an associated coord layout).

        Returns:
            Generated features (B, N, D_feat). D_feat is 32 for the default
            TRELLIS.2-4B config.
        """
        if self.feature_flow_model is None:
            raise RuntimeError(
                "feature_flow_model not set. Pass feature_flow_model=... "
                "to SLATRepainter(...) or pass a pipeline."
            )
        if self._pipeline is None:
            raise RuntimeError(
                "SLATRepainter requires a pipeline for image conditioning. "
                "Pass pipeline=... in the constructor."
            )

        pipe = self._pipeline
        processed = pipe.preprocess_image(target_image) if hasattr(pipe, "preprocess_image") else target_image
        cond = pipe.get_cond([processed], 512)

        if coords is None:
            coords = self._derive_coords_from_ss_latent(ss_latent)

        sampler_params = {"steps": num_steps, "guidance_strength": guidance_scale}

        # sample_shape_slat's exact signature (per generate_pairs.py:985):
        #   pipeline.sample_shape_slat(cond, flow_model, coords, params)
        slat = pipe.sample_shape_slat(
            cond,
            self.feature_flow_model,
            coords,
            sampler_params,
        )

        # Sampler returns either a SparseTensor (with .feats) or a tensor
        if hasattr(slat, "feats"):
            feats = slat.feats
            if feats.dim() == 2:  # (N, D) → (B, N, D)
                feats = feats.unsqueeze(0)
            return feats
        if isinstance(slat, torch.Tensor):
            return slat.unsqueeze(0) if slat.dim() == 2 else slat
        raise TypeError(f"Unexpected sample_shape_slat return type: {type(slat).__name__}")

    @staticmethod
    def _derive_coords_from_ss_latent(ss_latent):
        """Extract voxel coords from a SparseTensor SS latent or return
        a best-effort placeholder for dense tensors.

        The feature flow model needs to know which voxels are occupied;
        for a SparseTensor input that's already in .coords. For a dense
        tensor we can't recover this cheaply — the caller should pass
        coords explicitly.
        """
        if hasattr(ss_latent, "coords"):
            return ss_latent.coords
        raise ValueError(
            "Could not derive coords from ss_latent. Pass coords= explicitly "
            "to _generate_features when ss_latent is a dense tensor."
        )

    def _replay_source_trajectory(
        self,
        ss_latent: torch.Tensor,
        source_features: torch.Tensor,
        source_image: Image.Image | None,
        edit_mask: torch.Tensor,
        num_steps: int,
    ) -> torch.Tensor:
        """Replay the source flow trajectory for the post-edit voxel set.

        Two cases:
          A. The structure didn't change (N_new == N_old): return source
             features directly. No replay needed.
          B. The structure changed (voxels added/removed): the new voxel
             set has no direct correspondence to the source features.
             We regenerate features for the new structure using the
             **source** image as conditioning (not the target), so the
             unedited regions end up with features consistent with the
             source identity rather than the target.

        This is the key insight in Easy3E §3.3: unedited voxels get
        "replayed" features, meaning they re-run the same flow ODE that
        originally produced them — conditioned on the source image —
        so the final per-voxel features are consistent with the source.

        When ``source_image`` is None we fall back to the pad-or-truncate
        heuristic (the behavior from the old stub) with a warning.

        Args:
            ss_latent: Current (edited) structure latent.
            source_features: Original per-voxel features (B, N_old, D).
            source_image: Original source image for conditioning. If None,
                falls back to pad/truncate.
            edit_mask: Binary mask (1=edited). Unused in replay but
                available for callers that want to debug the blend.
            num_steps: ODE steps for the replay.

        Returns:
            Replayed features (B, N_new, D_feat).
        """
        B = source_features.shape[0] if source_features.dim() == 3 else 1
        N_new = ss_latent.shape[1] if ss_latent.dim() >= 2 else ss_latent.shape[0]
        N_old = source_features.shape[1] if source_features.dim() == 3 else source_features.shape[0]
        D_feat = source_features.shape[-1]

        # Case A: structure unchanged — identity is cheapest and best
        if N_new == N_old:
            return source_features

        # Case B: structure changed — need a true replay
        if source_image is None or self._pipeline is None:
            import warnings
            warnings.warn(
                f"[slat_repaint] No source_image/pipeline for replay "
                f"(N_old={N_old}, N_new={N_new}); falling back to pad/truncate. "
                f"Edits at the structure boundary may show seams."
            )
            if N_new < N_old:
                return source_features[:, :N_new, :] if source_features.dim() == 3 else source_features[:N_new]
            # Pad with zeros (neutral features)
            padding = torch.zeros(B, N_new - N_old, D_feat, device=source_features.device, dtype=source_features.dtype)
            if source_features.dim() == 2:
                source_features = source_features.unsqueeze(0)
            return torch.cat([source_features, padding], dim=1)

        # True replay: regenerate features for the new structure, conditioned on source image
        coords = self._derive_coords_from_ss_latent(ss_latent) if hasattr(ss_latent, 'coords') else None
        return self._generate_features(
            ss_latent=ss_latent,
            target_image=source_image,  # source, not target — this is the "replay" bit
            num_steps=num_steps,
            guidance_scale=self.config.guidance_scale,
            coords=coords,
        )
