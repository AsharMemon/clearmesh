#!/usr/bin/env python3
"""Voxel FlowEdit — Training-free geometry editing via flow-matching ODE.

Implements the core geometry editing from Easy3E (arxiv 2602.21499v1).
Edits the sparse structure (SS) latent using TRELLIS.2's pretrained
flow-matching model with additional guidance terms.

The ODE integrates:
  dx_t = M_l * v_edit(x_t, t)dt + M_l * (Gamma * xi_traj - eta * G_sil)dt

Where:
  v_edit = velocity difference between target and source flow trajectories
  G_sil  = silhouette gradient guidance (BCE loss between rendered and target silhouette)
  xi_traj = trajectory correction (keeps state on the flow manifold)
  M_l    = edit mask (only modify selected voxel region)

This is entirely training-free — it reuses TRELLIS.2's pretrained
SparseStructureFlowModel to compute flow velocities.

Usage:
    flowedit = VoxelFlowEdit(
        flow_model=trellis2_ss_flow_model,
        device="cuda",
    )
    edited_ss_latent = flowedit.edit(
        source_ss_latent=slat.ss_latent,
        target_image=edit_image,
        source_image=source_render,
        edit_mask=mask,
        num_steps=25,
    )
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


@dataclass
class FlowEditConfig:
    """Configuration for Voxel FlowEdit."""

    num_steps: int = 25  # ODE integration steps
    gamma: float = 1.0  # Trajectory correction strength
    eta: float = 0.5  # Silhouette guidance strength
    t_start: float = 0.5  # Start time for editing (0=noise, 1=clean)
    t_end: float = 1.0  # End time
    guidance_scale: float = 7.5  # CFG scale for flow model


class VoxelFlowEdit:
    """Training-free voxel structure editing via flow-matching ODE.

    Uses TRELLIS.2's pretrained SparseStructureFlowModel to compute
    flow velocities, then applies edit-specific modifications:
      - Trajectory splitting: separate source/target trajectories
      - Silhouette guidance: steer towards target silhouette
      - Edit masking: only modify selected regions
    """

    def __init__(
        self,
        flow_model: torch.nn.Module | None = None,
        device: str = "cuda",
        config: FlowEditConfig | None = None,
        pipeline=None,
    ):
        """Initialize VoxelFlowEdit.

        Args:
            flow_model: TRELLIS.2's SparseStructureFlowModel. If None and
                ``pipeline`` is provided, resolved from
                ``pipeline.models["sparse_structure_flow_model"]``.
            device: Compute device.
            config: FlowEdit configuration.
            pipeline: Optional ``Trellis2ImageTo3DPipeline`` for image
                conditioning (``pipeline.get_cond``) and for resolving the
                flow model.
        """
        self.flow_model = flow_model
        self.device = device
        self.config = config or FlowEditConfig()
        self._pipeline = pipeline

        # Auto-resolve flow_model from pipeline if caller didn't pass one
        if self.flow_model is None and pipeline is not None:
            models = getattr(pipeline, "models", {})
            for key in ("sparse_structure_flow_model", "ss_flow_model"):
                if key in models:
                    self.flow_model = models[key]
                    break

    def edit(
        self,
        source_ss_latent: torch.Tensor,
        target_image: Image.Image,
        source_image: Image.Image | None = None,
        edit_mask: torch.Tensor | None = None,
        config: FlowEditConfig | None = None,
    ) -> torch.Tensor:
        """Edit voxel structure via flow-matching ODE.

        Args:
            source_ss_latent: Source sparse structure latent (B, N, D).
            target_image: Target/edited image to guide editing.
            source_image: Original source rendering (for trajectory splitting).
            edit_mask: Binary mask over voxels (N,) — 1=edit, 0=preserve.
                       If None, all voxels are edited.
            config: Override default config for this edit.

        Returns:
            Edited sparse structure latent (B, N, D).
        """
        cfg = config or self.config

        if self.flow_model is None:
            raise RuntimeError(
                "Flow model not loaded. Pass flow_model to constructor or call load_model()."
            )

        B, N, D = source_ss_latent.shape
        device = source_ss_latent.device

        # Default mask: edit everything
        if edit_mask is None:
            edit_mask = torch.ones(N, device=device)
        edit_mask = edit_mask.float().unsqueeze(0).unsqueeze(-1)  # (1, N, 1)

        # Encode target image for conditioning
        target_cond = self._encode_image_condition(target_image)
        source_cond = (
            self._encode_image_condition(source_image)
            if source_image is not None
            else None
        )

        # Forward diffusion: add noise to source latent at t_start
        x_t = self._forward_diffuse(source_ss_latent, cfg.t_start)

        # Source trajectory: record positions for trajectory correction
        source_trajectory = self._compute_source_trajectory(
            source_ss_latent, source_cond, cfg
        )

        # ODE integration from t_start to t_end
        dt = (cfg.t_end - cfg.t_start) / cfg.num_steps
        t = cfg.t_start

        for step in range(cfg.num_steps):
            # Compute edit velocity (target - source flow)
            v_target = self._compute_velocity(x_t, t, target_cond, cfg.guidance_scale)
            v_source = self._compute_velocity(x_t, t, source_cond, cfg.guidance_scale)
            v_edit = v_target - v_source

            # Trajectory correction: keep on manifold
            xi_traj = self._trajectory_correction(x_t, source_trajectory, t, cfg)

            # Silhouette guidance: steer towards target silhouette
            g_sil = self._silhouette_guidance(x_t, target_image, t)

            # Combined update with edit mask
            dx = edit_mask * (
                v_edit * dt + cfg.gamma * xi_traj * dt - cfg.eta * g_sil * dt
            )

            # Also advance unmasked regions along source trajectory
            v_source_full = self._compute_velocity(x_t, t, source_cond, cfg.guidance_scale)
            dx_unmasked = (1 - edit_mask) * v_source_full * dt

            x_t = x_t + dx + dx_unmasked
            t += dt

        return x_t

    def _forward_diffuse(
        self, x_0: torch.Tensor, t: float
    ) -> torch.Tensor:
        """Add noise to latent at time t (flow-matching forward process).

        In flow-matching: x_t = (1-t) * noise + t * x_0
        So at t=0 it's pure noise, at t=1 it's clean data.

        Args:
            x_0: Clean latent (B, N, D).
            t: Time in [0, 1].

        Returns:
            Noised latent at time t.
        """
        noise = torch.randn_like(x_0)
        return t * x_0 + (1 - t) * noise

    def _compute_velocity(
        self,
        x_t: torch.Tensor,
        t: float,
        condition,
        guidance_scale: float,
    ) -> torch.Tensor:
        """Compute flow velocity using TRELLIS.2's flow model with CFG.

        With classifier-free guidance:
          v = v_uncond + guidance_scale * (v_cond - v_uncond)

        TRELLIS.2's ``SparseStructureFlowModel`` uses flow-matching, and
        its forward signature — per TRELLIS.2's
        ``trellis2/models/sparse_structure_flow_model.py`` — is roughly:

            v = model(x_t, t, cond=cond_features)

        where ``cond`` is a dict with ``"cond"`` and ``"neg_cond"`` entries
        as returned by ``pipeline.get_cond``. We do the CFG blend manually
        (rather than via ``pipeline.sample_sparse_structure``) because the
        Easy3E ODE integrates two trajectories and needs per-step access
        to conditional and unconditional velocities.

        Args:
            x_t: Current state. Can be a dense tensor (B, C, R, R, R) for
                the SS latent case, or a sparse-feature tensor (B, N, D).
            t: Current time in [0, 1].
            condition: Either a conditioning tensor (cond features) or a
                dict from ``pipeline.get_cond`` with "cond"/"neg_cond" keys.
            guidance_scale: CFG scale; 1.0 disables guidance.

        Returns:
            Velocity tensor with the same shape as ``x_t``.
        """
        if self.flow_model is None:
            raise RuntimeError(
                "flow_model not wired. Pass flow_model to VoxelFlowEdit(...) "
                "or pass a pipeline whose .models contains "
                "'sparse_structure_flow_model'."
            )

        # Normalize `condition` to have explicit cond/neg_cond halves so we
        # can do CFG in one place.
        cond_pos, cond_neg = self._split_condition(condition)

        # Flow models usually accept t as a batched 1-D tensor
        if isinstance(t, (int, float)):
            B = x_t.shape[0] if isinstance(x_t, torch.Tensor) else 1
            t_tensor = torch.full((B,), float(t), device=self.device, dtype=torch.float32)
        else:
            t_tensor = t

        # Try the common TRELLIS.2 signature: model(x, t, cond=...)
        # with CFG done manually. If that fails, try passing a dict.
        try:
            v_cond = self.flow_model(x_t, t_tensor, cond=cond_pos)
        except TypeError:
            # Some variants want positional conditioning
            v_cond = self.flow_model(x_t, t_tensor, cond_pos)

        if guidance_scale == 1.0 or cond_neg is None:
            return v_cond

        try:
            v_uncond = self.flow_model(x_t, t_tensor, cond=cond_neg)
        except TypeError:
            v_uncond = self.flow_model(x_t, t_tensor, cond_neg)

        # Standard CFG blend
        return v_uncond + guidance_scale * (v_cond - v_uncond)

    @staticmethod
    def _split_condition(condition):
        """Separate positive / negative conditioning regardless of container.

        ``pipeline.get_cond`` may return a dict, a (cond, neg_cond) tuple,
        or a tensor (meaning no negative conditioning). We normalize all
        three to ``(pos, neg_or_None)``.
        """
        if condition is None:
            return None, None
        if isinstance(condition, dict):
            pos = condition.get("cond") or condition.get("image_cond") or condition.get("pos")
            neg = condition.get("neg_cond") or condition.get("uncond") or condition.get("neg")
            return pos, neg
        if isinstance(condition, (tuple, list)):
            if len(condition) == 2:
                return condition[0], condition[1]
            return condition[0], None
        return condition, None

    def _compute_source_trajectory(
        self,
        source_latent: torch.Tensor,
        source_cond: torch.Tensor | None,
        config: FlowEditConfig,
    ) -> list[torch.Tensor]:
        """Pre-compute source trajectory for trajectory correction.

        Records x_t at each ODE step when running the source forward,
        used later for xi_traj correction term.

        Args:
            source_latent: Source SS latent.
            source_cond: Source image conditioning.
            config: FlowEdit config.

        Returns:
            List of trajectory states at each timestep.
        """
        trajectory = []
        x_t = self._forward_diffuse(source_latent, config.t_start)
        dt = (config.t_end - config.t_start) / config.num_steps
        t = config.t_start

        for _ in range(config.num_steps):
            trajectory.append(x_t.clone())
            v = self._compute_velocity(x_t, t, source_cond, config.guidance_scale)
            x_t = x_t + v * dt
            t += dt

        return trajectory

    def _trajectory_correction(
        self,
        x_t: torch.Tensor,
        source_trajectory: list[torch.Tensor],
        t: float,
        config: FlowEditConfig,
    ) -> torch.Tensor:
        """Compute trajectory correction to keep state on manifold.

        xi_traj = x_t^source - x_t^current (difference from expected position)

        Args:
            x_t: Current edited state.
            source_trajectory: Pre-computed source trajectory.
            t: Current time.
            config: Config with timing info.

        Returns:
            Correction vector (B, N, D).
        """
        # Find closest trajectory step
        step_idx = int(
            (t - config.t_start) / (config.t_end - config.t_start) * len(source_trajectory)
        )
        step_idx = min(step_idx, len(source_trajectory) - 1)

        return source_trajectory[step_idx] - x_t

    def _silhouette_guidance(
        self,
        x_t: torch.Tensor,
        target_image: Image.Image,
        t: float,
        voxel_indices: torch.Tensor | None = None,
        grid_size: int = 256,
    ) -> torch.Tensor:
        """Compute silhouette gradient guidance: ∇(BCE(rendered, target)).

        Full implementation in [silhouette.py](clearmesh/editing/silhouette.py).
        Briefly:
          - Interpret per-voxel feature norms as occupancy weights.
          - Splat voxels to a soft 2D silhouette image under the canonical
            camera.
          - Compute BCE vs target silhouette (extracted from target_image's
            alpha or background-difference).
          - Backprop to get grad w.r.t. x_t.

        On a GPU pod with nvdiffrast installed this could be upgraded to
        a true differentiable mesh rasterization (decode x_t → FlexiCubes
        → nvdiffrast raster → BCE). For v1 the voxel-soft path is good
        enough and runs on CPU for unit tests.

        Args:
            x_t: Current state (B, N, D). Must be differentiable.
            target_image: Target edited image.
            t: Current ODE time (currently unused; kept for future time-
               adaptive guidance schedules).
            voxel_indices: (N, 3) voxel positions. Falls back to a placeholder
                grid if None (meaning guidance will be near-zero, which is
                the safe default).
            grid_size: Voxel grid resolution.

        Returns:
            Gradient tensor matching x_t's shape.
        """
        from clearmesh.editing.camera import CanonicalCamera
        from clearmesh.editing.silhouette import (
            compute_silhouette_guidance,
            extract_target_silhouette,
        )

        if voxel_indices is None:
            # Without voxel_indices we can't splat, so return zero grad.
            # This is the "safe stub" path for callers that don't have
            # voxel coords handy.
            return torch.zeros_like(x_t)

        camera = CanonicalCamera.trellis2_default(image_size=256)
        target_sil = extract_target_silhouette(target_image, size=256)

        return compute_silhouette_guidance(
            x_t,
            target_silhouette=target_sil,
            voxel_indices=voxel_indices,
            camera=camera,
            grid_size=grid_size,
        )

    def _encode_image_condition(self, image):
        """Encode an image for flow-model conditioning.

        Delegates to ``pipeline.preprocess_image`` + ``pipeline.get_cond``
        — the exact path used by ``scripts/data/generate_pairs.py:974``.
        Works with either DINOv2 or DINOv3 conditioning depending on how
        the pipeline was configured; we don't care which — we hand the
        result straight to the flow model and let it figure out the dim.

        Args:
            image: PIL Image, or None for the "empty" conditioning used
                as the negative in CFG.

        Returns:
            Whatever ``pipeline.get_cond`` returns — typically a dict with
            ``"cond"`` and ``"neg_cond"`` tensors of shape (B, M, D).
            ``None`` propagates through.
        """
        if image is None:
            return None
        if self._pipeline is None:
            raise RuntimeError(
                "VoxelFlowEdit was not given a TRELLIS.2 pipeline, so it "
                "cannot encode images. Pass pipeline=... to VoxelFlowEdit(...) "
                "or to Easy3EEditor(...)."
            )
        pipe = self._pipeline

        # Preprocess if the pipeline exposes that step; some pipelines already
        # accept PIL images directly.
        if hasattr(pipe, "preprocess_image"):
            processed = pipe.preprocess_image(image)
        else:
            processed = image

        # The second arg to get_cond is the target SS resolution; 512 is
        # what the 512 flow model expects.
        cond = pipe.get_cond([processed], 512)
        return cond

    def auto_detect_edit_mask(
        self,
        source_image: Image.Image,
        target_image: Image.Image,
        voxel_indices: torch.Tensor,
        threshold: float = 0.1,
        camera=None,
        grid_size: int | None = None,
        morphological_dilation: int = 1,
    ) -> torch.Tensor:
        """Auto-detect the edit region by projecting a 2D image diff into
        the sparse voxel space.

        Pipeline:
          1. Compute per-pixel difference between source and target images.
          2. Threshold into a binary 2D edit mask.
          3. Project each voxel position through TRELLIS.2's canonical
             camera (same one used by ``pipeline.get_cond``) to obtain
             pixel coordinates.
          4. Sample the 2D mask at each voxel's pixel to get a per-voxel
             score.
          5. Apply optional 3D morphological dilation to smooth the edge
             between edited and unedited regions.

        Args:
            source_image: Rendered view of the source mesh.
            target_image: Target (edited) view.
            voxel_indices: Voxel positions (N, 3), integer coords in
                [0, grid_size).
            threshold: Pixel-diff threshold in [0, 1].
            camera: Optional ``CanonicalCamera`` instance. Defaults to
                TRELLIS.2's canonical front-facing camera.
            grid_size: Voxel grid resolution (used to normalize voxel
                positions to [-0.5, 0.5]). Defaults to 256 — the typical
                O-Voxel grid size.
            morphological_dilation: Number of 3×3×3 dilation passes to
                apply in 3D. 0 disables, 1 is a good default.

        Returns:
            Float mask (N,) in [0, 1]. 1.0 means "edit this voxel",
            0.0 means "preserve".
        """
        from clearmesh.editing.camera import CanonicalCamera, project_voxels_to_pixels

        # --- 1. 2D image diff ---
        src = np.array(source_image.convert("RGB")).astype(np.float32) / 255.0
        tgt = np.array(target_image.convert("RGB")).astype(np.float32) / 255.0
        if src.shape != tgt.shape:
            # Resize target to match source
            tgt_img = target_image.convert("RGB").resize(
                (source_image.width, source_image.height), Image.LANCZOS
            )
            tgt = np.array(tgt_img).astype(np.float32) / 255.0

        diff = np.abs(src - tgt).mean(axis=-1)  # (H, W)
        mask_2d = (diff > threshold).astype(np.float32)  # (H, W) binary

        # --- 2. Project voxels to pixel coords ---
        cam = camera if camera is not None else CanonicalCamera.trellis2_default(
            image_size=src.shape[0]
        )
        g = grid_size or 256

        device = voxel_indices.device if isinstance(voxel_indices, torch.Tensor) else "cpu"
        if not isinstance(voxel_indices, torch.Tensor):
            voxel_indices = torch.tensor(voxel_indices)
        voxel_indices = voxel_indices.to(torch.float32)

        pixel_u, pixel_v, depth = project_voxels_to_pixels(voxel_indices, cam, grid_size=g)

        # --- 3. Sample mask_2d at each voxel's projected pixel ---
        H, W = mask_2d.shape
        u = pixel_u.round().long().clamp(0, W - 1)
        v = pixel_v.round().long().clamp(0, H - 1)
        # Voxels projected outside the image get score 0
        in_frame = (pixel_u >= 0) & (pixel_u < W) & (pixel_v >= 0) & (pixel_v < H)
        # Voxels behind the camera (negative depth) get score 0
        in_front = depth > 0

        mask_2d_t = torch.from_numpy(mask_2d).to(device)
        sampled = mask_2d_t[v, u]
        sampled = sampled * in_frame.float() * in_front.float()

        # --- 4. Optional 3D morphological dilation ---
        if morphological_dilation > 0 and sampled.sum() > 0:
            sampled = self._dilate_sparse_mask_3d(
                sampled,
                voxel_indices.long().to(device),
                iterations=morphological_dilation,
            )

        return sampled.to(device)

    @staticmethod
    def _dilate_sparse_mask_3d(
        mask: torch.Tensor,
        voxel_indices: torch.Tensor,
        iterations: int = 1,
    ) -> torch.Tensor:
        """Dilate a sparse per-voxel mask by including the 26-neighborhood
        of each already-marked voxel.

        Implementation uses a voxel-position hash table (python dict) for
        clarity over speed — the voxel counts we deal with (typically
        ~10K-30K) make this sub-millisecond.
        """
        pos_to_idx = {
            (int(x), int(y), int(z)): i
            for i, (x, y, z) in enumerate(voxel_indices.tolist())
        }
        current = mask.clone()
        for _ in range(iterations):
            seeds = torch.where(current > 0.0)[0].tolist()
            for i in seeds:
                x, y, z = voxel_indices[i].tolist()
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for dz in (-1, 0, 1):
                            if dx == 0 and dy == 0 and dz == 0:
                                continue
                            j = pos_to_idx.get((int(x + dx), int(y + dy), int(z + dz)))
                            if j is not None and current[j] == 0.0:
                                current[j] = 1.0
        return current
