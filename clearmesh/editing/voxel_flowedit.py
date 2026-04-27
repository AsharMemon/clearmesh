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
        pipeline=None,
        resolution: int = 512,
        device: str = "cuda",
        config: FlowEditConfig | None = None,
        fingerprint: str | None = None,
    ):
        """Initialize VoxelFlowEdit.

        Args:
            flow_model: TRELLIS.2's ``SparseStructureFlowModel``. The
                public ``Trellis2ImageTo3DPipeline`` does not expose this
                as a callable velocity field — it is wrapped inside the
                pipeline's sampler. Pass ``pipeline=...`` instead unless
                you've loaded the flow model separately.
            pipeline: A loaded ``Trellis2ImageTo3DPipeline``. Used for
                ``get_cond`` (image → DINOv2 conditioning) and, when no
                ``flow_model`` is provided, for sampler-driven flow edits.
            resolution: Image conditioning resolution (512 or 1024).
                Matches how ``pipeline.get_cond`` is called elsewhere.
            device: Compute device.
            config: FlowEdit configuration.
            fingerprint: Optional short string identifying the current
                environment (e.g. ``f"{trellis2.__version__}:{model_dir}"``).
                Persisted alongside the winning flow-model signature to
                ``~/.cache/clearmesh/flow_sig.json`` on first successful
                probe. Lets us detect signature drift across TRELLIS.2
                builds without re-running the probe.
        """
        self.flow_model = flow_model
        self.pipeline = pipeline
        self.resolution = resolution
        self.device = device
        self.config = config or FlowEditConfig()
        self.fingerprint = fingerprint

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
        condition: torch.Tensor | None,
        guidance_scale: float,
    ) -> torch.Tensor:
        """Compute flow velocity using TRELLIS.2's flow model.

        With classifier-free guidance::

            v = v_uncond + guidance_scale * (v_cond - v_uncond)

        The flow model is called directly (not through the sampler) so we
        get per-timestep velocity for Easy3E's edit-flow ODE. Signature
        matches TRELLIS.2's DiT convention: ``model(x, t, cond)``. Time is
        passed as a tensor broadcast across the batch.

        If this raises ``TypeError`` on the first invocation, the installed
        TRELLIS.2 build uses a different kwarg name (e.g. ``context`` or
        ``encoder_hidden_states``) — fix by adjusting the kwargs below.

        Args:
            x_t: Current state. For SS edits this is ``(B, C, R, R, R)``;
                for SLAT repaint it's a ``SparseTensor``-wrapped latent.
            t: Current time in [0, 1].
            condition: Image conditioning tensor (DINOv2 features).
            guidance_scale: CFG scale.

        Returns:
            Velocity tensor, same shape as ``x_t``.
        """
        if self.flow_model is None:
            raise RuntimeError(
                "VoxelFlowEdit._compute_velocity requires flow_model=... "
                "The public Trellis2ImageTo3DPipeline does not expose the "
                "flow model as a direct callable. Load it manually, e.g.:\n"
                "    from trellis2 import models\n"
                "    ss_flow = models.from_pretrained("
                "    f'{model_dir}/{pipeline_json.sparse_structure_flow_model}'"
                "    )\n"
                "and pass it as flow_model to VoxelFlowEdit(...)."
            )

        B = x_t.shape[0] if hasattr(x_t, "shape") else 1
        t_tensor = torch.full((B,), float(t), device=self.device, dtype=torch.float32)

        # Conditional velocity. TRELLIS.2 flow models accept condition either
        # positionally (3rd arg) or as a kwarg — the name varies by build
        # ("cond", "context", "encoder_hidden_states"). Try the common
        # variants; cache the working signature so we only probe once.
        v_cond = self._flow_call(x_t, t_tensor, condition)

        if guidance_scale == 1.0 or condition is None:
            return v_cond

        # Unconditional branch — zero-tensor condition (TRELLIS.2 convention).
        neg_cond = torch.zeros_like(condition)
        v_uncond = self._flow_call(x_t, t_tensor, neg_cond)
        return v_uncond + guidance_scale * (v_cond - v_uncond)

    def _flow_call(
        self, x_t: torch.Tensor, t_tensor: torch.Tensor, condition: torch.Tensor | None
    ) -> torch.Tensor:
        """Invoke the flow model with kwarg-signature auto-detection.

        Caches the first working signature in ``self._flow_sig`` so we don't
        re-probe on every step. Order of attempts:
          1. positional  ``model(x, t, cond)``
          2. kwarg       ``model(x, t, context=cond)``
          3. kwarg       ``model(x, t, cond=cond)``
          4. kwarg       ``model(x, t, encoder_hidden_states=cond)``

        If ``condition`` is None (pure unconditional), fall through to
        positional-None (some models handle this natively).
        """
        # Lazy cache attribute
        sig = getattr(self, "_flow_sig", None)
        attempts = []
        if sig is None:
            attempts = [
                ("pos", None),
                ("kw", "context"),
                ("kw", "cond"),
                ("kw", "encoder_hidden_states"),
            ]
        else:
            attempts = [sig]

        last_exc: Exception | None = None
        for attempt in attempts:
            try:
                if attempt[0] == "pos":
                    out = self.flow_model(x_t, t_tensor, condition)
                else:
                    out = self.flow_model(x_t, t_tensor, **{attempt[1]: condition})
                # Cache the winning signature in-memory...
                first_time = sig is None
                self._flow_sig = attempt
                # ...and persist on first probe so we can detect signature
                # drift across TRELLIS.2 builds without re-running edits.
                if first_time:
                    self._persist_flow_sig(attempt)
                return out
            except TypeError as e:
                last_exc = e
                continue
        # All attempts failed — re-raise the last error with context.
        raise RuntimeError(
            f"VoxelFlowEdit._flow_call: flow model rejected all known "
            f"signatures (positional, context=, cond=, encoder_hidden_states=). "
            f"Last error: {last_exc}"
        )

    def _persist_flow_sig(self, sig: tuple) -> None:
        """Append the winning (fingerprint, sig) pair to a local cache.

        Best-effort — any IO failure is swallowed (logged only to stderr).
        Writes to ``~/.cache/clearmesh/flow_sig.json`` as a list of
        ``{timestamp, fingerprint, sig}`` entries so we can spot drift
        when the same code runs against different TRELLIS.2 builds.
        """
        import json
        import os
        import sys
        import time

        try:
            cache_dir = os.path.expanduser("~/.cache/clearmesh")
            os.makedirs(cache_dir, exist_ok=True)
            log_path = os.path.join(cache_dir, "flow_sig.json")

            history: list = []
            if os.path.exists(log_path):
                try:
                    with open(log_path) as f:
                        history = json.load(f)
                    if not isinstance(history, list):
                        history = []
                except (OSError, json.JSONDecodeError):
                    history = []

            entry = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "fingerprint": self.fingerprint,
                "sig": list(sig),  # tuples don't roundtrip JSON
            }
            history.append(entry)

            # Keep the log bounded — last 128 probes is more than enough.
            history = history[-128:]

            with open(log_path, "w") as f:
                json.dump(history, f, indent=2)
        except Exception as e:  # pragma: no cover — best-effort logging
            print(
                f"[VoxelFlowEdit] Could not persist flow_sig: {e}",
                file=sys.stderr,
            )

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
    ) -> torch.Tensor:
        """Compute silhouette gradient guidance.

        G_sil = gradient of BCE(rendered_silhouette, target_silhouette)
        This steers the structure towards matching the target's outline.

        Args:
            x_t: Current state (requires grad for gradient computation).
            target_image: Target image (silhouette extracted from alpha/edges).
            t: Current time.

        Returns:
            Silhouette gradient (B, N, D).
        """
        # TODO: Implement silhouette rendering and BCE loss gradient
        # 1. Decode x_t to approximate voxel occupancy
        # 2. Render silhouette from canonical views
        # 3. Compare with target silhouette (from target_image alpha)
        # 4. Backprop gradient through rendering
        return torch.zeros_like(x_t)

    def _encode_image_condition(
        self, image: Image.Image | None
    ) -> torch.Tensor | None:
        """Encode an image into DINOv2 conditioning features.

        Uses ``pipeline.preprocess_image`` (rembg + crop/pad) and
        ``pipeline.get_cond([proc], resolution)`` — the exact sequence
        used in ``scripts/data/generate_slat_pairs.py:177`` and the rest
        of the repo. Returns the ``'cond'`` tensor ``(1, M, 1024)``.

        Args:
            image: PIL Image to encode.

        Returns:
            Conditioning tensor, or None if image is None.
        """
        if image is None:
            return None

        if self.pipeline is None:
            raise RuntimeError(
                "VoxelFlowEdit._encode_image_condition requires pipeline=... "
                "Pass a loaded Trellis2ImageTo3DPipeline to the constructor."
            )

        processed = self.pipeline.preprocess_image(image)
        cond_dict = self.pipeline.get_cond([processed], self.resolution)

        # Normalize to the 'cond' tensor (matches extract_cond_feats in
        # generate_slat_pairs.py:124).
        if isinstance(cond_dict, dict):
            cond_tensor = cond_dict.get("cond", cond_dict.get("image_cond"))
            if cond_tensor is None:
                cond_tensor = next(iter(cond_dict.values()))
        elif isinstance(cond_dict, (list, tuple)):
            cond_tensor = cond_dict[0]
        else:
            cond_tensor = cond_dict

        return cond_tensor.to(self.device)

    def auto_detect_edit_mask(
        self,
        source_image: Image.Image,
        target_image: Image.Image,
        voxel_indices: torch.Tensor,
        threshold: float = 0.1,
        grid_size: int = 256,
        camera_eye: tuple[float, float, float] = (0.0, 0.0, 2.0),
        camera_up: tuple[float, float, float] = (0.0, 1.0, 0.0),
        fov_deg: float = 60.0,
        dilate_2d: int = 2,
    ) -> torch.Tensor:
        """Auto-detect edit region from image difference.

        Diffs source vs target images to get a 2D changed-pixel mask, then
        projects each occupied voxel to that view and keeps voxels whose
        projection falls inside the changed region.

        Camera convention matches ``ImageEditor._render_view`` (perspective
        camera looking at the origin; model normalized to unit-diameter).
        If you rendered with a different view, pass matching ``camera_eye``
        / ``camera_up``. ``fov_deg`` defaults to trimesh's 60° FOV.

        NOTE: This is a single-view projection — voxels occluded from this
        view are NOT detectable. For more robust masks, OR together masks
        from multiple views (front/back/left/right).

        Args:
            source_image: Original rendered view.
            target_image: Edited view (same camera as source_image).
            voxel_indices: Occupied voxel integer indices (N, 3) in
                [0, grid_size). Assumed to live in a voxel grid spanning
                the unit cube [-0.5, 0.5]^3 (matches SLATEncoder).
            threshold: Per-pixel L1 difference threshold in [0, 1].
            grid_size: Voxel grid resolution.
            camera_eye: Camera position in world space.
            camera_up: Camera up vector.
            fov_deg: Vertical field-of-view in degrees.
            dilate_2d: Pixel-radius dilation on the 2D diff mask before
                projection — tolerates small alignment slop.

        Returns:
            Binary mask (N,) over voxels. 1 = edit, 0 = preserve.
        """
        device = voxel_indices.device

        # --- 1. 2D diff mask from source vs target ---
        src = np.asarray(source_image.convert("RGB"), dtype=np.float32) / 255.0
        tgt_img = target_image.convert("RGB")
        # Match resolutions — editors sometimes change size.
        if tgt_img.size != source_image.size:
            tgt_img = tgt_img.resize(source_image.size, Image.LANCZOS)
        tgt = np.asarray(tgt_img, dtype=np.float32) / 255.0

        diff = np.abs(src - tgt).mean(axis=-1)  # (H, W)
        mask_2d = diff > threshold

        if dilate_2d > 0 and mask_2d.any():
            # Cheap square-kernel dilation via max-pool in numpy.
            r = int(dilate_2d)
            H, W = mask_2d.shape
            padded = np.pad(mask_2d.astype(np.uint8), r, mode="constant")
            dilated = np.zeros_like(mask_2d, dtype=bool)
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    dilated |= padded[r + dy : r + dy + H, r + dx : r + dx + W].astype(bool)
            mask_2d = dilated

        H, W = mask_2d.shape

        # --- 2. Voxel integer indices → world coordinates in [-0.5, 0.5] ---
        # Centers of voxels: (i + 0.5) / grid_size in [0,1], then shift to [-0.5, 0.5].
        vox = voxel_indices.to(torch.float32)
        world = (vox + 0.5) / float(grid_size) - 0.5  # (N, 3)

        # --- 3. World → camera transform (look-at) ---
        eye = torch.tensor(camera_eye, device=device, dtype=torch.float32)
        up = torch.tensor(camera_up, device=device, dtype=torch.float32)
        target = torch.zeros(3, device=device, dtype=torch.float32)

        forward = target - eye
        forward = forward / (forward.norm() + 1e-8)
        right = torch.linalg.cross(forward, up)
        right = right / (right.norm() + 1e-8)
        true_up = torch.linalg.cross(right, forward)

        # Project (world - eye) onto camera basis. cam_z = projection on
        # the forward axis, so cam_z > 0 ⇔ point is in front of the camera.
        rel = world.to(device) - eye  # (N, 3)
        cam_x = rel @ right            # (N,)
        cam_y = rel @ true_up          # (N,)
        cam_z = rel @ forward          # (N,) positive = in front of camera

        # --- 4. Perspective projection → normalized device coords ---
        in_front = cam_z > 1e-4
        f = 1.0 / np.tan(np.deg2rad(fov_deg) / 2.0)
        ndc_x = torch.zeros_like(cam_x)
        ndc_y = torch.zeros_like(cam_y)
        ndc_x[in_front] = (cam_x[in_front] * f) / cam_z[in_front]
        ndc_y[in_front] = (cam_y[in_front] * f) / cam_z[in_front]

        # NDC ([-1, 1]) → pixel indices. y flips (image y grows downward).
        px = ((ndc_x + 1.0) * 0.5 * (W - 1)).round().to(torch.int64)
        py = ((1.0 - (ndc_y + 1.0) * 0.5) * (H - 1)).round().to(torch.int64)

        in_view = in_front & (px >= 0) & (px < W) & (py >= 0) & (py < H)

        # --- 5. Sample 2D mask at projected pixel for each in-view voxel ---
        mask_2d_t = torch.from_numpy(mask_2d).to(device)
        N = voxel_indices.shape[0]
        out = torch.zeros(N, device=device, dtype=torch.float32)
        if in_view.any():
            idx = torch.where(in_view)[0]
            out[idx] = mask_2d_t[py[idx], px[idx]].to(torch.float32)

        # Edge case: if the 2D diff was empty (images identical within
        # threshold) don't return an all-zero mask — fall back to "edit
        # everything" so downstream doesn't silently no-op.
        if not mask_2d.any():
            out = torch.ones(N, device=device, dtype=torch.float32)

        return out
