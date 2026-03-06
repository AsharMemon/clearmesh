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
    ):
        """Initialize VoxelFlowEdit.

        Args:
            flow_model: TRELLIS.2's SparseStructureFlowModel.
                        Will be loaded lazily if None.
            device: Compute device.
            config: FlowEdit configuration.
        """
        self.flow_model = flow_model
        self.device = device
        self.config = config or FlowEditConfig()

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

        With classifier-free guidance:
          v = v_uncond + guidance_scale * (v_cond - v_uncond)

        Args:
            x_t: Current state (B, N, D).
            t: Current time.
            condition: Image conditioning tensor.
            guidance_scale: CFG scale.

        Returns:
            Velocity tensor (B, N, D).
        """
        # TODO: Call TRELLIS.2's SparseStructureFlowModel
        # The model takes (x_t, t, condition) and returns velocity
        # Need to investigate exact API on RunPod
        raise NotImplementedError(
            "Velocity computation requires TRELLIS.2's flow model. "
            "Investigate SparseStructureFlowModel API in trellis2/models/"
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
        """Encode an image for flow model conditioning.

        Uses DINOv2 to extract multi-view features, matching
        TRELLIS.2's conditioning format.

        Args:
            image: PIL Image to encode.

        Returns:
            Conditioning tensor or None.
        """
        if image is None:
            return None
        # TODO: Use TRELLIS.2's image encoder (DINOv2/DINOv3)
        # to produce conditioning features
        raise NotImplementedError(
            "Image conditioning requires TRELLIS.2's DINOv2 encoder. "
            "Investigate trellis2/pipelines/ for image encoding API."
        )

    def auto_detect_edit_mask(
        self,
        source_image: Image.Image,
        target_image: Image.Image,
        voxel_indices: torch.Tensor,
        threshold: float = 0.1,
    ) -> torch.Tensor:
        """Auto-detect edit region from image difference.

        Compares source and target images to find changed regions,
        then projects the 2D mask to 3D voxel space.

        Args:
            source_image: Original rendered view.
            target_image: Edited view.
            voxel_indices: Voxel positions (N, 3).
            threshold: Pixel difference threshold.

        Returns:
            Binary mask (N,) over voxels.
        """
        # Convert to numpy
        src = np.array(source_image.convert("RGB")).astype(np.float32) / 255
        tgt = np.array(target_image.convert("RGB")).astype(np.float32) / 255

        # Pixel-wise difference
        diff = np.abs(src - tgt).mean(axis=-1)
        mask_2d = (diff > threshold).astype(np.float32)

        # TODO: Project 2D mask to 3D voxel space
        # This requires camera projection matrices to map voxels → pixels
        # For now, return all-ones (edit everything)
        N = voxel_indices.shape[0]
        return torch.ones(N, device=voxel_indices.device)
