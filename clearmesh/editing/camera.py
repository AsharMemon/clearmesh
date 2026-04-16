"""Canonical camera model for Easy3E editing.

TRELLIS.2's image conditioning assumes a specific canonical view of the
object. To project voxels to pixels (for auto-detecting edit masks) and
to render silhouettes (for silhouette guidance), we need the same camera
model that the pipeline uses internally.

This module encodes the canonical camera pose and provides one-shot
voxel-to-pixel projection. It is designed to be PyTorch-native,
differentiable, and not dependent on nvdiffrast / pyrender.

The values match TRELLIS.2's conditioning preprocessor (front-facing,
yfov ≈ 40°, camera placed at z=2 looking at the origin, object
normalized to fit in a unit cube centered at the origin).

To confirm these match what the installed TRELLIS.2 actually uses, run
``scripts/setup/inspect_trellis2.py`` on the pod and compare the
``get_cond`` intermediate tensor shapes / camera pose dumps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class CanonicalCamera:
    """A pinhole camera defined by a world→camera transform and FOV."""

    # 4x4 world-to-camera matrix (extrinsic)
    view_matrix: torch.Tensor
    # Field of view in radians (vertical)
    yfov: float
    # Output image size (square)
    image_size: int = 512

    @classmethod
    def trellis2_default(cls, image_size: int = 512) -> "CanonicalCamera":
        """TRELLIS.2's canonical front-facing camera.

        - Object is normalized to fit in [-0.5, 0.5]^3, centered at origin.
        - Camera at (0, 0, 2), looking at origin, up = +Y.
        - 40° vertical field of view (roughly matches TRELLIS.2 conditioning
          at 512px; rerun inspect_trellis2.py to verify on the pod).
        """
        eye = np.array([0.0, 0.0, 2.0], dtype=np.float32)
        target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        # Build look-at (world→camera)
        forward = target - eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        true_up = np.cross(right, forward)

        # Camera coordinate frame: +X right, +Y up, -Z forward (OpenGL convention)
        rot = np.stack([right, true_up, -forward], axis=0)  # (3, 3) rows
        trans = -rot @ eye
        view = np.eye(4, dtype=np.float32)
        view[:3, :3] = rot
        view[:3, 3] = trans

        return cls(
            view_matrix=torch.from_numpy(view),
            yfov=math.radians(40.0),
            image_size=image_size,
        )

    def projection_matrix(self, near: float = 0.01, far: float = 100.0) -> torch.Tensor:
        """Return a 4x4 OpenGL-style perspective projection matrix."""
        f = 1.0 / math.tan(self.yfov / 2.0)
        aspect = 1.0  # square images
        proj = torch.zeros(4, 4, dtype=torch.float32)
        proj[0, 0] = f / aspect
        proj[1, 1] = f
        proj[2, 2] = -(far + near) / (far - near)
        proj[2, 3] = -2 * far * near / (far - near)
        proj[3, 2] = -1
        return proj


def project_voxels_to_pixels(
    voxel_indices: torch.Tensor,
    camera: CanonicalCamera,
    grid_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project voxel positions through the camera to pixel coords.

    Args:
        voxel_indices: (N, 3) integer voxel coords in [0, grid_size).
        camera: CanonicalCamera instance.
        grid_size: Voxel grid resolution (used to normalize to [-0.5, 0.5]).

    Returns:
        (pixel_u, pixel_v, depth):
            pixel_u: (N,) float, image-space X in [0, image_size).
                     Out-of-frame voxels have values outside [0, image_size).
            pixel_v: (N,) float, image-space Y in [0, image_size).
            depth: (N,) float, camera-space Z. Positive = in front of camera.
                   Negative depth means the voxel is behind the camera.

    The projection follows the standard pipeline:
        world = voxel_index / grid_size - 0.5 (recenter to unit cube)
        cam = view_matrix @ [world; 1]
        clip = projection @ cam
        ndc = clip[:3] / clip[3]
        pixel = (ndc[:2] + 1) * 0.5 * image_size  (flip y to image coords)
    """
    device = voxel_indices.device
    N = voxel_indices.shape[0]

    view = camera.view_matrix.to(device)
    proj = camera.projection_matrix().to(device)
    mvp = proj @ view  # (4, 4)

    # Voxel idx → world coords in [-0.5, 0.5]
    # Center of voxel (i, j, k): (i + 0.5) / grid_size - 0.5
    world = voxel_indices.float() / float(grid_size) - 0.5 + (0.5 / float(grid_size))

    # Homogeneous
    ones = torch.ones(N, 1, device=device, dtype=world.dtype)
    homog = torch.cat([world, ones], dim=-1)  # (N, 4)

    clip = homog @ mvp.T  # (N, 4)

    # Depth is -Z in camera space (OpenGL forward = -Z)
    cam_space = homog @ view.T
    depth = -cam_space[:, 2]

    # Perspective divide (guard against w=0 with tiny epsilon; any voxel at
    # the camera origin projects to nonsense anyway)
    w = clip[:, 3].clone()
    w = torch.where(w.abs() < 1e-8, torch.full_like(w, 1e-8), w)
    ndc_x = clip[:, 0] / w  # in [-1, 1] for in-frame
    ndc_y = clip[:, 1] / w

    # Map NDC to pixel coords (image origin at top-left, Y flips)
    pixel_u = (ndc_x + 1.0) * 0.5 * camera.image_size
    pixel_v = (1.0 - ndc_y) * 0.5 * camera.image_size

    return pixel_u, pixel_v, depth
