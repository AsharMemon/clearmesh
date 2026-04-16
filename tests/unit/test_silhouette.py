"""CPU-only unit tests for silhouette guidance.

Tests the voxel-soft backend (nvdiffrast path is GPU-only, tested on pod).
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from clearmesh.editing.camera import CanonicalCamera
from clearmesh.editing.silhouette import (
    compute_silhouette_guidance,
    extract_target_silhouette,
    silhouette_bce_loss,
    silhouette_from_voxels,
)


def test_extract_silhouette_alpha_channel():
    """Image with alpha channel → silhouette from alpha."""
    arr = np.zeros((64, 64, 4), dtype=np.uint8)
    arr[20:40, 20:40, 3] = 255  # square of alpha = 1
    arr[20:40, 20:40, :3] = (200, 100, 50)
    img = Image.fromarray(arr, mode="RGBA")

    sil = extract_target_silhouette(img, size=64)
    assert sil.shape == (64, 64)
    # Inside the alpha square → 1
    assert sil[30, 30] == 1.0
    # Outside → 0
    assert sil[5, 5] == 0.0


def test_extract_silhouette_rgb_bg_diff():
    """RGB-only image → silhouette from background difference."""
    # Grey background, red foreground blob
    arr = np.full((64, 64, 3), 128, dtype=np.uint8)
    arr[20:44, 20:44] = (255, 0, 0)
    img = Image.fromarray(arr)

    sil = extract_target_silhouette(img, size=64, threshold=0.05)
    # Foreground detected
    assert sil[30, 30] == 1.0
    # Background: 0
    assert sil[5, 5] == 0.0


def test_silhouette_from_voxels_uniform_weights():
    """Voxels with weight 1.0 projected into image should produce a dense
    splat region (non-zero area)."""
    cam = CanonicalCamera.trellis2_default(image_size=64)
    # A small cube of voxels near image center
    voxel_indices = torch.tensor(
        [[i, j, k] for i in range(120, 136) for j in range(120, 136) for k in range(120, 136)]
    )
    weights = torch.ones(voxel_indices.shape[0])
    rendered = silhouette_from_voxels(voxel_indices, weights, cam, grid_size=256, splat_radius=1.0)
    assert rendered.shape == (64, 64)
    assert rendered.max() > 0.0
    # Rendered silhouette should land near image center
    nonzero = (rendered > 0.1).nonzero(as_tuple=True)
    if len(nonzero[0]) > 0:
        center_y = nonzero[0].float().mean().item()
        center_x = nonzero[1].float().mean().item()
        assert 22 < center_y < 42
        assert 22 < center_x < 42


def test_silhouette_bce_loss_perfect_match_is_low():
    target = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    rendered = torch.tensor([[0.99, 0.99], [0.01, 0.01]])
    loss = silhouette_bce_loss(rendered, target)
    assert loss.item() < 0.1


def test_silhouette_bce_loss_total_mismatch_is_high():
    target = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    rendered = torch.tensor([[0.01, 0.01], [0.99, 0.99]])
    loss = silhouette_bce_loss(rendered, target)
    assert loss.item() > 1.0


def test_compute_silhouette_guidance_returns_shape_matches_x_t():
    cam = CanonicalCamera.trellis2_default(image_size=32)
    # Small x_t: 1 batch, 10 voxels, 8 dims
    x_t = torch.randn(1, 10, 8)
    voxel_indices = torch.randint(100, 150, (10, 3))
    target_sil = torch.zeros(32, 32)
    target_sil[10:22, 10:22] = 1.0

    grad = compute_silhouette_guidance(
        x_t, target_sil, voxel_indices, cam, grid_size=256
    )
    assert grad.shape == x_t.shape
    assert not torch.isnan(grad).any()


def test_compute_silhouette_guidance_matching_shape_smaller_grad():
    """A rendered silhouette that already matches the target should produce
    a much smaller gradient than a mismatched one."""
    cam = CanonicalCamera.trellis2_default(image_size=48)

    # Case A: voxels projected into the target region
    voxels_match = torch.tensor(
        [[i, j, 128] for i in range(120, 136) for j in range(120, 136)]
    )
    x_match = torch.ones(1, voxels_match.shape[0], 4)

    # Target silhouette: bright square in image center
    target = torch.zeros(48, 48)
    target[14:34, 14:34] = 1.0

    grad_match = compute_silhouette_guidance(x_match, target, voxels_match, cam, grid_size=256)

    # Case B: voxels projected somewhere else
    voxels_miss = torch.tensor(
        [[i, j, 128] for i in range(20, 36) for j in range(20, 36)]
    )
    x_miss = torch.ones(1, voxels_miss.shape[0], 4)
    grad_miss = compute_silhouette_guidance(x_miss, target, voxels_miss, cam, grid_size=256)

    # Exact-match grad norm should be <= mismatch grad norm
    assert grad_match.abs().sum() <= grad_miss.abs().sum() + 1e-6


def test_compute_silhouette_guidance_2d_input():
    """Should handle (N, D) input as well as (B, N, D)."""
    cam = CanonicalCamera.trellis2_default(image_size=32)
    x_t = torch.randn(10, 4)
    voxels = torch.randint(100, 150, (10, 3))
    target = torch.zeros(32, 32)
    grad = compute_silhouette_guidance(x_t, target, voxels, cam, grid_size=256)
    assert grad.shape == x_t.shape
