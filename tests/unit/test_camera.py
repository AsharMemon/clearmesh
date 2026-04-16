"""CPU-only unit tests for the canonical camera and voxel projection.

Tests the math in clearmesh/editing/camera.py without any GPU or
TRELLIS.2 dependency.
"""

from __future__ import annotations

import math
import torch

from clearmesh.editing.camera import CanonicalCamera, project_voxels_to_pixels


def test_default_camera_looks_at_origin():
    cam = CanonicalCamera.trellis2_default(image_size=512)
    # Camera at (0,0,2), looking toward origin
    assert cam.image_size == 512
    assert cam.view_matrix.shape == (4, 4)
    # Origin in world → projected behind the image center
    origin = torch.tensor([[128, 128, 128]])  # center voxel in 256 grid
    u, v, d = project_voxels_to_pixels(origin, cam, grid_size=256)
    # Projects near image center (within 5% of image_size/2)
    assert abs(u[0].item() - 256) < 25
    assert abs(v[0].item() - 256) < 25
    # Should be in front of camera (positive depth)
    assert d[0].item() > 0


def test_projection_matrix_shape():
    cam = CanonicalCamera.trellis2_default(image_size=256)
    proj = cam.projection_matrix()
    assert proj.shape == (4, 4)
    # [3, 2] should be -1 for OpenGL-style perspective
    assert proj[3, 2].item() == -1.0


def test_voxel_in_front_vs_behind():
    """Voxel between camera and origin: in front.
    Voxel behind camera: negative depth."""
    cam = CanonicalCamera.trellis2_default(image_size=256)
    # Voxel at grid center (128, 128, 128) = world origin
    in_front = torch.tensor([[128, 128, 128]])
    u1, v1, d1 = project_voxels_to_pixels(in_front, cam, grid_size=256)
    assert d1[0] > 0

    # Voxel at z = 255 (back of grid) = world z = +0.5 → still in front of cam at z=2
    back_of_grid = torch.tensor([[128, 128, 255]])
    _, _, d2 = project_voxels_to_pixels(back_of_grid, cam, grid_size=256)
    assert d2[0] > 0

    # Voxel at z = 0 (front of grid) = world z = -0.5 → further from camera at z=2
    front_of_grid = torch.tensor([[128, 128, 0]])
    _, _, d3 = project_voxels_to_pixels(front_of_grid, cam, grid_size=256)
    assert d3[0] > d2[0]  # further away = greater depth (positive)


def test_voxel_left_projects_to_left_half():
    """A voxel left of center should project to pixel_u < image_size/2."""
    cam = CanonicalCamera.trellis2_default(image_size=256)
    # X=0 (far left), middle of Y and Z
    left = torch.tensor([[0, 128, 128]])
    u, v, d = project_voxels_to_pixels(left, cam, grid_size=256)
    assert u[0].item() < 128  # left half


def test_voxel_right_projects_to_right_half():
    cam = CanonicalCamera.trellis2_default(image_size=256)
    right = torch.tensor([[255, 128, 128]])
    u, v, d = project_voxels_to_pixels(right, cam, grid_size=256)
    assert u[0].item() > 128  # right half


def test_voxel_top_projects_above_center():
    """Y in voxel coords increases upward (grid convention).
    After projection to image coords, pixel_v smaller = higher on image."""
    cam = CanonicalCamera.trellis2_default(image_size=256)
    top = torch.tensor([[128, 255, 128]])    # high Y = top in world
    u, v, _ = project_voxels_to_pixels(top, cam, grid_size=256)
    # Image Y increases downward, so "top of world" → small pixel_v
    assert v[0].item() < 128


def test_many_voxels_batched():
    cam = CanonicalCamera.trellis2_default(image_size=256)
    voxels = torch.randint(0, 256, (100, 3))
    u, v, d = project_voxels_to_pixels(voxels, cam, grid_size=256)
    assert u.shape == (100,)
    assert v.shape == (100,)
    assert d.shape == (100,)
    # All voxels inside the unit cube are in front of the camera at z=2
    assert (d > 0).all()


def test_voxel_indices_accept_non_int_tensor():
    """The projection should accept float tensors (needed for subvoxel coords)."""
    cam = CanonicalCamera.trellis2_default(image_size=256)
    voxels = torch.tensor([[128.5, 128.5, 128.5]], dtype=torch.float32)
    u, v, d = project_voxels_to_pixels(voxels, cam, grid_size=256)
    assert u.shape == (1,)


def test_auto_detect_edit_mask_isolates_top_region():
    """Test the full auto_detect_edit_mask integration: a target image with
    a bright spot in the top half should produce a mask with higher density
    in the upper voxels.
    """
    import numpy as np
    from PIL import Image
    from clearmesh.editing.voxel_flowedit import VoxelFlowEdit

    ve = VoxelFlowEdit.__new__(VoxelFlowEdit)
    ve.device = "cpu"

    # Source: uniform gray
    src = Image.new("RGB", (128, 128), (128, 128, 128))
    # Target: same gray except a bright strip in the TOP half
    arr = np.full((128, 128, 3), 128, dtype=np.uint8)
    arr[:40, :, :] = (255, 200, 200)  # top 30% is bright
    tgt = Image.fromarray(arr)

    # 1000 voxels spread evenly across the grid
    torch.manual_seed(0)
    voxels = torch.randint(0, 256, (1000, 3))

    mask = ve.auto_detect_edit_mask(
        src, tgt, voxels,
        threshold=0.1, grid_size=256, morphological_dilation=0,
    )
    assert mask.shape == (1000,)
    # Voxels in the upper half of the grid (Y > 128) should have higher mask density
    # because the target change is in the upper image half (Y high in world = low v in image)
    is_upper = (voxels[:, 1] > 128)
    is_lower = (voxels[:, 1] <= 128)
    upper_mean = mask[is_upper].mean().item()
    lower_mean = mask[is_lower].mean().item()
    assert upper_mean > lower_mean


def test_auto_detect_edit_mask_identical_images_returns_zero():
    import numpy as np
    from PIL import Image
    from clearmesh.editing.voxel_flowedit import VoxelFlowEdit

    ve = VoxelFlowEdit.__new__(VoxelFlowEdit)
    ve.device = "cpu"

    img = Image.new("RGB", (64, 64), (100, 100, 100))
    voxels = torch.randint(0, 256, (50, 3))
    mask = ve.auto_detect_edit_mask(img, img, voxels, threshold=0.01, grid_size=256)
    assert mask.shape == (50,)
    # Identical images → near-zero diff → empty mask
    assert mask.sum().item() == 0.0


def test_auto_detect_mask_dilation_grows_region():
    """morphological_dilation should grow a single-voxel mask."""
    import numpy as np
    import torch
    from PIL import Image
    from clearmesh.editing.voxel_flowedit import VoxelFlowEdit

    ve = VoxelFlowEdit.__new__(VoxelFlowEdit)
    ve.device = "cpu"

    # Create a 5x5x5 cluster of voxels; source all-grey, target differs
    # only at one position (center of projected image)
    src = Image.new("RGB", (128, 128), (128, 128, 128))
    arr = np.full((128, 128, 3), 128, dtype=np.uint8)
    arr[60:68, 60:68, :] = (255, 0, 0)  # small red spot
    tgt = Image.fromarray(arr)

    # Dense cluster of voxels near grid center
    voxels = torch.stack(torch.meshgrid(
        torch.arange(125, 132), torch.arange(125, 132), torch.arange(125, 132),
        indexing='ij'
    ), dim=-1).reshape(-1, 3)

    mask_no_dilate = ve.auto_detect_edit_mask(
        src, tgt, voxels, threshold=0.1, grid_size=256, morphological_dilation=0
    )
    mask_dilated = ve.auto_detect_edit_mask(
        src, tgt, voxels, threshold=0.1, grid_size=256, morphological_dilation=1
    )
    # Dilation should make the mask cover >= as many voxels
    assert mask_dilated.sum().item() >= mask_no_dilate.sum().item()
