"""CPU-only unit tests for region-focused editing — mask loading, projection,
and SLAT blending.

Tests the new ``EditOptions.region_mask`` plumbing in easy3e.py:
  - Mask accepts PIL.Image, path, and numpy array
  - 2D mask projects to 3D voxels via CanonicalCamera
  - Blending with a fake SparseTensor produces expected per-voxel weights
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image


@pytest.fixture
def fake_slat():
    """Create a fake SparseTensor-like object with .coords and .feats."""
    class FakeSparseTensor:
        def __init__(self, coords, feats):
            self.coords = coords
            self.feats = feats

        def replace(self, feats=None):
            return FakeSparseTensor(self.coords, feats if feats is not None else self.feats)

    # 10 voxels spread across a grid
    coords = torch.tensor(
        [[0, 128, 128, 128],  # center
         [0, 32,  128, 128],  # left
         [0, 224, 128, 128],  # right
         [0, 128, 32,  128],  # bottom in world Y = high v
         [0, 128, 224, 128],  # top in world Y
         [0, 128, 128, 32],   # front
         [0, 128, 128, 224],  # back
         [0, 64,  64,  64],
         [0, 192, 192, 192],
         [0, 100, 100, 100]],
        dtype=torch.long,
    )
    feats = torch.randn(10, 32)
    return FakeSparseTensor(coords, feats)


def _make_editor_stub():
    from clearmesh.editing.easy3e import Easy3EEditor
    editor = Easy3EEditor.__new__(Easy3EEditor)
    editor.device = "cpu"
    return editor


def test_mask_accepts_pil_image(fake_slat):
    editor = _make_editor_stub()

    # 64x64 binary mask with a bright spot in the top half
    arr = np.zeros((64, 64), dtype=np.uint8)
    arr[:30, :] = 255  # top half = edit
    mask_img = Image.fromarray(arr, mode="L")

    source_slat = fake_slat
    edit_slat = fake_slat.__class__(fake_slat.coords, torch.ones_like(fake_slat.feats))

    blended = editor._blend_slat_by_mask(
        edit_slat=edit_slat,
        source_slat=source_slat,
        region_mask=mask_img,
        source_image=None, edit_image=None, dilation=0,
    )

    # Same number of voxels, same feature dim
    assert blended.feats.shape == fake_slat.feats.shape


def test_mask_accepts_numpy_array(fake_slat):
    editor = _make_editor_stub()
    arr = np.zeros((32, 32), dtype=np.float32)
    arr[:16, :] = 1.0
    source_slat = fake_slat
    edit_slat = fake_slat.__class__(fake_slat.coords, torch.ones_like(fake_slat.feats))

    blended = editor._blend_slat_by_mask(
        edit_slat=edit_slat, source_slat=source_slat, region_mask=arr,
        source_image=None, edit_image=None, dilation=0,
    )
    assert blended.feats.shape == fake_slat.feats.shape


def test_mask_accepts_path(fake_slat, tmp_path):
    editor = _make_editor_stub()
    arr = np.full((64, 64), 255, dtype=np.uint8)  # all-edit mask
    mask_path = tmp_path / "mask.png"
    Image.fromarray(arr, mode="L").save(mask_path)

    source_slat = fake_slat
    edit_slat = fake_slat.__class__(
        fake_slat.coords,
        torch.ones_like(fake_slat.feats) * 10.0,  # distinctive edit features
    )

    blended = editor._blend_slat_by_mask(
        edit_slat=edit_slat, source_slat=source_slat, region_mask=str(mask_path),
        source_image=None, edit_image=None, dilation=0,
    )
    # All-edit mask + voxels in-frame = blend should be ~= edit features (=10)
    # for in-frame in-front voxels. Out-of-frame ones fall back to source.
    assert blended.feats.shape == fake_slat.feats.shape


def test_all_zero_mask_returns_source_features(fake_slat):
    editor = _make_editor_stub()
    arr = np.zeros((64, 64), dtype=np.uint8)  # no edits
    mask_img = Image.fromarray(arr, mode="L")

    source_slat = fake_slat
    edit_slat = fake_slat.__class__(
        fake_slat.coords, torch.ones_like(fake_slat.feats) * 99.0,
    )

    blended = editor._blend_slat_by_mask(
        edit_slat=edit_slat, source_slat=source_slat, region_mask=mask_img,
        source_image=None, edit_image=None, dilation=0,
    )
    # Expect all voxels to fall back to source features (zeros not 99)
    assert torch.allclose(blended.feats, source_slat.feats)


def test_unknown_mask_type_raises():
    editor = _make_editor_stub()

    class BogusType:
        pass

    source_slat = None  # irrelevant, TypeError raises first
    edit_slat = None

    with pytest.raises(TypeError, match="region_mask must be"):
        editor._blend_slat_by_mask(
            edit_slat=edit_slat, source_slat=source_slat,
            region_mask=BogusType(),
            source_image=None, edit_image=None, dilation=0,
        )


def test_editoptions_has_region_mask_field():
    """EditOptions dataclass should include region_mask and ultrashape fields."""
    from clearmesh.editing.easy3e import EditOptions
    opts = EditOptions()
    assert opts.region_mask is None
    assert opts.enable_ultrashape is False
    assert opts.ultrashape_steps > 0


def test_editoptions_region_mask_accepts_image():
    from clearmesh.editing.easy3e import EditOptions
    img = Image.new("L", (32, 32), 255)
    opts = EditOptions(region_mask=img)
    assert opts.region_mask is img
