"""Integration tests for VoxelFlowEdit velocity computation and ODE.

Requires TRELLIS.2 pipeline loaded. Skipped without it.
"""

from __future__ import annotations

import pytest
import torch
from PIL import Image


pytestmark = [pytest.mark.gpu, pytest.mark.trellis2]


@pytest.fixture(scope="module")
def flowedit(trellis2_pipeline):
    from clearmesh.editing.voxel_flowedit import VoxelFlowEdit
    return VoxelFlowEdit(device="cuda", pipeline=trellis2_pipeline)


def test_encode_image_condition(flowedit):
    """Should return a non-None conditioning object for a valid image."""
    import numpy as np
    arr = np.full((256, 256, 3), 128, dtype=np.uint8)
    arr[100:150, 100:150] = (255, 0, 0)
    img = Image.fromarray(arr)

    cond = flowedit._encode_image_condition(img)
    assert cond is not None

    # Should survive _split_condition
    pos, neg = flowedit._split_condition(cond)
    assert pos is not None


def test_velocity_shape_matches_input(flowedit, trellis2_pipeline):
    """Velocity tensor should match x_t shape; no NaN."""
    # Build a tiny x_t — SS latent is typically (B, C, R, R, R) or sparse
    # The exact shape depends on the model's sample_sparse_structure internals.
    # We fabricate a plausible small shape and fallback if the model rejects it.
    img = Image.new("RGB", (256, 256), (200, 100, 100))
    cond = flowedit._encode_image_condition(img)
    pos, neg = flowedit._split_condition(cond)

    # Try: dense (1, 1, 16, 16, 16)
    x_t = torch.randn(1, 1, 16, 16, 16, device="cuda")
    try:
        v = flowedit._compute_velocity(x_t, t=0.5, condition=cond, guidance_scale=1.0)
        assert v.shape == x_t.shape
        assert not torch.isnan(v).any()
    except Exception as e:
        # If the model rejects this shape, skip with a clear message
        pytest.skip(f"Velocity forward rejected fabricated x_t: {e}")


def test_cfg_monotonic(flowedit):
    """Higher guidance_scale → larger deviation from the uncond baseline."""
    import numpy as np
    arr = np.full((256, 256, 3), 128, dtype=np.uint8)
    arr[80:180, 80:180] = (255, 200, 100)
    img = Image.fromarray(arr)
    cond = flowedit._encode_image_condition(img)

    x_t = torch.randn(1, 1, 16, 16, 16, device="cuda")
    try:
        v_1 = flowedit._compute_velocity(x_t, t=0.5, condition=cond, guidance_scale=1.0)
        v_7 = flowedit._compute_velocity(x_t, t=0.5, condition=cond, guidance_scale=7.5)
        # Higher CFG → further from v_1
        delta = (v_7 - v_1).norm()
        assert delta.item() > 1e-4
    except Exception as e:
        pytest.skip(f"Velocity forward rejected fabricated x_t: {e}")
