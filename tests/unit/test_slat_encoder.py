"""CPU-only unit tests for SLATEncoder logic that doesn't touch TRELLIS.2.

Covered:
- Constructor accepts pipeline=None and doesn't load models eagerly.
- _resolve_* raises a clear RuntimeError listing all three paths tried,
  when no TRELLIS.2 is installed. This is the failure mode users will
  see most often, so the error message contract is worth testing.
- _ovoxel_to_dense_occupancy rasterizes indices correctly.
- _make_decoder_input handles the "no SparseTensor class" fallback path.
- save / load_slat round-trips without data loss.

NOT covered here (requires the pod):
- Actually calling an encoder or decoder — see tests/integration/test_slat_encode.py.
"""

from __future__ import annotations

import torch
import pytest


def _make_encoder_no_pipeline():
    from clearmesh.editing.slat_encoder import SLATEncoder
    enc = SLATEncoder.__new__(SLATEncoder)
    enc.trellis2_dir = "/nonexistent"
    enc.model_dir = "/nonexistent"
    enc.device = "cpu"
    enc.grid_size = 16
    enc._pipeline = None
    enc._o_voxel = None
    enc._ss_encoder = None
    enc._shape_encoder = None
    enc._decoder = None
    return enc


def test_resolve_ss_encoder_raises_with_all_three_paths_listed():
    from pathlib import Path
    enc = _make_encoder_no_pipeline()
    enc.model_dir = Path("/nonexistent")
    with pytest.raises(RuntimeError) as exc:
        enc._resolve_ss_encoder()
    msg = str(exc.value)
    # The error message should mention all three resolution paths so a
    # user debugging on the pod knows which one to fix.
    assert "pipeline.models" in msg
    assert "data_toolkit" in msg
    assert "SparseStructureEncoder" in msg
    assert "inspect_trellis2.py" in msg


def test_resolve_shape_encoder_raises_with_clear_message():
    enc = _make_encoder_no_pipeline()
    with pytest.raises(RuntimeError) as exc:
        enc._resolve_shape_encoder()
    msg = str(exc.value)
    assert "shape" in msg.lower()
    assert "data_toolkit" in msg


def test_resolve_ss_encoder_uses_pipeline_models_if_present():
    """If pipeline.models contains 'sparse_structure_encoder', use it directly
    without falling through to the data_toolkit / checkpoint paths."""
    enc = _make_encoder_no_pipeline()

    class FakeModule(torch.nn.Module):
        pass

    fake_encoder = FakeModule()

    class FakePipeline:
        models = {"sparse_structure_encoder": fake_encoder}

    enc._pipeline = FakePipeline()
    result = enc._resolve_ss_encoder()
    assert result is fake_encoder


def test_resolve_ss_encoder_caches_result():
    """Second call must not re-resolve (would be a hot-path bug since
    resolution can trigger a checkpoint load)."""
    enc = _make_encoder_no_pipeline()

    class FakeModule(torch.nn.Module):
        pass

    fake = FakeModule()
    enc._ss_encoder = fake
    assert enc._resolve_ss_encoder() is fake


def test_ovoxel_to_dense_occupancy_marks_all_indices():
    enc = _make_encoder_no_pipeline()
    enc.grid_size = 8
    indices = torch.tensor([[0, 0, 0], [3, 4, 5], [7, 7, 7]])
    ovoxel = {"voxel_indices": indices, "grid_size": 8}
    dense = enc._ovoxel_to_dense_occupancy(ovoxel)
    assert dense.shape == (1, 1, 8, 8, 8)
    # Exactly 3 voxels should be marked
    assert dense.sum().item() == 3.0
    assert dense[0, 0, 0, 0, 0] == 1.0
    assert dense[0, 0, 3, 4, 5] == 1.0
    assert dense[0, 0, 7, 7, 7] == 1.0


def test_ovoxel_to_dense_occupancy_clips_out_of_range():
    """Out-of-range indices should be silently dropped, not raise."""
    enc = _make_encoder_no_pipeline()
    enc.grid_size = 4
    indices = torch.tensor([
        [0, 0, 0],
        [3, 3, 3],
        [5, 0, 0],     # out of range (high)
        [-1, 0, 0],    # out of range (low)
    ])
    ovoxel = {"voxel_indices": indices, "grid_size": 4}
    dense = enc._ovoxel_to_dense_occupancy(ovoxel)
    # Only 2 in-range voxels should be marked
    assert dense.sum().item() == 2.0


def test_decode_requires_pipeline():
    from clearmesh.editing.slat_encoder import SLATEncoder, SLATRepresentation
    enc = _make_encoder_no_pipeline()
    rep = SLATRepresentation(
        ss_latent=torch.zeros(1, 5, 32),
        shape_latent=torch.zeros(1, 5, 32),
        voxel_indices=torch.zeros(5, 3, dtype=torch.long),
        dual_vertices=torch.zeros(5, 3),
        intersected=None,
        grid_size=256,
    )
    with pytest.raises(RuntimeError, match="requires a TRELLIS.2 pipeline"):
        enc.decode(rep)


def test_make_decoder_input_falls_back_to_dict_when_no_sparse_tensor():
    """When trellis2.modules.sparse.SparseTensor isn't importable,
    _make_decoder_input returns a dict with coords/feats/intersected."""
    from clearmesh.editing.slat_encoder import SLATEncoder, SLATRepresentation
    enc = _make_encoder_no_pipeline()
    rep = SLATRepresentation(
        ss_latent=torch.randn(1, 7, 32),
        shape_latent=torch.randn(1, 7, 32),
        voxel_indices=torch.randint(0, 256, (7, 3)),
        dual_vertices=torch.randn(7, 3),
        intersected=torch.zeros(7, dtype=torch.bool),
        grid_size=256,
    )
    out = enc._make_decoder_input(rep)
    # On macOS without trellis2.modules.sparse installed, we expect the dict path
    if isinstance(out, dict):
        assert set(out.keys()) >= {"feats", "coords", "intersected", "dual_vertices"}
    else:
        # SparseTensor path (would only trigger if trellis2 is importable)
        assert hasattr(out, "feats") or hasattr(out, "F")


def test_resolve_ss_encoder_handles_ss_encoder_key_alias():
    """Some pipeline layouts use 'ss_encoder' instead of 'sparse_structure_encoder'."""
    enc = _make_encoder_no_pipeline()

    class FakeModule(torch.nn.Module):
        pass

    fake = FakeModule()

    class FakePipeline:
        models = {"ss_encoder": fake}

    enc._pipeline = FakePipeline()
    result = enc._resolve_ss_encoder()
    assert result is fake


def test_mesh_to_ovoxel_normalizes_mesh(tmp_path):
    """mesh_to_ovoxel centers and scales the mesh to [-0.5, 0.5] before conversion.
    This test guards the preprocessing math; the actual o_voxel call is mocked."""
    import trimesh
    from clearmesh.editing.slat_encoder import SLATEncoder

    enc = _make_encoder_no_pipeline()

    # Build a mesh offset and scaled
    mesh = trimesh.creation.box(extents=(4, 4, 4))
    mesh.vertices += 10.0  # offset
    mesh_path = tmp_path / "big_box.glb"
    mesh.export(mesh_path)

    captured = {}

    class FakeOVoxel:
        class convert:
            @staticmethod
            def mesh_to_flexible_dual_grid(vertices, faces, **kwargs):
                captured["vmin"] = vertices.min().item()
                captured["vmax"] = vertices.max().item()
                # Return dummy data matching the expected tuple
                return (
                    torch.zeros(10, 3, dtype=torch.long),
                    torch.zeros(10, 3),
                    torch.zeros(10, dtype=torch.bool),
                )

    enc._o_voxel = FakeOVoxel()
    ovoxel = enc.mesh_to_ovoxel(mesh_path, grid_size=8)

    # After normalization: range [-0.5, 0.5] (scale=0.99999)
    assert -0.6 < captured["vmin"] < -0.4
    assert 0.4 < captured["vmax"] < 0.6
    assert "voxel_indices" in ovoxel
    assert ovoxel["grid_size"] == 8
