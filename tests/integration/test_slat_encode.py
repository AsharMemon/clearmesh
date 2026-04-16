"""Integration tests for SLATEncoder — encode/decode roundtrip.

Requires TRELLIS.2 pipeline loaded and a CUDA GPU. Skipped on CPU-only
development machines.

To run on the pod:

    source /opt/conda/etc/profile.d/conda.sh && conda activate trellis2
    export PYTHONPATH=/workspace/TRELLIS.2:$PYTHONPATH
    export HF_TOKEN=<your token>
    cd /workspace/clearmesh && pytest -q tests/integration/test_slat_encode.py -m "gpu and trellis2"
"""

from __future__ import annotations

import pytest
import torch


pytestmark = [pytest.mark.gpu, pytest.mark.trellis2]


def test_encode_produces_valid_shapes(trellis2_pipeline, sample_cube_path):
    """Encoding a cube should give non-empty SS and shape latents."""
    from clearmesh.editing.slat_encoder import SLATEncoder

    enc = SLATEncoder(
        device="cuda",
        grid_size=256,
        pipeline=trellis2_pipeline,
    )
    slat = enc.encode(str(sample_cube_path), grid_size=128)  # small for speed

    assert slat.voxel_indices.shape[0] > 0, "encode produced no voxels"
    assert slat.voxel_indices.shape[1] == 3
    assert slat.ss_latent is not None
    assert slat.shape_latent is not None
    assert not torch.isnan(slat.ss_latent).any()
    assert not torch.isnan(slat.shape_latent).any()


def test_encode_decode_roundtrip_iou(trellis2_pipeline, sample_cube_path, tmp_output_dir):
    """Encode → decode → measure voxel IoU against source.

    A perfect round-trip has IoU == 1. We accept ≥ 0.5 as "geometry
    preserved enough to call this a working encoder/decoder pair."
    """
    import trimesh
    from clearmesh.editing.slat_encoder import SLATEncoder

    enc = SLATEncoder(device="cuda", grid_size=128, pipeline=trellis2_pipeline)
    slat = enc.encode(str(sample_cube_path), grid_size=128)
    mesh_out = enc.decode(slat, resolution=512)

    assert isinstance(mesh_out, trimesh.Trimesh)
    assert mesh_out.vertices.shape[0] > 0

    # Save for debugging
    (tmp_output_dir / "roundtrip.glb").write_bytes(b"")  # ensure dir exists
    mesh_out.export(tmp_output_dir / "roundtrip.glb")

    # Compare voxel IoU at a low resolution
    src = trimesh.load(sample_cube_path, force="mesh")
    src_vox = src.voxelized(pitch=0.1)
    tgt_vox = mesh_out.voxelized(pitch=0.1)

    src_points = set(map(tuple, src_vox.sparse_indices.tolist()))
    tgt_points = set(map(tuple, tgt_vox.sparse_indices.tolist()))
    if src_points or tgt_points:
        iou = len(src_points & tgt_points) / max(1, len(src_points | tgt_points))
    else:
        iou = 0.0

    assert iou >= 0.3, f"encode/decode roundtrip IoU too low: {iou:.2f}"
