from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import trimesh

from clearmesh.mesh_heads.face_indexed import encode_mesh_to_indexed_face_tokens
from clearmesh.mesh_heads.localizable_face import (
    assert_patch_face_coverage,
    build_local_face_patches,
    face_centroid_anchors,
    reconstruct_source_faces_from_patches,
    reconstruct_source_faces_from_packed_arrays,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _indexed_box(num_bins: int = 32):
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    return mesh, encode_mesh_to_indexed_face_tokens(
        mesh,
        num_bins=num_bins,
        face_order="boundary_growth",
    )


def _write_indexed_npz(path: Path, mesh: trimesh.Trimesh, num_bins: int = 32) -> None:
    sequence = encode_mesh_to_indexed_face_tokens(mesh, num_bins=num_bins, face_order="boundary_growth")
    points = sequence.transform.normalize(mesh.triangles_center).astype(np.float32)
    normals = np.asarray(mesh.face_normals, dtype=np.float32)
    np.savez_compressed(
        path,
        indexed_vertices=np.asarray(sequence.vertices, dtype=np.int16),
        indexed_faces=np.asarray(sequence.faces, dtype=np.int32),
        num_bins=np.asarray([sequence.num_bins], dtype=np.int32),
        surface_points=points,
        surface_normals=normals,
    )


def test_face_centroid_anchors_are_bounded_and_deterministic() -> None:
    _, sequence = _indexed_box(num_bins=32)
    first = face_centroid_anchors(sequence.vertices, sequence.faces, num_bins=32, voxel_resolution=4)
    second = face_centroid_anchors(sequence.vertices, sequence.faces, num_bins=32, voxel_resolution=4)

    assert first.shape == (len(sequence.faces), 3)
    assert np.all(first >= 0)
    assert np.all(first < 4)
    np.testing.assert_array_equal(first, second)


def test_local_face_patches_cover_every_source_face_once() -> None:
    mesh, sequence = _indexed_box(num_bins=32)
    points = sequence.transform.normalize(mesh.triangles_center).astype(np.float32)
    patches = build_local_face_patches(
        vertices=sequence.vertices,
        faces=sequence.faces,
        num_bins=sequence.num_bins,
        voxel_resolution=4,
        max_faces_per_patch=2,
        surface_points=points,
        point_samples_per_patch=1,
    )

    assert patches
    assert_patch_face_coverage(patches, len(sequence.faces))
    assert max(patch.face_count for patch in patches) <= 2
    covered = np.concatenate([patch.source_face_indices for patch in patches])
    np.testing.assert_array_equal(np.sort(covered), np.arange(len(sequence.faces)))
    reconstructed = reconstruct_source_faces_from_patches(patches, face_count=len(sequence.faces))
    np.testing.assert_array_equal(reconstructed, sequence.faces)


def test_build_localizable_face_patch_dataset_writes_packed_sources(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    _write_indexed_npz(dataset_dir / "box.npz", trimesh.creation.box(), num_bins=32)
    _write_indexed_npz(dataset_dir / "wide_box.npz", trimesh.creation.box(extents=(2.0, 1.0, 0.5)), num_bins=32)

    output_dir = tmp_path / "patches"
    standard_dir = tmp_path / "standard_patch_samples"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/research/build_localizable_face_patch_dataset.py"),
            "--dataset-dir",
            str(dataset_dir),
            "--output-dir",
            str(output_dir),
            "--voxel-resolution",
            "4",
            "--max-faces-per-patch",
            "2",
            "--point-samples-per-patch",
            "1",
            "--standard-patch-sample-dir",
            str(standard_dir),
        ],
        cwd=REPO_ROOT,
        check=True,
    )

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["input_npz"] == 2
    assert summary["written_sources"] == 2
    assert summary["failed_sources"] == 0
    assert summary["source_faces"] == 24
    assert summary["faces"] == 24
    assert summary["patches"] >= 12
    assert summary["standard_patch_samples"] == summary["patches"]
    assert len(list(standard_dir.glob("*.npz"))) == summary["patches"]

    first_patch_file = sorted((output_dir / "patches").glob("*.npz"))[0]
    with np.load(first_patch_file) as data:
        source_face_indices = np.asarray(data["source_face_indices_flat"], dtype=np.int64)
        face_offsets = np.asarray(data["patch_face_offsets"], dtype=np.int64)
        anchors = np.asarray(data["anchor_coords"], dtype=np.int64)
        assert face_offsets[0] == 0
        assert face_offsets[-1] == len(source_face_indices)
        np.testing.assert_array_equal(np.sort(source_face_indices), np.arange(12))
        assert np.all(anchors >= 0)
        assert np.all(anchors < 4)
        reconstructed = reconstruct_source_faces_from_packed_arrays(
            patch_faces_flat=np.asarray(data["patch_faces_flat"], dtype=np.int64),
            patch_face_offsets=face_offsets,
            global_vertex_indices_flat=np.asarray(data["global_vertex_indices_flat"], dtype=np.int64),
            patch_vertex_offsets=np.asarray(data["patch_vertex_offsets"], dtype=np.int64),
            source_face_indices_flat=source_face_indices,
            face_count=12,
        )
        np.testing.assert_array_equal(reconstructed, np.asarray(np.load(dataset_dir / "box.npz")["indexed_faces"]))

    report_path = tmp_path / "verify_report.json"
    mesh_dir = tmp_path / "debug_meshes"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/research/verify_localizable_face_patch_dataset.py"),
            "--manifest",
            str(output_dir / "manifest.jsonl"),
            "--report",
            str(report_path),
            "--export-debug-mesh-dir",
            str(mesh_dir),
            "--debug-mesh-limit",
            "1",
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["verified_sources"] == 2
    assert report["failed_sources"] == 0
    assert report["source_exact"] == 2
    assert len(list(mesh_dir.glob("*.obj"))) == 1

    stitch_report_path = tmp_path / "stitch_report.json"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/research/eval_localizable_face_patch_stitch.py"),
            "--manifest",
            str(output_dir / "manifest.jsonl"),
            "--report",
            str(stitch_report_path),
            "--strategy",
            "teacher_identity",
            "--pair-samples",
            "0",
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    stitch_report = json.loads(stitch_report_path.read_text(encoding="utf-8"))
    stitch_summary = stitch_report["summary"]["teacher_identity"]
    assert stitch_summary["sources"] == 2
    assert stitch_summary["teacher_identity_exact_count"] == 2
    assert stitch_summary["mean_ordered_source_face_coverage"] == 1.0
    assert stitch_summary["mean_unordered_source_face_coverage"] == 1.0
    assert stitch_summary["predicted_watertight_count"] == 2
