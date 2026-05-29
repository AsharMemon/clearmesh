#!/usr/bin/env python3
"""Build voxel-anchored local FACE patch shards from indexed FACE NPZs."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clearmesh.mesh_heads.localizable_face import (
    LocalFacePatch,
    assert_patch_face_coverage,
    build_local_face_patches,
    patch_summary,
    vertex_anchor_coords,
)


def _build_one_source_worker(task: dict[str, object]) -> dict[str, object]:
    """Build one packed localizable source file.

    Kept at module scope so it is picklable for ``ProcessPoolExecutor``.
    """

    source_index = int(task["source_index"])
    path = Path(str(task["path"]))
    output_dir = Path(str(task["output_dir"]))
    standard_dir_text = str(task.get("standard_patch_sample_dir") or "")
    standard_dir = Path(standard_dir_text) if standard_dir_text else None
    data = _load_indexed_face_npz(path)
    vertices = np.asarray(data["vertices"], dtype=np.int64)
    faces = np.asarray(data["faces"], dtype=np.int64)
    num_bins = int(data["num_bins"])
    points = data["points"]
    normals = data["normals"]
    if points is not None and normals is not None and len(points) != len(normals):
        raise ValueError(f"{path} has mismatched surface point/normal counts")
    out_path = output_dir / "patches" / f"{source_index:08d}_{path.stem}.npz"
    return _write_patch_source(
        source_path=path,
        out_path=out_path,
        vertices=vertices,
        faces=faces,
        num_bins=num_bins,
        voxel_resolution=int(task["voxel_resolution"]),
        max_faces_per_patch=int(task["max_faces_per_patch"]),
        point_samples_per_patch=int(task["point_samples_per_patch"]),
        surface_points=points if isinstance(points, np.ndarray) else None,
        surface_normals=normals if isinstance(normals, np.ndarray) else None,
        source_index=source_index,
        standard_patch_sample_dir=standard_dir,
    )


def _iter_npz_paths(dataset_dir: Path, limit: int) -> list[Path]:
    paths = sorted(path for path in dataset_dir.rglob("*.npz") if not path.name.startswith("._"))
    if limit > 0:
        paths = paths[:limit]
    return paths


def _load_indexed_face_npz(path: Path) -> dict[str, np.ndarray | int]:
    with np.load(path) as data:
        required = {"indexed_vertices", "indexed_faces", "num_bins"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"{path} is missing indexed FACE arrays: {missing}")
        vertices = np.asarray(data["indexed_vertices"], dtype=np.int64)
        faces = np.asarray(data["indexed_faces"], dtype=np.int64)
        num_bins = int(np.asarray(data["num_bins"]).reshape(-1)[0])
        points = np.asarray(data["surface_points"], dtype=np.float32) if "surface_points" in data else None
        normals = np.asarray(data["surface_normals"], dtype=np.float32) if "surface_normals" in data else None
    return {
        "vertices": vertices,
        "faces": faces,
        "num_bins": num_bins,
        "points": points,
        "normals": normals,
    }


def _offsets_from_lengths(lengths: list[int]) -> np.ndarray:
    offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
    if lengths:
        offsets[1:] = np.cumsum(np.asarray(lengths, dtype=np.int64))
    return offsets


def _pack_patch_arrays(
    patches: list[LocalFacePatch],
    *,
    surface_points: np.ndarray | None,
    surface_normals: np.ndarray | None,
) -> dict[str, np.ndarray]:
    vertex_lengths = [patch.vertex_count for patch in patches]
    face_lengths = [patch.face_count for patch in patches]
    point_lengths = [int(len(patch.point_indices)) for patch in patches]
    vertices_flat = (
        np.concatenate([patch.vertices for patch in patches], axis=0).astype(np.int16)
        if patches
        else np.zeros((0, 3), dtype=np.int16)
    )
    faces_flat = (
        np.concatenate([patch.faces for patch in patches], axis=0).astype(np.int32)
        if patches
        else np.zeros((0, 3), dtype=np.int32)
    )
    global_vertex_indices_flat = (
        np.concatenate([patch.global_vertex_indices for patch in patches], axis=0).astype(np.int32)
        if patches
        else np.zeros((0,), dtype=np.int32)
    )
    source_face_indices_flat = (
        np.concatenate([patch.source_face_indices for patch in patches], axis=0).astype(np.int32)
        if patches
        else np.zeros((0,), dtype=np.int32)
    )
    point_indices_flat = (
        np.concatenate([patch.point_indices for patch in patches], axis=0).astype(np.int32)
        if patches
        else np.zeros((0,), dtype=np.int32)
    )
    if surface_points is not None and len(point_indices_flat):
        patch_points_flat = np.asarray(surface_points[point_indices_flat], dtype=np.float32)
    else:
        patch_points_flat = np.zeros((0, 3), dtype=np.float32)
    if surface_normals is not None and len(point_indices_flat):
        patch_normals_flat = np.asarray(surface_normals[point_indices_flat], dtype=np.float32)
    else:
        patch_normals_flat = np.zeros((0, 3), dtype=np.float32)

    return {
        "anchor_coords": np.asarray([patch.anchor for patch in patches], dtype=np.int16).reshape(-1, 3),
        "patch_vertex_offsets": _offsets_from_lengths(vertex_lengths),
        "patch_face_offsets": _offsets_from_lengths(face_lengths),
        "patch_point_offsets": _offsets_from_lengths(point_lengths),
        "patch_vertices_flat": vertices_flat,
        "patch_faces_flat": faces_flat,
        "global_vertex_indices_flat": global_vertex_indices_flat,
        "source_face_indices_flat": source_face_indices_flat,
        "point_indices_flat": point_indices_flat,
        "patch_points_flat": patch_points_flat,
        "patch_normals_flat": patch_normals_flat,
    }


def _write_patch_source(
    *,
    source_path: Path,
    out_path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    num_bins: int,
    voxel_resolution: int,
    max_faces_per_patch: int,
    point_samples_per_patch: int,
    surface_points: np.ndarray | None,
    surface_normals: np.ndarray | None,
    source_index: int,
    standard_patch_sample_dir: Path | None = None,
) -> dict[str, object]:
    patches = build_local_face_patches(
        vertices=vertices,
        faces=faces,
        num_bins=num_bins,
        voxel_resolution=voxel_resolution,
        max_faces_per_patch=max_faces_per_patch,
        surface_points=surface_points,
        point_samples_per_patch=point_samples_per_patch,
    )
    assert_patch_face_coverage(patches, len(faces))
    packed = _pack_patch_arrays(patches, surface_points=surface_points, surface_normals=surface_normals)
    vertex_anchors = vertex_anchor_coords(
        vertices,
        num_bins=num_bins,
        voxel_resolution=voxel_resolution,
    )
    standard_samples_written = 0
    if standard_patch_sample_dir is not None:
        standard_patch_sample_dir.mkdir(parents=True, exist_ok=True)
        standard_samples_written = _write_standard_patch_samples(
            patches,
            output_dir=standard_patch_sample_dir,
            source_index=source_index,
            source_stem=source_path.stem,
            num_bins=num_bins,
            surface_points=surface_points,
            surface_normals=surface_normals,
        )
    summary = patch_summary(patches)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        num_bins=np.asarray([num_bins], dtype=np.int32),
        voxel_resolution=np.asarray([voxel_resolution], dtype=np.int32),
        max_faces_per_patch=np.asarray([max_faces_per_patch], dtype=np.int32),
        source_vertices=np.asarray(vertices, dtype=np.int16),
        source_vertex_anchor_coords=vertex_anchors,
        source_face_count=np.asarray([len(faces)], dtype=np.int32),
        **packed,
    )
    return {
        "path": str(out_path),
        "source_path": str(source_path),
        "num_bins": int(num_bins),
        "voxel_resolution": int(voxel_resolution),
        "max_faces_per_patch": int(max_faces_per_patch),
        "point_samples_per_patch": int(point_samples_per_patch),
        "standard_patch_samples": int(standard_samples_written),
        "source_faces": int(len(faces)),
        **summary,
    }


def _write_standard_patch_samples(
    patches: list[LocalFacePatch],
    *,
    output_dir: Path,
    source_index: int,
    source_stem: str,
    num_bins: int,
    surface_points: np.ndarray | None,
    surface_normals: np.ndarray | None,
) -> int:
    written = 0
    for patch_index, patch in enumerate(patches):
        point_indices = np.asarray(patch.point_indices, dtype=np.int64)
        if surface_points is not None and len(point_indices):
            points = np.asarray(surface_points[point_indices], dtype=np.float32)
        else:
            points = np.zeros((1, 3), dtype=np.float32)
        if surface_normals is not None and len(point_indices):
            normals = np.asarray(surface_normals[point_indices], dtype=np.float32)
        else:
            normals = np.zeros((len(points), 3), dtype=np.float32)
        if len(normals) != len(points):
            normals = np.zeros((len(points), 3), dtype=np.float32)
        out_path = output_dir / f"{source_index:08d}_{patch_index:04d}_{source_stem}.npz"
        np.savez_compressed(
            out_path,
            indexed_vertices=np.asarray(patch.vertices, dtype=np.int16),
            indexed_faces=np.asarray(patch.faces, dtype=np.int32),
            surface_points=points,
            surface_normals=normals,
            num_bins=np.asarray([num_bins], dtype=np.int32),
            anchor_coord=np.asarray(patch.anchor, dtype=np.int16),
            source_face_indices=np.asarray(patch.source_face_indices, dtype=np.int32),
            global_vertex_indices=np.asarray(patch.global_vertex_indices, dtype=np.int32),
        )
        written += 1
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--voxel-resolution", type=int, default=32)
    parser.add_argument("--max-faces-per-patch", type=int, default=128)
    parser.add_argument("--point-samples-per-patch", type=int, default=64)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel source builders. Use 1 for deterministic single-process debugging.",
    )
    parser.add_argument(
        "--standard-patch-sample-dir",
        type=Path,
        default=None,
        help="Optional directory of one-NPZ-per-patch samples compatible with the existing indexed FACE trainer.",
    )
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    patch_dir = args.output_dir / "patches"
    manifest_path = args.output_dir / "manifest.jsonl"
    summary_path = args.output_dir / "summary.json"
    paths = _iter_npz_paths(args.dataset_dir, args.limit)
    rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    tasks = [
        {
            "source_index": source_index,
            "path": str(path),
            "output_dir": str(args.output_dir),
            "voxel_resolution": int(args.voxel_resolution),
            "max_faces_per_patch": int(args.max_faces_per_patch),
            "point_samples_per_patch": int(args.point_samples_per_patch),
            "standard_patch_sample_dir": str(args.standard_patch_sample_dir) if args.standard_patch_sample_dir else "",
        }
        for source_index, path in enumerate(paths)
    ]
    completed: dict[int, dict[str, object]] = {}

    def handle_result(source_index: int, path: Path, result: concurrent.futures.Future[dict[str, object]] | dict[str, object]) -> None:
        try:
            row = result.result() if isinstance(result, concurrent.futures.Future) else result
        except Exception as exc:
            failure = {"path": str(path), "error": f"{type(exc).__name__}: {exc}"}
            failures.append(failure)
            if args.fail_fast:
                raise
            return
        completed[source_index] = row
        if len(completed) % 500 == 0:
            print(json.dumps({"processed": len(completed), "failed": len(failures)}, sort_keys=True), flush=True)

    worker_count = max(1, int(args.workers))
    if worker_count == 1:
        for task in tasks:
            handle_result(int(task["source_index"]), Path(str(task["path"])), _build_one_source_worker(task))
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as pool:
            future_map = {
                pool.submit(_build_one_source_worker, task): (int(task["source_index"]), Path(str(task["path"])))
                for task in tasks
            }
            for future in concurrent.futures.as_completed(future_map):
                source_index, path = future_map[future]
                handle_result(source_index, path, future)

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for source_index in sorted(completed):
            row = completed[source_index]
            rows.append(row)
            manifest.write(json.dumps(row, sort_keys=True) + "\n")

    totals = {
        "input_npz": int(len(paths)),
        "written_sources": int(len(rows)),
        "failed_sources": int(len(failures)),
        "workers": int(worker_count),
        "patches": int(sum(int(row["patches"]) for row in rows)),
        "faces": int(sum(int(row["faces"]) for row in rows)),
        "source_faces": int(sum(int(row["source_faces"]) for row in rows)),
        "vertices_in_patches": int(sum(int(row["vertices"]) for row in rows)),
        "standard_patch_samples": int(sum(int(row["standard_patch_samples"]) for row in rows)),
        "voxel_resolution": int(args.voxel_resolution),
        "max_faces_per_patch": int(args.max_faces_per_patch),
        "point_samples_per_patch": int(args.point_samples_per_patch),
        "manifest": str(manifest_path),
        "patch_dir": str(patch_dir),
        "failures": failures[:20],
    }
    summary_path.write_text(json.dumps(totals, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(totals, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
