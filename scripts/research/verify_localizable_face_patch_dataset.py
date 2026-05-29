#!/usr/bin/env python3
"""Verify packed localizable FACE patch manifests.

This is the M1 gate that proves local patches are lossless with respect to the
indexed FACE target: each source face must appear exactly once, and when the
original source NPZ is available the packed local faces must reconstruct the
exact original indexed face table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clearmesh.mesh_heads.localizable_face import reconstruct_source_faces_from_packed_arrays


def _manifest_rows(manifest: Path, limit_sources: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    root = manifest.parent
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        path = Path(str(row["path"]))
        if not path.is_absolute() and not path.exists():
            row["path"] = str(root / path)
        rows.append(row)
        if limit_sources and len(rows) >= int(limit_sources):
            break
    return rows


def _load_original_faces(path_text: str) -> np.ndarray | None:
    path = Path(path_text)
    if not path.exists():
        return None
    with np.load(path) as data:
        if "indexed_faces" not in data.files:
            raise ValueError(f"{path} is missing indexed_faces")
        return np.asarray(data["indexed_faces"], dtype=np.int64)


def _check_offsets(name: str, offsets: np.ndarray, flat_len: int) -> None:
    arr = np.asarray(offsets, dtype=np.int64)
    if arr.ndim != 1 or len(arr) == 0:
        raise ValueError(f"{name} must be a non-empty 1D offset array")
    if int(arr[0]) != 0:
        raise ValueError(f"{name} must start at zero")
    if np.any(np.diff(arr) < 0):
        raise ValueError(f"{name} must be monotonic")
    if int(arr[-1]) != int(flat_len):
        raise ValueError(f"{name} final offset {int(arr[-1])} != flat length {flat_len}")


def _export_debug_mesh(path: Path, *, vertices: np.ndarray, faces: np.ndarray, num_bins: int) -> None:
    try:
        import trimesh
    except Exception as exc:  # pragma: no cover - optional diagnostic path.
        raise RuntimeError("trimesh is required for --export-debug-mesh-dir") from exc
    coords = ((np.asarray(vertices, dtype=np.float32) + 0.5) / float(num_bins)) * 2.0 - 1.0
    mesh = trimesh.Trimesh(vertices=coords, faces=np.asarray(faces, dtype=np.int64), process=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path)


def _verify_row(
    row: dict[str, object],
    *,
    compare_source: bool,
    export_debug_mesh_dir: Path | None,
    debug_mesh_limit: int,
    debug_mesh_count: int,
) -> tuple[dict[str, object], int]:
    packed_path = Path(str(row["path"]))
    source_path = str(row.get("source_path") or "")
    with np.load(packed_path) as data:
        required = {
            "num_bins",
            "source_vertices",
            "source_face_count",
            "anchor_coords",
            "patch_vertex_offsets",
            "patch_face_offsets",
            "patch_point_offsets",
            "patch_vertices_flat",
            "patch_faces_flat",
            "global_vertex_indices_flat",
            "source_face_indices_flat",
            "patch_points_flat",
        }
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"{packed_path} is missing required arrays: {missing}")
        source_vertices = np.asarray(data["source_vertices"], dtype=np.int64)
        face_count = int(np.asarray(data["source_face_count"]).reshape(-1)[0])
        num_bins = int(np.asarray(data["num_bins"]).reshape(-1)[0])
        patch_vertex_offsets = np.asarray(data["patch_vertex_offsets"], dtype=np.int64)
        patch_face_offsets = np.asarray(data["patch_face_offsets"], dtype=np.int64)
        patch_point_offsets = np.asarray(data["patch_point_offsets"], dtype=np.int64)
        patch_faces_flat = np.asarray(data["patch_faces_flat"], dtype=np.int64)
        patch_vertices_flat = np.asarray(data["patch_vertices_flat"], dtype=np.int64)
        global_vertex_indices_flat = np.asarray(data["global_vertex_indices_flat"], dtype=np.int64)
        source_face_indices_flat = np.asarray(data["source_face_indices_flat"], dtype=np.int64)
        patch_points_flat = np.asarray(data["patch_points_flat"], dtype=np.float32)
        anchor_coords = np.asarray(data["anchor_coords"], dtype=np.int64)

        _check_offsets("patch_vertex_offsets", patch_vertex_offsets, len(patch_vertices_flat))
        _check_offsets("patch_face_offsets", patch_face_offsets, len(patch_faces_flat))
        _check_offsets("patch_point_offsets", patch_point_offsets, len(patch_points_flat))
        if len(patch_face_offsets) != len(anchor_coords) + 1:
            raise ValueError("anchor count does not match patch count")
        if len(source_vertices) == 0:
            raise ValueError("source_vertices must not be empty")

        reconstructed = reconstruct_source_faces_from_packed_arrays(
            patch_faces_flat=patch_faces_flat,
            patch_face_offsets=patch_face_offsets,
            global_vertex_indices_flat=global_vertex_indices_flat,
            patch_vertex_offsets=patch_vertex_offsets,
            source_face_indices_flat=source_face_indices_flat,
            face_count=face_count,
        )

        source_compared = False
        source_exact = None
        original_faces = _load_original_faces(source_path) if compare_source and source_path else None
        if original_faces is not None:
            source_compared = True
            source_exact = bool(np.array_equal(reconstructed, original_faces))
            if not source_exact:
                mismatch = int(np.flatnonzero(np.any(reconstructed != original_faces, axis=1))[0])
                raise ValueError(f"{packed_path} does not reconstruct source face row {mismatch}")

        if export_debug_mesh_dir is not None and debug_mesh_count < int(debug_mesh_limit):
            out_name = f"{debug_mesh_count:04d}_{packed_path.stem}.obj"
            _export_debug_mesh(
                export_debug_mesh_dir / out_name,
                vertices=source_vertices,
                faces=reconstructed,
                num_bins=num_bins,
            )
            debug_mesh_count += 1

        patch_face_counts = np.diff(patch_face_offsets)
        info = {
            "path": str(packed_path),
            "source_path": source_path,
            "source_faces": int(face_count),
            "patches": int(len(anchor_coords)),
            "max_faces_per_patch": int(patch_face_counts.max()) if len(patch_face_counts) else 0,
            "mean_faces_per_patch": float(patch_face_counts.mean()) if len(patch_face_counts) else 0.0,
            "source_compared": source_compared,
            "source_exact": source_exact,
        }
        return info, debug_mesh_count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--limit-sources", type=int, default=0)
    parser.add_argument("--no-source-compare", action="store_true")
    parser.add_argument("--export-debug-mesh-dir", type=Path, default=None)
    parser.add_argument("--debug-mesh-limit", type=int, default=4)
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    rows = _manifest_rows(args.manifest, args.limit_sources)
    verified: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    debug_mesh_count = 0
    for row in rows:
        try:
            info, debug_mesh_count = _verify_row(
                row,
                compare_source=not args.no_source_compare,
                export_debug_mesh_dir=args.export_debug_mesh_dir,
                debug_mesh_limit=args.debug_mesh_limit,
                debug_mesh_count=debug_mesh_count,
            )
        except Exception as exc:
            failures.append({"path": str(row.get("path")), "error": f"{type(exc).__name__}: {exc}"})
            if args.fail_fast:
                raise
            continue
        verified.append(info)
        if len(verified) % 500 == 0:
            print(json.dumps({"verified": len(verified), "failed": len(failures)}, sort_keys=True), flush=True)

    report = {
        "manifest": str(args.manifest),
        "input_sources": int(len(rows)),
        "verified_sources": int(len(verified)),
        "failed_sources": int(len(failures)),
        "source_compared": int(sum(1 for item in verified if item["source_compared"])),
        "source_exact": int(sum(1 for item in verified if item["source_exact"] is True)),
        "faces": int(sum(int(item["source_faces"]) for item in verified)),
        "patches": int(sum(int(item["patches"]) for item in verified)),
        "max_faces_per_patch": int(max((int(item["max_faces_per_patch"]) for item in verified), default=0)),
        "debug_meshes_exported": int(debug_mesh_count),
        "failures": failures[:20],
    }
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
