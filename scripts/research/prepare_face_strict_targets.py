#!/usr/bin/env python3
"""Convert proxy meshes into FACE training targets.

This is the target-side counterpart to `prepare_face_fixture_meshes.py`.
Fragments can be useful conditioning diagnostics, but FACE supervision usually
needs a bounded, topology-aware target mesh. This script runs the existing
coarse/reference adapter, exports accepted target meshes, and writes the target
gate report. By default it preserves the historical strict/watertight policy;
callers can opt into a paper-matching high-volume lane by relaxing the
watertight/boundary/nonmanifold thresholds explicitly.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clearmesh.mesh.coarse_adapter import CoarseAdapterOptions, adapt_coarse_mesh_file

MESH_SUFFIXES = {".glb", ".gltf", ".obj", ".ply", ".stl"}


def _iter_meshes(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.rglob("*") if path.suffix.lower() in MESH_SUFFIXES)


def _iter_manifest_rows(manifest: Path) -> list[dict]:
    if manifest.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "records" in data:
        return [row for row in data["records"] if row.get("accepted", row.get("status") == "accepted")]
    if isinstance(data, list):
        return data
    raise ValueError(f"unsupported manifest format: {manifest}")


def _manifest_path(row: dict) -> Path | None:
    for key in ("target_path", "path", "local_path", "source_path"):
        value = row.get(key)
        if value:
            path = Path(str(value))
            if path.suffix.lower() in MESH_SUFFIXES:
                return path
    return None


def _iter_manifest_meshes(manifest: Path) -> list[Path]:
    paths = []
    for row in _iter_manifest_rows(manifest):
        path = _manifest_path(row)
        if path is not None:
            paths.append(path)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None, help="JSON/JSONL manifest with path/target_path rows.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--engine", choices=["auto", "voxel_shell", "voxel", "poisson", "cleanup", "convex_hull"], default="convex_hull")
    parser.add_argument(
        "--target-faces",
        type=int,
        default=1400,
        help="Simplify outputs above this face count. Use 0 to disable simplification and control density via voxel resolution.",
    )
    parser.add_argument(
        "--max-target-face-ratio",
        type=float,
        default=1.25,
        help="Reject targets above target_faces * ratio so missing decimation cannot silently pass.",
    )
    parser.add_argument("--sample-points", type=int, default=120_000)
    parser.add_argument("--voxel-resolution", type=int, default=160)
    parser.add_argument("--voxel-dilate", type=int, default=2)
    parser.add_argument("--voxel-close", type=int, default=1)
    parser.add_argument("--mesh-voxel-max-faces", type=int, default=75_000)
    parser.add_argument("--poisson-depth", type=int, default=8)
    parser.add_argument("--fallback", choices=["convex_hull", "cleanup", ""], default="convex_hull")
    parser.add_argument(
        "--max-output-components",
        type=int,
        default=1,
        help="Reject adapted targets with more than this many connected components.",
    )
    parser.add_argument(
        "--max-boundary-loops",
        type=int,
        default=0,
        help="Reject adapted targets with more than this many boundary loops.",
    )
    parser.add_argument(
        "--max-nonmanifold-edges",
        type=int,
        default=0,
        help="Reject adapted targets with more than this many non-manifold edges.",
    )
    parser.add_argument(
        "--require-watertight",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require adapted targets to be watertight. Use --no-require-watertight for paper-matching high-volume lanes.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=25, help="Print JSON progress every N candidates. Use 0 to disable.")
    args = parser.parse_args()
    if args.input_dir is None and args.manifest is None:
        parser.error("one of --input-dir or --manifest is required")

    mesh_dir = args.output_dir / "meshes"
    report_dir = args.output_dir / "reports"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    options = CoarseAdapterOptions(
        engine=args.engine,
        target_faces=args.target_faces,
        max_target_face_ratio=args.max_target_face_ratio,
        sample_points=args.sample_points,
        max_output_components=args.max_output_components,
        max_boundary_loops=args.max_boundary_loops,
        max_nonmanifold_edges=args.max_nonmanifold_edges,
        require_watertight=args.require_watertight,
        voxel_resolution=args.voxel_resolution,
        voxel_dilate=args.voxel_dilate,
        voxel_close=args.voxel_close,
        mesh_voxel_max_faces=args.mesh_voxel_max_faces,
        poisson_depth=args.poisson_depth,
        fallback=args.fallback,
    )

    records = []
    accepted = 0
    rejected = 0
    errors = 0
    candidates = _iter_manifest_meshes(args.manifest) if args.manifest else _iter_meshes(args.input_dir)
    for index, path in enumerate(candidates):
        if args.limit and accepted >= args.limit:
            records.append({"source_path": str(path), "status": "not_attempted_limit_reached"})
            continue
        target_path = mesh_dir / f"{accepted:04d}_{path.stem}_strict.glb"
        report_path = report_dir / f"{accepted:04d}_{path.stem}.json"
        try:
            report = adapt_coarse_mesh_file(path, target_path, options)
            report_path.write_text(json.dumps(asdict(report), indent=2, sort_keys=True), encoding="utf-8")
            records.append(
                {
                    "source_path": str(path),
                    "target_path": str(target_path),
                    "report_path": str(report_path),
                    "status": "accepted" if report.accepted else "rejected",
                    "engine": report.engine,
                    "accepted": bool(report.accepted),
                    "output_metrics": report.output_metrics,
                }
            )
            if report.accepted:
                accepted += 1
            else:
                rejected += 1
        except Exception as exc:  # noqa: BLE001 - keep processing candidate meshes.
            errors += 1
            records.append({"source_path": str(path), "status": "error", "error": f"{type(exc).__name__}: {exc}"})
        if args.progress_every and ((index + 1) % args.progress_every == 0 or index + 1 == len(candidates)):
            print(
                json.dumps(
                    {
                        "accepted": accepted,
                        "candidate_count": len(candidates),
                        "errors": errors,
                        "processed": index + 1,
                        "rejected": rejected,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    summary = {
        "input_dir": str(args.input_dir) if args.input_dir else None,
        "manifest": str(args.manifest) if args.manifest else None,
        "mesh_dir": str(mesh_dir),
        "candidate_count": len(candidates),
        "accepted": accepted,
        "options": asdict(options),
        "records": records,
    }
    manifest_path = args.output_dir / "strict_target_manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, indent=2, sort_keys=True))
    print(f"Wrote {manifest_path}")
    return 0 if accepted > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
