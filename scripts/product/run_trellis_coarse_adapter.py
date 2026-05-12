#!/usr/bin/env python3
"""Adapt a TRELLIS/TRELLIS.2 mesh into a coherent coarse UltraShape proxy."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clearmesh.mesh.coarse_adapter import CoarseAdapterOptions, adapt_coarse_mesh_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--engine", default="auto", choices=["auto", "voxel_shell", "voxel", "poisson", "cleanup", "convex_hull"])
    parser.add_argument("--target-faces", type=int, default=150_000)
    parser.add_argument("--sample-points", type=int, default=180_000)
    parser.add_argument("--min-component-faces", type=int, default=64)
    parser.add_argument("--min-component-face-ratio", type=float, default=0.0001)
    parser.add_argument("--keep-largest-components", type=int, default=128)
    parser.add_argument("--max-output-components", type=int, default=1)
    parser.add_argument("--max-boundary-loops", type=int, default=0)
    parser.add_argument("--max-nonmanifold-edges", type=int, default=0)
    parser.add_argument("--allow-open", action="store_true")
    parser.add_argument("--voxel-resolution", type=int, default=192)
    parser.add_argument("--voxel-dilate", type=int, default=2)
    parser.add_argument("--voxel-close", type=int, default=1)
    parser.add_argument("--voxel-pad-ratio", type=float, default=0.08)
    parser.add_argument("--mesh-voxel-max-faces", type=int, default=75_000)
    parser.add_argument("--hull-max-points", type=int, default=20_000)
    parser.add_argument("--poisson-depth", type=int, default=8)
    parser.add_argument("--poisson-density-quantile", type=float, default=0.01)
    parser.add_argument("--orient-normals", action="store_true")
    parser.add_argument("--fallback", default="convex_hull", choices=["convex_hull", "cleanup", ""])
    args = parser.parse_args()

    options = CoarseAdapterOptions(
        engine=args.engine,
        target_faces=args.target_faces,
        sample_points=args.sample_points,
        min_component_faces=args.min_component_faces,
        min_component_face_ratio=args.min_component_face_ratio,
        keep_largest_components=args.keep_largest_components,
        max_output_components=args.max_output_components,
        max_boundary_loops=args.max_boundary_loops,
        max_nonmanifold_edges=args.max_nonmanifold_edges,
        require_watertight=not args.allow_open,
        voxel_resolution=args.voxel_resolution,
        voxel_dilate=args.voxel_dilate,
        voxel_close=args.voxel_close,
        voxel_pad_ratio=args.voxel_pad_ratio,
        mesh_voxel_max_faces=args.mesh_voxel_max_faces,
        hull_max_points=args.hull_max_points,
        poisson_depth=args.poisson_depth,
        poisson_density_quantile=args.poisson_density_quantile,
        orient_normals=args.orient_normals,
        fallback=args.fallback,
    )
    report = adapt_coarse_mesh_file(args.input, args.output, options)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(asdict(report), indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(asdict(report), indent=2, sort_keys=True))
    return 0 if report.accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
