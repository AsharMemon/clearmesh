#!/usr/bin/env python3
"""Normalize a generator mesh into a bounded control surface."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clearmesh.mesh.normalization import SurfaceNormalizationOptions, normalize_surface_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--engine", default="auto", choices=["auto", "poisson", "cleanup"])
    parser.add_argument("--target-faces", type=int, default=50_000)
    parser.add_argument("--sample-points", type=int, default=120_000)
    parser.add_argument("--poisson-depth", type=int, default=8)
    parser.add_argument("--density-quantile", type=float, default=0.02)
    parser.add_argument("--orient-normals", action="store_true")
    args = parser.parse_args()

    report = normalize_surface_file(
        args.input,
        args.output,
        SurfaceNormalizationOptions(
            engine=args.engine,
            target_faces=args.target_faces,
            sample_points=args.sample_points,
            poisson_depth=args.poisson_depth,
            density_quantile=args.density_quantile,
            orient_normals=args.orient_normals,
        ),
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(asdict(report), indent=2, sort_keys=True), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
