#!/usr/bin/env python3
"""Stitch chart remesh outputs into one quad candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clearmesh.retopology.chart_stitch import ChartStitchOptions, stitch_chart_remesh_outputs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--weld-tolerance", type=float, default=1e-6)
    parser.add_argument("--min-quad-ratio", type=float, default=0.85)
    parser.add_argument("--max-components", type=int, default=16)
    parser.add_argument("--max-boundary-loops", type=int, default=128)
    parser.add_argument("--require-watertight", action="store_true")
    args = parser.parse_args()

    report = stitch_chart_remesh_outputs(
        args.manifest,
        args.output,
        ChartStitchOptions(
            weld_tolerance=args.weld_tolerance,
            min_quad_ratio=args.min_quad_ratio,
            max_components=args.max_components,
            max_boundary_loops=args.max_boundary_loops,
            require_watertight=args.require_watertight,
        ),
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
