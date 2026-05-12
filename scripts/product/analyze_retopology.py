#!/usr/bin/env python3
"""Analyze a mesh into generic retopology charts and feature constraints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clearmesh.retopology.feature_graph import (  # noqa: E402
    RetopologyPlanningOptions,
    analyze_retopology_file,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--crease-angle", type=float, default=45.0)
    parser.add_argument("--target-quads", type=int, default=5_000)
    parser.add_argument("--min-chart-faces", type=int, default=16)
    parser.add_argument("--max-report-charts", type=int, default=256)
    parser.add_argument("--no-merge-small-charts", action="store_true")
    parser.add_argument("--merge-min-faces", type=int, default=0)
    parser.add_argument("--merge-max-passes", type=int, default=4)
    parser.add_argument("--no-face-indices", action="store_true")
    args = parser.parse_args()

    plan = analyze_retopology_file(
        args.input,
        RetopologyPlanningOptions(
            crease_angle_degrees=args.crease_angle,
            target_quads=args.target_quads,
            min_chart_faces=args.min_chart_faces,
            max_report_charts=args.max_report_charts,
            merge_small_charts=not args.no_merge_small_charts,
            merge_min_faces=args.merge_min_faces,
            merge_max_passes=args.merge_max_passes,
            include_chart_face_indices=not args.no_face_indices,
        ),
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(plan.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    print(args.report)


if __name__ == "__main__":
    main()
