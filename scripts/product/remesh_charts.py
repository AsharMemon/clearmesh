#!/usr/bin/env python3
"""Run bounded chart-level remeshing from a retopology plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clearmesh.retopology.chart_remesh import ChartRemeshOptions, remesh_plan_charts  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--engine", default="auto", choices=["auto", "pyinstantmeshes", "instant_meshes_cli", "quadriflow_cli", "template_cage"])
    parser.add_argument("--max-charts", type=int, default=8)
    parser.add_argument("--min-chart-faces", type=int, default=16)
    parser.add_argument("--target-quads", type=int, default=5_000)
    parser.add_argument("--allow-cleanup-charts", action="store_true")
    parser.add_argument("--instant-meshes-path", default=None)
    parser.add_argument("--quadriflow-path", default=None)
    parser.add_argument("--quadriflow-no-sharp", action="store_true")
    parser.add_argument("--quadriflow-mcf", action="store_true")
    parser.add_argument("--weld-tolerance", type=float, default=1e-8)
    parser.add_argument("--skip-weld-metrics", action="store_true")
    args = parser.parse_args()

    manifest = remesh_plan_charts(
        args.input,
        args.plan,
        args.output_dir,
        ChartRemeshOptions(
            engine=args.engine,
            max_charts=args.max_charts,
            min_chart_faces=args.min_chart_faces,
            allow_cleanup_charts=args.allow_cleanup_charts,
            target_quads_total=args.target_quads,
            instant_meshes_path=args.instant_meshes_path,
            quadriflow_path=args.quadriflow_path,
            quadriflow_sharp=not args.quadriflow_no_sharp,
            quadriflow_mcf=args.quadriflow_mcf,
            weld_tolerance=args.weld_tolerance,
            measure_weld_metrics=not args.skip_weld_metrics,
        ),
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    print(args.manifest)


if __name__ == "__main__":
    main()
