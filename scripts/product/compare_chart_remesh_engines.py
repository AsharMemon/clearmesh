#!/usr/bin/env python3
"""Compare chart-level quad engines on one retopology plan."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clearmesh.retopology.chart_remesh import ChartRemeshOptions, remesh_plan_charts  # noqa: E402
from clearmesh.retopology.chart_stitch import ChartStitchOptions, stitch_chart_remesh_outputs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--engines", default="pyinstantmeshes,quadriflow_cli,template_cage")
    parser.add_argument("--max-charts", type=int, default=8)
    parser.add_argument("--min-chart-faces", type=int, default=16)
    parser.add_argument("--target-quads", type=int, default=5000)
    parser.add_argument("--instant-meshes-path")
    parser.add_argument("--quadriflow-path")
    parser.add_argument("--weld-tolerance", type=float, default=1e-6)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for engine in [item.strip() for item in args.engines.split(",") if item.strip()]:
        results.append(_run_engine(engine, args))
    payload = {"input": str(args.input), "plan": str(args.plan), "results": results}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(args.report)


def _run_engine(engine: str, args: argparse.Namespace) -> dict:
    started = time.time()
    output_dir = args.output_dir / engine
    manifest_path = output_dir / "chart_remesh_manifest.json"
    stitch_path = output_dir / "stitched_chart_quads.obj"
    try:
        manifest = remesh_plan_charts(
            args.input,
            args.plan,
            output_dir / "charts",
            ChartRemeshOptions(
                engine=engine,
                max_charts=args.max_charts,
                min_chart_faces=args.min_chart_faces,
                target_quads_total=args.target_quads,
                instant_meshes_path=args.instant_meshes_path,
                quadriflow_path=args.quadriflow_path,
                weld_tolerance=args.weld_tolerance,
            ),
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        stitch = stitch_chart_remesh_outputs(
            manifest_path,
            stitch_path,
            ChartStitchOptions(weld_tolerance=args.weld_tolerance),
        )
        return {
            "engine": engine,
            "status": "succeeded",
            "runtime_seconds": time.time() - started,
            "manifest_path": str(manifest_path),
            "manifest_summary": manifest.summary,
            "stitch_report": stitch.to_dict(),
        }
    except Exception as exc:  # noqa: BLE001 - comparisons should keep going.
        return {"engine": engine, "status": "failed", "runtime_seconds": time.time() - started, "error": f"{type(exc).__name__}: {exc}"}


if __name__ == "__main__":
    main()
