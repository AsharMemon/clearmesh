#!/usr/bin/env python3
"""Run an optional quad remeshing benchmark on a control mesh."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clearmesh.retopology.quad_remesh import QuadRemeshOptions, quad_remesh_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--engine", default="auto", choices=["auto", "pyinstantmeshes", "instant_meshes_cli", "quadriflow_cli", "template_cage"])
    parser.add_argument("--target-faces", type=int, default=5_000)
    parser.add_argument("--target-vertices", type=int, default=-1)
    parser.add_argument("--allow-dominant", action="store_true", help="Allow tri/quad dominant output instead of pure quads")
    parser.add_argument("--instant-meshes-path", default=None)
    parser.add_argument("--quadriflow-path", default=None)
    parser.add_argument("--quadriflow-no-sharp", action="store_true")
    parser.add_argument("--quadriflow-mcf", action="store_true")
    parser.add_argument("--cage-subdivisions", type=int)
    parser.add_argument("--weld-tolerance", type=float, default=1e-8)
    parser.add_argument("--skip-weld-metrics", action="store_true")
    args = parser.parse_args()

    report = quad_remesh_file(
        args.input,
        args.output,
        QuadRemeshOptions(
            engine=args.engine,
            target_faces=args.target_faces,
            target_vertices=args.target_vertices,
            pure_quad=not args.allow_dominant,
            instant_meshes_path=args.instant_meshes_path,
            quadriflow_path=args.quadriflow_path,
            quadriflow_sharp=not args.quadriflow_no_sharp,
            quadriflow_mcf=args.quadriflow_mcf,
            cage_subdivisions=args.cage_subdivisions,
            weld_tolerance=args.weld_tolerance,
            measure_weld_metrics=not args.skip_weld_metrics,
        ),
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(asdict(report), indent=2, sort_keys=True), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
