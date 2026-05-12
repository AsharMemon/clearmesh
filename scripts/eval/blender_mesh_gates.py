#!/usr/bin/env python3
"""Run Blender production-promotion gates on a mesh."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clearmesh.eval.blender_gates import BlenderGateOptions, run_blender_mesh_gates  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--blender", default="blender")
    parser.add_argument("--timeout-seconds", type=int, default=180)
    args = parser.parse_args()

    report = run_blender_mesh_gates(
        args.input,
        args.output_dir,
        BlenderGateOptions(blender=args.blender, timeout_seconds=args.timeout_seconds),
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    print(args.report)


if __name__ == "__main__":
    main()
