#!/usr/bin/env python3
"""Project a clean control mesh onto a high-fidelity target mesh."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clearmesh.mesh.shrinkwrap import ShrinkwrapOptions, shrinkwrap_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Clean control/cage mesh")
    parser.add_argument("--target", required=True, type=Path, help="High-fidelity visual target mesh")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--sample-points", type=int, default=100_000)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--attraction", type=float, default=0.65)
    parser.add_argument("--smoothing", type=float, default=0.12)
    parser.add_argument("--max-step-ratio", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    report = shrinkwrap_file(
        args.source,
        args.target,
        args.output,
        ShrinkwrapOptions(
            sample_points=args.sample_points,
            iterations=args.iterations,
            attraction=args.attraction,
            smoothing=args.smoothing,
            max_step_ratio=args.max_step_ratio,
            seed=args.seed,
        ),
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(asdict(report), indent=2, sort_keys=True), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
