#!/usr/bin/env python3
"""Refresh generated-vs-teacher pair metrics in an eval report.

Older FACE eval artifacts may only contain raw Chamfer/Hausdorff metrics. Raw
distances are hard to compare across differently scaled assets, so this script
recomputes pair metrics from exported GLBs and adds normalized distances.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clearmesh.eval.mesh_quality import evaluate_mesh_pair


def _find_pair(export_dir: Path, index: int) -> tuple[Path, Path]:
    prefix = f"{index:04d}_"
    generated = sorted(export_dir.glob(f"{prefix}*_generated.glb"))
    teacher = sorted(export_dir.glob(f"{prefix}*_teacher.glb"))
    if not generated:
        raise FileNotFoundError(f"no generated GLB found for prefix {prefix} in {export_dir}")
    if not teacher:
        raise FileNotFoundError(f"no teacher GLB found for prefix {prefix} in {export_dir}")
    return generated[0], teacher[0]


def _mean(items: list[dict[str, Any]], key: str) -> float | None:
    values = [float(item[key]) for item in items if item.get(key) is not None]
    return float(np.mean(values)) if values else None


def _max(items: list[dict[str, Any]], key: str) -> float | None:
    values = [float(item[key]) for item in items if item.get(key) is not None]
    return float(np.max(values)) if values else None


def refresh_report(report: dict[str, Any], export_dir: Path, *, samples: int, seed: int) -> dict[str, Any]:
    results = list(report.get("results") or [])
    refreshed = 0
    failures: list[dict[str, str]] = []
    for index, item in enumerate(results):
        try:
            generated_path, teacher_path = _find_pair(export_dir, index)
            item.update(evaluate_mesh_pair(generated_path, teacher_path, samples=samples, seed=seed + index))
            item["pair_metrics_refreshed_from"] = {
                "generated": str(generated_path),
                "teacher": str(teacher_path),
                "samples": int(samples),
                "seed": int(seed + index),
            }
            refreshed += 1
        except Exception as exc:  # noqa: BLE001 - keep partial refreshes inspectable.
            failures.append({"index": str(index), "error": f"{type(exc).__name__}: {exc}"})

    summary = dict(report.get("summary") or {})
    for key in (
        "chamfer_l2",
        "hausdorff_l2",
        "chamfer_l2_normalized",
        "hausdorff_l2_normalized",
        "normal_consistency",
    ):
        summary[f"mean_{key}"] = _mean(results, key)
        summary[f"max_{key}"] = _max(results, key)
    report["summary"] = summary
    report["pair_metrics_refresh"] = {
        "export_dir": str(export_dir),
        "requested": len(results),
        "refreshed": refreshed,
        "failed": len(failures),
        "failures": failures,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    refreshed = refresh_report(report, args.export_dir, samples=args.samples, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(refreshed, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(refreshed["pair_metrics_refresh"], indent=2, sort_keys=True))
    return 0 if refreshed["pair_metrics_refresh"]["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
