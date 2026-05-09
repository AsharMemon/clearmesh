#!/usr/bin/env python3
"""Analyze FACE paper-run autoregressive failure patterns.

The scale gate tells us whether to promote. This helper explains why a run did
not promote: early divergence, face-count bands, boundary-edge concentration,
and train/test memorization gaps.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


EVAL_NAMES = (
    "train_teacher_forced",
    "train_autoregressive",
    "test_teacher_forced",
    "test_autoregressive",
)


FACE_BINS = ((0, 128), (129, 256), (257, 384), (385, 512), (513, 10_000_000))


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric):
        return None
    return numeric


def _first_present(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "p50": None, "max": None, "mean": None}
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "p50": _median(ordered),
        "max": ordered[-1],
        "mean": _mean(ordered),
    }


def _metric(rows: list[dict[str, Any]], key: str) -> dict[str, float | None]:
    return _quantiles([value for row in rows if (value := _num(row.get(key))) is not None])


def _face_count(row: dict[str, Any]) -> int:
    for key in ("reference_face_count", "face_count", "uncapped_reference_face_count"):
        value = _num(row.get(key))
        if value is not None:
            return int(value)
    return 0


def _bin_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    binned = []
    for lo, hi in FACE_BINS:
        members = [row for row in rows if lo <= _face_count(row) <= hi]
        if not members:
            continue
        binned.append(
            {
                "face_range": [lo, hi],
                "count": len(members),
                "watertight": sum(1 for row in members if row.get("watertight")),
                "watertight_rate": sum(1 for row in members if row.get("watertight")) / len(members),
                "mean_generated_token_accuracy": _mean(
                    [
                        value
                        for row in members
                        if (value := _num(row.get("generated_token_accuracy"))) is not None
                    ]
                ),
                "mean_boundary_edges": _mean(
                    [
                        value
                        for row in members
                        if (value := _num(_first_present(row, "token_boundary_edge_count", "boundary_edges")))
                        is not None
                    ]
                ),
                "mean_edge_pairing_ratio": _mean(
                    [
                        value
                        for row in members
                        if (value := _num(row.get("token_edge_pairing_ratio"))) is not None
                    ]
                ),
            }
        )
    return binned


def _top_rows(rows: list[dict[str, Any]], *, best: bool, limit: int) -> list[dict[str, Any]]:
    def score(row: dict[str, Any]) -> tuple[float, float, float]:
        watertight = 1.0 if row.get("watertight") else 0.0
        accuracy = _num(row.get("generated_token_accuracy")) or 0.0
        boundary = _num(_first_present(row, "token_boundary_edge_count", "boundary_edges")) or 0.0
        return (watertight, accuracy, -boundary)

    selected = sorted(rows, key=score, reverse=best)[:limit]
    return [
        {
            "path": row.get("path"),
            "file": Path(str(row.get("path", ""))).name,
            "watertight": bool(row.get("watertight")),
            "generated_token_accuracy": _num(row.get("generated_token_accuracy")),
            "teacher_forced_accuracy": _num(row.get("teacher_forced_accuracy")),
            "reference_face_count": _face_count(row),
            "boundary_edges": _num(_first_present(row, "token_boundary_edge_count", "boundary_edges")),
            "edge_pairing_ratio": _num(row.get("token_edge_pairing_ratio")),
            "first_divergent_face_index": row.get("first_divergent_face_index"),
            "first_divergent_coord_slot": row.get("first_divergent_coord_slot"),
        }
        for row in selected
    ]


def _early_divergence(rows: list[dict[str, Any]]) -> dict[str, Any]:
    divergent = [row for row in rows if row.get("first_divergent_face_index") is not None]
    face_indices = [
        value
        for row in divergent
        if (value := _num(row.get("first_divergent_face_index"))) is not None
    ]
    coord_slots = [
        value
        for row in divergent
        if (value := _num(row.get("first_divergent_coord_slot"))) is not None
    ]
    early = [
        row
        for row in divergent
        if (value := _num(row.get("first_divergent_face_index"))) is not None and value <= 8
    ]
    return {
        "divergent_count": len(divergent),
        "early_face_le_8_count": len(early),
        "early_face_le_8_rate": len(early) / len(rows) if rows else None,
        "first_divergent_face_index": _quantiles(face_indices),
        "first_divergent_coord_slot": _quantiles(coord_slots),
    }


def _eval_report(payload: dict[str, Any] | None, *, limit: int) -> dict[str, Any] | None:
    if payload is None:
        return None
    rows = list(payload.get("results") or [])
    watertight = sum(1 for row in rows if row.get("watertight"))
    return {
        "attempted": len(rows),
        "watertight": watertight,
        "watertight_rate": watertight / len(rows) if rows else None,
        "metrics": {
            "generated_token_accuracy": _metric(rows, "generated_token_accuracy"),
            "teacher_forced_accuracy": _metric(rows, "teacher_forced_accuracy"),
            "boundary_edges": _metric(rows, "token_boundary_edge_count"),
            "edge_pairing_ratio": _metric(rows, "token_edge_pairing_ratio"),
            "reference_face_count": _metric(rows, "reference_face_count"),
            "normal_consistency": _metric(rows, "normal_consistency"),
            "chamfer_l2_normalized": _metric(rows, "chamfer_l2_normalized"),
        },
        "early_divergence": _early_divergence(rows),
        "face_count_bins": _bin_rows(rows),
        "best": _top_rows(rows, best=True, limit=limit),
        "worst": _top_rows(rows, best=False, limit=limit),
    }


def analyze(run_dir: Path, *, limit: int = 8) -> dict[str, Any]:
    eval_dir = run_dir / "eval"
    evals = {
        name: _eval_report(_load_json(eval_dir / f"{name}.json"), limit=limit)
        for name in EVAL_NAMES
    }
    train_ar = evals.get("train_autoregressive") or {}
    train_tf = evals.get("train_teacher_forced") or {}
    test_tf = evals.get("test_teacher_forced") or {}
    test_ar = evals.get("test_autoregressive") or {}

    return {
        "run_dir": str(run_dir),
        "evals": evals,
        "diagnosis": {
            "train_ar_closed": bool(
                train_ar
                and (train_ar.get("watertight_rate") or 0.0) >= 0.8
                and (((train_ar.get("metrics") or {}).get("boundary_edges") or {}).get("mean") or 0.0) <= 5.0
            ),
            "teacher_forced_topology_gap": {
                "train_teacher_watertight_rate": train_tf.get("watertight_rate"),
                "train_ar_watertight_rate": train_ar.get("watertight_rate"),
                "train_teacher_mean_boundary_edges": ((train_tf.get("metrics") or {}).get("boundary_edges") or {}).get("mean"),
                "train_ar_mean_boundary_edges": ((train_ar.get("metrics") or {}).get("boundary_edges") or {}).get("mean"),
            },
            "generalization_gap": {
                "train_teacher_accuracy_mean": ((train_tf.get("metrics") or {}).get("teacher_forced_accuracy") or {}).get("mean"),
                "test_teacher_accuracy_mean": ((test_tf.get("metrics") or {}).get("teacher_forced_accuracy") or {}).get("mean"),
                "train_ar_accuracy_mean": ((train_ar.get("metrics") or {}).get("generated_token_accuracy") or {}).get("mean"),
                "test_ar_accuracy_mean": ((test_ar.get("metrics") or {}).get("generated_token_accuracy") or {}).get("mean"),
            },
            "next_debug_hint": _next_debug_hint(train_tf, train_ar, test_tf, test_ar),
        },
    }


def _next_debug_hint(
    train_tf: dict[str, Any],
    train_ar: dict[str, Any],
    test_tf: dict[str, Any],
    test_ar: dict[str, Any],
) -> str:
    train_ar_wat = train_ar.get("watertight_rate") or 0.0
    train_ar_acc = ((train_ar.get("metrics") or {}).get("generated_token_accuracy") or {}).get("mean") or 0.0
    train_tf_acc = ((train_tf.get("metrics") or {}).get("teacher_forced_accuracy") or {}).get("mean") or 0.0
    test_tf_acc = ((test_tf.get("metrics") or {}).get("teacher_forced_accuracy") or {}).get("mean") or 0.0
    early_rate = ((train_ar.get("early_divergence") or {}).get("early_face_le_8_rate")) or 0.0
    if train_ar_wat < 0.8 and train_ar_acc >= 0.85 and early_rate >= 0.25:
        return "train memorization is close but exposure bias remains: prioritize longer same-slice closure or stricter first-token/early-face diagnostics before scaling"
    if train_tf_acc >= 0.95 and train_ar_acc < 0.85:
        return "teacher-forced fit is not surviving free-run rollout: inspect decode/feed-back mismatch and greedy sampling path"
    if train_tf_acc >= 0.95 and test_tf_acc < 0.20:
        return "bounded slice is memorizing without held-out generalization: do not scale until train AR closes, then increase curated data gradually"
    return "no single dominant failure mode detected; rely on scale_readiness gates and visual contact sheets"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = analyze(args.run_dir, limit=args.limit)
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
