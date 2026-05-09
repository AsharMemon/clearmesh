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
    "train_autoregressive_predicted_count",
    "test_autoregressive_predicted_count",
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
    first_face = [
        row
        for row in divergent
        if (value := _num(row.get("first_divergent_face_index"))) is not None and value == 0
    ]
    return {
        "divergent_count": len(divergent),
        "divergent_rate": len(divergent) / len(rows) if rows else None,
        "first_face_count": len(first_face),
        "first_face_rate": len(first_face) / len(rows) if rows else None,
        "early_face_le_8_count": len(early),
        "early_face_le_8_rate": len(early) / len(rows) if rows else None,
        "first_divergent_face_index": _quantiles(face_indices),
        "first_divergent_coord_slot": _quantiles(coord_slots),
    }


def _eval_report(payload: dict[str, Any] | None, *, limit: int) -> dict[str, Any] | None:
    if payload is None:
        return None
    rows = list(payload.get("results") or [])
    summary = dict(payload.get("summary") or {})
    watertight = sum(1 for row in rows if row.get("watertight"))
    return {
        "attempted": len(rows),
        "watertight": watertight,
        "watertight_rate": watertight / len(rows) if rows else None,
        "summary_metrics": {
            "mean_predicted_to_reference_face_ratio": summary.get("mean_predicted_to_reference_face_ratio"),
            "mean_teacher_forced_loss": summary.get("mean_teacher_forced_loss"),
            "mean_teacher_forced_eos_accuracy": summary.get("mean_teacher_forced_eos_accuracy"),
        },
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
    train_pred = evals.get("train_autoregressive_predicted_count") or {}
    test_pred = evals.get("test_autoregressive_predicted_count") or {}

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
            "failure_modes": _failure_modes(train_tf, train_ar, test_tf, test_ar, train_pred, test_pred),
            "next_debug_hint": _next_debug_hint(train_tf, train_ar, test_tf, test_ar),
        },
    }


def _metric_mean(report: dict[str, Any], metric: str) -> float | None:
    return ((report.get("metrics") or {}).get(metric) or {}).get("mean")


def _metric_p50(report: dict[str, Any], metric: str) -> float | None:
    return ((report.get("metrics") or {}).get(metric) or {}).get("p50")


def _first_face_rate(report: dict[str, Any]) -> float:
    return float(((report.get("early_divergence") or {}).get("first_face_rate")) or 0.0)


def _failure_modes(
    train_tf: dict[str, Any],
    train_ar: dict[str, Any],
    test_tf: dict[str, Any],
    test_ar: dict[str, Any],
    train_pred: dict[str, Any],
    test_pred: dict[str, Any],
) -> list[dict[str, Any]]:
    modes: list[dict[str, Any]] = []

    train_tf_acc = _metric_mean(train_tf, "teacher_forced_accuracy") or 0.0
    test_tf_acc = _metric_mean(test_tf, "teacher_forced_accuracy") or 0.0
    train_ar_acc = _metric_mean(train_ar, "generated_token_accuracy") or 0.0
    test_ar_acc = _metric_mean(test_ar, "generated_token_accuracy") or 0.0
    train_ar_wat = train_ar.get("watertight_rate") or 0.0
    test_ar_wat = test_ar.get("watertight_rate") or 0.0
    train_tf_wat = train_tf.get("watertight_rate") or 0.0
    train_tf_boundary = _metric_mean(train_tf, "boundary_edges") or 0.0
    train_ar_boundary = _metric_mean(train_ar, "boundary_edges") or 0.0
    train_tf_chamfer = _metric_p50(train_tf, "chamfer_l2_normalized")

    if train_tf_acc < 0.90:
        modes.append(
            {
                "name": "teacher_forced_underfit",
                "severity": "blocker",
                "evidence": {
                    "train_teacher_accuracy_mean": train_tf_acc,
                    "test_teacher_accuracy_mean": test_tf_acc,
                },
                "interpretation": "The model has not learned the supervised face-token reconstruction problem yet; scaling decisions based on AR samples are premature.",
            }
        )

    first_face_train = _first_face_rate(train_ar)
    first_face_test = _first_face_rate(test_ar)
    if max(first_face_train, first_face_test) >= 0.80 and max(train_ar_acc, test_ar_acc) < 0.25:
        modes.append(
            {
                "name": "first_face_collapse",
                "severity": "blocker",
                "evidence": {
                    "train_first_face_divergence_rate": first_face_train,
                    "test_first_face_divergence_rate": first_face_test,
                    "train_ar_token_accuracy_mean": train_ar_acc,
                    "test_ar_token_accuracy_mean": test_ar_acc,
                },
                "interpretation": "Free-running rollout usually chooses the wrong first face, then the whole ordered sequence no longer matches the target mesh.",
            }
        )

    if train_tf_wat <= 0.05 and train_tf_boundary >= 100 and train_tf_chamfer is not None and train_tf_chamfer <= 0.01:
        modes.append(
            {
                "name": "coordinate_close_but_topologically_broken",
                "severity": "blocker",
                "evidence": {
                    "train_teacher_watertight_rate": train_tf_wat,
                    "train_teacher_boundary_edges_mean": train_tf_boundary,
                    "train_teacher_chamfer_l2_normalized_p50": train_tf_chamfer,
                },
                "interpretation": "Teacher-forced geometry is near the surface, but exact shared vertices are not being recovered, so the edge graph opens.",
            }
        )

    if train_ar_wat > train_tf_wat and train_ar_boundary < max(32.0, train_tf_boundary * 0.1) and train_ar_acc < 0.20:
        modes.append(
            {
                "name": "generic_closed_mesh_not_target_reconstruction",
                "severity": "blocker",
                "evidence": {
                    "train_ar_watertight_rate": train_ar_wat,
                    "test_ar_watertight_rate": test_ar_wat,
                    "train_ar_boundary_edges_mean": train_ar_boundary,
                    "train_ar_token_accuracy_mean": train_ar_acc,
                },
                "interpretation": "The decoder can emit self-consistent mesh-like sequences, but they are not conditioned tightly enough on the target shape/order.",
            }
        )

    predicted_ratio = (test_pred.get("summary_metrics") or {}).get("mean_predicted_to_reference_face_ratio")
    if _num(predicted_ratio) is not None and float(predicted_ratio) >= 2.0:
        modes.append(
            {
                "name": "predicted_count_overrun",
                "severity": "blocker",
                "evidence": {"test_predicted_to_reference_face_ratio_mean": float(predicted_ratio)},
                "interpretation": "The model often fails to terminate at the right face count in product-style predicted-count inference.",
            }
        )

    return modes


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
    first_face_rate = ((train_ar.get("early_divergence") or {}).get("first_face_rate")) or 0.0
    if train_tf_acc < 0.85:
        return "teacher-forced reconstruction is still underfit: keep this as a capacity/optimization/data-quality diagnostic, and add first-token rank/entropy plus teacher-prefix AR probes before a larger run"
    if train_ar_wat < 0.8 and train_ar_acc >= 0.85 and early_rate >= 0.25:
        return "train memorization is close but exposure bias remains: prioritize longer same-slice closure or stricter first-token/early-face diagnostics before scaling"
    if train_tf_acc >= 0.95 and train_ar_acc < 0.85:
        return "teacher-forced fit is not surviving free-run rollout: inspect decode/feed-back mismatch and greedy sampling path"
    if first_face_rate >= 0.80 and train_ar_acc < 0.25:
        return "free-run is collapsing on the first face: inspect canonical face ordering, first-face logits/rank, and latent conditioning before scaling"
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
