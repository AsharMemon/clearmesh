#!/usr/bin/env python3
"""Conservative scale-readiness gate for the paper-faithful FACE lane.

This gate is intentionally stricter than a loss-only training monitor. FACE can
memorize token sequences while still producing open or spiky free-running meshes,
so promotion requires separate train/test and teacher-forced/autoregressive
signals.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class GateCheck:
    name: str
    passed: bool
    value: Any
    threshold: Any
    severity: str = "blocker"
    detail: str = ""


@dataclass
class ReadinessReport:
    scale_ready: bool
    topology_ready: bool
    generalization_ready: bool
    paper_knobs_ready: bool
    recommendation: str
    checks: list[GateCheck] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


def _load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _summary(report: dict[str, Any] | None) -> dict[str, Any]:
    if not report:
        return {}
    return dict(report.get("summary") or {})


def _results(report: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not report:
        return []
    return list(report.get("results") or [])


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and not math.isnan(float(value)):
        return float(value)
    return None


def _numbers(items: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for item in items:
        value = _number(item.get(key))
        if value is not None:
            values.append(value)
    return values


def _quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "p95": None, "max": None}
    ordered = sorted(values)
    p95_idx = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return {
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "p95": float(ordered[p95_idx]),
        "max": float(max(values)),
    }


def _rate(count: Any, total: Any) -> float | None:
    c = _number(count)
    t = _number(total)
    if c is None or t in (None, 0.0):
        return None
    return float(c / t)


def _eval_metrics(report: dict[str, Any] | None) -> dict[str, Any]:
    summary = _summary(report)
    results = _results(report)
    attempted = int(summary.get("attempted") or len(results) or 0)
    watertight = int(summary.get("watertight") or sum(1 for item in results if item.get("watertight")) or 0)
    boundary_edges = _numbers(results, "token_boundary_edge_count") or _numbers(results, "boundary_edges")
    nonmanifold_edges = _numbers(results, "nonmanifold_edges")
    edge_pairing = _numbers(results, "token_edge_pairing_ratio")
    chamfer = _numbers(results, "chamfer_l2_normalized") or _numbers(results, "chamfer_l2")
    hausdorff = _numbers(results, "hausdorff_l2_normalized") or _numbers(results, "hausdorff_l2")
    normal = _numbers(results, "normal_consistency")
    sample_sec = _numbers(results, "sample_sec")
    truncated = int(sum(1 for item in results if item.get("truncated_by_generation_face_limit")))
    stopped_on_eos = int(sum(1 for item in results if item.get("stopped_on_eos")))
    return {
        "attempted": attempted,
        "watertight": watertight,
        "watertight_rate": _rate(watertight, attempted),
        "mean_teacher_forced_accuracy": summary.get("mean_teacher_forced_accuracy"),
        "mean_teacher_forced_loss": summary.get("mean_teacher_forced_loss"),
        "mean_teacher_forced_eos_accuracy": summary.get("mean_teacher_forced_eos_accuracy"),
        "mean_generated_token_accuracy": summary.get("mean_generated_token_accuracy")
        if summary.get("mean_generated_token_accuracy") is not None
        else (statistics.fmean(_numbers(results, "generated_token_accuracy")) if _numbers(results, "generated_token_accuracy") else None),
        "mean_generated_vertex_exact_ratio": summary.get("mean_generated_vertex_exact_ratio")
        if summary.get("mean_generated_vertex_exact_ratio") is not None
        else (statistics.fmean(_numbers(results, "generated_vertex_exact_ratio")) if _numbers(results, "generated_vertex_exact_ratio") else None),
        "mean_generated_edge_exact_ratio": summary.get("mean_generated_edge_exact_ratio")
        if summary.get("mean_generated_edge_exact_ratio") is not None
        else (statistics.fmean(_numbers(results, "generated_edge_exact_ratio")) if _numbers(results, "generated_edge_exact_ratio") else None),
        "mean_generated_face_exact_ratio": summary.get("mean_generated_face_exact_ratio")
        if summary.get("mean_generated_face_exact_ratio") is not None
        else (statistics.fmean(_numbers(results, "generated_face_exact_ratio")) if _numbers(results, "generated_face_exact_ratio") else None),
        "mean_generated_edge_set_precision": summary.get("mean_generated_edge_set_precision")
        if summary.get("mean_generated_edge_set_precision") is not None
        else (statistics.fmean(_numbers(results, "generated_edge_set_precision")) if _numbers(results, "generated_edge_set_precision") else None),
        "mean_generated_edge_set_recall": summary.get("mean_generated_edge_set_recall")
        if summary.get("mean_generated_edge_set_recall") is not None
        else (statistics.fmean(_numbers(results, "generated_edge_set_recall")) if _numbers(results, "generated_edge_set_recall") else None),
        "mean_generated_edge_set_f1": summary.get("mean_generated_edge_set_f1")
        if summary.get("mean_generated_edge_set_f1") is not None
        else (statistics.fmean(_numbers(results, "generated_edge_set_f1")) if _numbers(results, "generated_edge_set_f1") else None),
        "mean_boundary_edges": summary.get("mean_boundary_edges")
        if summary.get("mean_boundary_edges") is not None
        else (statistics.fmean(boundary_edges) if boundary_edges else None),
        "max_boundary_edges": max(boundary_edges) if boundary_edges else None,
        "mean_nonmanifold_edges": statistics.fmean(nonmanifold_edges) if nonmanifold_edges else None,
        "max_nonmanifold_edges": max(nonmanifold_edges) if nonmanifold_edges else None,
        "mean_edge_pairing_ratio": summary.get("mean_edge_pairing_ratio")
        if summary.get("mean_edge_pairing_ratio") is not None
        else (statistics.fmean(edge_pairing) if edge_pairing else None),
        "mean_predicted_to_reference_face_ratio": summary.get("mean_predicted_to_reference_face_ratio"),
        "faces_per_sec": summary.get("faces_per_sec"),
        "sample_sec": _quantiles(sample_sec),
        "chamfer_l2_normalized": _quantiles(chamfer),
        "hausdorff_l2_normalized": _quantiles(hausdorff),
        "normal_consistency": _quantiles(normal),
        "truncated_samples": truncated,
        "stopped_on_eos": stopped_on_eos,
        "generation_face_limit": report.get("generation_face_limit") if report else None,
        "face_count_mode": report.get("face_count_mode") if report else None,
        "generation_mode": report.get("generation_mode") if report else None,
        "causal_mlp_variant": report.get("causal_mlp_variant") if report else None,
        "decode_head": report.get("decode_head") if report else None,
        "encoder_backend": report.get("encoder_backend") if report else None,
    }


def _add_check(
    report: ReadinessReport,
    *,
    name: str,
    passed: bool,
    value: Any,
    threshold: Any,
    detail: str = "",
    severity: str = "blocker",
    blocker: str | None = None,
    warning: str | None = None,
) -> None:
    report.checks.append(GateCheck(name=name, passed=bool(passed), value=value, threshold=threshold, severity=severity, detail=detail))
    if passed:
        return
    if severity == "warning":
        report.warnings.append(warning or blocker or name)
    else:
        report.blockers.append(blocker or name)


def _approx_equal(a: Any, b: float, tol: float = 1e-9) -> bool:
    value = _number(a)
    return value is not None and abs(value - b) <= tol


def _run_settings(run_summary: dict[str, Any] | None) -> dict[str, Any]:
    if not run_summary:
        return {}
    if "settings" in run_summary:
        return dict(run_summary.get("settings") or {})
    nested = run_summary.get("run_summary")
    if isinstance(nested, dict):
        return dict(nested.get("settings") or {})
    return {}


def _dataset_counts(run_summary: dict[str, Any] | None, train_teacher: dict[str, Any] | None, test_teacher: dict[str, Any] | None) -> dict[str, int]:
    train = int((_summary(train_teacher).get("attempted") or 0))
    test = int((_summary(test_teacher).get("attempted") or 0))

    if not run_summary:
        return {"train": train, "test": test, "total": train + test}

    # Full runs often evaluate a bounded train/test subset, so eval attempts can
    # badly undercount the real corpus. Prefer the integrity-checked split
    # counts written by the strict token-hash split gate when available.
    split_integrity = run_summary.get("split_integrity")
    if isinstance(split_integrity, dict):
        train_info = split_integrity.get("train")
        test_info = split_integrity.get("test")
        if isinstance(train_info, dict):
            train = int(train_info.get("sample_count") or train_info.get("valid_count") or train)
        if isinstance(test_info, dict):
            test = int(test_info.get("sample_count") or test_info.get("valid_count") or test)

    nested = run_summary.get("split")
    if isinstance(nested, dict):
        train = int(nested.get("train_count") or nested.get("train") or train)
        test = int(nested.get("test_count") or nested.get("test") or test)
    return {"train": train, "test": test, "total": train + test}


def assess(
    *,
    run_summary: dict[str, Any] | None,
    train_teacher: dict[str, Any] | None,
    train_ar: dict[str, Any] | None,
    test_teacher: dict[str, Any] | None,
    test_ar: dict[str, Any] | None,
    min_dataset_samples: int = 64,
    min_teacher_samples: int = 16,
    min_ar_samples: int = 10,
    require_paper_knobs: bool = True,
    min_train_teacher_accuracy: float = 0.95,
    max_train_teacher_loss: float = 0.3,
    min_train_ar_accuracy: float = 0.90,
    min_train_ar_watertight_rate: float = 0.80,
    max_train_ar_mean_boundary_edges: float = 5.0,
    min_train_ar_edge_pairing_ratio: float = 0.995,
    min_test_teacher_accuracy: float = 0.20,
    max_test_teacher_loss: float = 3.0,
    min_test_ar_watertight_rate: float = 0.20,
    max_test_ar_mean_boundary_edges: float = 64.0,
    min_test_ar_edge_pairing_ratio: float = 0.75,
    max_test_ar_median_chamfer_l2_normalized: float = 0.10,
    min_test_ar_median_normal_consistency: float = 0.65,
) -> ReadinessReport:
    report = ReadinessReport(
        scale_ready=False,
        topology_ready=False,
        generalization_ready=False,
        paper_knobs_ready=False,
        recommendation="hold",
    )
    settings = _run_settings(run_summary)
    counts = _dataset_counts(run_summary, train_teacher, test_teacher)
    metrics = {
        "dataset": counts,
        "settings": settings,
        "train_teacher_forced": _eval_metrics(train_teacher),
        "train_autoregressive": _eval_metrics(train_ar),
        "test_teacher_forced": _eval_metrics(test_teacher),
        "test_autoregressive": _eval_metrics(test_ar),
    }
    report.metrics = metrics

    _add_check(
        report,
        name="dataset_size",
        passed=counts["total"] >= min_dataset_samples,
        value=counts["total"],
        threshold=f">={min_dataset_samples}",
        blocker="bounded corpus is too small to justify promotion",
    )
    _add_check(
        report,
        name="train_teacher_sample_count",
        passed=metrics["train_teacher_forced"]["attempted"] >= min_teacher_samples,
        value=metrics["train_teacher_forced"]["attempted"],
        threshold=f">={min_teacher_samples}",
        blocker="train teacher-forced eval sample count is too small",
    )
    _add_check(
        report,
        name="test_teacher_sample_count",
        passed=metrics["test_teacher_forced"]["attempted"] >= max(1, min_teacher_samples // 4),
        value=metrics["test_teacher_forced"]["attempted"],
        threshold=f">={max(1, min_teacher_samples // 4)}",
        blocker="held-out teacher-forced eval sample count is too small",
    )
    for label in ("train_autoregressive", "test_autoregressive"):
        _add_check(
            report,
            name=f"{label}_sample_count",
            passed=metrics[label]["attempted"] >= min_ar_samples,
            value=metrics[label]["attempted"],
            threshold=f">={min_ar_samples}",
            blocker=f"{label} eval sample count is too small",
        )

    if require_paper_knobs:
        required = {
            "point_samples": 8192,
            "vecset_tokens": 2048,
            "latent_dim": 64,
            "optimizer": "muon",
            "lr": 0.0006,
            "weight_decay": 0.1,
            "precision": "bf16",
            "disable_augment": False,
            "augment_rotation": "so3",
            "encoder_backend": "shape2vecset",
            "decode_head": "causal",
            "causal_mlp_variant": "legacy_concat",
            "face_embedding_variant": "token_concat_project",
        }
        for key, expected in required.items():
            value = settings.get(key)
            if isinstance(expected, float):
                passed = _approx_equal(value, expected, tol=1e-8)
            else:
                passed = value == expected
            _add_check(
                report,
                name=f"paper_knob_{key}",
                passed=passed,
                value=value,
                threshold=expected,
                blocker=f"paper-knob mismatch: {key}",
            )
        if "model_max_faces" in settings and int(settings.get("model_max_faces") or 0) < 512:
            _add_check(
                report,
                name="paper_knob_model_max_faces_floor",
                passed=False,
                value=settings.get("model_max_faces"),
                threshold=">=512 for current bounded gate; 4000 for true paper-scale",
                blocker="model face cap is below the current strict target budget",
            )
        # Capacity is intentionally a warning, not a blocker. A reduced model can
        # prove tokenization/data/free-running health and justify the next corpus
        # rung, but it is not evidence that we have matched the full paper-scale
        # 500M-class setting.
        capacity_floor = {
            "hidden_size": 1024,
            "encoder_hidden_size": 768,
            "encoder_layers": 8,
            "decoder_layers": 24,
            "heads": 16,
        }
        reduced = {
            key: settings.get(key)
            for key, floor in capacity_floor.items()
            if key in settings and int(settings.get(key) or 0) < floor
        }
        _add_check(
            report,
            name="paper_capacity_full_scale",
            passed=not reduced,
            value=reduced or {key: settings.get(key) for key in capacity_floor if key in settings},
            threshold={key: f">={value}" for key, value in capacity_floor.items()},
            severity="warning",
            warning="model capacity is below the paper-scale profile; treat this as a bounded validation, not a final reproduction claim",
        )
    else:
        report.warnings.append("paper knob checks are relaxed for this diagnostic gate")

    report.paper_knobs_ready = not any(check.name.startswith("paper_knob_") and not check.passed for check in report.checks)

    for label in ("train_teacher_forced", "train_autoregressive", "test_teacher_forced", "test_autoregressive"):
        m = metrics[label]
        _add_check(
            report,
            name=f"{label}_not_truncated",
            passed=int(m.get("truncated_samples") or 0) == 0 and int(m.get("generation_face_limit") or 0) == 0,
            value={"truncated_samples": m.get("truncated_samples"), "generation_face_limit": m.get("generation_face_limit")},
            threshold="no face-limit truncation",
            blocker=f"{label} is truncated; full-face AR evidence is required",
        )
        if m.get("face_count_mode") == "gt":
            _add_check(
                report,
                name=f"{label}_gt_face_count_mode",
                passed=False,
                value="gt",
                threshold="predicted-count probe still needed before product inference",
                severity="warning",
                warning=f"{label} uses ground-truth face count; keep this as reconstruction evidence, not final production termination evidence",
            )

    train_tf = metrics["train_teacher_forced"]
    train_ar_m = metrics["train_autoregressive"]
    test_tf = metrics["test_teacher_forced"]
    test_ar_m = metrics["test_autoregressive"]

    _add_check(
        report,
        name="train_teacher_accuracy",
        passed=_number(train_tf.get("mean_teacher_forced_accuracy")) is not None
        and float(train_tf["mean_teacher_forced_accuracy"]) >= min_train_teacher_accuracy,
        value=train_tf.get("mean_teacher_forced_accuracy"),
        threshold=f">={min_train_teacher_accuracy}",
        blocker="train teacher-forced accuracy is not high enough; capacity/optimization gate failed",
    )
    _add_check(
        report,
        name="train_teacher_loss",
        passed=_number(train_tf.get("mean_teacher_forced_loss")) is not None
        and float(train_tf["mean_teacher_forced_loss"]) <= max_train_teacher_loss,
        value=train_tf.get("mean_teacher_forced_loss"),
        threshold=f"<={max_train_teacher_loss}",
        blocker="train teacher-forced loss is too high; do not scale an underfit run",
    )
    _add_check(
        report,
        name="train_ar_generated_token_accuracy",
        passed=_number(train_ar_m.get("mean_generated_token_accuracy")) is not None
        and float(train_ar_m["mean_generated_token_accuracy"]) >= min_train_ar_accuracy,
        value=train_ar_m.get("mean_generated_token_accuracy"),
        threshold=f">={min_train_ar_accuracy}",
        blocker="train autoregressive free-run tokens do not match the learned targets closely enough",
    )
    _add_check(
        report,
        name="train_ar_watertight_rate",
        passed=_number(train_ar_m.get("watertight_rate")) is not None
        and float(train_ar_m["watertight_rate"]) >= min_train_ar_watertight_rate,
        value=train_ar_m.get("watertight_rate"),
        threshold=f">={min_train_ar_watertight_rate}",
        blocker="train autoregressive meshes are not mostly watertight",
    )
    _add_check(
        report,
        name="train_ar_boundary_edges",
        passed=_number(train_ar_m.get("mean_boundary_edges")) is not None
        and float(train_ar_m["mean_boundary_edges"]) <= max_train_ar_mean_boundary_edges,
        value=train_ar_m.get("mean_boundary_edges"),
        threshold=f"<={max_train_ar_mean_boundary_edges}",
        blocker="train autoregressive meshes still have too many boundary edges",
    )
    _add_check(
        report,
        name="train_ar_edge_pairing",
        passed=_number(train_ar_m.get("mean_edge_pairing_ratio")) is not None
        and float(train_ar_m["mean_edge_pairing_ratio"]) >= min_train_ar_edge_pairing_ratio,
        value=train_ar_m.get("mean_edge_pairing_ratio"),
        threshold=f">={min_train_ar_edge_pairing_ratio}",
        blocker="train autoregressive token edge graph is not fully paired",
    )

    _add_check(
        report,
        name="test_teacher_accuracy",
        passed=_number(test_tf.get("mean_teacher_forced_accuracy")) is not None
        and float(test_tf["mean_teacher_forced_accuracy"]) >= min_test_teacher_accuracy,
        value=test_tf.get("mean_teacher_forced_accuracy"),
        threshold=f">={min_test_teacher_accuracy}",
        blocker="held-out teacher-forced accuracy is too low; no generalization signal yet",
    )
    _add_check(
        report,
        name="test_teacher_loss",
        passed=_number(test_tf.get("mean_teacher_forced_loss")) is not None
        and float(test_tf["mean_teacher_forced_loss"]) <= max_test_teacher_loss,
        value=test_tf.get("mean_teacher_forced_loss"),
        threshold=f"<={max_test_teacher_loss}",
        blocker="held-out teacher-forced loss is too high",
    )
    _add_check(
        report,
        name="test_ar_watertight_rate",
        passed=_number(test_ar_m.get("watertight_rate")) is not None
        and float(test_ar_m["watertight_rate"]) >= min_test_ar_watertight_rate,
        value=test_ar_m.get("watertight_rate"),
        threshold=f">={min_test_ar_watertight_rate}",
        blocker="held-out autoregressive meshes are not closing",
    )
    _add_check(
        report,
        name="test_ar_boundary_edges",
        passed=_number(test_ar_m.get("mean_boundary_edges")) is not None
        and float(test_ar_m["mean_boundary_edges"]) <= max_test_ar_mean_boundary_edges,
        value=test_ar_m.get("mean_boundary_edges"),
        threshold=f"<={max_test_ar_mean_boundary_edges}",
        blocker="held-out autoregressive meshes have too many boundary edges",
    )
    _add_check(
        report,
        name="test_ar_edge_pairing",
        passed=_number(test_ar_m.get("mean_edge_pairing_ratio")) is not None
        and float(test_ar_m["mean_edge_pairing_ratio"]) >= min_test_ar_edge_pairing_ratio,
        value=test_ar_m.get("mean_edge_pairing_ratio"),
        threshold=f">={min_test_ar_edge_pairing_ratio}",
        blocker="held-out autoregressive edge pairing is too low",
    )
    test_chamfer_median = (test_ar_m.get("chamfer_l2_normalized") or {}).get("median")
    _add_check(
        report,
        name="test_ar_median_chamfer_normalized",
        passed=_number(test_chamfer_median) is not None
        and float(test_chamfer_median) <= max_test_ar_median_chamfer_l2_normalized,
        value=test_chamfer_median,
        threshold=f"<={max_test_ar_median_chamfer_l2_normalized}",
        blocker="held-out autoregressive geometry error is too high",
    )
    test_normal_median = (test_ar_m.get("normal_consistency") or {}).get("median")
    _add_check(
        report,
        name="test_ar_median_normal_consistency",
        passed=_number(test_normal_median) is not None
        and float(test_normal_median) >= min_test_ar_median_normal_consistency,
        value=test_normal_median,
        threshold=f">={min_test_ar_median_normal_consistency}",
        blocker="held-out autoregressive normals are not consistent enough",
    )

    topology_check_names = {
        "train_ar_watertight_rate",
        "train_ar_boundary_edges",
        "train_ar_edge_pairing",
        "train_ar_generated_token_accuracy",
        "test_ar_watertight_rate",
        "test_ar_boundary_edges",
        "test_ar_edge_pairing",
    }
    generalization_check_names = {
        "test_teacher_accuracy",
        "test_teacher_loss",
        "test_ar_watertight_rate",
        "test_ar_boundary_edges",
        "test_ar_edge_pairing",
        "test_ar_median_chamfer_normalized",
        "test_ar_median_normal_consistency",
    }
    report.topology_ready = all(check.passed for check in report.checks if check.name in topology_check_names)
    report.generalization_ready = all(check.passed for check in report.checks if check.name in generalization_check_names)
    report.scale_ready = not report.blockers
    if report.scale_ready:
        report.recommendation = "promote_to_next_corpus_rung"
    elif not report.paper_knobs_ready:
        report.recommendation = "fix_paper_knob_mismatch_before_more_gpu"
    elif not report.topology_ready and train_tf.get("mean_teacher_forced_accuracy") and float(train_tf["mean_teacher_forced_accuracy"]) >= min_train_teacher_accuracy:
        report.recommendation = "debug_free_running_topology_before_scaling"
    elif not report.generalization_ready:
        report.recommendation = "increase_curated_data_or_training_curve_only_after_train_ar_gate"
    else:
        report.recommendation = "hold_and_diagnose"
    return report


def _infer_paths(run_dir: Path) -> dict[str, Path]:
    return {
        "run_summary": run_dir / "summary.json",
        "train_teacher": run_dir / "eval" / "train_teacher_forced.json",
        "train_ar": run_dir / "eval" / "train_autoregressive.json",
        "test_teacher": run_dir / "eval" / "test_teacher_forced.json",
        "test_ar": run_dir / "eval" / "test_autoregressive.json",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=None, help="Infer summary/eval paths from a FACE paper run directory.")
    parser.add_argument("--run-summary", type=Path, default=None)
    parser.add_argument("--train-teacher", type=Path, default=None)
    parser.add_argument("--train-ar", type=Path, default=None)
    parser.add_argument("--test-teacher", type=Path, default=None)
    parser.add_argument("--test-ar", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--relax-paper-knobs", action="store_true")
    parser.add_argument("--min-dataset-samples", type=int, default=64)
    parser.add_argument("--min-teacher-samples", type=int, default=16)
    parser.add_argument("--min-ar-samples", type=int, default=10)
    parser.add_argument("--min-train-teacher-accuracy", type=float, default=0.95)
    parser.add_argument("--max-train-teacher-loss", type=float, default=0.3)
    parser.add_argument("--min-train-ar-accuracy", type=float, default=0.90)
    parser.add_argument("--min-train-ar-watertight-rate", type=float, default=0.80)
    parser.add_argument("--max-train-ar-mean-boundary-edges", type=float, default=5.0)
    parser.add_argument("--min-train-ar-edge-pairing-ratio", type=float, default=0.995)
    parser.add_argument("--min-test-teacher-accuracy", type=float, default=0.20)
    parser.add_argument("--max-test-teacher-loss", type=float, default=3.0)
    parser.add_argument("--min-test-ar-watertight-rate", type=float, default=0.20)
    parser.add_argument("--max-test-ar-mean-boundary-edges", type=float, default=64.0)
    parser.add_argument("--min-test-ar-edge-pairing-ratio", type=float, default=0.75)
    parser.add_argument("--max-test-ar-median-chamfer-l2-normalized", type=float, default=0.10)
    parser.add_argument("--min-test-ar-median-normal-consistency", type=float, default=0.65)
    parser.add_argument("--fail-on-not-ready", action="store_true")
    args = parser.parse_args()

    inferred: dict[str, Path] = {}
    if args.run_dir is not None:
        inferred = _infer_paths(args.run_dir)
    run_summary_path = args.run_summary or inferred.get("run_summary")
    train_teacher_path = args.train_teacher or inferred.get("train_teacher")
    train_ar_path = args.train_ar or inferred.get("train_ar")
    test_teacher_path = args.test_teacher or inferred.get("test_teacher")
    test_ar_path = args.test_ar or inferred.get("test_ar")

    report = assess(
        run_summary=_load_json(run_summary_path),
        train_teacher=_load_json(train_teacher_path),
        train_ar=_load_json(train_ar_path),
        test_teacher=_load_json(test_teacher_path),
        test_ar=_load_json(test_ar_path),
        min_dataset_samples=args.min_dataset_samples,
        min_teacher_samples=args.min_teacher_samples,
        min_ar_samples=args.min_ar_samples,
        require_paper_knobs=not args.relax_paper_knobs,
        min_train_teacher_accuracy=args.min_train_teacher_accuracy,
        max_train_teacher_loss=args.max_train_teacher_loss,
        min_train_ar_accuracy=args.min_train_ar_accuracy,
        min_train_ar_watertight_rate=args.min_train_ar_watertight_rate,
        max_train_ar_mean_boundary_edges=args.max_train_ar_mean_boundary_edges,
        min_train_ar_edge_pairing_ratio=args.min_train_ar_edge_pairing_ratio,
        min_test_teacher_accuracy=args.min_test_teacher_accuracy,
        max_test_teacher_loss=args.max_test_teacher_loss,
        min_test_ar_watertight_rate=args.min_test_ar_watertight_rate,
        max_test_ar_mean_boundary_edges=args.max_test_ar_mean_boundary_edges,
        min_test_ar_edge_pairing_ratio=args.min_test_ar_edge_pairing_ratio,
        max_test_ar_median_chamfer_l2_normalized=args.max_test_ar_median_chamfer_l2_normalized,
        min_test_ar_median_normal_consistency=args.min_test_ar_median_normal_consistency,
    )
    payload = asdict(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"scale_ready": report.scale_ready, "recommendation": report.recommendation, "blockers": report.blockers[:8]}, indent=2, sort_keys=True))
    if args.fail_on_not_ready and not report.scale_ready:
        return 30
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
