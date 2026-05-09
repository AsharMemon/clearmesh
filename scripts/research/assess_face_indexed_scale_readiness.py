#!/usr/bin/env python3
"""Assess whether FACE-indexed runs are ready to scale.

This script is intentionally conservative. A closed mesh is not automatically a
good mesh, so the gate separates topology readiness from geometry-fidelity
readiness and reports the exact blockers before we spend larger GPU time.
"""

from __future__ import annotations

import argparse
import json
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
    detail: str = ""


@dataclass
class ReadinessReport:
    scale_ready: bool
    topology_ready: bool
    geometry_ready: bool
    recommendation: str
    checks: list[GateCheck] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


def _load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _as_rate(count: int | float | None, total: int | float | None) -> float | None:
    if count is None or total in (None, 0):
        return None
    return float(count) / float(total)


def _numbers(items: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for item in items:
        value = item.get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def _fill_numbers(items: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for item in items:
        fill = item.get("boundary_fill_report")
        if not isinstance(fill, dict):
            continue
        value = fill.get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
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


def _add_check(
    report: ReadinessReport,
    *,
    name: str,
    passed: bool,
    value: Any,
    threshold: Any,
    blocker: str | None = None,
    warning: str | None = None,
    detail: str = "",
) -> None:
    report.checks.append(GateCheck(name=name, passed=passed, value=value, threshold=threshold, detail=detail))
    if not passed and blocker:
        report.blockers.append(blocker)
    elif not passed and warning:
        report.warnings.append(warning)


def _eval_metrics(eval_report: dict[str, Any] | None) -> dict[str, Any]:
    if not eval_report:
        return {}
    results = list(eval_report.get("results") or [])
    summary = dict(eval_report.get("summary") or {})
    attempted = int(summary.get("attempted") or len(results))
    watertight = int(summary.get("watertight") or sum(1 for item in results if item.get("watertight")))
    boundary_edges = _numbers(results, "boundary_edges")
    nonmanifold_edges = _numbers(results, "nonmanifold_edges")
    nonmanifold_vertices = _numbers(results, "nonmanifold_vertices")
    if not nonmanifold_vertices:
        nonmanifold_vertices = _numbers(results, "nonmanifold_vertex_count")
    edge_pairing = _numbers(results, "token_edge_pairing_ratio")
    chamfer = _numbers(results, "chamfer_l2_normalized")
    if not chamfer:
        chamfer = _numbers(results, "chamfer_l2")
    normal = _numbers(results, "normal_consistency")
    decode_sec = _numbers(results, "decode_elapsed_sec")
    fill_input_boundary_edges = _fill_numbers(results, "input_boundary_edges")
    fill_filled_faces = _fill_numbers(results, "filled_faces")
    fill_input_faces = _fill_numbers(results, "input_faces")
    fill_face_ratios = [
        filled / max(input_faces, 1.0)
        for filled, input_faces in zip(fill_filled_faces, fill_input_faces, strict=False)
    ]
    return {
        "attempted": attempted,
        "watertight": watertight,
        "watertight_rate": _as_rate(watertight, attempted),
        "mean_boundary_edges": summary.get("mean_boundary_edges"),
        "max_boundary_edges": max(boundary_edges) if boundary_edges else None,
        "mean_nonmanifold_edges": summary.get("mean_nonmanifold_edges"),
        "max_nonmanifold_edges": max(nonmanifold_edges) if nonmanifold_edges else None,
        "mean_nonmanifold_vertices": summary.get("mean_nonmanifold_vertices")
        if summary.get("mean_nonmanifold_vertices") is not None
        else (statistics.fmean(nonmanifold_vertices) if nonmanifold_vertices else None),
        "max_nonmanifold_vertices": max(nonmanifold_vertices) if nonmanifold_vertices else None,
        "mean_edge_pairing_ratio": summary.get("mean_edge_pairing_ratio")
        if summary.get("mean_edge_pairing_ratio") is not None
        else (statistics.fmean(edge_pairing) if edge_pairing else None),
        "teacher_forced_token_accuracy": summary.get("mean_teacher_forced_token_accuracy"),
        "teacher_forced_face_exact_ratio": summary.get("mean_teacher_forced_face_exact_ratio"),
        "chamfer_l2_normalized": _quantiles(chamfer),
        "chamfer_l2_normalized_available": any(item.get("chamfer_l2_normalized") is not None for item in results),
        "normal_consistency": _quantiles(normal),
        "decode_elapsed_sec": _quantiles(decode_sec),
        "boundary_fill_input_boundary_edges": _quantiles(fill_input_boundary_edges),
        "boundary_fill_filled_faces": _quantiles(fill_filled_faces),
        "boundary_fill_face_ratio": _quantiles(fill_face_ratios),
    }


def assess(
    *,
    curation: dict[str, Any] | None,
    teacher_eval: dict[str, Any] | None,
    free_run_eval: dict[str, Any] | None,
    min_dataset_samples: int,
    min_eval_samples: int,
    min_watertight_rate: float,
    min_teacher_token_accuracy: float,
    min_teacher_face_exact: float,
    min_edge_pairing_ratio: float,
    max_mean_boundary_edges: float,
    max_mean_nonmanifold_edges: float,
    max_mean_nonmanifold_vertices: float,
    max_median_chamfer_l2_normalized: float,
    max_p95_chamfer_l2_normalized: float,
    min_median_normal_consistency: float,
    teacher_gate_mode: str = "memorization",
    max_mean_boundary_fill_edges: float = 0.0,
    max_mean_boundary_fill_face_ratio: float = 0.0,
) -> ReadinessReport:
    report = ReadinessReport(
        scale_ready=False,
        topology_ready=False,
        geometry_ready=False,
        recommendation="hold",
    )

    curation_metrics: dict[str, Any] = {}
    if curation:
        written = int(curation.get("written") or 0)
        failed = int(curation.get("failed") or 0)
        zero_closure = int(curation.get("zero_closure_after_first_sum") or 0)
        curation_metrics = {
            "written": written,
            "failed": failed,
            "max_faces": curation.get("max_faces"),
            "max_vertices": curation.get("max_vertices"),
            "mean_faces": curation.get("mean_faces"),
            "zero_closure_after_first_sum": zero_closure,
        }
        _add_check(
            report,
            name="curated_dataset_size",
            passed=written >= min_dataset_samples,
            value=written,
            threshold=f">={min_dataset_samples}",
            blocker="curated dataset is too small for the requested scale gate",
        )
        _add_check(
            report,
            name="curated_boundary_growth_order",
            passed=zero_closure == 0,
            value=zero_closure,
            threshold="0 extra zero-closure jumps",
            blocker="dataset order is not fully aligned with boundary-growth decoding",
        )
        if failed:
            report.warnings.append(f"curation rejected {failed} source assets; keep rejection reasons when expanding corpus")
    else:
        report.warnings.append("no curation summary supplied")

    teacher_metrics = _eval_metrics(teacher_eval)
    free_metrics = _eval_metrics(free_run_eval)
    strict_teacher_gate = teacher_gate_mode == "memorization"
    report.metrics = {
        "teacher_gate_mode": teacher_gate_mode,
        "curation": curation_metrics,
        "teacher_forced": teacher_metrics,
        "free_run": free_metrics,
    }

    for label, metrics in (("teacher_forced", teacher_metrics), ("free_run", free_metrics)):
        teacher_diagnostic_only = label == "teacher_forced" and not strict_teacher_gate
        attempted = int(metrics.get("attempted") or 0)
        watertight_rate = metrics.get("watertight_rate")
        mean_boundary = metrics.get("mean_boundary_edges")
        mean_nonmanifold = metrics.get("mean_nonmanifold_edges")
        mean_nonmanifold_vertices = metrics.get("mean_nonmanifold_vertices")
        edge_pairing = metrics.get("mean_edge_pairing_ratio")
        fill_edges = (metrics.get("boundary_fill_input_boundary_edges") or {}).get("mean")
        fill_ratio = (metrics.get("boundary_fill_face_ratio") or {}).get("mean")
        _add_check(
            report,
            name=f"{label}_sample_count",
            passed=attempted >= min_eval_samples,
            value=attempted,
            threshold=f">={min_eval_samples}",
            blocker=None if teacher_diagnostic_only else f"{label} eval sample count is too small",
            warning=f"{label} eval sample count is too small" if teacher_diagnostic_only else None,
            detail="diagnostic only for held-out/generalization gates" if teacher_diagnostic_only else "",
        )
        _add_check(
            report,
            name=f"{label}_watertight_rate",
            passed=watertight_rate is not None and watertight_rate >= min_watertight_rate,
            value=watertight_rate,
            threshold=f">={min_watertight_rate}",
            blocker=None if teacher_diagnostic_only else f"{label} watertight rate is below threshold",
            warning=f"{label} watertight rate is below threshold" if teacher_diagnostic_only else None,
            detail="diagnostic only for held-out/generalization gates" if teacher_diagnostic_only else "",
        )
        _add_check(
            report,
            name=f"{label}_mean_boundary_edges",
            passed=mean_boundary is not None and float(mean_boundary) <= max_mean_boundary_edges,
            value=mean_boundary,
            threshold=f"<={max_mean_boundary_edges}",
            blocker=None if teacher_diagnostic_only else f"{label} still has boundary edges",
            warning=f"{label} still has boundary edges" if teacher_diagnostic_only else None,
            detail="diagnostic only for held-out/generalization gates" if teacher_diagnostic_only else "",
        )
        _add_check(
            report,
            name=f"{label}_mean_nonmanifold_edges",
            passed=mean_nonmanifold is not None and float(mean_nonmanifold) <= max_mean_nonmanifold_edges,
            value=mean_nonmanifold,
            threshold=f"<={max_mean_nonmanifold_edges}",
            blocker=None if teacher_diagnostic_only else f"{label} still has nonmanifold edges",
            warning=f"{label} still has nonmanifold edges" if teacher_diagnostic_only else None,
            detail="diagnostic only for held-out/generalization gates" if teacher_diagnostic_only else "",
        )
        _add_check(
            report,
            name=f"{label}_mean_nonmanifold_vertices",
            passed=mean_nonmanifold_vertices is not None
            and float(mean_nonmanifold_vertices) <= max_mean_nonmanifold_vertices,
            value=mean_nonmanifold_vertices,
            threshold=f"<={max_mean_nonmanifold_vertices}",
            blocker=None if teacher_diagnostic_only else f"{label} still has nonmanifold/pinched vertices",
            warning=f"{label} still has nonmanifold/pinched vertices" if teacher_diagnostic_only else None,
            detail="diagnostic only for held-out/generalization gates" if teacher_diagnostic_only else "",
        )
        _add_check(
            report,
            name=f"{label}_edge_pairing",
            passed=edge_pairing is not None and float(edge_pairing) >= min_edge_pairing_ratio,
            value=edge_pairing,
            threshold=f">={min_edge_pairing_ratio}",
            blocker=None if teacher_diagnostic_only else f"{label} token edge graph is not fully paired",
            warning=f"{label} token edge graph is not fully paired" if teacher_diagnostic_only else None,
            detail="diagnostic only for held-out/generalization gates" if teacher_diagnostic_only else "",
        )
        _add_check(
            report,
            name=f"{label}_raw_boundary_fill_edges",
            passed=fill_edges is None or float(fill_edges) <= max_mean_boundary_fill_edges,
            value=fill_edges,
            threshold=f"<={max_mean_boundary_fill_edges}",
            blocker=None if teacher_diagnostic_only else f"{label} relies on boundary fill to close too many raw edges",
            warning=f"{label} relies on boundary fill to close too many raw edges" if teacher_diagnostic_only else None,
            detail="raw decoder should be nearly closed before repair" if not teacher_diagnostic_only else "diagnostic only for held-out/generalization gates",
        )
        _add_check(
            report,
            name=f"{label}_boundary_fill_face_ratio",
            passed=fill_ratio is None or float(fill_ratio) <= max_mean_boundary_fill_face_ratio,
            value=fill_ratio,
            threshold=f"<={max_mean_boundary_fill_face_ratio}",
            blocker=None if teacher_diagnostic_only else f"{label} adds too many faces during boundary fill",
            warning=f"{label} adds too many faces during boundary fill" if teacher_diagnostic_only else None,
            detail="raw decoder should not need large synthetic caps" if not teacher_diagnostic_only else "diagnostic only for held-out/generalization gates",
        )

    teacher_token = teacher_metrics.get("teacher_forced_token_accuracy")
    teacher_face = teacher_metrics.get("teacher_forced_face_exact_ratio")
    _add_check(
        report,
        name="teacher_forced_token_accuracy",
        passed=teacher_token is not None and float(teacher_token) >= min_teacher_token_accuracy,
        value=teacher_token,
        threshold=f">={min_teacher_token_accuracy}" if strict_teacher_gate else "diagnostic only",
        blocker="teacher-forced token accuracy is not close enough to memorization" if strict_teacher_gate else None,
        warning="held-out teacher-forced token accuracy is low; inspect as a learning-curve signal" if not strict_teacher_gate else None,
        detail="strict only for train/memorization gates" if not strict_teacher_gate else "",
    )
    _add_check(
        report,
        name="teacher_forced_face_exact_ratio",
        passed=teacher_face is not None and float(teacher_face) >= min_teacher_face_exact,
        value=teacher_face,
        threshold=f">={min_teacher_face_exact}" if strict_teacher_gate else "diagnostic only",
        blocker="teacher-forced face exact ratio is not close enough to memorization" if strict_teacher_gate else None,
        warning="held-out teacher-forced face exact ratio is low; inspect as a learning-curve signal" if not strict_teacher_gate else None,
        detail="strict only for train/memorization gates" if not strict_teacher_gate else "",
    )

    free_chamfer = free_metrics.get("chamfer_l2_normalized") or {}
    free_normals = free_metrics.get("normal_consistency") or {}
    median_chamfer = free_chamfer.get("median")
    p95_chamfer = free_chamfer.get("p95")
    median_normal = free_normals.get("median")
    _add_check(
        report,
        name="free_run_median_chamfer_l2_normalized",
        passed=median_chamfer is not None and float(median_chamfer) <= max_median_chamfer_l2_normalized,
        value=median_chamfer,
        threshold=f"<={max_median_chamfer_l2_normalized}",
        blocker="free-run median geometry error is too high",
    )
    _add_check(
        report,
        name="free_run_p95_chamfer_l2_normalized",
        passed=p95_chamfer is not None and float(p95_chamfer) <= max_p95_chamfer_l2_normalized,
        value=p95_chamfer,
        threshold=f"<={max_p95_chamfer_l2_normalized}",
        blocker="free-run tail geometry error is too high",
    )
    _add_check(
        report,
        name="free_run_median_normal_consistency",
        passed=median_normal is not None and float(median_normal) >= min_median_normal_consistency,
        value=median_normal,
        threshold=f">={min_median_normal_consistency}",
        blocker="free-run median normal consistency is too low",
    )

    topology_check_names = {
        "curated_dataset_size",
        "curated_boundary_growth_order",
        "free_run_sample_count",
        "free_run_watertight_rate",
        "free_run_mean_boundary_edges",
        "free_run_mean_nonmanifold_edges",
        "free_run_mean_nonmanifold_vertices",
        "free_run_edge_pairing",
        "free_run_raw_boundary_fill_edges",
        "free_run_boundary_fill_face_ratio",
    }
    if strict_teacher_gate:
        topology_check_names.update(
            {
                "teacher_forced_sample_count",
                "teacher_forced_watertight_rate",
                "teacher_forced_mean_boundary_edges",
                "teacher_forced_mean_nonmanifold_edges",
                "teacher_forced_mean_nonmanifold_vertices",
                "teacher_forced_edge_pairing",
                "teacher_forced_raw_boundary_fill_edges",
                "teacher_forced_boundary_fill_face_ratio",
                "teacher_forced_token_accuracy",
                "teacher_forced_face_exact_ratio",
            }
        )
    geometry_check_names = {
        "free_run_median_chamfer_l2_normalized",
        "free_run_p95_chamfer_l2_normalized",
        "free_run_median_normal_consistency",
    }
    by_name = {check.name: check for check in report.checks}
    report.topology_ready = all(by_name[name].passed for name in topology_check_names if name in by_name)
    report.geometry_ready = all(by_name[name].passed for name in geometry_check_names if name in by_name)
    report.scale_ready = bool(report.topology_ready and report.geometry_ready)
    if report.scale_ready:
        report.recommendation = "promote to the next larger curated run"
    elif report.topology_ready:
        report.recommendation = "do not scale corpus yet; topology is green but geometry fidelity needs the next bounded probe"
    else:
        report.recommendation = "do not scale; fix the listed topology/training blockers first"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curation-summary", type=Path)
    parser.add_argument("--teacher-eval", type=Path)
    parser.add_argument("--free-run-eval", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-dataset-samples", type=int, default=8)
    parser.add_argument("--min-eval-samples", type=int, default=8)
    parser.add_argument("--min-watertight-rate", type=float, default=1.0)
    parser.add_argument("--min-teacher-token-accuracy", type=float, default=0.995)
    parser.add_argument("--min-teacher-face-exact", type=float, default=0.99)
    parser.add_argument("--min-edge-pairing-ratio", type=float, default=1.0)
    parser.add_argument("--max-mean-boundary-edges", type=float, default=0.0)
    parser.add_argument("--max-mean-nonmanifold-edges", type=float, default=0.0)
    parser.add_argument("--max-mean-nonmanifold-vertices", type=float, default=0.0)
    parser.add_argument("--max-median-chamfer-l2-normalized", type=float, default=0.02)
    parser.add_argument("--max-p95-chamfer-l2-normalized", type=float, default=0.08)
    parser.add_argument("--min-median-normal-consistency", type=float, default=0.75)
    parser.add_argument("--teacher-gate-mode", choices=["memorization", "generalization"], default="memorization")
    parser.add_argument("--max-mean-boundary-fill-edges", type=float, default=0.0)
    parser.add_argument("--max-mean-boundary-fill-face-ratio", type=float, default=0.0)
    args = parser.parse_args()

    report = assess(
        curation=_load_json(args.curation_summary),
        teacher_eval=_load_json(args.teacher_eval),
        free_run_eval=_load_json(args.free_run_eval),
        min_dataset_samples=args.min_dataset_samples,
        min_eval_samples=args.min_eval_samples,
        min_watertight_rate=args.min_watertight_rate,
        min_teacher_token_accuracy=args.min_teacher_token_accuracy,
        min_teacher_face_exact=args.min_teacher_face_exact,
        min_edge_pairing_ratio=args.min_edge_pairing_ratio,
        max_mean_boundary_edges=args.max_mean_boundary_edges,
        max_mean_nonmanifold_edges=args.max_mean_nonmanifold_edges,
        max_mean_nonmanifold_vertices=args.max_mean_nonmanifold_vertices,
        max_median_chamfer_l2_normalized=args.max_median_chamfer_l2_normalized,
        max_p95_chamfer_l2_normalized=args.max_p95_chamfer_l2_normalized,
        min_median_normal_consistency=args.min_median_normal_consistency,
        teacher_gate_mode=args.teacher_gate_mode,
        max_mean_boundary_fill_edges=args.max_mean_boundary_fill_edges,
        max_mean_boundary_fill_face_ratio=args.max_mean_boundary_fill_face_ratio,
    )
    payload = {
        **asdict(report),
        "checks": [asdict(check) for check in report.checks],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "scale_ready": report.scale_ready,
        "topology_ready": report.topology_ready,
        "geometry_ready": report.geometry_ready,
        "recommendation": report.recommendation,
        "blockers": report.blockers,
        "warnings": report.warnings,
    }, indent=2, sort_keys=True))
    return 0 if report.scale_ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
