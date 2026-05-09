from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "research" / "analyze_face_paper_ar_failures.py"
    spec = importlib.util.spec_from_file_location("analyze_face_paper_ar_failures", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _row(
    *,
    watertight: bool,
    acc: float,
    teacher_acc: float = 0.99,
    boundary: int = 0,
    faces: int = 512,
    first_face: int | None = None,
):
    return {
        "path": f"/tmp/sample_{acc}.npz",
        "watertight": watertight,
        "generated_token_accuracy": acc,
        "teacher_forced_accuracy": teacher_acc,
        "token_boundary_edge_count": boundary,
        "token_edge_pairing_ratio": 1.0 if boundary == 0 else 0.7,
        "reference_face_count": faces,
        "normal_consistency": 0.8,
        "chamfer_l2_normalized": 0.01,
        "first_divergent_face_index": first_face,
        "first_divergent_coord_slot": None if first_face is None else 2,
    }


def _write_eval(path: Path, rows: list[dict], *, summary: dict | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload_summary = {"attempted": len(rows)}
    if summary:
        payload_summary.update(summary)
    path.write_text(json.dumps({"results": rows, "summary": payload_summary}), encoding="utf-8")


def test_analyzer_reports_early_divergence_and_generalization_gap(tmp_path):
    module = _load_module()
    run = tmp_path / "run"
    _write_eval(
        run / "eval" / "train_teacher_forced.json",
        [_row(watertight=True, acc=1.0), _row(watertight=False, acc=0.98, boundary=8)],
    )
    _write_eval(
        run / "eval" / "train_autoregressive.json",
        [
            _row(watertight=True, acc=1.0),
            _row(watertight=False, acc=0.9, boundary=100, first_face=0),
            _row(watertight=False, acc=0.8, boundary=200, first_face=4),
        ],
    )
    _write_eval(
        run / "eval" / "test_teacher_forced.json",
        [_row(watertight=False, acc=0.05, teacher_acc=0.05, boundary=1000, first_face=0)],
    )
    _write_eval(
        run / "eval" / "test_autoregressive.json",
        [_row(watertight=False, acc=0.02, teacher_acc=0.05, boundary=1000, first_face=0)],
    )

    report = module.analyze(run)
    train_ar = report["evals"]["train_autoregressive"]

    assert train_ar["watertight_rate"] == 1 / 3
    assert train_ar["early_divergence"]["early_face_le_8_count"] == 2
    assert train_ar["face_count_bins"][0]["count"] == 3
    assert report["diagnosis"]["generalization_gap"]["test_teacher_accuracy_mean"] == 0.05
    assert "exposure bias" in report["diagnosis"]["next_debug_hint"]


def test_analyzer_identifies_underfit_first_face_and_generic_closed_mesh(tmp_path):
    module = _load_module()
    run = tmp_path / "run"
    _write_eval(
        run / "eval" / "train_teacher_forced.json",
        [
            _row(watertight=False, acc=0.75, teacher_acc=0.75, boundary=900, first_face=0),
            _row(watertight=False, acc=0.76, teacher_acc=0.76, boundary=850, first_face=0),
        ],
    )
    _write_eval(
        run / "eval" / "test_teacher_forced.json",
        [_row(watertight=False, acc=0.75, teacher_acc=0.75, boundary=900, first_face=0)],
    )
    _write_eval(
        run / "eval" / "train_autoregressive.json",
        [
            _row(watertight=True, acc=0.03, teacher_acc=0.75, boundary=0, first_face=0),
            _row(watertight=False, acc=0.04, teacher_acc=0.75, boundary=8, first_face=0),
        ],
    )
    _write_eval(
        run / "eval" / "test_autoregressive.json",
        [_row(watertight=True, acc=0.08, teacher_acc=0.75, boundary=0, first_face=0)],
    )
    _write_eval(
        run / "eval" / "test_autoregressive_predicted_count.json",
        [_row(watertight=True, acc=0.08, teacher_acc=0.75, boundary=0, first_face=0)],
        summary={"mean_predicted_to_reference_face_ratio": 12.0},
    )

    report = module.analyze(run)
    modes = {mode["name"] for mode in report["diagnosis"]["failure_modes"]}

    assert "teacher_forced_underfit" in modes
    assert "first_face_collapse" in modes
    assert "coordinate_close_but_topologically_broken" in modes
    assert "generic_closed_mesh_not_target_reconstruction" in modes
    assert "predicted_count_overrun" in modes
    assert "teacher-forced reconstruction is still underfit" in report["diagnosis"]["next_debug_hint"]


def test_analyzer_reports_teacher_prefix_diagnostics(tmp_path):
    module = _load_module()
    run = tmp_path / "run"
    _write_eval(
        run / "eval" / "train_teacher_forced.json",
        [_row(watertight=True, acc=0.999, teacher_acc=0.999)],
    )
    _write_eval(
        run / "eval" / "test_teacher_forced.json",
        [_row(watertight=False, acc=0.1, teacher_acc=0.1, boundary=1000, first_face=0)],
    )
    _write_eval(
        run / "eval" / "train_autoregressive.json",
        [
            _row(watertight=False, acc=0.95, teacher_acc=0.999, boundary=32, first_face=12),
            _row(watertight=True, acc=1.0, teacher_acc=1.0),
        ],
    )
    _write_eval(
        run / "eval" / "train_autoregressive_prefix1.json",
        [
            _row(watertight=False, acc=0.96, teacher_acc=0.999, boundary=30, first_face=12),
            _row(watertight=True, acc=1.0, teacher_acc=1.0),
        ],
    )
    _write_eval(
        run / "eval" / "test_autoregressive.json",
        [_row(watertight=False, acc=0.03, teacher_acc=0.1, boundary=700, first_face=0)],
    )
    _write_eval(
        run / "eval" / "test_autoregressive_prefix1.json",
        [_row(watertight=False, acc=0.05, teacher_acc=0.1, boundary=650, first_face=0)],
    )

    report = module.analyze(run)
    diagnosis = report["diagnosis"]
    train_prefix = diagnosis["prefix_diagnostics"]["train"]
    modes = {mode["name"] for mode in diagnosis["failure_modes"]}

    assert train_prefix["best_by_accuracy"]["teacher_prefix_faces"] == 1
    assert train_prefix["best_by_accuracy"]["generated_token_accuracy_gain"] > 0
    assert "prefix_does_not_close_train_topology" in modes
    assert "prefix_does_not_rescue_heldout" in modes
