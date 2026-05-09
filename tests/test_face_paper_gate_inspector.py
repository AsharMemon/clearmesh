from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "research" / "inspect_face_paper_gate.py"
    spec = importlib.util.spec_from_file_location("inspect_face_paper_gate", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_inspector_waits_when_readiness_is_missing(tmp_path):
    module = _load_module()
    run = tmp_path / "runs" / "paper_gate"
    (run / "logs").mkdir(parents=True)
    (run / "logs" / "train.log").write_text('{"step": 100, "loss": 3.0}\n', encoding="utf-8")

    report = module.inspect(tmp_path)

    assert report["state"] == "running_or_incomplete"
    assert report["next_action"] == "wait_for_training_eval_archive"
    assert report["latest_train_rows"][-1]["step"] == 100
    assert report["progress"]["state"] == "training_or_eval_in_progress"
    assert report["progress"]["latest_step"] == 100


def test_inspector_promotes_only_after_metrics_and_visuals(tmp_path):
    module = _load_module()
    run = tmp_path / "runs" / "paper_gate"
    _write_json(run / "summary.json", {"settings": {"hidden_size": 384}})
    _write_json(
        run / "scale_readiness.json",
        {
            "scale_ready": True,
            "recommendation": "promote_to_next_corpus_rung",
            "blockers": [],
            "warnings": ["reduced capacity"],
        },
    )
    (tmp_path / "train_ar_contact_sheet.png").write_bytes(b"png")
    (tmp_path / "test_ar_contact_sheet.png").write_bytes(b"png")

    report = module.inspect(tmp_path)

    assert report["state"] == "ready_for_visual_review"
    assert report["next_action"] == "promote_next_corpus_rung_after_visual_review"
    assert report["scale_ready"] is True


def test_inspector_holds_when_scale_gate_blocks(tmp_path):
    module = _load_module()
    run = tmp_path / "runs" / "paper_gate"
    _write_json(run / "summary.json", {"settings": {"hidden_size": 384}})
    _write_json(
        run / "scale_readiness.json",
        {
            "scale_ready": False,
            "recommendation": "debug_free_running_topology_before_scaling",
            "blockers": ["train autoregressive meshes are not mostly watertight"],
            "warnings": [],
        },
    )

    report = module.inspect(tmp_path)

    assert report["state"] == "not_ready"
    assert report["next_action"] == "debug_free_running_topology_before_scaling"
    assert report["blockers"] == ["train autoregressive meshes are not mostly watertight"]


def test_inspector_reports_selection_progress(tmp_path):
    module = _load_module()
    run = tmp_path / "runs" / "paper_gate"
    (run / "logs").mkdir(parents=True)
    (run / "logs" / "train.log").write_text(
        "\n".join(
            [
                '{"step": 1, "loss": 0.4, "selection_loss": 0.4, "eta_sec": 99.0, "steps_per_sec": 1.0}',
                '{"step": 100, "loss": 0.2}',
                '{"step": 200, "loss": 0.3, "selection_loss": 0.25, "eta_sec": 300.0, "steps_per_sec": 2.0}',
                '{"step": 300, "loss": 0.1, "selection_loss": 0.2, "eta_sec": 200.0, "steps_per_sec": 2.0}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = module.inspect(tmp_path)
    progress = report["progress"]

    assert progress["latest_step"] == 300
    assert progress["selection_count"] == 3
    assert progress["first_selection_loss"] == 0.4
    assert progress["latest_selection_loss"] == 0.2
    assert progress["best_selection_loss"] == 0.2
    assert progress["best_selection_step"] == 300
    assert progress["selection_improvement"] == 0.2
    assert progress["inferred_target_step"] == 700
