from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "thunder" / "promote_face_paper_next_rung.sh"


def _run(tmp_path: Path, payload: dict, **env: str):
    inspection = tmp_path / "inspection.json"
    inspection.write_text(json.dumps(payload), encoding="utf-8")
    merged_env = os.environ.copy()
    merged_env.update(env)
    merged_env["INSPECTION_JSON"] = str(inspection)
    return subprocess.run(
        [str(SCRIPT)],
        check=False,
        env=merged_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _ready_payload():
    return {
        "scale_ready": True,
        "state": "ready_for_visual_review",
        "next_action": "promote_next_corpus_rung_after_visual_review",
        "blockers": [],
        "warnings": ["reduced capacity"],
        "galleries": {
            "train_ar": "/tmp/train.png",
            "test_ar": "/tmp/test.png",
        },
    }


def test_promoter_holds_when_inspection_is_not_ready(tmp_path):
    result = _run(
        tmp_path,
        {
            "scale_ready": False,
            "state": "not_ready",
            "next_action": "debug_free_running_topology_before_scaling",
            "blockers": ["train autoregressive meshes are not mostly watertight"],
            "galleries": {},
        },
    )
    assert result.returncode == 0
    decision = json.loads(result.stdout)
    assert decision["ready_to_launch_after_human_visual_review"] is False
    assert "train autoregressive meshes are not mostly watertight" in decision["reasons"]


def test_promoter_requires_visual_review_even_when_metrics_are_ready(tmp_path):
    result = _run(tmp_path, _ready_payload())
    assert result.returncode == 0
    decision = json.loads(result.stdout)
    assert decision["ready_to_launch_after_human_visual_review"] is True
    assert "VISUAL_REVIEW_PASSED=1" in result.stderr


def test_promoter_blocks_large_rung_without_explicit_large_allowance(tmp_path):
    result = _run(tmp_path, _ready_payload(), NEXT_SELECT_TARGET="20000", VISUAL_REVIEW_PASSED="1")
    assert result.returncode == 0
    decision = json.loads(result.stdout)
    assert decision["ready_to_launch_after_human_visual_review"] is False
    assert any("exceeds MAX_UNCONFIRMED_TARGET" in reason for reason in decision["reasons"])
