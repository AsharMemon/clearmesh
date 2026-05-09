from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


pytest.importorskip("trimesh")


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "research" / "eval_face_paper_faithful.py"
    spec = importlib.util.spec_from_file_location("eval_face_paper_faithful", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_generated_vs_teacher_metrics_track_free_run_token_drift():
    module = _load_module()
    teacher = np.asarray(
        [
            [0, 0, 0, 0, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 1, 0, 1, 0, 0],
        ],
        dtype=np.int64,
    )
    generated = teacher.copy()
    generated[1, 8] = 1

    metrics = module._generated_vs_teacher_metrics(generated, teacher)

    assert metrics["generated_compared_face_count"] == 2
    assert metrics["generated_token_accuracy"] == pytest.approx(17 / 18)
    assert metrics["generated_face_exact_ratio"] == pytest.approx(0.5)
    assert metrics["generated_vertex_exact_ratio"] == pytest.approx(5 / 6)
    assert metrics["first_divergent_face_index"] == 1
    assert metrics["first_divergent_coord_slot"] == 8
    assert metrics["first_divergent_token_index"] == 17
    assert metrics["generated_edge_set_f1"] < 1.0


def test_first_face_logit_metrics_report_rank_and_entropy():
    torch = pytest.importorskip("torch")
    module = _load_module()
    logits = torch.zeros((1, 1, 9, 8), dtype=torch.float32)
    targets = torch.arange(9, dtype=torch.long).reshape(1, 1, 9) % 8
    for slot in range(9):
        logits[0, 0, slot, int(targets[0, 0, slot])] = 10.0

    metrics = module._first_face_logit_metrics(logits, targets, num_bins=8)

    assert metrics["first_face_teacher_rank_by_slot"] == [1] * 9
    assert metrics["first_face_teacher_top1_accuracy"] == pytest.approx(1.0)
    assert metrics["first_face_teacher_target_prob_mean"] > 0.99


def test_teacher_prefix_generation_forces_initial_faces():
    torch = pytest.importorskip("torch")
    module = _load_module()

    class FakeModel:
        def init_incremental_cache(self, point_tensor):
            return {}

        def incremental_hidden_step(self, previous_face, position, cache):
            return torch.zeros((1, 1, 4), dtype=torch.float32)

        def greedy_face_from_hidden(self, hidden, limit_bins=None):
            return torch.full((1, 9), 7, dtype=torch.long)

    teacher = np.asarray(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2, 2, 2, 2, 2],
        ],
        dtype=np.int64,
    )
    generated, meta = module._generate_tokens(
        FakeModel(),
        np.zeros((8, 6), dtype=np.float32),
        face_count=4,
        num_bins=8,
        device=torch.device("cpu"),
        decode_head="causal",
        teacher_prefix_tokens=teacher,
    )

    np.testing.assert_array_equal(generated[:2], teacher)
    np.testing.assert_array_equal(generated[2:], np.full((2, 9), 7, dtype=np.int64))
    assert meta["teacher_prefix_faces"] == 2
