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
