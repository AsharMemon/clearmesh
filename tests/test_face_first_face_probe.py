from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "research" / "probe_face_first_face_decode.py"
    spec = importlib.util.spec_from_file_location("probe_face_first_face_decode", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_candidate_summary_reports_exact_teacher_in_beam():
    module = _load_module()
    teacher = np.asarray([0, 1, 2, 0, 1, 3, 0, 2, 3], dtype=np.int64)
    candidates = [
        module.FaceCandidate(tokens=tuple([0, 1, 2, 0, 1, 4, 0, 2, 3]), logprob=-0.1, rank=1),
        module.FaceCandidate(tokens=tuple(teacher.tolist()), logprob=-1.0, rank=2),
    ]
    annotated = module._annotate_candidates(
        candidates,
        points_xyz=np.asarray(
            [
                [-1.0, -1.0, -1.0],
                [-1.0, -1.0, -0.95],
                [-1.0, -0.95, -1.0],
                [-0.95, -1.0, -1.0],
            ],
            dtype=np.float64,
        ),
        num_bins=8,
        teacher_first=teacher,
    )

    summary = module._candidate_selection_summary(annotated)

    assert summary["teacher_exact_in_beam"] is True
    assert summary["teacher_exact_best_rank"] == 2
    assert summary["best_oracle"]["teacher_exact"] is True
    assert summary["best_oracle"]["teacher_l1"] == pytest.approx(0.0)


def test_topology_score_rewards_closing_boundary_edge():
    module = _load_module()
    state = module.TopologyDecodeState()
    first = [0, 0, 0, 0, 0, 4, 0, 4, 0]
    state.add(first)

    closes_one_edge = [0, 0, 0, 0, 4, 0, 4, 0, 0]
    disconnected = [7, 7, 7, 7, 7, 6, 7, 6, 7]

    close_score = module._topology_score(closes_one_edge, state, position=1)
    disconnected_score = module._topology_score(disconnected, state, position=1)

    assert close_score > disconnected_score
