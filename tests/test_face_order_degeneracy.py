from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "research" / "analyze_face_order_degeneracy.py"
    spec = importlib.util.spec_from_file_location("analyze_face_order_degeneracy", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_order_degeneracy_counts_same_min_anchor_faces():
    module = _load_module()
    tokens = np.asarray(
        [
            [0, 0, 0, 0, 0, 4, 0, 4, 0],
            [0, 0, 0, 0, 4, 0, 4, 0, 0],
            [3, 3, 3, 3, 3, 4, 4, 3, 3],
        ],
        dtype=np.int64,
    )

    row = module.analyze_tokens(tokens, near_bins=(1, 4))

    assert row["same_min_count"] == 2
    assert row["same_slot0_anchor_count"] == 2
    assert row["near_min_counts"]["1"] == 2
    assert row["near_min_counts"]["4"] == 3
    assert row["second_min_linf_gap"] == 0


def test_order_degeneracy_aggregate_reports_tie_rates():
    module = _load_module()
    rows = [
        {"ordered_by_min_zyx": True, "same_min_count": 2, "same_slot0_anchor_count": 2, "second_min_linf_gap": 0, "second_min_l1_gap": 0, "near_min_counts": {"1": 2}},
        {"ordered_by_min_zyx": True, "same_min_count": 1, "same_slot0_anchor_count": 1, "second_min_linf_gap": 3, "second_min_l1_gap": 6, "near_min_counts": {"1": 1}},
    ]

    summary = module._aggregate(rows, near_bins=(1,))

    assert summary["same_min_gt1_rate"] == pytest.approx(0.5)
    assert summary["near_min_gt1_rates"]["1"] == pytest.approx(0.5)
    assert summary["second_min_linf_gap_mean"] == pytest.approx(1.5)
