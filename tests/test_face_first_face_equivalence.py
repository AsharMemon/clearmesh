from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "research" / "analyze_face_first_face_equivalence.py"
    spec = importlib.util.spec_from_file_location("analyze_face_first_face_equivalence", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_first_face_equivalence_accepts_same_anchor_non_row0(tmp_path: Path):
    module = _load_module()
    path = tmp_path / "sample.npz"
    tokens = np.asarray(
        [
            [0, 0, 0, 0, 0, 4, 0, 4, 0],
            [0, 0, 0, 0, 4, 0, 4, 0, 0],
            [3, 3, 3, 3, 3, 4, 4, 3, 3],
        ],
        dtype=np.int64,
    )
    np.savez_compressed(path, paper_tokens=tokens)
    probe = {
        "results": [
            {
                "path": str(path),
                "first_face": {
                    "best_logprob": {"tokens": tokens[1].tolist()},
                    "best_hybrid": {"tokens": tokens[2].tolist()},
                },
            }
        ]
    }

    report = module.analyze(probe, selectors=("best_logprob", "best_hybrid"))
    summary = report["summary"]

    assert summary["best_logprob_exact_row0_rate"] == pytest.approx(0.0)
    assert summary["best_logprob_in_same_anchor_group_rate"] == pytest.approx(1.0)
    assert summary["best_hybrid_in_same_anchor_group_rate"] == pytest.approx(0.0)
