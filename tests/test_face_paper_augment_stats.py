from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "research" / "train_face_paper_faithful.py"
    spec = importlib.util.spec_from_file_location("train_face_paper_faithful", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


train_face = _load_module()


def _sample() -> train_face.PaperFaceSample:
    return train_face.PaperFaceSample(
        path=Path("sample.npz"),
        point_features=np.asarray(
            [
                [-0.5, -0.5, -0.5, 0.0, 0.0, 1.0],
                [0.5, -0.5, -0.5, 0.0, 1.0, 0.0],
                [-0.5, 0.5, 0.5, 1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        tokens=np.asarray([[0, 0, 0, 0, 0, 7, 7, 7, 0]], dtype=np.int64),
        paper_within_face_order="rotate_min_zyx",
    )


def _augment(sample: train_face.PaperFaceSample, diagnostics: train_face.AugmentDiagnostics):
    return train_face._augment_sample(
        sample,
        num_bins=8,
        rng=np.random.default_rng(123),
        rotation="none",
        scale_min=1.0,
        scale_max=1.0,
        flip_prob=0.0,
        diagnostics=diagnostics,
    )


def test_augment_diagnostics_records_success(monkeypatch):
    sample = _sample()
    diagnostics = train_face.AugmentDiagnostics()

    def canonicalize(vertices, faces, *, num_bins, within_face_order):
        assert within_face_order == "rotate_min_zyx"
        return sample.tokens.copy(), faces.copy()

    monkeypatch.setattr(train_face, "canonicalize_mesh_faces_paper_zyx", canonicalize)

    _, tokens = _augment(sample, diagnostics)

    snapshot = diagnostics.snapshot()
    assert np.array_equal(tokens, sample.tokens)
    assert snapshot["attempts"] == 1
    assert snapshot["successes"] == 1
    assert snapshot["fallbacks"] == 0
    assert snapshot["avg_original_faces"] == 1.0
    assert snapshot["avg_augmented_faces"] == 1.0


def test_augment_diagnostics_records_exception_fallback(monkeypatch):
    sample = _sample()
    diagnostics = train_face.AugmentDiagnostics()

    def fail_canonicalize(vertices, faces, *, num_bins, within_face_order):
        assert within_face_order == "rotate_min_zyx"
        raise RuntimeError("forced tokenization failure")

    monkeypatch.setattr(train_face, "canonicalize_mesh_faces_paper_zyx", fail_canonicalize)

    points, tokens = _augment(sample, diagnostics)

    snapshot = diagnostics.snapshot()
    assert points is sample.point_features
    assert tokens is sample.tokens
    assert snapshot["attempts"] == 1
    assert snapshot["successes"] == 0
    assert snapshot["fallbacks"] == 1
    assert snapshot["fallback_reasons"] == {"tokenize_exception": 1}
    assert snapshot["exception_types"] == {"RuntimeError": 1}
    assert snapshot["last_exception_message"] == "forced tokenization failure"


def test_augment_diagnostics_records_empty_token_fallback(monkeypatch):
    sample = _sample()
    diagnostics = train_face.AugmentDiagnostics()

    def empty_canonicalize(vertices, faces, *, num_bins, within_face_order):
        assert within_face_order == "rotate_min_zyx"
        return np.empty((0, 9), dtype=np.int64), faces[:0]

    monkeypatch.setattr(train_face, "canonicalize_mesh_faces_paper_zyx", empty_canonicalize)

    points, tokens = _augment(sample, diagnostics)

    snapshot = diagnostics.snapshot()
    assert points is sample.point_features
    assert tokens is sample.tokens
    assert snapshot["attempts"] == 1
    assert snapshot["successes"] == 0
    assert snapshot["fallbacks"] == 1
    assert snapshot["fallback_reasons"] == {"empty_tokens": 1}
