from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "research" / "assess_face_paper_scale_readiness.py"
    spec = importlib.util.spec_from_file_location("assess_face_paper_scale_readiness", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _settings(*, disable_augment: bool = False):
    return {
        "settings": {
            "point_samples": 8192,
            "vecset_tokens": 2048,
            "latent_dim": 64,
            "optimizer": "muon",
            "lr": 0.0006,
            "weight_decay": 0.1,
            "precision": "bf16",
            "disable_augment": disable_augment,
            "augment_rotation": "so3",
            "encoder_backend": "shape2vecset",
            "decode_head": "causal",
            "causal_mlp_variant": "legacy_concat",
            "face_embedding_variant": "token_concat_project",
            "model_max_faces": 512,
        }
    }


def _settings_with_split_integrity(*, train_count: int, test_count: int):
    run_summary = _settings()
    run_summary["split_integrity"] = {
        "ok": True,
        "train": {"sample_count": train_count, "valid_count": train_count},
        "test": {"sample_count": test_count, "valid_count": test_count},
    }
    return run_summary


def _settings_with_capacity(**capacity):
    settings = _settings()
    settings["settings"].update(capacity)
    return settings


def _eval_report(
    *,
    attempted: int,
    watertight: int,
    accuracy: float,
    loss: float,
    mean_boundary_edges: float,
    edge_pairing: float,
    generated_accuracy: float | None = None,
    chamfer: float = 0.02,
    normal: float = 0.8,
    generation_mode: str = "autoregressive",
):
    if generated_accuracy is None:
        generated_accuracy = accuracy
    results = []
    for idx in range(attempted):
        results.append(
            {
                "watertight": idx < watertight,
                "token_boundary_edge_count": mean_boundary_edges,
                "boundary_edges": mean_boundary_edges,
                "nonmanifold_edges": 0,
                "token_edge_pairing_ratio": edge_pairing,
                "teacher_forced_accuracy": accuracy,
                "teacher_forced_loss": loss,
                "generated_token_accuracy": generated_accuracy,
                "generated_vertex_exact_ratio": generated_accuracy,
                "generated_edge_exact_ratio": generated_accuracy,
                "generated_face_exact_ratio": generated_accuracy,
                "generated_edge_set_precision": edge_pairing,
                "generated_edge_set_recall": edge_pairing,
                "generated_edge_set_f1": edge_pairing,
                "chamfer_l2_normalized": chamfer,
                "hausdorff_l2_normalized": chamfer * 2.0,
                "normal_consistency": normal,
                "sample_sec": 1.0,
                "truncated_by_generation_face_limit": False,
            }
        )
    return {
        "generation_mode": generation_mode,
        "generation_face_limit": 0,
        "face_count_mode": "gt",
        "causal_mlp_variant": "legacy_concat",
        "decode_head": "causal",
        "encoder_backend": "shape2vecset",
        "summary": {
            "attempted": attempted,
            "watertight": watertight,
            "mean_teacher_forced_accuracy": accuracy,
            "mean_teacher_forced_loss": loss,
            "mean_generated_token_accuracy": generated_accuracy,
            "mean_generated_vertex_exact_ratio": generated_accuracy,
            "mean_generated_edge_exact_ratio": generated_accuracy,
            "mean_generated_face_exact_ratio": generated_accuracy,
            "mean_generated_edge_set_precision": edge_pairing,
            "mean_generated_edge_set_recall": edge_pairing,
            "mean_generated_edge_set_f1": edge_pairing,
            "mean_boundary_edges": mean_boundary_edges,
            "mean_edge_pairing_ratio": edge_pairing,
            "mean_predicted_to_reference_face_ratio": 1.0,
            "faces_per_sec": 100.0,
        },
        "results": results,
    }


def test_paper_scale_readiness_promotes_only_when_train_and_holdout_pass():
    module = _load_module()
    report = module.assess(
        run_summary=_settings(),
        train_teacher=_eval_report(attempted=64, watertight=64, accuracy=0.99, loss=0.02, mean_boundary_edges=0, edge_pairing=1.0, generation_mode="teacher_forced"),
        train_ar=_eval_report(attempted=10, watertight=9, accuracy=0.98, loss=0.03, mean_boundary_edges=1, edge_pairing=0.998),
        test_teacher=_eval_report(attempted=16, watertight=8, accuracy=0.24, loss=2.0, mean_boundary_edges=20, edge_pairing=0.8, generation_mode="teacher_forced"),
        test_ar=_eval_report(attempted=10, watertight=3, accuracy=0.24, loss=2.0, mean_boundary_edges=20, edge_pairing=0.8),
    )
    assert report.scale_ready
    assert report.topology_ready
    assert report.generalization_ready
    assert report.recommendation == "promote_to_next_corpus_rung"


def test_paper_scale_readiness_blocks_no_aug_diagnostic_as_non_paper_knob():
    module = _load_module()
    report = module.assess(
        run_summary=_settings(disable_augment=True),
        train_teacher=_eval_report(attempted=88, watertight=39, accuracy=0.999, loss=0.006, mean_boundary_edges=10.8, edge_pairing=0.989, generation_mode="teacher_forced"),
        train_ar=_eval_report(attempted=10, watertight=2, accuracy=0.999, loss=0.005, mean_boundary_edges=14.2, edge_pairing=0.989),
        test_teacher=_eval_report(attempted=22, watertight=0, accuracy=0.074, loss=8.3, mean_boundary_edges=1250, edge_pairing=0.0, generation_mode="teacher_forced"),
        test_ar=_eval_report(attempted=10, watertight=0, accuracy=0.052, loss=8.6, mean_boundary_edges=983, edge_pairing=0.245),
    )
    assert not report.scale_ready
    assert not report.paper_knobs_ready
    assert "paper-knob mismatch: disable_augment" in report.blockers
    assert report.recommendation == "fix_paper_knob_mismatch_before_more_gpu"


def test_paper_scale_readiness_blocks_truncated_ar_prefixes():
    module = _load_module()
    train_ar = _eval_report(attempted=10, watertight=10, accuracy=1.0, loss=0.0, mean_boundary_edges=0, edge_pairing=1.0)
    train_ar["generation_face_limit"] = 128
    for row in train_ar["results"]:
        row["truncated_by_generation_face_limit"] = True
    report = module.assess(
        run_summary=_settings(),
        train_teacher=_eval_report(attempted=64, watertight=64, accuracy=1.0, loss=0.0, mean_boundary_edges=0, edge_pairing=1.0, generation_mode="teacher_forced"),
        train_ar=train_ar,
        test_teacher=_eval_report(attempted=16, watertight=16, accuracy=1.0, loss=0.0, mean_boundary_edges=0, edge_pairing=1.0, generation_mode="teacher_forced"),
        test_ar=_eval_report(attempted=10, watertight=10, accuracy=1.0, loss=0.0, mean_boundary_edges=0, edge_pairing=1.0),
    )
    assert not report.scale_ready
    assert "train_autoregressive is truncated; full-face AR evidence is required" in report.blockers


def test_paper_scale_readiness_warns_for_reduced_capacity_without_blocking_next_rung():
    module = _load_module()
    report = module.assess(
        run_summary=_settings_with_capacity(
            hidden_size=384,
            encoder_hidden_size=384,
            encoder_layers=4,
            decoder_layers=8,
            heads=8,
        ),
        train_teacher=_eval_report(attempted=64, watertight=64, accuracy=0.99, loss=0.02, mean_boundary_edges=0, edge_pairing=1.0, generation_mode="teacher_forced"),
        train_ar=_eval_report(attempted=10, watertight=9, accuracy=0.98, loss=0.03, mean_boundary_edges=1, edge_pairing=0.998),
        test_teacher=_eval_report(attempted=16, watertight=8, accuracy=0.24, loss=2.0, mean_boundary_edges=20, edge_pairing=0.8, generation_mode="teacher_forced"),
        test_ar=_eval_report(attempted=10, watertight=3, accuracy=0.24, loss=2.0, mean_boundary_edges=20, edge_pairing=0.8),
    )
    assert report.scale_ready
    assert "model capacity is below the paper-scale profile; treat this as a bounded validation, not a final reproduction claim" in report.warnings


def test_paper_scale_readiness_uses_true_free_run_accuracy_for_ar_gate():
    module = _load_module()
    report = module.assess(
        run_summary=_settings(),
        train_teacher=_eval_report(
            attempted=64,
            watertight=64,
            accuracy=0.99,
            loss=0.02,
            mean_boundary_edges=0,
            edge_pairing=1.0,
            generation_mode="teacher_forced",
        ),
        train_ar=_eval_report(
            attempted=10,
            watertight=9,
            accuracy=0.99,
            generated_accuracy=0.25,
            loss=0.03,
            mean_boundary_edges=1,
            edge_pairing=0.998,
        ),
        test_teacher=_eval_report(
            attempted=16,
            watertight=8,
            accuracy=0.24,
            loss=2.0,
            mean_boundary_edges=20,
            edge_pairing=0.8,
            generation_mode="teacher_forced",
        ),
        test_ar=_eval_report(attempted=10, watertight=3, accuracy=0.24, loss=2.0, mean_boundary_edges=20, edge_pairing=0.8),
    )
    assert not report.scale_ready
    assert "train autoregressive free-run tokens do not match the learned targets closely enough" in report.blockers


def test_paper_scale_readiness_prefers_split_integrity_counts_over_eval_attempts():
    module = _load_module()
    report = module.assess(
        run_summary=_settings_with_split_integrity(train_count=4362, test_count=1091),
        train_teacher=_eval_report(
            attempted=256,
            watertight=256,
            accuracy=0.99,
            loss=0.02,
            mean_boundary_edges=0,
            edge_pairing=1.0,
            generation_mode="teacher_forced",
        ),
        train_ar=_eval_report(attempted=32, watertight=30, accuracy=0.98, loss=0.03, mean_boundary_edges=1, edge_pairing=0.998),
        test_teacher=_eval_report(
            attempted=256,
            watertight=128,
            accuracy=0.30,
            loss=2.0,
            mean_boundary_edges=20,
            edge_pairing=0.8,
            generation_mode="teacher_forced",
        ),
        test_ar=_eval_report(attempted=32, watertight=10, accuracy=0.30, loss=2.0, mean_boundary_edges=20, edge_pairing=0.8),
        min_dataset_samples=1000,
    )

    assert report.metrics["dataset"] == {"train": 4362, "test": 1091, "total": 5453}
    assert "bounded corpus is too small to justify promotion" not in report.blockers
