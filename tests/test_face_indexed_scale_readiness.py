from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import trimesh


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "research" / "assess_face_indexed_scale_readiness.py"
    spec = importlib.util.spec_from_file_location("assess_face_indexed_scale_readiness", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_refresh_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "research" / "refresh_eval_pair_metrics.py"
    spec = importlib.util.spec_from_file_location("refresh_eval_pair_metrics", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _eval_report(*, attempted: int, watertight: int, chamfers: list[float], token_accuracy: float | None = None):
    results = []
    for idx in range(attempted):
        results.append(
            {
                "watertight": idx < watertight,
                "boundary_edges": 0,
                "nonmanifold_edges": 0,
                "nonmanifold_vertices": 0,
                "token_edge_pairing_ratio": 1.0,
                "chamfer_l2_normalized": chamfers[min(idx, len(chamfers) - 1)],
                "normal_consistency": 0.9,
                "decode_elapsed_sec": 0.1,
                "teacher_forced_token_accuracy": token_accuracy,
                "teacher_forced_face_exact_ratio": 1.0 if token_accuracy is not None else None,
            }
        )
    return {
        "summary": {
            "attempted": attempted,
            "watertight": watertight,
            "mean_boundary_edges": 0.0,
            "mean_nonmanifold_edges": 0.0,
            "mean_nonmanifold_vertices": 0.0,
            "mean_edge_pairing_ratio": 1.0,
            "mean_teacher_forced_token_accuracy": token_accuracy,
            "mean_teacher_forced_face_exact_ratio": 1.0 if token_accuracy is not None else None,
        },
        "results": results,
    }


def test_readiness_blocks_watertight_but_geometrically_bad_run():
    module = _load_module()
    report = module.assess(
        curation={"written": 8, "failed": 0, "zero_closure_after_first_sum": 0},
        teacher_eval=_eval_report(attempted=8, watertight=8, chamfers=[0.001], token_accuracy=1.0),
        free_run_eval=_eval_report(attempted=8, watertight=8, chamfers=[0.001, 5.0]),
        min_dataset_samples=8,
        min_eval_samples=8,
        min_watertight_rate=1.0,
        min_teacher_token_accuracy=0.995,
        min_teacher_face_exact=0.99,
        min_edge_pairing_ratio=1.0,
        max_mean_boundary_edges=0.0,
        max_mean_nonmanifold_edges=0.0,
        max_mean_nonmanifold_vertices=0.0,
        max_median_chamfer_l2_normalized=0.1,
        max_p95_chamfer_l2_normalized=1.0,
        min_median_normal_consistency=0.75,
    )
    assert report.topology_ready
    assert not report.geometry_ready
    assert not report.scale_ready
    assert "free-run median geometry error is too high" in report.blockers


def test_readiness_promotes_when_topology_and_geometry_pass():
    module = _load_module()
    report = module.assess(
        curation={"written": 16, "failed": 0, "zero_closure_after_first_sum": 0},
        teacher_eval=_eval_report(attempted=16, watertight=16, chamfers=[0.001], token_accuracy=1.0),
        free_run_eval=_eval_report(attempted=16, watertight=16, chamfers=[0.01, 0.02]),
        min_dataset_samples=8,
        min_eval_samples=8,
        min_watertight_rate=1.0,
        min_teacher_token_accuracy=0.995,
        min_teacher_face_exact=0.99,
        min_edge_pairing_ratio=1.0,
        max_mean_boundary_edges=0.0,
        max_mean_nonmanifold_edges=0.0,
        max_mean_nonmanifold_vertices=0.0,
        max_median_chamfer_l2_normalized=0.1,
        max_p95_chamfer_l2_normalized=1.0,
        min_median_normal_consistency=0.75,
    )
    assert report.topology_ready
    assert report.geometry_ready
    assert report.scale_ready


def test_refresh_eval_pair_metrics_adds_normalized_distances(tmp_path: Path):
    module = _load_refresh_module()
    export_dir = tmp_path / "exports"
    export_dir.mkdir()
    mesh = trimesh.creation.box(extents=(3, 2, 1))
    mesh.export(export_dir / "0000_box_generated.glb")
    mesh.export(export_dir / "0000_box_teacher.glb")
    report = {"results": [{}], "summary": {"attempted": 1}}

    refreshed = module.refresh_report(report, export_dir, samples=200, seed=3)

    item = refreshed["results"][0]
    assert refreshed["pair_metrics_refresh"]["refreshed"] == 1
    assert item["chamfer_l2_normalized"] is not None
    assert item["hausdorff_l2_normalized"] is not None
    assert refreshed["summary"]["mean_chamfer_l2_normalized"] == item["chamfer_l2_normalized"]
