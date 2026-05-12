from __future__ import annotations

from pathlib import Path

import trimesh

from clearmesh.product.artifacts import ArtifactStore
from clearmesh.product.billing import CreditLedger
from clearmesh.product.jobs import JobService
from clearmesh.product.models import GenerationRequest, JobRecord, JobStatus
from clearmesh.product.pipeline_worker import PipelineWorker, _production_gate_decision
from clearmesh.product.store import JsonJobStore


def test_worker_emits_optional_quad_sidecar_without_replacing_final(tmp_path: Path):
    proxy_path = tmp_path / "proxy.obj"
    trimesh.creation.box(extents=(1, 2, 3)).export(proxy_path)

    store = JsonJobStore(tmp_path / "state")
    ledger = CreditLedger(tmp_path / "credits.json")
    service = JobService(store=store, ledger=ledger)
    artifacts = ArtifactStore(tmp_path / "artifacts")
    request = GenerationRequest(
        input_uri=str(proxy_path),
        point_budgets=[64],
        metadata={
            "project_id": "test",
            "proxy_mesh_path": str(proxy_path),
            "coarse_adapter_enabled": True,
            "coarse_adapter_engine": "convex_hull",
            "coarse_adapter_target_faces": 128,
            "surface_normalization_enabled": False,
            "reference_refinement_enabled": False,
            "chart_remesh_enabled": True,
            "chart_remesh_engine": "template_cage",
            "chart_remesh_max_charts": 1,
            "chart_remesh_min_chart_faces": 1,
            "chart_remesh_allow_cleanup_charts": True,
            "chart_stitch_enabled": True,
            "chart_stitch_prefer_for_projection": True,
            "quad_remesh_enabled": True,
            "quad_remesh_engine": "template_cage",
            "quad_cage_subdivisions": 2,
            "feature_projection_enabled": True,
            "shrinkwrap_iterations": 1,
            "mesh_head_policy": "never",
            "cleanup_enabled": False,
        },
    )
    job = store.create_job(JobRecord.create("team", "user", request))

    result = PipelineWorker(
        store=store,
        service=service,
        artifacts=artifacts,
        execute_heavy=True,
        preferred_point_budget=64,
    ).run_job(job.id)

    assert result.status == JobStatus.SUCCEEDED
    completed = store.get_job(job.id)
    adapter_step = next(step for step in completed.steps if step.name == "coarse_adapter")
    plan_step = next(step for step in completed.steps if step.name == "retopology_planning")
    chart_step = next(step for step in completed.steps if step.name == "chart_remesh")
    stitch_step = next(step for step in completed.steps if step.name == "chart_stitch")
    quad_step = next(step for step in completed.steps if step.name == "quad_remesh")
    projection_step = next(step for step in completed.steps if step.name == "feature_projection")
    gate_step = next(step for step in completed.steps if step.name == "production_gate")
    assert adapter_step.status.value == "succeeded"
    assert Path(adapter_step.artifacts["mesh"]).exists()
    assert plan_step.status.value == "succeeded"
    assert Path(plan_step.artifacts["report"]).exists()
    assert chart_step.status.value == "succeeded"
    assert Path(chart_step.artifacts["manifest"]).exists()
    assert stitch_step.status.value == "succeeded"
    assert Path(stitch_step.artifacts["mesh"]).exists()
    assert quad_step.status.value == "succeeded"
    assert Path(quad_step.artifacts["mesh"]).exists()
    assert projection_step.status.value == "succeeded"
    assert Path(projection_step.artifacts["mesh"]).exists()
    assert gate_step.status.value == "succeeded"
    assert Path(gate_step.artifacts["report"]).exists()

    plan_assets = [asset for asset in completed.assets if asset.kind == "retopology_plan"]
    adapter_assets = [asset for asset in completed.assets if asset.kind == "coarse_proxy_mesh"]
    chart_assets = [asset for asset in completed.assets if asset.kind == "chart_remesh_manifest"]
    stitch_assets = [asset for asset in completed.assets if asset.kind == "chart_stitched_mesh"]
    quad_assets = [asset for asset in completed.assets if asset.kind == "quad_mesh"]
    projected_assets = [asset for asset in completed.assets if asset.kind == "projected_quad_mesh"]
    gate_assets = [asset for asset in completed.assets if asset.kind == "production_gate_report"]
    final_assets = [asset for asset in completed.assets if asset.kind == "export_mesh"]
    assert adapter_assets
    assert plan_assets
    assert chart_assets
    assert stitch_assets
    assert quad_assets
    assert projected_assets
    assert gate_assets
    assert final_assets
    assert quad_assets[-1].metadata["as_final"] == "false"
    assert final_assets[-1].metadata["source"] != quad_assets[-1].uri


def test_production_gate_blocks_pinched_watertight_vertices():
    metrics = {
        "ok": True,
        "watertight": True,
        "connected_components": 1,
        "boundary_loop_count": 0,
        "nonmanifold_edge_count": 0,
        "nonmanifold_vertex_count": 2,
    }

    gate = _production_gate_decision({}, metrics, {}, None)

    assert not gate["promoted"]
    assert not gate["watertight_ready"]
    assert "nonmanifold_vertices" in gate["reasons"]


def test_production_gate_can_relax_pinched_vertex_limit_for_diagnostics():
    metrics = {
        "ok": True,
        "watertight": True,
        "connected_components": 1,
        "boundary_loop_count": 0,
        "nonmanifold_edge_count": 0,
        "nonmanifold_vertex_count": 1,
    }

    gate = _production_gate_decision({"production_max_nonmanifold_vertices": 1}, metrics, {}, None)

    assert gate["promoted"]
    assert gate["watertight_ready"]
