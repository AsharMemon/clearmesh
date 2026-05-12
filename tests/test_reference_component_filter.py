from __future__ import annotations

from pathlib import Path

import trimesh

from clearmesh.eval.mesh_quality import evaluate_mesh
from clearmesh.product.artifacts import ArtifactStore
from clearmesh.product.billing import CreditLedger
from clearmesh.product.jobs import JobService
from clearmesh.product.models import GenerationRequest, JobRecord, JobStepStatus
from clearmesh.product.pipeline_worker import PipelineWorker
from clearmesh.product.store import JsonJobStore


def _triangle(offset: float) -> trimesh.Trimesh:
    vertices = [[offset, 0, 0], [offset + 0.1, 0, 0], [offset, 0.1, 0]]
    faces = [[0, 1, 2]]
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def test_reference_refinement_filters_dominant_ultrashape_component(tmp_path: Path):
    proxy_path = tmp_path / "proxy.obj"
    reference_path = tmp_path / "ultrashape_raw.obj"
    trimesh.creation.box(extents=(1, 1, 1)).export(proxy_path)
    trimesh.util.concatenate(
        [trimesh.creation.icosphere(subdivisions=2, radius=1.0), *[_triangle(3 + i * 0.2) for i in range(16)]]
    ).export(reference_path)

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
            "coarse_adapter_enabled": False,
            "reference_refinement_enabled": True,
            "reference_mesh_path": str(reference_path),
            "reference_component_filter_enabled": True,
            "reference_dominant_component_face_ratio": 0.8,
        },
    )
    job = store.create_job(JobRecord.create("team", "user", request))
    worker = PipelineWorker(store=store, service=service, artifacts=artifacts, execute_heavy=True, preferred_point_budget=64)

    filtered = worker._run_reference_refinement(job.id, proxy_path)
    completed = store.get_job(job.id)
    reference_step = next(step for step in completed.steps if step.name == "reference_refinement")
    metrics = evaluate_mesh(filtered)

    assert filtered is not None
    assert Path(filtered).exists()
    assert reference_step.status == JobStepStatus.SUCCEEDED
    assert reference_step.artifacts["component_filter_applied"] == "true"
    assert Path(reference_step.artifacts["component_filter_report"]).exists()
    assert metrics["connected_components"] == 1
    assert metrics["tiny_component_count"] == 0
    assert metrics["watertight"]
