"""User-facing job progress and artifact selection helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .models import AssetRecord, JobRecord, JobStatus, JobStepStatus


@dataclass(frozen=True)
class JobProgress:
    phase: str
    percent: int
    message: str
    preview_ready: bool
    final_ready: bool
    rig_ready: bool
    preview_asset_id: str | None
    preview_mesh_asset_id: str | None
    reference_mesh_asset_id: str | None
    retopology_plan_asset_id: str | None
    chart_remesh_manifest_asset_id: str | None
    chart_stitched_mesh_asset_id: str | None
    quad_mesh_asset_id: str | None
    projected_quad_mesh_asset_id: str | None
    final_asset_id: str | None
    quality_report_asset_id: str | None
    production_gate_asset_id: str | None

    def to_dict(self) -> dict:
        return asdict(self)


def summarize_job_progress(job: JobRecord) -> JobProgress:
    preview_asset = _latest_asset(job, "preview_image")
    preview_mesh = _latest_asset(job, "control_mesh") or _latest_asset(job, "cleaned_mesh")
    reference_mesh = _latest_asset(job, "reference_mesh") or _latest_asset(job, "refined_reference_mesh")
    retopology_plan = _latest_asset(job, "retopology_plan")
    chart_remesh_manifest = _latest_asset(job, "chart_remesh_manifest")
    chart_stitched_mesh = _latest_asset(job, "chart_stitched_mesh")
    quad_mesh = _latest_asset(job, "quad_mesh")
    projected_quad_mesh = _latest_asset(job, "projected_quad_mesh")
    final_asset = _latest_asset(job, "export_mesh")
    rig_asset = _latest_asset(job, "rigged_mesh")
    quality_report = _latest_asset(job, "mesh_eval_report")
    production_gate = _latest_asset(job, "production_gate_report")

    preview_ready = preview_asset is not None or preview_mesh is not None
    final_ready = final_asset is not None and job.status == JobStatus.SUCCEEDED
    if job.status == JobStatus.FAILED:
        phase = "failed"
        percent = _completed_percent(job)
        message = job.error or "Generation failed."
    elif job.status == JobStatus.CANCELED:
        phase = "canceled"
        percent = _completed_percent(job)
        message = "Generation was canceled."
    elif final_ready:
        phase = "final_ready"
        percent = 100
        message = "Final editable mesh is ready."
    elif preview_ready:
        phase = "preview_ready"
        percent = max(45, _completed_percent(job))
        message = "Preview mesh is ready; high-resolution finishing is continuing."
    elif job.status == JobStatus.RUNNING:
        phase = "running"
        percent = _completed_percent(job)
        message = _running_message(job)
    else:
        phase = "queued"
        percent = 0
        message = "Queued for generation."

    return JobProgress(
        phase=phase,
        percent=percent,
        message=message,
        preview_ready=preview_ready,
        final_ready=final_ready,
        rig_ready=rig_asset is not None,
        preview_asset_id=preview_asset.id if preview_asset else None,
        preview_mesh_asset_id=preview_mesh.id if preview_mesh else None,
        reference_mesh_asset_id=reference_mesh.id if reference_mesh else None,
        retopology_plan_asset_id=retopology_plan.id if retopology_plan else None,
        chart_remesh_manifest_asset_id=chart_remesh_manifest.id if chart_remesh_manifest else None,
        chart_stitched_mesh_asset_id=chart_stitched_mesh.id if chart_stitched_mesh else None,
        quad_mesh_asset_id=quad_mesh.id if quad_mesh else None,
        projected_quad_mesh_asset_id=projected_quad_mesh.id if projected_quad_mesh else None,
        final_asset_id=final_asset.id if final_asset else None,
        quality_report_asset_id=quality_report.id if quality_report else None,
        production_gate_asset_id=production_gate.id if production_gate else None,
    )


def _latest_asset(job: JobRecord, kind: str) -> AssetRecord | None:
    matches = [asset for asset in job.assets if asset.kind == kind]
    return matches[-1] if matches else None


def _completed_percent(job: JobRecord) -> int:
    if not job.steps:
        return 0
    done = sum(1 for step in job.steps if step.status in {JobStepStatus.SUCCEEDED, JobStepStatus.SKIPPED})
    failed = sum(1 for step in job.steps if step.status == JobStepStatus.FAILED)
    return int(round(((done + failed) / len(job.steps)) * 100))


def _running_message(job: JobRecord) -> str:
    running = next((step for step in job.steps if step.status == JobStepStatus.RUNNING), None)
    if running is None:
        return "Generation is starting."
    labels = {
        "input_validation": "Checking the input.",
        "easy3e_edit": "Applying the edit.",
        "trellis_proxy": "Building the visual 3D asset.",
        "coarse_adapter": "Converting the proxy into a coherent coarse surface.",
        "reference_refinement": "Refining the watertight reference surface.",
        "mesh_passport": "Checking mesh structure.",
        "surface_normalization": "Creating the editable control surface.",
        "shrinkwrap_projection": "Projecting the control surface when enabled.",
        "retopology_planning": "Planning generic retopology charts.",
        "chart_remesh": "Testing chart-level quad topology.",
        "chart_stitch": "Stitching chart quads into a promotion candidate.",
        "quad_remesh": "Testing quad topology.",
        "feature_projection": "Projecting quad features to the reference surface.",
        "preview_publish": "Preparing the preview.",
        "point_cloud_bridge": "Preparing refinement conditioning.",
        "part_structure": "Finding semantic parts.",
        "mesh_head": "Refining the artist mesh.",
        "mesh_cleanup": "Cleaning the mesh.",
        "repair_validation": "Validating the final asset.",
        "production_gate": "Checking production promotion gates.",
        "export_package": "Packaging exports.",
        "autorigging": "Adding the optional rig.",
    }
    return labels.get(running.name, f"Running {running.name}.")
