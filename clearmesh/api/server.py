"""FastAPI app for the ClearMesh production scaffold."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile, status
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel, Field
from typing_extensions import Annotated

from clearmesh.product.artifacts import ArtifactStore
from clearmesh.product.auth import Principal, authenticate_api_key
from clearmesh.product.billing import CreditLedger
from clearmesh.product.jobs import JobService
from clearmesh.product.models import GenerationRequest
from clearmesh.product.profiles import runtime_quote
from clearmesh.product.status import summarize_job_progress
from clearmesh.product.store import JsonJobStore

STATE_ROOT = os.getenv("CLEARMESH_STATE_ROOT", ".clearmesh_state")
ARTIFACT_ROOT = os.getenv("CLEARMESH_ARTIFACT_ROOT", "artifacts")


def build_job_store():
    postgres_dsn = os.getenv("CLEARMESH_POSTGRES_DSN")
    if postgres_dsn:
        from clearmesh.product.postgres_store import PostgresJobStore

        return PostgresJobStore(postgres_dsn)
    return JsonJobStore(STATE_ROOT)


def build_artifact_store():
    bucket = os.getenv("CLEARMESH_S3_BUCKET")
    if bucket:
        from clearmesh.product.s3_artifacts import S3ArtifactStore

        return S3ArtifactStore(
            ARTIFACT_ROOT,
            bucket=bucket,
            prefix=os.getenv("CLEARMESH_S3_PREFIX", ""),
        )
    return ArtifactStore(ARTIFACT_ROOT)


_store = build_job_store()
_ledger = CreditLedger(os.path.join(STATE_ROOT, "credits.json"))
_jobs = JobService(store=_store, ledger=_ledger)
_artifacts = build_artifact_store()

app = FastAPI(title="ClearMesh API", version="0.1.0")


class JobCreateRequest(BaseModel):
    input_uri: str
    mode: str = Field(default="image_to_3d", pattern="^(image_to_3d|text_to_3d|edit_image|edit_text)$")
    prompt: str | None = None
    output_formats: list[str] = Field(default_factory=lambda: ["glb", "obj"])
    point_budgets: list[int] = Field(default_factory=lambda: [16_384, 40_960, 100_000])
    enable_parts: bool = True
    enable_rigging: bool = False
    quality_tier: str = Field(default="standard", pattern="^(draft|standard|high)$")
    metadata: dict[str, Any] = Field(default_factory=dict)


class CreditGrantRequest(BaseModel):
    team_id: str
    credits: int
    reason: str = "manual_grant"


class UploadResponse(BaseModel):
    uri: str
    filename: str
    content_type: str | None = None
    size_bytes: int


RUNTIME_METADATA_PREFIXES = (
    "trellis_",
    "part_structure_",
    "easy3e_",
    "autorigging_",
)
RUNTIME_METADATA_KEYS = {
    "proxy_mesh_path",
    "artist_mesh_path",
    "edited_mesh_path",
}


def sanitize_client_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Keep public API metadata descriptive, not executable."""

    if os.getenv("CLEARMESH_ALLOW_CLIENT_COMMAND_METADATA") == "1":
        return dict(metadata)
    sanitized: dict[str, Any] = {}
    for key, value in metadata.items():
        if key in RUNTIME_METADATA_KEYS or any(key.startswith(prefix) for prefix in RUNTIME_METADATA_PREFIXES):
            continue
        sanitized[key] = value
    return sanitized


def require_principal(authorization: Annotated[str | None, Header()] = None) -> Principal:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="missing bearer token")
    raw_key = authorization.split(" ", 1)[1]
    principal = authenticate_api_key(raw_key)
    if principal is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid API key")
    return principal


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True}


@app.post("/v1/jobs", status_code=202)
def create_job(payload: JobCreateRequest, principal: Principal = Depends(require_principal)) -> dict:
    payload_dict = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
    payload_dict["metadata"] = sanitize_client_metadata(payload_dict.get("metadata", {}))
    request = GenerationRequest(**payload_dict)
    try:
        job = _jobs.create_job(team_id=principal.team_id, user_id=principal.user_id, request=request)
    except ValueError as exc:
        raise HTTPException(status_code=402, detail=str(exc)) from exc
    return enrich_job_response(job)


@app.post("/v1/billing/quote")
def quote_job(payload: JobCreateRequest, principal: Principal = Depends(require_principal)) -> dict:
    payload_dict = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
    payload_dict["metadata"] = sanitize_client_metadata(payload_dict.get("metadata", {}))
    request = GenerationRequest(**payload_dict)
    account = _ledger.get_account(principal.team_id)
    credits = _jobs.estimate_cost(request)
    return {
        "estimated_credits": credits,
        "balance": account.balance,
        "reserved": account.reserved,
        "available": account.balance - account.reserved,
        "runtime": runtime_quote(
            request.quality_tier,
            includes_trellis="proxy_mesh_path" not in request.metadata,
            enable_parts=request.enable_parts,
            part_count=request.metadata.get("estimated_part_count"),
        ),
    }


@app.post("/v1/uploads", status_code=201)
def upload_input(file: UploadFile = File(...), principal: Principal = Depends(require_principal)) -> UploadResponse:
    safe_name = Path(file.filename or "upload.bin").name
    upload_id = f"upload_{uuid4().hex}"
    destination = Path(ARTIFACT_ROOT) / "uploads" / principal.team_id / upload_id / safe_name
    destination.parent.mkdir(parents=True, exist_ok=True)

    size = 0
    max_bytes = int(os.getenv("CLEARMESH_MAX_UPLOAD_BYTES", str(100 * 1024 * 1024)))
    with destination.open("wb") as handle:
        while chunk := file.file.read(1024 * 1024):
            size += len(chunk)
            if size > max_bytes:
                handle.close()
                destination.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="upload too large")
            handle.write(chunk)

    return UploadResponse(
        uri=f"local://uploads/{principal.team_id}/{upload_id}/{safe_name}",
        filename=safe_name,
        content_type=file.content_type,
        size_bytes=size,
    )


@app.get("/v1/jobs")
def list_jobs(principal: Principal = Depends(require_principal)) -> dict:
    return {"jobs": [enrich_job_response(job) for job in _store.list_jobs(team_id=principal.team_id)]}


@app.get("/v1/jobs/{job_id}")
def get_job(job_id: str, principal: Principal = Depends(require_principal)) -> dict:
    try:
        job = _store.get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="job not found") from exc
    if job.team_id != principal.team_id:
        raise HTTPException(status_code=404, detail="job not found")
    return enrich_job_response(job)


@app.get("/v1/jobs/{job_id}/status")
def get_job_status(job_id: str, principal: Principal = Depends(require_principal)) -> dict:
    try:
        job = _store.get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="job not found") from exc
    if job.team_id != principal.team_id:
        raise HTTPException(status_code=404, detail="job not found")
    progress = summarize_job_progress(job).to_dict()
    return {
        "job_id": job.id,
        "status": job.status.value,
        "updated_at": job.updated_at,
        "progress": progress,
        "asset_urls": asset_urls_for_progress(job, progress),
    }


@app.get("/v1/jobs/{job_id}/assets")
def list_assets(job_id: str, principal: Principal = Depends(require_principal)) -> dict:
    job = get_job(job_id, principal)
    assets = []
    for asset in job["assets"]:
        enriched = dict(asset)
        enriched["download_url"] = f"/v1/jobs/{job_id}/assets/{asset['id']}/download"
        if asset.get("uri"):
            enriched["local_url"] = _artifacts.signed_url_placeholder(asset["uri"])
        assets.append(enriched)
    return {"assets": assets}


@app.get("/v1/jobs/{job_id}/assets/{asset_id}/download")
def download_asset(job_id: str, asset_id: str, principal: Principal = Depends(require_principal)) -> FileResponse:
    job = get_job(job_id, principal)
    asset = next((item for item in job["assets"] if item["id"] == asset_id), None)
    if asset is None:
        raise HTTPException(status_code=404, detail="asset not found")
    if str(asset.get("uri", "")).startswith("s3://"):
        return RedirectResponse(_artifacts.signed_url_placeholder(asset["uri"]))
    try:
        path = _artifacts.resolve_asset_path(asset["uri"])
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="asset file not found") from exc
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="asset file not found")
    return FileResponse(path, media_type=asset.get("content_type") or "application/octet-stream", filename=path.name)


@app.post("/v1/jobs/{job_id}/cancel")
def cancel_job(job_id: str, principal: Principal = Depends(require_principal)) -> dict:
    try:
        job = _store.get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="job not found") from exc
    if job.team_id != principal.team_id:
        raise HTTPException(status_code=404, detail="job not found")
    return enrich_job_response(_jobs.cancel_job(job_id))


@app.get("/v1/billing/credits")
def credits(principal: Principal = Depends(require_principal)) -> dict:
    return _ledger.get_account(principal.team_id).__dict__


@app.get("/v1/billing/events")
def billing_events(limit: int = 100, principal: Principal = Depends(require_principal)) -> dict:
    safe_limit = max(1, min(limit, 500))
    return {"events": [event.__dict__ for event in _ledger.list_events(principal.team_id, safe_limit)]}


@app.post("/admin/credits/grant")
def grant_credits(payload: CreditGrantRequest, authorization: Annotated[str | None, Header()] = None) -> dict:
    admin_key = os.getenv("CLEARMESH_ADMIN_KEY")
    if admin_key is None:
        raise HTTPException(status_code=404, detail="admin endpoint disabled")
    if not authorization or authorization != f"Bearer {admin_key}":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid admin token")
    account = _ledger.grant(payload.team_id, payload.credits, payload.reason)
    return account.__dict__


def enrich_job_response(job) -> dict:
    data = job.to_dict()
    data["progress"] = summarize_job_progress(job).to_dict()
    return data


def asset_urls_for_progress(job, progress: dict) -> dict[str, str]:
    urls: dict[str, str] = {}
    for key in (
        "preview_asset_id",
        "preview_mesh_asset_id",
        "reference_mesh_asset_id",
        "retopology_plan_asset_id",
        "chart_remesh_manifest_asset_id",
        "chart_stitched_mesh_asset_id",
        "quad_mesh_asset_id",
        "projected_quad_mesh_asset_id",
        "final_asset_id",
        "quality_report_asset_id",
        "production_gate_asset_id",
    ):
        asset_id = progress.get(key)
        if not asset_id:
            continue
        asset = next((item for item in job.assets if item.id == asset_id), None)
        if asset is not None:
            url_key = key[:-3] if key.endswith("_id") else key
            urls[f"{url_key}_url"] = f"/v1/jobs/{job.id}/assets/{asset.id}/download"
    return urls
