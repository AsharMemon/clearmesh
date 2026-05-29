"""FastAPI app for the ClearMesh production scaffold."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import Cookie, Depends, FastAPI, File, Header, HTTPException, Request, Response, UploadFile, status
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing_extensions import Annotated

from clearmesh.product.artifacts import ArtifactStore
from clearmesh.product.accounts import AccountContext, ProductAccountStore
from clearmesh.product.auth import Principal, authenticate_api_key
from clearmesh.product.billing import CreditLedger
from clearmesh.product.jobs import JobService
from clearmesh.product.models import GenerationRequest
from clearmesh.product.oauth import build_authorization_url, exchange_code_for_profile, provider_status
from clearmesh.product.profiles import runtime_quote
from clearmesh.product.status import summarize_job_progress
from clearmesh.product.store import JsonJobStore
from clearmesh.product.subscriptions import (
    create_billing_portal_session,
    create_checkout_session,
    plan_for_tier,
    public_plans,
)

STATE_ROOT = os.getenv("CLEARMESH_STATE_ROOT", ".clearmesh_state")
ARTIFACT_ROOT = os.getenv("CLEARMESH_ARTIFACT_ROOT", "artifacts")
DASHBOARD_DIR = Path(os.getenv("CLEARMESH_DASHBOARD_DIR", Path(__file__).resolve().parents[2] / "dashboard"))
SESSION_COOKIE = "cm_session"


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
_accounts = ProductAccountStore(os.path.join(STATE_ROOT, "accounts.json"))
_ledger = CreditLedger(os.path.join(STATE_ROOT, "credits.json"))
_jobs = JobService(store=_store, ledger=_ledger)
_artifacts = build_artifact_store()

app = FastAPI(title="ClearMesh API", version="0.1.0")
if (DASHBOARD_DIR / "static").exists():
    app.mount("/static", StaticFiles(directory=DASHBOARD_DIR / "static"), name="static")


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


class SignupRequest(BaseModel):
    email: str
    password: str = Field(min_length=8)
    name: str | None = None
    team_name: str | None = None


class LoginRequest(BaseModel):
    email: str
    password: str


class SettingsPatchRequest(BaseModel):
    profile: dict[str, Any] = Field(default_factory=dict)
    user_settings: dict[str, Any] = Field(default_factory=dict)
    team_settings: dict[str, Any] = Field(default_factory=dict)


class ApiKeyCreateRequest(BaseModel):
    label: str = "Production API key"


class CheckoutRequest(BaseModel):
    tier: str
    success_url: str | None = None
    cancel_url: str | None = None


class InferenceBridgeRequest(BaseModel):
    prompt: str | None = None
    input_uri: str | None = None
    mode: str = Field(default="text_to_3d", pattern="^(image_to_3d|text_to_3d|edit_image|edit_text)$")
    output_formats: list[str] = Field(default_factory=lambda: ["glb", "obj"])
    quality_tier: str = Field(default="standard", pattern="^(draft|standard|high)$")
    pipeline: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


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


def dashboard_file(name: str = "product.html") -> FileResponse:
    path = DASHBOARD_DIR / name
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{name} not found")
    return FileResponse(path)


def public_base_url(request: Request) -> str:
    return os.getenv("CLEARMESH_PUBLIC_URL", str(request.base_url).rstrip("/")).rstrip("/")


def oauth_redirect_uri(request: Request, provider: str) -> str:
    return f"{public_base_url(request)}/v1/auth/oauth/{provider}/callback"


def set_session_cookie(response: Response, raw_session: str) -> None:
    max_age = int(os.getenv("CLEARMESH_SESSION_DAYS", "30")) * 24 * 60 * 60
    response.set_cookie(
        SESSION_COOKIE,
        raw_session,
        max_age=max_age,
        httponly=True,
        secure=os.getenv("CLEARMESH_COOKIE_SECURE", "0") == "1",
        samesite="lax",
    )


def clear_session_cookie(response: Response) -> None:
    response.delete_cookie(SESSION_COOKIE)


def account_response(context: AccountContext) -> dict:
    payload = context.to_public_dict()
    payload["credits"] = _ledger.get_account(context.team["id"]).__dict__
    payload["oauth_providers"] = provider_status()
    return payload


def grant_trial_credits(team_id: str) -> None:
    trial_credits = int(os.getenv("CLEARMESH_TRIAL_CREDITS", "25"))
    if trial_credits > 0:
        _ledger.grant(team_id, trial_credits, reason="trial_signup")


def require_principal(
    authorization: Annotated[str | None, Header()] = None,
    session_cookie: Annotated[str | None, Cookie(alias=SESSION_COOKIE)] = None,
) -> Principal:
    if authorization and authorization.lower().startswith("bearer "):
        raw_key = authorization.split(" ", 1)[1]
        principal = authenticate_api_key(raw_key) or _accounts.authenticate_api_key(raw_key)
        if principal is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid API key")
        return principal
    context = _accounts.authenticate_session(session_cookie)
    if context is not None:
        return context.principal
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="missing bearer token or session")


def require_account(session_cookie: Annotated[str | None, Cookie(alias=SESSION_COOKIE)] = None) -> AccountContext:
    context = _accounts.authenticate_session(session_cookie)
    if context is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="missing session")
    return context


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'
        '<rect width="64" height="64" rx="12" fill="#eef7fa"/>'
        '<path d="M18 40 31 14l15 9-4 24-21-2z" fill="none" stroke="#2f6f9f" stroke-width="5" '
        'stroke-linejoin="round"/>'
        '<path d="M18 40h24M31 14l11 33M46 23 21 45" stroke="#71806f" stroke-width="2"/>'
        "</svg>"
    )
    return Response(content=svg, media_type="image/svg+xml")


@app.get("/", include_in_schema=False)
def home_app() -> FileResponse:
    return dashboard_file("home.html")


@app.get("/app", include_in_schema=False)
@app.get("/studio", include_in_schema=False)
@app.get("/settings", include_in_schema=False)
def product_app() -> FileResponse:
    return dashboard_file("product.html")


@app.get("/pricing", include_in_schema=False)
def pricing_app() -> FileResponse:
    return dashboard_file("pricing.html")


@app.get("/index.html", include_in_schema=False)
def legacy_index() -> RedirectResponse:
    return RedirectResponse("/app")


@app.get("/home.html", include_in_schema=False)
def legacy_home() -> RedirectResponse:
    return RedirectResponse("/")


@app.get("/pricing.html", include_in_schema=False)
def legacy_pricing() -> RedirectResponse:
    return RedirectResponse("/pricing")


@app.head("/", include_in_schema=False)
@app.head("/app", include_in_schema=False)
@app.head("/pricing", include_in_schema=False)
@app.head("/settings", include_in_schema=False)
def product_app_head() -> Response:
    return Response(media_type="text/html")


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True}


def inference_bridge_config() -> dict[str, Any]:
    base_url = os.getenv("CLEARMESH_INFERENCE_BASE_URL", "").rstrip("/")
    public_url = os.getenv("CLEARMESH_INFERENCE_PUBLIC_URL", base_url).rstrip("/")
    return {
        "available": bool(base_url),
        "stage": "remote" if base_url else "local_preview",
        "public_url": public_url or None,
        "pipeline": {
            "text_to_image": os.getenv("CLEARMESH_TEXT_TO_IMAGE_MODEL", "open-source configurable"),
            "trellis": os.getenv("CLEARMESH_TRELLIS_MODEL", "microsoft/TRELLIS.2-4B"),
            "faceq": os.getenv("CLEARMESH_FACEQ_MODEL_NAME", "faceq-1p4b-65k1024-effective100k"),
            "easy3e": os.getenv("CLEARMESH_EASY3E_ENABLED", "0") == "1",
        },
        "model_bundle_b2_prefix": os.getenv(
            "CLEARMESH_FACEQ_MODEL_B2_PREFIX",
            "face-runs/faceq-1p4b-65k1024-257k-full/vast-20260523_lr1e4_clip1_100k/"
            "runs/faceq_merged_scale_full_100k_init_from025k_remaining75k/model_bundle_effective100k",
        ),
    }


def post_to_inference_bridge(payload: dict[str, Any]) -> dict[str, Any]:
    base_url = os.getenv("CLEARMESH_INFERENCE_BASE_URL", "").rstrip("/")
    if not base_url:
        return {
            "ok": True,
            "status": "local_preview",
            "job_id": f"demo_{uuid4().hex[:12]}",
            "message": "Remote inference GPU is not connected yet; the Studio is showing a local preview.",
            "pipeline": "Text/Image -> Trellis -> FACE-Q -> Easy3E-ready",
        }

    path = os.getenv("CLEARMESH_INFERENCE_SUBMIT_PATH", "/v1/pipeline/jobs")
    url = urllib.parse.urljoin(f"{base_url}/", path.lstrip("/"))
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    token = os.getenv("CLEARMESH_INFERENCE_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    timeout = float(os.getenv("CLEARMESH_INFERENCE_TIMEOUT_SECONDS", "30"))
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise HTTPException(status_code=502, detail=f"inference bridge rejected request: {detail}") from exc
    except urllib.error.URLError as exc:
        raise HTTPException(status_code=502, detail=f"inference bridge unavailable: {exc.reason}") from exc
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=502, detail="inference bridge returned invalid JSON") from exc


def get_from_inference_bridge(path: str) -> tuple[bytes, str]:
    base_url = os.getenv("CLEARMESH_INFERENCE_BASE_URL", "").rstrip("/")
    if not base_url:
        raise HTTPException(status_code=404, detail="remote inference bridge is not connected")
    url = urllib.parse.urljoin(f"{base_url}/", path.lstrip("/"))
    headers = {"Accept": "application/json"}
    token = os.getenv("CLEARMESH_INFERENCE_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers, method="GET")
    timeout = float(os.getenv("CLEARMESH_INFERENCE_TIMEOUT_SECONDS", "30"))
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read(), response.headers.get_content_type()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        status_code = exc.code if exc.code in {400, 401, 403, 404} else 502
        raise HTTPException(status_code=status_code, detail=detail or "inference bridge request failed") from exc
    except urllib.error.URLError as exc:
        raise HTTPException(status_code=502, detail=f"inference bridge unavailable: {exc.reason}") from exc


@app.get("/v1/inference/config")
def get_inference_config() -> dict[str, Any]:
    return inference_bridge_config()


@app.post("/v1/inference/demo", status_code=202)
def submit_inference_demo(payload: InferenceBridgeRequest) -> dict[str, Any]:
    payload_dict = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
    payload_dict["metadata"] = sanitize_client_metadata(payload_dict.get("metadata", {}))
    payload_dict["request_id"] = f"ui_{uuid4().hex}"
    payload_dict["model_bundle_b2_prefix"] = inference_bridge_config()["model_bundle_b2_prefix"]
    response = post_to_inference_bridge(payload_dict)
    response.setdefault("ok", True)
    response.setdefault("pipeline", "Text/Image -> Trellis -> FACE-Q -> Easy3E-ready")
    return response


@app.get("/v1/inference/jobs/{job_id}")
def get_inference_job(job_id: str) -> dict[str, Any]:
    safe_job_id = urllib.parse.quote(job_id, safe="")
    body, _content_type = get_from_inference_bridge(f"/v1/pipeline/jobs/{safe_job_id}")
    try:
        job = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=502, detail="inference bridge returned invalid JSON") from exc
    artifacts = job.get("artifacts") or {}
    job["artifact_urls"] = {
        name: f"/v1/inference/jobs/{safe_job_id}/artifacts/{urllib.parse.quote(name, safe='')}" for name in artifacts
    }
    return job


@app.get("/v1/inference/jobs/{job_id}/artifacts/{name}")
def get_inference_artifact(job_id: str, name: str) -> Response:
    safe_job_id = urllib.parse.quote(job_id, safe="")
    safe_name = urllib.parse.quote(name, safe="")
    body, content_type = get_from_inference_bridge(f"/v1/pipeline/jobs/{safe_job_id}/artifacts/{safe_name}")
    return Response(content=body, media_type=content_type)


@app.get("/v1/auth/providers")
def auth_providers() -> dict:
    return {"providers": provider_status()}


@app.get("/v1/auth/me")
def me(context: AccountContext = Depends(require_account)) -> dict:
    return account_response(context)


@app.post("/v1/auth/signup", status_code=201)
def signup(payload: SignupRequest, response: Response) -> dict:
    try:
        context, created = _accounts.create_email_user(
            email=payload.email,
            password=payload.password,
            name=payload.name,
            team_name=payload.team_name,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if created:
        grant_trial_credits(context.team["id"])
    raw_session, _session_id = _accounts.create_session(context.user["id"], context.team["id"])
    set_session_cookie(response, raw_session)
    return account_response(context)


@app.post("/v1/auth/login")
def login(payload: LoginRequest, response: Response) -> dict:
    context = _accounts.authenticate_password(payload.email, payload.password)
    if context is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid email or password")
    raw_session, _session_id = _accounts.create_session(context.user["id"], context.team["id"])
    set_session_cookie(response, raw_session)
    return account_response(context)


@app.post("/v1/auth/logout")
def logout(
    response: Response,
    session_cookie: Annotated[str | None, Cookie(alias=SESSION_COOKIE)] = None,
) -> dict:
    _accounts.delete_session(session_cookie)
    clear_session_cookie(response)
    return {"ok": True}


@app.get("/v1/auth/oauth/{provider}/start")
def oauth_start(provider: str, request: Request, next: str = "/") -> RedirectResponse:
    try:
        state = _accounts.create_oauth_state(provider, next)
        authorize_url = build_authorization_url(provider, redirect_uri=oauth_redirect_uri(request, provider), state=state)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    return RedirectResponse(authorize_url)


@app.get("/v1/auth/oauth/{provider}/callback")
def oauth_callback(provider: str, request: Request, code: str | None = None, state: str | None = None, error: str | None = None):
    if error:
        return RedirectResponse(f"/?auth_error={error}")
    if not code or not state:
        return RedirectResponse("/?auth_error=missing_oauth_code")
    try:
        next_path = _accounts.consume_oauth_state(state, provider)
        profile = exchange_code_for_profile(provider, code=code, redirect_uri=oauth_redirect_uri(request, provider))
        context, created = _accounts.upsert_oauth_user(
            provider=provider,
            provider_user_id=profile.provider_user_id,
            email=profile.email,
            name=profile.name,
            avatar_url=profile.avatar_url,
        )
    except Exception as exc:  # provider failures should land back in the app.
        return RedirectResponse(f"/?auth_error={type(exc).__name__}")
    if created:
        grant_trial_credits(context.team["id"])
    raw_session, _session_id = _accounts.create_session(context.user["id"], context.team["id"])
    redirect = RedirectResponse(next_path)
    set_session_cookie(redirect, raw_session)
    return redirect


@app.patch("/v1/settings")
def update_settings(payload: SettingsPatchRequest, context: AccountContext = Depends(require_account)) -> dict:
    updated = _accounts.update_settings(
        user_id=context.user["id"],
        team_id=context.team["id"],
        profile=payload.profile,
        user_settings=payload.user_settings,
        team_settings=payload.team_settings,
    )
    return account_response(updated)


@app.get("/v1/settings/api-keys")
def list_api_keys(context: AccountContext = Depends(require_account)) -> dict:
    return {"api_keys": _accounts.list_api_keys(context.team["id"])}


@app.post("/v1/settings/api-keys", status_code=201)
def create_api_key(payload: ApiKeyCreateRequest, context: AccountContext = Depends(require_account)) -> dict:
    key = _accounts.create_api_key(context.user["id"], context.team["id"], payload.label)
    return {"api_key": key}


@app.delete("/v1/settings/api-keys/{key_id}")
def delete_api_key(key_id: str, context: AccountContext = Depends(require_account)) -> dict:
    deleted = _accounts.delete_api_key(context.team["id"], key_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="api key not found")
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


@app.get("/v1/billing/plans")
def billing_plans() -> dict:
    return {"plans": public_plans()}


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


@app.get("/v1/billing/subscription")
def subscription(context: AccountContext = Depends(require_account)) -> dict:
    team = context.to_public_dict()["team"]
    return {
        "team": team,
        "plan": plan_for_tier(team.get("plan", "free")),
        "credits": _ledger.get_account(context.team["id"]).__dict__,
    }


@app.post("/v1/billing/checkout")
def checkout(payload: CheckoutRequest, request: Request, context: AccountContext = Depends(require_account)) -> dict:
    base = public_base_url(request)
    success_url = payload.success_url or f"{base}/settings?checkout=success"
    cancel_url = payload.cancel_url or f"{base}/pricing?checkout=cancel"
    try:
        return create_checkout_session(
            tier=payload.tier,
            team_id=context.team["id"],
            user_id=context.user["id"],
            customer_email=context.user["email"],
            success_url=success_url,
            cancel_url=cancel_url,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/billing/portal")
def billing_portal(request: Request, context: AccountContext = Depends(require_account)) -> dict:
    subscription_data = context.team.get("subscription", {})
    return create_billing_portal_session(
        customer_id=subscription_data.get("stripe_customer_id"),
        return_url=f"{public_base_url(request)}/settings",
    )


@app.post("/v1/billing/webhook")
async def stripe_webhook(request: Request) -> dict:
    body = await request.body()
    webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET")
    signature = request.headers.get("stripe-signature")
    if webhook_secret:
        import stripe

        try:
            event = stripe.Webhook.construct_event(body, signature, webhook_secret)
        except Exception as exc:
            raise HTTPException(status_code=400, detail="invalid stripe webhook") from exc
    elif os.getenv("CLEARMESH_ALLOW_UNSIGNED_STRIPE_WEBHOOKS") == "1":
        event = json.loads(body.decode("utf-8"))
    else:
        raise HTTPException(status_code=404, detail="stripe webhook disabled")

    event_type = event.get("type")
    obj = event.get("data", {}).get("object", {})
    metadata = obj.get("metadata") or {}
    team_id = metadata.get("team_id")
    tier = metadata.get("tier") or "free"
    if not team_id:
        return {"ok": True, "ignored": "missing team metadata"}

    if event_type == "checkout.session.completed":
        _accounts.set_team_subscription(
            team_id=team_id,
            tier=tier,
            status="active",
            stripe_customer_id=obj.get("customer"),
            stripe_subscription_id=obj.get("subscription"),
        )
        monthly_credits = plan_for_tier(tier).get("monthly_credits")
        if monthly_credits:
            _ledger.grant_once(
                team_id,
                int(monthly_credits),
                reason=f"stripe_{tier}_credits",
                idempotency_key=event.get("id") or obj.get("id") or "",
            )
    elif event_type in {"customer.subscription.updated", "customer.subscription.created"}:
        _accounts.set_team_subscription(
            team_id=team_id,
            tier=tier,
            status=obj.get("status") or "active",
            stripe_customer_id=obj.get("customer"),
            stripe_subscription_id=obj.get("id"),
            current_period_end=epoch_to_iso(obj.get("current_period_end")),
        )
    elif event_type == "customer.subscription.deleted":
        _accounts.set_team_subscription(
            team_id=team_id,
            tier="free",
            status="canceled",
            stripe_customer_id=obj.get("customer"),
            stripe_subscription_id=obj.get("id"),
            current_period_end=epoch_to_iso(obj.get("current_period_end")),
        )
    return {"ok": True}


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


def epoch_to_iso(value: Any) -> str | None:
    if value in {None, ""}:
        return None
    try:
        return datetime.fromtimestamp(int(value), tz=timezone.utc).isoformat()
    except (TypeError, ValueError, OSError):
        return None
