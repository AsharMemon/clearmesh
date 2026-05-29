from __future__ import annotations

import importlib
import sys
from pathlib import Path
from uuid import uuid4

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from clearmesh.product.models import AssetRecord, JobStatus  # noqa: E402


def load_server(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    state_root = tmp_path / "state"
    artifact_root = tmp_path / "artifacts"
    monkeypatch.setenv("CLEARMESH_STATE_ROOT", str(state_root))
    monkeypatch.setenv("CLEARMESH_ARTIFACT_ROOT", str(artifact_root))
    monkeypatch.setenv("CLEAR_MESH_API_KEYS", "dev:key_user_dev:user_dev:team_dev,other:key_other:user_other:team_other")
    monkeypatch.setenv("CLEARMESH_ADMIN_KEY", "admin_dev")
    monkeypatch.delenv("CLEARMESH_ALLOW_CLIENT_COMMAND_METADATA", raising=False)
    sys.modules.pop("clearmesh.api.server", None)
    return importlib.import_module("clearmesh.api.server")


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    server = load_server(monkeypatch, tmp_path)
    return server, TestClient(server.app)


def auth(key: str = "key_user_dev") -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def test_auth_billing_quote_create_cancel_flow(client):
    server, api = client

    assert api.get("/healthz").json() == {"ok": True}
    assert api.get("/").status_code == 200
    assert api.get("/app").status_code == 200
    assert api.get("/pricing").status_code == 200
    assert api.get("/v1/inference/config").json()["available"] is False
    demo = api.post("/v1/inference/demo", json={"prompt": "clean sci-fi courier drone"})
    assert demo.status_code == 202
    assert demo.json()["status"] == "local_preview"
    assert api.get("/v1/inference/jobs/demo_missing").status_code == 404
    assert api.get("/v1/inference/jobs/demo_missing/artifacts/final_mesh").status_code == 404
    assert api.get("/v1/jobs").status_code == 401

    grant = api.post(
        "/admin/credits/grant",
        headers={"Authorization": "Bearer admin_dev"},
        json={"team_id": "team_dev", "credits": 10, "reason": "test"},
    )
    assert grant.status_code == 200
    assert grant.json()["balance"] == 10

    quote = api.post(
        "/v1/billing/quote",
        headers=auth(),
        json={"input_uri": "local://uploads/example.png", "quality_tier": "draft", "enable_rigging": True},
    )
    assert quote.status_code == 200
    assert quote.json()["estimated_credits"] == 3

    create = api.post(
        "/v1/jobs",
        headers=auth(),
        json={
            "input_uri": "local://uploads/example.png",
            "quality_tier": "draft",
            "metadata": {
                "case_id": "safe_case",
                "trellis_command": "do-not-allow",
                "easy3e_command": "do-not-allow",
                "proxy_mesh_path": "/tmp/escape.obj",
            },
        },
    )
    assert create.status_code == 202
    job = create.json()
    assert job["request"]["metadata"] == {"case_id": "safe_case"}
    assert job["status"] == "queued"

    cancel = api.post(f"/v1/jobs/{job['id']}/cancel", headers=auth())
    assert cancel.status_code == 200
    assert cancel.json()["status"] == "canceled"

    credits = api.get("/v1/billing/credits", headers=auth()).json()
    assert credits["balance"] == 10
    assert credits["reserved"] == 0

    events = api.get("/v1/billing/events", headers=auth()).json()["events"]
    assert [event["kind"] for event in events][-3:] == ["test", "reserve", "canceled_release"]

    # Keep static analyzers honest that the fixture returns the module too.
    assert server.STATE_ROOT


def test_browser_signup_session_settings_and_api_key_flow(client):
    _server, api = client

    signup = api.post(
        "/v1/auth/signup",
        json={
            "email": "artist@example.com",
            "password": "correct horse",
            "name": "Mesh Artist",
            "team_name": "Blue Studio",
        },
    )
    assert signup.status_code == 201
    body = signup.json()
    assert body["user"]["email"] == "artist@example.com"
    assert body["team"]["name"] == "Blue Studio"
    assert body["credits"]["balance"] == 25

    me = api.get("/v1/auth/me")
    assert me.status_code == 200
    assert me.json()["user"]["name"] == "Mesh Artist"

    settings = api.patch(
        "/v1/settings",
        json={
            "profile": {"name": "Lead Artist"},
            "user_settings": {"default_quality_tier": "draft", "auto_rigging": True},
            "team_settings": {"workspace_name": "ClearMesh Lab", "model_endpoint": "https://models.example.test"},
        },
    )
    assert settings.status_code == 200
    assert settings.json()["user"]["settings"]["default_quality_tier"] == "draft"
    assert settings.json()["user"]["settings"]["auto_rigging"] is True
    assert settings.json()["team"]["name"] == "ClearMesh Lab"

    key = api.post("/v1/settings/api-keys", json={"label": "CI key"})
    assert key.status_code == 201
    raw_key = key.json()["api_key"]["token"]
    listed = api.get("/v1/settings/api-keys").json()["api_keys"]
    assert listed[0]["label"] == "CI key"

    job = api.post(
        "/v1/jobs",
        json={"input_uri": "prompt://clockwork-fox", "mode": "text_to_3d", "quality_tier": "draft"},
    )
    assert job.status_code == 202
    assert job.json()["team_id"] == body["team"]["id"]

    api.post("/v1/auth/logout")
    assert api.get("/v1/auth/me").status_code == 401

    api_key_jobs = api.get("/v1/jobs", headers={"Authorization": f"Bearer {raw_key}"})
    assert api_key_jobs.status_code == 200
    assert len(api_key_jobs.json()["jobs"]) == 1


def test_oauth_and_subscription_scaffolds(client):
    _server, api = client

    providers = api.get("/v1/auth/providers").json()["providers"]
    assert {provider["id"] for provider in providers} == {"github", "google"}
    assert all(provider["configured"] is False for provider in providers)
    assert api.get("/v1/auth/oauth/github/start").status_code == 501

    plans = api.get("/v1/billing/plans").json()["plans"]
    assert [plan["tier"] for plan in plans] == ["free", "creative", "studio", "enterprise"]

    api.post(
        "/v1/auth/signup",
        json={"email": "billing@example.com", "password": "correct horse", "name": "Billing"},
    )
    checkout = api.post("/v1/billing/checkout", json={"tier": "creative"})
    assert checkout.status_code == 200
    assert checkout.json()["status"] == "not_configured"


def test_upload_and_asset_download_are_team_scoped_and_root_safe(client, tmp_path: Path):
    server, api = client
    api.post(
        "/admin/credits/grant",
        headers={"Authorization": "Bearer admin_dev"},
        json={"team_id": "team_dev", "credits": 5, "reason": "test"},
    )

    upload = api.post(
        "/v1/uploads",
        headers=auth(),
        files={"file": ("../unsafe name.png", b"image-bytes", "image/png")},
    )
    assert upload.status_code == 201
    assert upload.json()["filename"] == "unsafe name.png"
    assert upload.json()["uri"].startswith("local://uploads/team_dev/")

    job = api.post(
        "/v1/jobs",
        headers=auth(),
        json={"input_uri": upload.json()["uri"], "quality_tier": "draft"},
    ).json()

    asset_path = Path(server.ARTIFACT_ROOT) / "projects" / "test" / "jobs" / job["id"] / "exports" / "mesh.obj"
    asset_path.parent.mkdir(parents=True, exist_ok=True)
    asset_path.write_text("o mesh\n", encoding="utf-8")
    asset_id = f"asset_{uuid4().hex}"
    server._store.add_asset(job["id"], AssetRecord(id=asset_id, job_id=job["id"], kind="export_mesh", uri=str(asset_path)))

    download = api.get(f"/v1/jobs/{job['id']}/assets/{asset_id}/download", headers=auth())
    assert download.status_code == 200
    assert download.text == "o mesh\n"

    forbidden = api.get(f"/v1/jobs/{job['id']}", headers=auth("key_other"))
    assert forbidden.status_code == 404

    escape_id = f"asset_{uuid4().hex}"
    outside = tmp_path / "outside.obj"
    outside.write_text("o outside\n", encoding="utf-8")
    server._store.add_asset(job["id"], AssetRecord(id=escape_id, job_id=job["id"], kind="export_mesh", uri=str(outside)))
    escaped = api.get(f"/v1/jobs/{job['id']}/assets/{escape_id}/download", headers=auth())
    assert escaped.status_code == 404


def test_job_status_exposes_preview_and_final_progress(client):
    server, api = client
    api.post(
        "/admin/credits/grant",
        headers={"Authorization": "Bearer admin_dev"},
        json={"team_id": "team_dev", "credits": 5, "reason": "test"},
    )
    job = api.post(
        "/v1/jobs",
        headers=auth(),
        json={"input_uri": "local://uploads/example.png", "quality_tier": "draft"},
    ).json()

    preview_path = Path(server.ARTIFACT_ROOT) / "projects" / "test" / "jobs" / job["id"] / "previews" / "preview.png"
    reference_path = Path(server.ARTIFACT_ROOT) / "projects" / "test" / "jobs" / job["id"] / "reference_refinement" / "reference.obj"
    plan_path = Path(server.ARTIFACT_ROOT) / "projects" / "test" / "jobs" / job["id"] / "reports" / "retopology_plan.json"
    chart_manifest_path = Path(server.ARTIFACT_ROOT) / "projects" / "test" / "jobs" / job["id"] / "reports" / "chart_remesh_manifest.json"
    chart_stitched_path = Path(server.ARTIFACT_ROOT) / "projects" / "test" / "jobs" / job["id"] / "chart_stitch" / "stitched.obj"
    quad_path = Path(server.ARTIFACT_ROOT) / "projects" / "test" / "jobs" / job["id"] / "quad_remesh" / "mesh.obj"
    projected_path = Path(server.ARTIFACT_ROOT) / "projects" / "test" / "jobs" / job["id"] / "feature_projection" / "mesh.obj"
    gate_path = Path(server.ARTIFACT_ROOT) / "projects" / "test" / "jobs" / job["id"] / "reports" / "production_gate.json"
    final_path = Path(server.ARTIFACT_ROOT) / "projects" / "test" / "jobs" / job["id"] / "exports" / "mesh.obj"
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    chart_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    chart_stitched_path.parent.mkdir(parents=True, exist_ok=True)
    quad_path.parent.mkdir(parents=True, exist_ok=True)
    projected_path.parent.mkdir(parents=True, exist_ok=True)
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path.write_bytes(b"png")
    reference_path.write_text("o reference\n", encoding="utf-8")
    plan_path.write_text('{"summary":{}}\n', encoding="utf-8")
    chart_manifest_path.write_text('{"entries":[]}\n', encoding="utf-8")
    chart_stitched_path.write_text("f 1 2 3 4\n", encoding="utf-8")
    quad_path.write_text("f 1 2 3 4\n", encoding="utf-8")
    projected_path.write_text("f 1 2 3 4\n", encoding="utf-8")
    gate_path.write_text('{"promotion":{}}\n', encoding="utf-8")
    final_path.write_text("o mesh\n", encoding="utf-8")
    server._store.add_asset(job["id"], AssetRecord(id="asset_preview", job_id=job["id"], kind="preview_image", uri=str(preview_path)))
    server._store.add_asset(job["id"], AssetRecord(id="asset_reference", job_id=job["id"], kind="reference_mesh", uri=str(reference_path)))
    server._store.add_asset(job["id"], AssetRecord(id="asset_plan", job_id=job["id"], kind="retopology_plan", uri=str(plan_path)))
    server._store.add_asset(job["id"], AssetRecord(id="asset_chart", job_id=job["id"], kind="chart_remesh_manifest", uri=str(chart_manifest_path)))
    server._store.add_asset(job["id"], AssetRecord(id="asset_stitched", job_id=job["id"], kind="chart_stitched_mesh", uri=str(chart_stitched_path)))
    server._store.add_asset(job["id"], AssetRecord(id="asset_quad", job_id=job["id"], kind="quad_mesh", uri=str(quad_path)))
    server._store.add_asset(job["id"], AssetRecord(id="asset_projected", job_id=job["id"], kind="projected_quad_mesh", uri=str(projected_path)))
    server._store.add_asset(job["id"], AssetRecord(id="asset_gate", job_id=job["id"], kind="production_gate_report", uri=str(gate_path)))

    status = api.get(f"/v1/jobs/{job['id']}/status", headers=auth()).json()
    assert status["progress"]["preview_ready"]
    assert not status["progress"]["final_ready"]
    assert status["asset_urls"]["preview_asset_url"].endswith("/assets/asset_preview/download")
    assert status["asset_urls"]["reference_mesh_asset_url"].endswith("/assets/asset_reference/download")
    assert status["asset_urls"]["retopology_plan_asset_url"].endswith("/assets/asset_plan/download")
    assert status["asset_urls"]["chart_remesh_manifest_asset_url"].endswith("/assets/asset_chart/download")
    assert status["asset_urls"]["chart_stitched_mesh_asset_url"].endswith("/assets/asset_stitched/download")
    assert status["asset_urls"]["quad_mesh_asset_url"].endswith("/assets/asset_quad/download")
    assert status["asset_urls"]["projected_quad_mesh_asset_url"].endswith("/assets/asset_projected/download")
    assert status["asset_urls"]["production_gate_asset_url"].endswith("/assets/asset_gate/download")

    server._store.add_asset(job["id"], AssetRecord(id="asset_final", job_id=job["id"], kind="export_mesh", uri=str(final_path)))
    server._store.set_status(job["id"], JobStatus.SUCCEEDED)
    status = api.get(f"/v1/jobs/{job['id']}/status", headers=auth()).json()
    assert status["progress"]["final_ready"]
    assert status["asset_urls"]["final_asset_url"].endswith("/assets/asset_final/download")
