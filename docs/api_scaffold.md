# API Scaffold

The current API is intentionally small and async-job oriented.

## Endpoints

```text
GET  /healthz
POST /v1/uploads
POST /v1/jobs
GET  /v1/jobs
GET  /v1/jobs/{job_id}
GET  /v1/jobs/{job_id}/status
POST /v1/jobs/{job_id}/cancel
GET  /v1/jobs/{job_id}/assets
GET  /v1/jobs/{job_id}/assets/{asset_id}/download
POST /v1/billing/quote
GET  /v1/billing/credits
GET  /v1/billing/events
POST /admin/credits/grant
```

## Auth

Client endpoints use bearer API keys from `CLEAR_MESH_API_KEYS`:

```bash
export CLEAR_MESH_API_KEYS='dev:key_user_dev:user_dev:team_dev'
```

Admin credit grants use `CLEARMESH_ADMIN_KEY` and are disabled when that env var is absent.

## Safety Defaults

Public job creation strips executable worker metadata by default, including TRELLIS, Easy3E, OmniPart, autorigging commands, and direct mesh-path overrides. Trusted local scripts can still create GPU jobs with command metadata. If you intentionally need command metadata through the API in a private deployment, set:

```bash
export CLEARMESH_ALLOW_CLIENT_COMMAND_METADATA=1
```

Do not enable that flag for public traffic.

## Local Run

```bash
pip install -r requirements-product.txt
CLEAR_MESH_API_KEYS='dev:key_user_dev:user_dev:team_dev' \
uvicorn clearmesh.api.server:app --reload --port 8000
```

Grant local credits:

```bash
curl -X POST http://localhost:8000/admin/credits/grant \
  -H 'Authorization: Bearer admin_dev' \
  -H 'Content-Type: application/json' \
  -d '{"team_id":"team_dev","credits":100,"reason":"dev"}'
```

Create a job:

```bash
curl -X POST http://localhost:8000/v1/jobs \
  -H 'Authorization: Bearer key_user_dev' \
  -H 'Content-Type: application/json' \
  -d '{"input_uri":"local://uploads/example.png","quality_tier":"draft"}'
```

Poll user-facing status:

```bash
curl http://localhost:8000/v1/jobs/job_xxx/status \
  -H 'Authorization: Bearer key_user_dev'
```

The status response is intentionally UI-friendly:

```json
{
  "job_id": "job_xxx",
  "status": "running",
  "progress": {
    "phase": "preview_ready",
    "percent": 55,
    "message": "Preview mesh is ready; high-resolution finishing is continuing.",
    "preview_ready": true,
    "final_ready": false,
    "preview_asset_id": "asset_preview",
    "preview_mesh_asset_id": "asset_control",
    "reference_mesh_asset_id": "asset_reference",
    "retopology_plan_asset_id": "asset_retopology_plan",
    "chart_remesh_manifest_asset_id": "asset_chart_manifest",
    "quad_mesh_asset_id": "asset_quad",
    "projected_quad_mesh_asset_id": "asset_projected_quad",
    "quality_report_asset_id": "asset_quality",
    "production_gate_asset_id": "asset_gate",
    "final_asset_id": null
  },
  "asset_urls": {
    "preview_asset_url": "/v1/jobs/job_xxx/assets/asset_preview/download",
    "preview_mesh_asset_url": "/v1/jobs/job_xxx/assets/asset_control/download",
    "reference_mesh_asset_url": "/v1/jobs/job_xxx/assets/asset_reference/download",
    "retopology_plan_asset_url": "/v1/jobs/job_xxx/assets/asset_retopology_plan/download",
    "chart_remesh_manifest_asset_url": "/v1/jobs/job_xxx/assets/asset_chart_manifest/download",
    "quad_mesh_asset_url": "/v1/jobs/job_xxx/assets/asset_quad/download",
    "projected_quad_mesh_asset_url": "/v1/jobs/job_xxx/assets/asset_projected_quad/download",
    "quality_report_asset_url": "/v1/jobs/job_xxx/assets/asset_quality/download",
    "production_gate_asset_url": "/v1/jobs/job_xxx/assets/asset_gate/download"
  }
}
```

The frontend should poll this endpoint for the Claude-like “latest status
message” and use `preview_ready` / `final_ready` instead of parsing internal
worker step names.

## Production Migration

- Replace JSON state with Postgres.
- Replace local artifacts with S3/R2 signed URLs.
- Add Stripe checkout/subscription webhooks around the credit ledger.
- Put workers behind a queue such as SQS, Redis, or Temporal.
- Add org/user management and API-key rotation.

## Test Coverage

Endpoint tests live in `tests/test_api_server.py` and cover:

- bearer auth required for client endpoints
- admin credit grants
- billing quote and event ledger
- job creation and cancellation
- preview/final progress status
- optional quad sidecar status URLs
- optional retopology plan status URLs
- stripping executable worker metadata from public job requests
- upload filename sanitization
- team-scoped job access
- artifact-root-safe downloads

Run with:

```bash
python -m pytest tests/test_api_server.py -q
```
