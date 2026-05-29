# API Scaffold

The current API is intentionally small and async-job oriented.

## Endpoints

```text
GET  /
GET  /app
GET  /pricing
GET  /settings
GET  /healthz
GET  /v1/auth/providers
GET  /v1/auth/me
POST /v1/auth/signup
POST /v1/auth/login
POST /v1/auth/logout
GET  /v1/auth/oauth/{provider}/start
GET  /v1/auth/oauth/{provider}/callback
POST /v1/uploads
POST /v1/jobs
GET  /v1/jobs
GET  /v1/jobs/{job_id}
GET  /v1/jobs/{job_id}/status
POST /v1/jobs/{job_id}/cancel
GET  /v1/jobs/{job_id}/assets
GET  /v1/jobs/{job_id}/assets/{asset_id}/download
PATCH /v1/settings
GET  /v1/settings/api-keys
POST /v1/settings/api-keys
DELETE /v1/settings/api-keys/{key_id}
GET  /v1/billing/plans
POST /v1/billing/quote
GET  /v1/billing/credits
GET  /v1/billing/events
GET  /v1/billing/subscription
POST /v1/billing/checkout
POST /v1/billing/portal
POST /v1/billing/webhook
POST /admin/credits/grant
```

## Auth

Client endpoints accept either browser sessions from `/v1/auth/login` /
`/v1/auth/signup`, user-issued API keys from `/v1/settings/api-keys`, or bearer
API keys from `CLEAR_MESH_API_KEYS`:

```bash
export CLEAR_MESH_API_KEYS='dev:key_user_dev:user_dev:team_dev'
```

Admin credit grants use `CLEARMESH_ADMIN_KEY` and are disabled when that env var is absent.

OAuth buttons are enabled when these variables are present:

```bash
export CLEARMESH_PUBLIC_URL='https://app.example.com'
export CLEARMESH_GITHUB_CLIENT_ID='...'
export CLEARMESH_GITHUB_CLIENT_SECRET='...'
export CLEARMESH_GOOGLE_CLIENT_ID='...'
export CLEARMESH_GOOGLE_CLIENT_SECRET='...'
```

Provider callback URLs:

```text
https://app.example.com/v1/auth/oauth/github/callback
https://app.example.com/v1/auth/oauth/google/callback
```

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

Open the web app at `http://localhost:8000/`. The default route is the
chat-and-mesh generation surface. `/pricing` and `/settings` are served by the
same app shell.

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

## Subscription Setup

The pricing page reads `/v1/billing/plans`. Stripe checkout is active only when
the relevant price IDs are configured:

```bash
export STRIPE_SECRET_KEY='sk_live_...'
export STRIPE_WEBHOOK_SECRET='whsec_...'
export CLEARMESH_STRIPE_PRICE_CREATIVE='price_...'
export CLEARMESH_STRIPE_PRICE_STUDIO='price_...'
```

Checkout sessions include `team_id`, `user_id`, and `tier` metadata. The webhook
updates the team subscription and grants plan credits idempotently for completed
checkout events.

## Production Migration

- Replace JSON state with Postgres.
- Replace local artifacts with S3/R2 signed URLs.
- Move user/team/session/auth tables from JSON into Postgres.
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
- browser signup/login/session auth
- settings and user-issued API keys
- OAuth provider and Stripe checkout scaffolds

Run with:

```bash
python -m pytest tests/test_api_server.py -q
```
