# Production Infrastructure Target

The local product scaffold still uses JSON state and local artifacts for fast iteration. The production target is:

```text
FastAPI API
  -> Postgres job/credit ledger
  -> Redis/SQS/Temporal queue
  -> GPU worker pool
  -> S3/R2 artifact storage
  -> Stripe billing webhooks
```

## Prod-Lite Compose

A first deployable scaffold lives at:

```bash
deploy/docker-compose.prod-lite.yml
```

It starts:

```text
postgres: durable job, account, usage, API key state
redis: queue/worker coordination placeholder
minio: S3-compatible artifact store for local production testing
api: FastAPI service
worker: local scaffold worker
```

Run:

```bash
docker compose -f deploy/docker-compose.prod-lite.yml up --build
```

This is not the final GPU deployment. Heavy TRELLIS.2/MeshRipple/Easy3E workers should run in GPU images or Thunder/RunPod-style hosts and consume the same job/queue/artifact contracts.

## Migration Order

1. Add Postgres-backed `JobStore` and `CreditLedger` implementations behind the current interfaces.
2. Add S3/R2-backed `ArtifactStore` while preserving local filesystem mode for development.
3. Replace polling `claim_next_queued()` with Redis/SQS/Temporal queue claims.
4. Add Stripe checkout and webhook tables/events.
5. Add API-key rotation, team membership, and admin console.

## Safety Rules

- Public API must never accept executable command metadata.
- GPU command metadata is internal-only and generated server-side.
- Artifacts should be addressed by stable object IDs, not arbitrary paths.
- Failed infrastructure jobs release credits; successful jobs consume reserved credits.

## Current Backend Switches

The API and worker now keep local JSON/files as the default, but can switch to production-style services with env vars:

```bash
CLEARMESH_POSTGRES_DSN=postgresql://user:pass@host:5432/clearmesh
CLEARMESH_S3_BUCKET=clearmesh-artifacts
CLEARMESH_S3_PREFIX=prod
AWS_ENDPOINT_URL=http://minio:9000  # optional for MinIO/R2-compatible testing
```

`PostgresJobStore` stores full job payloads in JSONB while preserving the existing job-store interface. `S3ArtifactStore` keeps local execution paths for GPU commands, then uploads exported assets and returns signed download URLs.

## Runtime Quote UX

`POST /v1/billing/quote` now returns both credit and runtime estimates. The quote intentionally marks `standard` and `high` as hidden candidates until 2k/5k and part-aware quality are validated.
