#!/usr/bin/env bash
set -euo pipefail

STATE_ROOT="${STATE_ROOT:-$(mktemp -d /tmp/clearmesh_state.XXXXXX)}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$(mktemp -d /tmp/clearmesh_artifacts.XXXXXX)}"
PROXY_MESH="${PROXY_MESH:-$(pwd)/experiments/ultrashape/inputs/coarse_meshes/test_mug.glb}"

JOB_ID=$(python scripts/product/create_local_job.py \
  --state-root "$STATE_ROOT" \
  --input-uri file://local-smoke/input.png \
  --team-id team_dev \
  --user-id user_dev \
  --grant-credits 20 \
  --project-id local_smoke \
  --proxy-mesh-path "$PROXY_MESH" \
  --artist-mesh-path "$PROXY_MESH" \
  --disable-parts \
  --quality-tier draft)

python scripts/product/run_pipeline_worker.py \
  --state-root "$STATE_ROOT" \
  --artifact-root "$ARTIFACT_ROOT" \
  --once

python - <<PY
import json
from pathlib import Path

job = json.loads((Path("$STATE_ROOT") / "jobs" / "$JOB_ID.json").read_text())
print(json.dumps({
    "job_id": "$JOB_ID",
    "status": job["status"],
    "steps": {step["name"]: step["status"] for step in job["steps"]},
    "asset_kinds": [asset["kind"] for asset in job["assets"]],
    "state_root": "$STATE_ROOT",
    "artifact_root": "$ARTIFACT_ROOT",
}, indent=2, sort_keys=True))
PY
