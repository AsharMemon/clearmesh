#!/usr/bin/env bash
set -euo pipefail

INSTANCE_ID="${1:-${THUNDER_INSTANCE_ID:-0}}"
REMOTE_DIR="${REMOTE_DIR:-/home/ubuntu/clearmesh}"
STATE_ROOT="${STATE_ROOT:-/tmp/clearmesh_pipeline_state}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/tmp/clearmesh_pipeline_artifacts}"
PROXY_MESH="${PROXY_MESH:-/home/ubuntu/clearmesh/experiments/ultrashape/inputs/coarse_meshes/test_mug.glb}"
CONFIG_JSON="${CONFIG_JSON:-/home/ubuntu/clearmesh/configs/meshripple.thunder.smoke.json}"

"$(dirname "$0")/run_remote.sh" "$INSTANCE_ID" <<EOF
set -euo pipefail
cd "$REMOTE_DIR"
rm -rf "$STATE_ROOT" "$ARTIFACT_ROOT"

JOB_ID=\$(/home/ubuntu/clearmesh-venv/bin/python scripts/product/create_local_job.py \\
  --state-root "$STATE_ROOT" \\
  --team-id team_dev \\
  --user-id user_dev \\
  --input-uri local://smoke/mug.png \\
  --grant-credits 20 \\
  --project-id thunder_smoke \\
  --proxy-mesh-path "$PROXY_MESH" \\
  --disable-parts \\
  --quality-tier draft)

/home/ubuntu/clearmesh-venv/bin/python scripts/product/run_pipeline_worker.py \\
  --state-root "$STATE_ROOT" \\
  --artifact-root "$ARTIFACT_ROOT" \\
  --once \\
  --execute-heavy \\
  --mesh-head meshripple \\
  --mesh-head-config-json "$CONFIG_JSON" \\
  --preferred-point-budget 40960

/home/ubuntu/clearmesh-venv/bin/python - <<PY
import json
from pathlib import Path

job_id = "\$JOB_ID"
state_root = Path("$STATE_ROOT")
job = json.loads((state_root / "jobs" / f"{job_id}.json").read_text())
assets = job["assets"]
summary = {
    "job_id": job_id,
    "status": job["status"],
    "steps": {step["name"]: step["status"] for step in job["steps"]},
    "asset_kinds": [asset["kind"] for asset in assets],
    "export_manifest": next(
        (step["artifacts"].get("manifest") for step in job["steps"] if step["name"] == "export_package"),
        None,
    ),
    "mesh_eval_report": next((asset["uri"] for asset in assets if asset["kind"] == "mesh_eval_report"), None),
    "export_mesh": next((asset["uri"] for asset in assets if asset["kind"] == "export_mesh"), None),
}
print(json.dumps(summary, indent=2, sort_keys=True))
PY
EOF
