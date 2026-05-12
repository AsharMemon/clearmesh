#!/usr/bin/env bash
set -euo pipefail

INSTANCE_ID="${1:-${THUNDER_INSTANCE_ID:-0}}"
REMOTE_DIR="${REMOTE_DIR:-/home/ubuntu/clearmesh}"
STATE_ROOT="${STATE_ROOT:-/tmp/clearmesh_trellis2_pipeline_state}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/tmp/clearmesh_trellis2_pipeline_artifacts}"
INPUT_IMAGE="${INPUT_IMAGE:-/home/ubuntu/TRELLIS.2/assets/example_image/0e4984a9b3765ce80e9853443f9319ecedf90885c74b56cccfebc09402740f8a.webp}"
CONFIG_JSON="${CONFIG_JSON:-/home/ubuntu/clearmesh/configs/meshripple.thunder.smoke.json}"
REMOTE_ENV_PATH="${REMOTE_ENV_PATH:-/home/ubuntu/.clearmesh_hf.env}"

"$(dirname "$0")/run_remote.sh" "$INSTANCE_ID" <<EOF
set -euo pipefail
if [ -f "$REMOTE_ENV_PATH" ]; then
  source "$REMOTE_ENV_PATH"
fi
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH=/home/ubuntu/TRELLIS.2:${PYTHONPATH:-}
export ATTN_BACKEND=flash_attn

cd "$REMOTE_DIR"
rm -rf "$STATE_ROOT" "$ARTIFACT_ROOT"
mkdir -p "$ARTIFACT_ROOT/uploads/smoke"
cp "$INPUT_IMAGE" "$ARTIFACT_ROOT/uploads/smoke/input.webp"

METADATA_JSON=\$(mktemp /tmp/clearmesh_trellis2_metadata.XXXXXX.json)
cat > "\$METADATA_JSON" <<'JSON'
{
  "trellis_command": "/home/ubuntu/trellis2-venv/bin/python -u /home/ubuntu/clearmesh/scripts/product/run_trellis2_proxy.py --input {input_path} --output-dir {output_dir} --output-name trellis_proxy.glb --decimation-target 250000 --texture-size 1024 --seed 0",
  "trellis_cwd": "/home/ubuntu/TRELLIS.2",
  "trellis_timeout_seconds": 7200
}
JSON

JOB_ID=\$(/home/ubuntu/clearmesh-venv/bin/python scripts/product/create_local_job.py \\
  --state-root "$STATE_ROOT" \\
  --team-id team_dev \\
  --user-id user_dev \\
  --input-uri local://uploads/smoke/input.webp \\
  --grant-credits 30 \\
  --project-id thunder_trellis2_smoke \\
  --metadata-json "\$METADATA_JSON" \\
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
    "trellis_proxy": next((asset["uri"] for asset in assets if asset["kind"] == "trellis_proxy_mesh"), None),
    "mesh_eval_report": next((asset["uri"] for asset in assets if asset["kind"] == "mesh_eval_report"), None),
    "export_mesh": next((asset["uri"] for asset in assets if asset["kind"] == "export_mesh"), None),
}
print(json.dumps(summary, indent=2, sort_keys=True))
PY
EOF
