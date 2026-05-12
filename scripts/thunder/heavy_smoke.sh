#!/usr/bin/env bash
set -euo pipefail

INSTANCE_ID="${1:-${THUNDER_INSTANCE_ID:-0}}"
REMOTE_DIR="${REMOTE_DIR:-/home/ubuntu/clearmesh}"
REMOTE_VENV="${REMOTE_VENV:-/home/ubuntu/clearmesh-venv}"
TNR_BIN="${TNR_BIN:-/Users/Ashar/.tnr/bin/tnr}"
PROXY_MESH="${PROXY_MESH:-$REMOTE_DIR/experiments/ultrashape/inputs/coarse_meshes/test_mug.glb}"
INPUT_IMAGE="${INPUT_IMAGE:-$REMOTE_DIR/experiments/ultrashape/inputs/images/test_mug.png}"
SMOKE_ROOT="${SMOKE_ROOT:-/tmp/clearmesh_heavy_smoke}"

cat <<EOF | "$TNR_BIN" connect "$INSTANCE_ID"
set -euo pipefail
cd "$REMOTE_DIR"
. "$REMOTE_VENV/bin/activate"
rm -rf "$SMOKE_ROOT"
mkdir -p "$SMOKE_ROOT"
job=\$(python scripts/product/create_local_job.py \
  --state-root "$SMOKE_ROOT/state" \
  --input-uri "local://$INPUT_IMAGE" \
  --team-id team_dev \
  --user-id user_dev \
  --grant-credits 20 \
  --proxy-mesh-path "$PROXY_MESH" \
  --artist-mesh-path "$PROXY_MESH")
python scripts/product/run_pipeline_worker.py \
  --state-root "$SMOKE_ROOT/state" \
  --artifact-root "$SMOKE_ROOT/artifacts" \
  --execute-heavy \
  --once
python - <<PY "\$job" "$SMOKE_ROOT"
import json, pathlib, sys
job=sys.argv[1]
root=pathlib.Path(sys.argv[2])
data=json.loads((root/'state/jobs'/f'{job}.json').read_text())
print('job_id', job)
print('status', data['status'])
print('steps', [(s['name'], s['status']) for s in data['steps']])
print('assets', [(a['kind'], a['uri']) for a in data['assets']])
PY
exit
EOF
