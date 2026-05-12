#!/usr/bin/env bash
set -euo pipefail

INSTANCE_ID="${1:-${THUNDER_INSTANCE_ID:-0}}"
REMOTE_DIR="${REMOTE_DIR:-/home/ubuntu/clearmesh}"
TNR_BIN="${TNR_BIN:-/Users/Ashar/.tnr/bin/tnr}"

cat <<EOF | "$TNR_BIN" connect "$INSTANCE_ID"
set -euo pipefail
cd "$REMOTE_DIR"
. /home/ubuntu/meshripple-venv/bin/activate
python scripts/product/preflight_meshripple.py \
  --repo-dir /home/ubuntu/mesh-heads/MeshRipple \
  --python /home/ubuntu/meshripple-venv/bin/python \
  --checkpoint-dir /home/ubuntu/mesh-heads/MeshRipple/ckpt
exit
EOF
