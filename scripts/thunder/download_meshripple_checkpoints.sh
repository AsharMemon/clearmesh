#!/usr/bin/env bash
set -euo pipefail

INSTANCE_ID="${1:-${THUNDER_INSTANCE_ID:-0}}"
REMOTE_DIR="${REMOTE_DIR:-/home/ubuntu/clearmesh}"
TNR_BIN="${TNR_BIN:-/Users/Ashar/.tnr/bin/tnr}"

cat <<EOF | "$TNR_BIN" connect "$INSTANCE_ID"
set -euo pipefail
cd "$REMOTE_DIR"
. /home/ubuntu/meshripple-venv/bin/activate
PYTHON_BIN=/home/ubuntu/meshripple-venv/bin/python \
MESHRIPPLE_DIR=/home/ubuntu/mesh-heads/MeshRipple \
CHECKPOINT_DIR=/home/ubuntu/mesh-heads/MeshRipple/ckpt \
scripts/setup/download_meshripple_checkpoints.sh
exit
EOF
