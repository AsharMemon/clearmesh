#!/usr/bin/env bash
set -euo pipefail

INSTANCE_ID="${1:-${THUNDER_INSTANCE_ID:-0}}"
REMOTE_DIR="${REMOTE_DIR:-/home/ubuntu/clearmesh}"
TNR_BIN="${TNR_BIN:-/Users/Ashar/.tnr/bin/tnr}"

cat <<EOF | "$TNR_BIN" connect "$INSTANCE_ID"
set -euo pipefail
cd "$REMOTE_DIR"
MESHRIPPLE_DIR=/home/ubuntu/mesh-heads/MeshRipple \
MESHRIPPLE_VENV=/home/ubuntu/meshripple-venv \
bash scripts/setup/install_meshripple_env.sh
exit
EOF
