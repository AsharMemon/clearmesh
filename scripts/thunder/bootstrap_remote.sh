#!/usr/bin/env bash
set -euo pipefail

INSTANCE_ID="${1:-${THUNDER_INSTANCE_ID:-0}}"
REMOTE_DIR="${REMOTE_DIR:-/home/ubuntu/clearmesh}"
REMOTE_VENV="${REMOTE_VENV:-/home/ubuntu/clearmesh-venv}"
TNR_BIN="${TNR_BIN:-/Users/Ashar/.tnr/bin/tnr}"
INSTALL_MESH_HEAD_REPOS="${INSTALL_MESH_HEAD_REPOS:-1}"

if [ -z "${THUNDER_TOKEN:-}" ]; then
  echo "THUNDER_TOKEN is not set. Export it before bootstrapping." >&2
  exit 1
fi

cat <<EOF | "$TNR_BIN" connect "$INSTANCE_ID"
set -euo pipefail
cd "$REMOTE_DIR"
python3 -m venv "$REMOTE_VENV"
. "$REMOTE_VENV/bin/activate"
python -m pip install --upgrade pip wheel setuptools
python -m pip install -r requirements-product.txt
if [ "$INSTALL_MESH_HEAD_REPOS" = "1" ]; then
  MESH_HEAD_ROOT=/home/ubuntu/mesh-heads INSTALL_ENV="${INSTALL_MESH_HEAD_ENVS:-0}" bash scripts/setup/install_mesh_heads.sh
fi
python -m py_compile clearmesh/product/pipeline_worker.py clearmesh/mesh_heads/*.py scripts/product/*.py
exit
EOF
