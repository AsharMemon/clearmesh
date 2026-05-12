#!/usr/bin/env bash
set -euo pipefail

INSTANCE_ID="${1:-${THUNDER_INSTANCE_ID:-0}}"
REMOTE_DIR="${REMOTE_DIR:-/home/ubuntu/clearmesh}"

"$(dirname "$0")/run_remote.sh" "$INSTANCE_ID" <<EOF
set -euo pipefail
cd "$REMOTE_DIR"
TRELLIS2_DIR=/home/ubuntu/TRELLIS.2 \\
TRELLIS2_VENV=/home/ubuntu/trellis2-venv \\
CUDA_HOME=/usr/local/cuda \\
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-1}" \\
INSTALL_EXTENSIONS="${INSTALL_EXTENSIONS:-1}" \\
scripts/setup/install_trellis2_env.sh
EOF
