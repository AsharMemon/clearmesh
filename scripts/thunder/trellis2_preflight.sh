#!/usr/bin/env bash
set -euo pipefail

INSTANCE_ID="${1:-${THUNDER_INSTANCE_ID:-0}}"
REMOTE_DIR="${REMOTE_DIR:-/home/ubuntu/clearmesh}"

"$(dirname "$0")/run_remote.sh" "$INSTANCE_ID" <<EOF
set -euo pipefail
cd "$REMOTE_DIR"
/home/ubuntu/clearmesh-venv/bin/python scripts/product/preflight_trellis2.py \\
  --repo-dir /home/ubuntu/TRELLIS.2 \\
  --python /home/ubuntu/trellis2-venv/bin/python
EOF
