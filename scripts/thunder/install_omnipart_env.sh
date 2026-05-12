#!/usr/bin/env bash
set -euo pipefail
INSTANCE_ID="${1:-${THUNDER_INSTANCE_ID:-0}}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$SCRIPT_DIR/run_remote.sh" "$INSTANCE_ID" bash /home/ubuntu/clearmesh/scripts/setup/install_omnipart_env.sh
