#!/usr/bin/env bash
set -euo pipefail
INSTANCE_ID="${1:-${THUNDER_INSTANCE_ID:-0}}"
CONFIG_JSON="${CONFIG_JSON:-/home/ubuntu/clearmesh/configs/omnipart.thunder.example.json}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$SCRIPT_DIR/run_remote.sh" "$INSTANCE_ID" /home/ubuntu/clearmesh-venv/bin/python /home/ubuntu/clearmesh/scripts/product/preflight_omnipart.py --config-json "$CONFIG_JSON"
