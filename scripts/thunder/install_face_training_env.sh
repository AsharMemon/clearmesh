#!/usr/bin/env bash
set -euo pipefail

INSTANCE_ID="${1:-${THUNDER_INSTANCE_ID:-0}}"
REMOTE_DIR="${REMOTE_DIR:-/home/ubuntu/clearmesh}"
TNR_BIN="${TNR_BIN:-/Users/Ashar/.tnr/bin/tnr}"

if [ -z "${THUNDER_TOKEN:-}" ]; then
  echo "THUNDER_TOKEN is not set. Export it before installing FACE training deps." >&2
  exit 1
fi

cat <<EOF | "$TNR_BIN" connect "$INSTANCE_ID"
set -euo pipefail
cd "$REMOTE_DIR"
PIP_NO_CACHE_DIR=1 python -m pip install -r requirements-data.txt
python - <<'PY'
import importlib.util

required = ["datasets", "huggingface_hub", "objaverse", "fast_simplification"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"missing FACE data dependencies: {missing}")
print("FACE training/data dependencies ready")
PY
exit
EOF
