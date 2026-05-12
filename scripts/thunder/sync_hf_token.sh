#!/usr/bin/env bash
set -euo pipefail

INSTANCE_ID="${1:-${THUNDER_INSTANCE_ID:-0}}"
TNR_BIN="${TNR_BIN:-/Users/Ashar/.tnr/bin/tnr}"
REMOTE_ENV_PATH="${REMOTE_ENV_PATH:-/home/ubuntu/.clearmesh_hf.env}"

if [ -z "${THUNDER_TOKEN:-}" ]; then
  echo "THUNDER_TOKEN is not set. Export it before syncing secrets." >&2
  exit 1
fi
if [ -z "${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}" ]; then
  echo "HF_TOKEN or HUGGINGFACE_HUB_TOKEN is not set locally." >&2
  exit 1
fi

token="${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}"
tmp_env="$(mktemp -t clearmesh-hf.XXXXXX.env)"
trap 'rm -f "$tmp_env"' EXIT
umask 077
printf 'export HF_TOKEN=%q\nexport HUGGINGFACE_HUB_TOKEN=%q\n' "$token" "$token" > "$tmp_env"

"$TNR_BIN" scp "$tmp_env" "$INSTANCE_ID:$REMOTE_ENV_PATH"
printf 'chmod 600 %q && test -s %q && echo hf-token-file-ready\nexit\n' "$REMOTE_ENV_PATH" "$REMOTE_ENV_PATH" | "$TNR_BIN" connect "$INSTANCE_ID"
