#!/usr/bin/env bash
# Watch a Lambda FACE-Q gate, harvest lightweight outputs, then terminate.
set -euo pipefail

RUN_INFO="${1:?usage: watch_faceq_gate_and_terminate.sh RUN_INFO_JSON [OUT_DIR]}"
OUT_DIR="${2:-$(dirname "$RUN_INFO")}"
LAMBDA_ENV_FILE="${LAMBDA_ENV_FILE:-.codex_secrets/lambda.env}"
B2_ENV_FILE="${B2_ENV_FILE:-.codex_secrets/b2.env}"
SSH_KEY_FILE="${SSH_KEY_FILE:-$HOME/.ssh/id_ed25519}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MAX_SECONDS="${MAX_SECONDS:-21600}"
API_BASE="${LAMBDA_API_BASE:-https://cloud.lambdalabs.com/api/v1}"

mkdir -p "$OUT_DIR/fetched"

json_get() {
  python3 - "$RUN_INFO" "$1" <<'PY'
import json, sys
data=json.load(open(sys.argv[1], encoding="utf-8"))
print(data.get(sys.argv[2], ""))
PY
}

INSTANCE_ID="$(json_get instance_id)"
HOST="$(json_get host)"
REMOTE_USER="$(json_get remote_user)"
REMOTE_ROOT="$(json_get remote_lab_root)"
REMOTE_PID="$(json_get remote_pid)"
B2_RUN_PREFIX="$(json_get b2_run_prefix)"
B2_BUCKET="${B2_BUCKET:-clearmesh-pairs}"

if [[ -z "$INSTANCE_ID" || -z "$HOST" || -z "$REMOTE_ROOT" ]]; then
  echo "run_info is missing instance_id, host, or remote_lab_root" >&2
  exit 2
fi
if [[ ! -f "$LAMBDA_ENV_FILE" ]]; then
  echo "missing $LAMBDA_ENV_FILE" >&2
  exit 2
fi
source "$LAMBDA_ENV_FILE"
: "${LAMBDA_KEY:?missing LAMBDA_KEY}"

SSH_OPTS=(
  -i "$SSH_KEY_FILE"
  -o BatchMode=yes
  -o ConnectTimeout=20
  -o StrictHostKeyChecking=accept-new
  -o UserKnownHostsFile="$HOME/.ssh/known_hosts"
)
SSH_TARGET="${REMOTE_USER:-ubuntu}@$HOST"

terminate_instance() {
  local raw="$OUT_DIR/terminate_response.raw.json"
  local redacted="$OUT_DIR/terminate_response.json"
  curl -fsS -L --max-time 30 \
    -X POST \
    -u "$LAMBDA_KEY:" \
    -H 'Accept: application/json' \
    -H 'Content-Type: application/json' \
    --data "{\"instance_ids\":[\"$INSTANCE_ID\"]}" \
    "$API_BASE/instance-operations/terminate" > "$raw" || true
  python3 - "$raw" "$redacted" <<'PY'
import json, sys
src, dst = sys.argv[1:3]
try:
    data = json.load(open(src, encoding="utf-8"))
except Exception as exc:
    data = {"error": str(exc), "redacted": True}
for item in data.get("data", {}).get("terminated_instances", []):
    if "jupyter_token" in item:
        item["jupyter_token"] = "REDACTED"
    if "jupyter_url" in item:
        item["jupyter_url"] = "REDACTED"
open(dst, "w", encoding="utf-8").write(json.dumps(data, indent=2, sort_keys=True) + "\n")
PY
  rm -f "$raw"
}

fetch_lightweight() {
  ssh "${SSH_OPTS[@]}" "$SSH_TARGET" \
    "test -f '$REMOTE_ROOT/status.jsonl' && cat '$REMOTE_ROOT/status.jsonl' || true" \
    > "$OUT_DIR/fetched/status.jsonl" 2>/dev/null || true
  ssh "${SSH_OPTS[@]}" "$SSH_TARGET" \
    "test -f '$REMOTE_ROOT/ablation_summary.json' && cat '$REMOTE_ROOT/ablation_summary.json' || true" \
    > "$OUT_DIR/fetched/ablation_summary.json" 2>/dev/null || true
  if [[ -f "$B2_ENV_FILE" && -n "$B2_RUN_PREFIX" ]]; then
    (
      source "$B2_ENV_FILE"
      export RCLONE_CONFIG_B2ENV_TYPE=b2
      export RCLONE_CONFIG_B2ENV_ACCOUNT="${B2_KEY_ID:-${B2_KEYID:-${B2_APPLICATION_KEY_ID:-}}}"
      export RCLONE_CONFIG_B2ENV_KEY="${B2_APP_KEY:-${B2_APPKEY:-${B2_APPLICATION_KEY:-}}}"
      rclone copy "b2env:$B2_BUCKET/$B2_RUN_PREFIX/ablation_summary.json" "$OUT_DIR/fetched/" --stats 0 || true
      rclone copy "b2env:$B2_BUCKET/$B2_RUN_PREFIX/status.jsonl" "$OUT_DIR/fetched/" --stats 0 || true
    )
  fi
}

deadline=$(( $(date +%s) + MAX_SECONDS ))
while [[ $(date +%s) -lt "$deadline" ]]; do
  status_text="$(
    ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "set -e
      pid=\$(cat '$REMOTE_PID' 2>/dev/null || true)
      alive=0
      if [[ -n \"\$pid\" ]] && kill -0 \"\$pid\" 2>/dev/null; then alive=1; fi
      completed=0
      if grep -q '\"event\": \"b2_upload_completed\"' '$REMOTE_ROOT/status.jsonl' 2>/dev/null; then completed=1; fi
      failed=0
      if [[ \"\$alive\" = 0 && ! -f '$REMOTE_ROOT/ablation_summary.json' ]]; then failed=1; fi
      echo alive=\$alive completed=\$completed failed=\$failed
    " 2>/dev/null || echo "ssh_failed=1"
  )"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $status_text"
  if [[ "$status_text" == *"completed=1"* || "$status_text" == *"failed=1"* ]]; then
    fetch_lightweight
    terminate_instance
    exit 0
  fi
  sleep "$POLL_SECONDS"
done

echo "watch timed out; fetching lightweight artifacts and terminating to avoid idle spend" >&2
fetch_lightweight
terminate_instance
