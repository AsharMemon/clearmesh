#!/usr/bin/env bash
# Poll a Vast Localizable FACE run and destroy the instance after completion/failure.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

RUN_INFO="${RUN_INFO:-${1:-}}"
VAST_BIN="${VAST_BIN:-/Users/Ashar/Library/Python/3.14/bin/vastai}"
SSH_KEY_FILE="${SSH_KEY_FILE:-$HOME/.ssh/id_ed25519}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-120}"
MAX_SECONDS="${MAX_SECONDS:-21600}"
REQUIRE_EVAL_REPORT="${REQUIRE_EVAL_REPORT:-0}"

if [[ -z "$RUN_INFO" || ! -f "$RUN_INFO" ]]; then
  echo "Usage: RUN_INFO=/path/to/run_info.json $0" >&2
  exit 2
fi
if [[ -z "${VAST_API:-}" ]]; then
  echo "VAST_API is not set." >&2
  exit 2
fi

instance_id="$(python3 - "$RUN_INFO" <<'PY'
import json, sys
data=json.load(open(sys.argv[1]))
print(data.get("instance_id") or data.get("vast_instance_id") or "")
PY
)"
ssh_probe="$(python3 - "$RUN_INFO" <<'PY'
import json, sys
data=json.load(open(sys.argv[1]))
print(data.get("ssh_probe_file") or "")
PY
)"
remote_root="$(python3 - "$RUN_INFO" <<'PY'
import json, sys
data=json.load(open(sys.argv[1]))
print(data.get("remote_lab_root") or "")
PY
)"
if [[ -z "$instance_id" || -z "$ssh_probe" || -z "$remote_root" ]]; then
  echo "run_info is missing instance_id, ssh_probe_file, or remote_lab_root" >&2
  exit 2
fi
if [[ ! -f "$ssh_probe" ]]; then
  echo "Missing SSH probe: $ssh_probe" >&2
  exit 2
fi

source "$ssh_probe"
REMOTE_USER="${REMOTE_USER:-root}"
SSH_PORT="${SSH_PORT:-22}"
watch_dir="$(dirname "$RUN_INFO")/watch"
mkdir -p "$watch_dir"
log_file="$watch_dir/watch.log"

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$log_file"
}

ssh_probe_run() {
  ssh -i "$SSH_KEY_FILE" -p "$SSH_PORT" \
    -o StrictHostKeyChecking=accept-new \
    -o UserKnownHostsFile="$HOME/.ssh/known_hosts" \
    -o ConnectTimeout=20 \
    "$REMOTE_USER@$HOST" "$@"
}

fetch_lightweight_logs() {
  mkdir -p "$watch_dir/harvest"
  scp -i "$SSH_KEY_FILE" -P "$SSH_PORT" \
    -o StrictHostKeyChecking=accept-new \
    -o UserKnownHostsFile="$HOME/.ssh/known_hosts" \
    "$REMOTE_USER@$HOST:$remote_root/status.jsonl" "$watch_dir/harvest/" >/dev/null 2>&1 || true
  scp -i "$SSH_KEY_FILE" -P "$SSH_PORT" \
    -o StrictHostKeyChecking=accept-new \
    -o UserKnownHostsFile="$HOME/.ssh/known_hosts" \
    -r "$REMOTE_USER@$HOST:$remote_root/logs" "$watch_dir/harvest/" >/dev/null 2>&1 || true
  scp -i "$SSH_KEY_FILE" -P "$SSH_PORT" \
    -o StrictHostKeyChecking=accept-new \
    -o UserKnownHostsFile="$HOME/.ssh/known_hosts" \
    "$REMOTE_USER@$HOST:$remote_root/localizable_patches/summary.json" "$watch_dir/harvest/" >/dev/null 2>&1 || true
  scp -i "$SSH_KEY_FILE" -P "$SSH_PORT" \
    -o StrictHostKeyChecking=accept-new \
    -o UserKnownHostsFile="$HOME/.ssh/known_hosts" \
    "$REMOTE_USER@$HOST:$remote_root/localizable_patches/verify_report.json" "$watch_dir/harvest/" >/dev/null 2>&1 || true
}

destroy_instance() {
  log "destroying Vast instance $instance_id"
  "$VAST_BIN" --api-key "$VAST_API" destroy instance "$instance_id" >> "$log_file" 2>&1 || true
}

deadline=$(( $(date +%s) + MAX_SECONDS ))
log "watching instance=$instance_id remote_root=$remote_root interval=${INTERVAL_SECONDS}s"
while [[ "$(date +%s)" -lt "$deadline" ]]; do
  status_text="$(ssh_probe_run "tail -n 80 '$remote_root/status.jsonl' 2>/dev/null || true" 2>>"$log_file" || true)"
  printf '%s\n' "$status_text" > "$watch_dir/latest_status_tail.jsonl"
  has_b2_upload=0
  has_eval=0
  if printf '%s\n' "$status_text" | grep -q '"event": "b2_upload_completed"'; then
    has_b2_upload=1
  fi
  if printf '%s\n' "$status_text" | grep -q '"event": "eval_completed"'; then
    has_eval=1
  fi
  if [[ "$has_b2_upload" -eq 1 && ( "$REQUIRE_EVAL_REPORT" != "1" || "$has_eval" -eq 1 ) ]]; then
    log "run completed and B2 upload finished"
    fetch_lightweight_logs
    destroy_instance
    exit 0
  fi
  if [[ "$has_b2_upload" -eq 1 && "$REQUIRE_EVAL_REPORT" == "1" && "$has_eval" -ne 1 ]]; then
    log "B2 upload completed but required eval report is not complete yet; holding instance"
    sleep "$INTERVAL_SECONDS"
    continue
  fi
  ps_text="$(ssh_probe_run "pgrep -af 'remote_localizable_face_smoke|train_localizable_face_patch_tiny|build_localizable_face_patch_dataset|verify_localizable_face_patch_dataset|eval_localizable_face_patch_tiny|rclone copy' || true" 2>>"$log_file" || true)"
  printf '%s\n' "$ps_text" > "$watch_dir/latest_ps.txt"
  if [[ -z "$ps_text" ]]; then
    log "run process is gone before b2_upload_completed; harvesting lightweight logs and destroying"
    fetch_lightweight_logs
    destroy_instance
    exit 1
  fi
  sleep "$INTERVAL_SECONDS"
done

log "watch timed out after ${MAX_SECONDS}s; harvesting lightweight logs and destroying to avoid idle spend"
fetch_lightweight_logs
destroy_instance
exit 1
