#!/usr/bin/env bash
# Monitor a launched FACE paper A100 probe, fetch its archive, and optionally
# delete the Thunder instance when the run is complete or failed.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TNR_BIN="${TNR_BIN:-/Users/Ashar/.tnr/bin/tnr}"
RUN_INFO="${RUN_INFO:-${1:-}}"
INSTANCE_ID="${THUNDER_INSTANCE_ID:-}"
REMOTE_LAB_ROOT="${REMOTE_LAB_ROOT:-}"
REMOTE_LOG="${REMOTE_LOG:-/tmp/clearmesh_face_a100_probe.nohup.log}"
REMOTE_PID="${REMOTE_PID:-/tmp/clearmesh_face_a100_probe.pid}"
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-}"
POLL_SEC="${POLL_SEC:-60}"
TIMEOUT_SEC="${TIMEOUT_SEC:-43200}"
DELETE_ON_COMPLETE="${DELETE_ON_COMPLETE:-1}"
DELETE_ON_FAILURE="${DELETE_ON_FAILURE:-1}"
EXTRACT="${EXTRACT:-1}"

if [ -z "${THUNDER_TOKEN:-}" ]; then
  echo "THUNDER_TOKEN is not set." >&2
  exit 1
fi
if [ ! -x "$TNR_BIN" ]; then
  echo "tnr binary not found or not executable: $TNR_BIN" >&2
  exit 1
fi
if [ -n "$RUN_INFO" ]; then
  if [ ! -f "$RUN_INFO" ]; then
    echo "RUN_INFO not found: $RUN_INFO" >&2
    exit 2
  fi
  INSTANCE_ID="$(python3 - "$RUN_INFO" <<'PY'
import json, sys
payload=json.load(open(sys.argv[1]))
print(payload.get('instance_id') or '')
PY
)"
  REMOTE_LAB_ROOT="$(python3 - "$RUN_INFO" <<'PY'
import json, sys
payload=json.load(open(sys.argv[1]))
print(payload.get('remote_lab_root') or '')
PY
)"
  REMOTE_LOG="$(python3 - "$RUN_INFO" <<'PY'
import json, sys
payload=json.load(open(sys.argv[1]))
print(payload.get('remote_log') or '/tmp/clearmesh_face_a100_probe.nohup.log')
PY
)"
  REMOTE_PID="$(python3 - "$RUN_INFO" <<'PY'
import json, sys
payload=json.load(open(sys.argv[1]))
print(payload.get('remote_pid') or '/tmp/clearmesh_face_a100_probe.pid')
PY
)"
  if [ -z "$DOWNLOAD_ROOT" ]; then
    DOWNLOAD_ROOT="$(python3 - "$RUN_INFO" <<'PY'
import json, sys
payload=json.load(open(sys.argv[1]))
print(payload.get('download_root') or '')
PY
)"
  fi
fi
if [ -z "$INSTANCE_ID" ] || [ -z "$REMOTE_LAB_ROOT" ]; then
  echo "Usage: RUN_INFO=/path/run_info.json scripts/thunder/monitor_face_paper_probe.sh" >&2
  echo "   or: THUNDER_INSTANCE_ID=<id> REMOTE_LAB_ROOT=/tmp/run scripts/thunder/monitor_face_paper_probe.sh" >&2
  exit 2
fi
if [ -z "$DOWNLOAD_ROOT" ]; then
  DOWNLOAD_ROOT="$REPO_ROOT/.codex_outputs/face_paper_monitor_$(date -u +%Y%m%d_%H%M%S)"
fi
mkdir -p "$DOWNLOAD_ROOT"
STATUS_FILE="$DOWNLOAD_ROOT/monitor_status.jsonl"
REMOTE_ARCHIVE="$REMOTE_LAB_ROOT.tar.gz"

append_status() {
  python3 - "$STATUS_FILE" "$@" <<'PY'
import json, sys, time
from pathlib import Path
path=Path(sys.argv[1])
item={'time': time.time()}
for raw in sys.argv[2:]:
    key, _, value = raw.partition('=')
    item[key]=value
path.parent.mkdir(parents=True, exist_ok=True)
with path.open('a', encoding='utf-8') as handle:
    handle.write(json.dumps(item, sort_keys=True)+'\n')
print(json.dumps(item, sort_keys=True))
PY
}

remote_probe() {
  cat <<REMOTE | "$TNR_BIN" connect "$INSTANCE_ID" 2>&1 || true
set -euo pipefail
REMOTE_LAB_ROOT=$(printf '%q' "$REMOTE_LAB_ROOT")
REMOTE_ARCHIVE=$(printf '%q' "$REMOTE_ARCHIVE")
REMOTE_PID=$(printf '%q' "$REMOTE_PID")
REMOTE_LOG=$(printf '%q' "$REMOTE_LOG")
status=unknown
pid=""
if [ -f "\$REMOTE_PID" ]; then
  pid="\$(cat "\$REMOTE_PID" 2>/dev/null || true)"
fi
if [ -n "\$pid" ] && kill -0 "\$pid" 2>/dev/null; then
  status=running
elif [ -f "\$REMOTE_ARCHIVE" ]; then
  status=completed
elif [ -n "\$pid" ]; then
  status=exited_no_archive
fi
archive_present=0
[ -f "\$REMOTE_ARCHIVE" ] && archive_present=1
printf 'CLEARMESH_MONITOR_STATUS=%s\n' "\$status"
printf 'CLEARMESH_MONITOR_PID=%s\n' "\$pid"
printf 'CLEARMESH_MONITOR_ARCHIVE=%s\n' "\$archive_present"
printf 'CLEARMESH_MONITOR_ARCHIVE_PATH=%s\n' "\$REMOTE_ARCHIVE"
if [ -f "\$REMOTE_LAB_ROOT/status.jsonl" ]; then
  echo '--- remote status tail ---'
  tail -n 10 "\$REMOTE_LAB_ROOT/status.jsonl" || true
fi
if [ -f "\$REMOTE_LOG" ]; then
  echo '--- remote log tail ---'
  tail -n 40 "\$REMOTE_LOG" || true
fi
exit
REMOTE
}

parse_field() {
  python3 - "$1" "$2" <<'PY'
import re
import sys
key=sys.argv[1]
text=sys.argv[2]
ansi = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
value = ""
prefix = f"{key}="
for raw in text.splitlines():
    clean = ansi.sub("", raw).replace("\r", "").strip()
    # Thunder echoes the remote script before executing it. Only accept the
    # actual status lines, not the echoed printf commands containing the key.
    if clean.startswith(prefix):
        value = clean[len(prefix):].strip()
print(value)
PY
}

fetch_archive() {
  "$REPO_ROOT/scripts/thunder/fetch_remote_artifact.sh" "$INSTANCE_ID" "$REMOTE_ARCHIVE" "$DOWNLOAD_ROOT"
}

delete_instance() {
  "$TNR_BIN" delete "$INSTANCE_ID" --yes || true
}

append_status event=start instance_id="$INSTANCE_ID" remote_lab_root="$REMOTE_LAB_ROOT" download_root="$DOWNLOAD_ROOT"
deadline=$(( $(date +%s) + TIMEOUT_SEC ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  probe_output="$(remote_probe)"
  printf '%s\n' "$probe_output" > "$DOWNLOAD_ROOT/remote_probe.latest.log"
  status="$(parse_field CLEARMESH_MONITOR_STATUS "$probe_output")"
  archive="$(parse_field CLEARMESH_MONITOR_ARCHIVE "$probe_output")"
  append_status event=poll status="$status" archive="$archive"
  if [ "$archive" = "1" ] || [ "$status" = "completed" ]; then
    append_status event=fetch_started remote_archive="$REMOTE_ARCHIVE"
    fetch_archive
    append_status event=fetch_completed
    if [ "$DELETE_ON_COMPLETE" = "1" ]; then
      append_status event=delete_started reason=complete
      delete_instance
      append_status event=delete_completed reason=complete
    fi
    exit 0
  fi
  if [ "$status" = "exited_no_archive" ]; then
    append_status event=failed reason=exited_no_archive
    if [ "$DELETE_ON_FAILURE" = "1" ]; then
      append_status event=delete_started reason=failure
      delete_instance
      append_status event=delete_completed reason=failure
    fi
    exit 30
  fi
  sleep "$POLL_SEC"
done
append_status event=timeout timeout_sec="$TIMEOUT_SEC"
exit 31
