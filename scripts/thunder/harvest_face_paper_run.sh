#!/usr/bin/env bash
# One-shot status/fetch/delete helper for a FACE paper Thunder run.
#
# This is designed for Codex heartbeats/cron-style checks: run once, report JSON,
# fetch the archive if present, and optionally delete the Thunder instance.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TNR_BIN="${TNR_BIN:-/Users/Ashar/.tnr/bin/tnr}"
RUN_INFO="${RUN_INFO:-${1:-}}"
INSTANCE_ID="${THUNDER_INSTANCE_ID:-}"
REMOTE_LAB_ROOT="${REMOTE_LAB_ROOT:-}"
REMOTE_LOG="${REMOTE_LOG:-/tmp/clearmesh_face_paper_corpus_gate.nohup.log}"
REMOTE_PID="${REMOTE_PID:-/tmp/clearmesh_face_paper_corpus_gate.pid}"
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-}"
DELETE_ON_COMPLETE="${DELETE_ON_COMPLETE:-0}"
DELETE_ON_FAILURE="${DELETE_ON_FAILURE:-0}"
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
print(payload.get("instance_id") or "")
PY
)"
  REMOTE_LAB_ROOT="$(python3 - "$RUN_INFO" <<'PY'
import json, sys
payload=json.load(open(sys.argv[1]))
print(payload.get("remote_lab_root") or "")
PY
)"
  REMOTE_LOG="$(python3 - "$RUN_INFO" <<'PY'
import json, sys
payload=json.load(open(sys.argv[1]))
print(payload.get("remote_log") or "/tmp/clearmesh_face_paper_corpus_gate.nohup.log")
PY
)"
  REMOTE_PID="$(python3 - "$RUN_INFO" <<'PY'
import json, sys
payload=json.load(open(sys.argv[1]))
print(payload.get("remote_pid") or "/tmp/clearmesh_face_paper_corpus_gate.pid")
PY
)"
  if [ -z "$DOWNLOAD_ROOT" ]; then
    DOWNLOAD_ROOT="$(python3 - "$RUN_INFO" <<'PY'
import json, sys
payload=json.load(open(sys.argv[1]))
print(payload.get("download_root") or "")
PY
)"
  fi
fi
if [ -z "$INSTANCE_ID" ] || [ -z "$REMOTE_LAB_ROOT" ]; then
  echo "Usage: RUN_INFO=/path/run_info.json scripts/thunder/harvest_face_paper_run.sh" >&2
  echo "   or: THUNDER_INSTANCE_ID=<id> REMOTE_LAB_ROOT=/tmp/run scripts/thunder/harvest_face_paper_run.sh" >&2
  exit 2
fi
if [ -z "$DOWNLOAD_ROOT" ]; then
  DOWNLOAD_ROOT="$REPO_ROOT/.codex_outputs/face_paper_harvest_$(date -u +%Y%m%d_%H%M%S)"
fi
mkdir -p "$DOWNLOAD_ROOT"

REMOTE_ARCHIVE="$REMOTE_LAB_ROOT.tar.gz"
PROBE_LOG="$DOWNLOAD_ROOT/harvest_probe.latest.log"

probe_output="$(
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
printf 'CLEARMESH_HARVEST_STATUS=%s\n' "\$status"
printf 'CLEARMESH_HARVEST_PID=%s\n' "\$pid"
printf 'CLEARMESH_HARVEST_ARCHIVE=%s\n' "\$archive_present"
printf 'CLEARMESH_HARVEST_ARCHIVE_PATH=%s\n' "\$REMOTE_ARCHIVE"
if [ -f "\$REMOTE_LAB_ROOT/status.jsonl" ]; then
  echo '--- remote status tail ---'
  tail -n 20 "\$REMOTE_LAB_ROOT/status.jsonl" || true
fi
if [ -f "\$REMOTE_LOG" ]; then
  echo '--- remote log tail ---'
  tail -n 60 "\$REMOTE_LOG" || true
fi
exit
REMOTE
)"
printf '%s\n' "$probe_output" > "$PROBE_LOG"

parse_field() {
  python3 - "$1" "$2" <<'PY'
import re
import sys
key=sys.argv[1]
text=sys.argv[2]
ansi = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
prefix = f"{key}="
value = ""
for raw in text.splitlines():
    clean = ansi.sub("", raw).replace("\r", "").strip()
    if clean.startswith(prefix):
        value = clean[len(prefix):].strip()
print(value)
PY
}

status="$(parse_field CLEARMESH_HARVEST_STATUS "$probe_output")"
archive="$(parse_field CLEARMESH_HARVEST_ARCHIVE "$probe_output")"
archive_path="$(parse_field CLEARMESH_HARVEST_ARCHIVE_PATH "$probe_output")"
fetched_archive=""
deleted="0"

if [ "$status" = "exited_no_archive" ] && [ "$archive" != "1" ]; then
  force_archive_output="$(
  cat <<REMOTE | "$TNR_BIN" connect "$INSTANCE_ID" 2>&1 || true
set -euo pipefail
REMOTE_LAB_ROOT=$(printf '%q' "$REMOTE_LAB_ROOT")
REMOTE_ARCHIVE=$(printf '%q' "$REMOTE_ARCHIVE")
if [ -d "\$REMOTE_LAB_ROOT" ]; then
  tar -czf "\$REMOTE_ARCHIVE" -C "\$(dirname "\$REMOTE_LAB_ROOT")" "\$(basename "\$REMOTE_LAB_ROOT")" || true
fi
if [ -f "\$REMOTE_ARCHIVE" ]; then
  echo "CLEARMESH_FORCED_ARCHIVE=1"
else
  echo "CLEARMESH_FORCED_ARCHIVE=0"
fi
exit
REMOTE
  )"
  printf '%s\n' "$force_archive_output" > "$DOWNLOAD_ROOT/harvest_force_archive.latest.log"
  forced_archive="$(parse_field CLEARMESH_FORCED_ARCHIVE "$force_archive_output")"
  if [ "$forced_archive" = "1" ]; then
    archive="1"
    archive_path="$REMOTE_ARCHIVE"
    status="failed_archived"
  fi
fi

if [ "$archive" = "1" ] || [ "$status" = "completed" ]; then
  fetch_output="$(EXTRACT="$EXTRACT" "$REPO_ROOT/scripts/thunder/fetch_remote_artifact.sh" "$INSTANCE_ID" "$archive_path" "$DOWNLOAD_ROOT")"
  printf '%s\n' "$fetch_output" > "$DOWNLOAD_ROOT/harvest_fetch.latest.log"
  fetched_archive="$(printf '%s\n' "$fetch_output" | awk -F= '/^downloaded=/{print $2}' | tail -n 1)"
  if [ "$DELETE_ON_COMPLETE" = "1" ] || { [ "$status" = "failed_archived" ] && [ "$DELETE_ON_FAILURE" = "1" ]; }; then
    "$TNR_BIN" delete "$INSTANCE_ID" --yes || true
    deleted="1"
  fi
elif [ "$status" = "exited_no_archive" ] && [ "$DELETE_ON_FAILURE" = "1" ]; then
  "$TNR_BIN" delete "$INSTANCE_ID" --yes || true
  deleted="1"
fi

python3 - "$DOWNLOAD_ROOT" "$INSTANCE_ID" "$REMOTE_LAB_ROOT" "$status" "$archive" "$archive_path" "$fetched_archive" "$deleted" <<'PY'
import json, sys, time
payload = {
    "time": time.time(),
    "download_root": sys.argv[1],
    "instance_id": sys.argv[2],
    "remote_lab_root": sys.argv[3],
    "status": sys.argv[4],
    "archive": sys.argv[5] == "1",
    "archive_path": sys.argv[6],
    "fetched_archive": sys.argv[7] or None,
    "deleted": sys.argv[8] == "1",
}
print(json.dumps(payload, indent=2, sort_keys=True))
PY
