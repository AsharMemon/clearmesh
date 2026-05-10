#!/usr/bin/env bash
# Metadata-only harvest for FACE paper Thunder runs.
#
# The full FACE corpus archive can be tens of GB because it contains raw corpus
# assets and checkpoints. This helper creates a lightweight remote archive of
# logs, eval JSON, summaries, contact sheets, and small exported meshes while
# excluding corpus caches and model weights.
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
FETCH_WHILE_RUNNING="${FETCH_WHILE_RUNNING:-0}"

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
payload = json.load(open(sys.argv[1]))
print(payload.get("instance_id") or "")
PY
)"
  REMOTE_LAB_ROOT="$(python3 - "$RUN_INFO" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
print(payload.get("remote_lab_root") or "")
PY
)"
  REMOTE_LOG="$(python3 - "$RUN_INFO" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
print(payload.get("remote_log") or "/tmp/clearmesh_face_paper_corpus_gate.nohup.log")
PY
)"
  REMOTE_PID="$(python3 - "$RUN_INFO" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
print(payload.get("remote_pid") or "/tmp/clearmesh_face_paper_corpus_gate.pid")
PY
)"
  if [ -z "$DOWNLOAD_ROOT" ]; then
    DOWNLOAD_ROOT="$(python3 - "$RUN_INFO" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
print(payload.get("download_root") or "")
PY
)"
  fi
fi
if [ -z "$INSTANCE_ID" ] || [ -z "$REMOTE_LAB_ROOT" ]; then
  echo "Usage: RUN_INFO=/path/run_info.json scripts/thunder/harvest_face_paper_run_metadata.sh" >&2
  echo "   or: THUNDER_INSTANCE_ID=<id> REMOTE_LAB_ROOT=/tmp/run scripts/thunder/harvest_face_paper_run_metadata.sh" >&2
  exit 2
fi
if [ -z "$DOWNLOAD_ROOT" ]; then
  DOWNLOAD_ROOT="$REPO_ROOT/.codex_outputs/face_paper_metadata_harvest_$(date -u +%Y%m%d_%H%M%S)"
fi
mkdir -p "$DOWNLOAD_ROOT"

REMOTE_META_ARCHIVE="${REMOTE_META_ARCHIVE:-/tmp/$(basename "$REMOTE_LAB_ROOT").metadata.tar.gz}"
PROBE_LOG="$DOWNLOAD_ROOT/metadata_harvest_probe.latest.log"

probe_output="$(
cat <<REMOTE | "$TNR_BIN" connect "$INSTANCE_ID" 2>&1 || true
set -euo pipefail
REMOTE_LAB_ROOT=$(printf '%q' "$REMOTE_LAB_ROOT")
REMOTE_META_ARCHIVE=$(printf '%q' "$REMOTE_META_ARCHIVE")
REMOTE_PID=$(printf '%q' "$REMOTE_PID")
REMOTE_LOG=$(printf '%q' "$REMOTE_LOG")
FETCH_WHILE_RUNNING=$(printf '%q' "$FETCH_WHILE_RUNNING")
status=unknown
pid=""
if [ -f "\$REMOTE_PID" ]; then
  pid="\$(cat "\$REMOTE_PID" 2>/dev/null || true)"
fi
if [ -n "\$pid" ] && kill -0 "\$pid" 2>/dev/null; then
  status=running
elif [ -d "\$REMOTE_LAB_ROOT" ]; then
  if [ -f "\$REMOTE_LAB_ROOT/status.jsonl" ] && grep -q '"status": "completed".*"step": "corpus_gate"' "\$REMOTE_LAB_ROOT/status.jsonl"; then
    status=completed
  else
    status=exited_no_archive
  fi
fi
printf 'CLEARMESH_METADATA_STATUS=%s\n' "\$status"
printf 'CLEARMESH_METADATA_PID=%s\n' "\$pid"
printf 'CLEARMESH_METADATA_ARCHIVE_PATH=%s\n' "\$REMOTE_META_ARCHIVE"
if [ -f "\$REMOTE_LAB_ROOT/status.jsonl" ]; then
  echo '--- remote status tail ---'
  tail -n 30 "\$REMOTE_LAB_ROOT/status.jsonl" || true
fi
if [ -f "\$REMOTE_LOG" ]; then
  echo '--- remote log tail ---'
  tail -n 80 "\$REMOTE_LOG" || true
fi
if [ -d "\$REMOTE_LAB_ROOT" ] && { [ "\$status" != "running" ] || [ "\$FETCH_WHILE_RUNNING" = "1" ]; }; then
  rm -f "\$REMOTE_META_ARCHIVE"
  tar \
    --exclude='corpus' \
    --exclude='*.pt' \
    --exclude='*.pth' \
    --exclude='*.safetensors' \
    --exclude='*.ckpt' \
    --exclude='*.npy' \
    --exclude='*.npz' \
    --exclude='__pycache__' \
    -czf "\$REMOTE_META_ARCHIVE" \
    -C "\$(dirname "\$REMOTE_LAB_ROOT")" "\$(basename "\$REMOTE_LAB_ROOT")"
  printf 'CLEARMESH_METADATA_ARCHIVE=1\n'
else
  printf 'CLEARMESH_METADATA_ARCHIVE=0\n'
fi
exit
REMOTE
)"
printf '%s\n' "$probe_output" > "$PROBE_LOG"

parse_field() {
  python3 - "$1" "$2" <<'PY'
import re
import sys
key = sys.argv[1]
text = sys.argv[2]
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

status="$(parse_field CLEARMESH_METADATA_STATUS "$probe_output")"
archive="$(parse_field CLEARMESH_METADATA_ARCHIVE "$probe_output")"
archive_path="$(parse_field CLEARMESH_METADATA_ARCHIVE_PATH "$probe_output")"
fetched_archive=""
deleted="0"

if [ "$archive" = "1" ]; then
  fetch_output="$(EXTRACT="$EXTRACT" "$REPO_ROOT/scripts/thunder/fetch_remote_artifact.sh" "$INSTANCE_ID" "$archive_path" "$DOWNLOAD_ROOT")"
  printf '%s\n' "$fetch_output" > "$DOWNLOAD_ROOT/metadata_harvest_fetch.latest.log"
  fetched_archive="$(printf '%s\n' "$fetch_output" | awk -F= '/^downloaded=/{print $2}' | tail -n 1)"
  if [ "$DELETE_ON_COMPLETE" = "1" ] && [ "$status" = "completed" ]; then
    "$TNR_BIN" delete "$INSTANCE_ID" --yes || true
    deleted="1"
  elif [ "$DELETE_ON_FAILURE" = "1" ] && [ "$status" = "exited_no_archive" ]; then
    "$TNR_BIN" delete "$INSTANCE_ID" --yes || true
    deleted="1"
  fi
fi

inspect_out=""
analysis_out=""
if [ "$EXTRACT" = "1" ] && [ "$archive" = "1" ]; then
  lab_root="$(find "$DOWNLOAD_ROOT" -mindepth 1 -maxdepth 1 -type d -name 'clearmesh_face_*' -print -quit)"
  if [ -n "$lab_root" ]; then
    inspect_out="$DOWNLOAD_ROOT/face_gate_inspection.json"
    python3 "$REPO_ROOT/scripts/research/inspect_face_paper_gate.py" "$lab_root" --output "$inspect_out" >/dev/null || inspect_out=""
    if [ -n "$inspect_out" ]; then
      run_dir="$(python3 - "$inspect_out" <<'PY'
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload.get("run_dir") or "")
PY
)"
      if [ -n "$run_dir" ] && [ -d "$run_dir/eval" ]; then
        analysis_out="$DOWNLOAD_ROOT/face_ar_failure_analysis.json"
        python3 "$REPO_ROOT/scripts/research/analyze_face_paper_ar_failures.py" "$run_dir" --output "$analysis_out" >/dev/null || analysis_out=""
      fi
    fi
  fi
fi

python3 - "$DOWNLOAD_ROOT" "$INSTANCE_ID" "$REMOTE_LAB_ROOT" "$status" "$archive" "$archive_path" "$fetched_archive" "$deleted" "$inspect_out" "$analysis_out" <<'PY'
import json
import sys
import time
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
    "inspection": sys.argv[9] or None,
    "ar_failure_analysis": sys.argv[10] or None,
}
print(json.dumps(payload, indent=2, sort_keys=True))
PY
