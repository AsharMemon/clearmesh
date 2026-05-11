#!/usr/bin/env bash
# Fetch lightweight FACE run metadata plus the latest checkpoint files from a
# Thunder run. This is intentionally narrower than a full archive harvest: it
# avoids raw corpora while preserving enough state to resume after preemption or
# credit exhaustion.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TNR_BIN="${TNR_BIN:-/Users/Ashar/.tnr/bin/tnr}"
RUN_INFO="${RUN_INFO:-${1:-}}"
INSTANCE_ID="${THUNDER_INSTANCE_ID:-}"
REMOTE_LAB_ROOT="${REMOTE_LAB_ROOT:-}"
REMOTE_LOG="${REMOTE_LOG:-/tmp/clearmesh_face_paper_corpus_gate.nohup.log}"
REMOTE_PID="${REMOTE_PID:-/tmp/clearmesh_face_paper_corpus_gate.pid}"
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-}"
FETCH_METADATA="${FETCH_METADATA:-1}"
FETCH_CHECKPOINTS="${FETCH_CHECKPOINTS:-1}"
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
  echo "Usage: RUN_INFO=/path/run_info.json scripts/thunder/preserve_face_paper_run_artifacts.sh" >&2
  echo "   or: THUNDER_INSTANCE_ID=<id> REMOTE_LAB_ROOT=/tmp/run scripts/thunder/preserve_face_paper_run_artifacts.sh" >&2
  exit 2
fi
if [ -z "$DOWNLOAD_ROOT" ]; then
  DOWNLOAD_ROOT="$REPO_ROOT/.codex_outputs/face_paper_preserve_$(date -u +%Y%m%d_%H%M%S)"
fi

mkdir -p "$DOWNLOAD_ROOT/checkpoints"
PROBE_LOG="$DOWNLOAD_ROOT/preserve_probe.latest.log"

probe_output="$(
cat <<REMOTE | "$TNR_BIN" connect "$INSTANCE_ID" 2>&1 || true
set -euo pipefail
REMOTE_LAB_ROOT=$(printf '%q' "$REMOTE_LAB_ROOT")
REMOTE_PID=$(printf '%q' "$REMOTE_PID")
REMOTE_LOG=$(printf '%q' "$REMOTE_LOG")
status=unknown
pid=""
if [ -f "\$REMOTE_PID" ]; then
  pid="\$(cat "\$REMOTE_PID" 2>/dev/null || true)"
fi
if [ -n "\$pid" ] && kill -0 "\$pid" 2>/dev/null; then
  status=running
elif [ -d "\$REMOTE_LAB_ROOT" ]; then
  if [ -f "\$REMOTE_LAB_ROOT/status.jsonl" ] && grep -q '"status": "completed".*"step": "corpus_gate"\\|"status": "completed".*"step": "existing_split_gate"' "\$REMOTE_LAB_ROOT/status.jsonl"; then
    status=completed
  else
    status=exited
  fi
fi
printf 'CLEARMESH_PRESERVE_STATUS=%s\n' "\$status"
printf 'CLEARMESH_PRESERVE_PID=%s\n' "\$pid"
if [ -f "\$REMOTE_LAB_ROOT/status.jsonl" ]; then
  echo '--- remote status tail ---'
  tail -n 30 "\$REMOTE_LAB_ROOT/status.jsonl" || true
fi
if [ -f "\$REMOTE_LOG" ]; then
  echo '--- remote log tail ---'
  tail -n 80 "\$REMOTE_LOG" || true
fi
if [ -d "\$REMOTE_LAB_ROOT" ]; then
  echo '--- checkpoints ---'
  find "\$REMOTE_LAB_ROOT/runs" -maxdepth 2 -type f \( -name 'checkpoint.pt' -o -name 'checkpoint.latest.pt' -o -name 'checkpoint.current.pt' \) -printf 'CLEARMESH_CHECKPOINT\t%p\t%s\t%T@\n' 2>/dev/null | sort -k4,4nr || true
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

status="$(parse_field CLEARMESH_PRESERVE_STATUS "$probe_output")"
metadata_out=""
if [ "$FETCH_METADATA" = "1" ]; then
  metadata_out="$(
    RUN_INFO="$RUN_INFO" \
    THUNDER_INSTANCE_ID="$INSTANCE_ID" \
    REMOTE_LAB_ROOT="$REMOTE_LAB_ROOT" \
    REMOTE_LOG="$REMOTE_LOG" \
    REMOTE_PID="$REMOTE_PID" \
    DOWNLOAD_ROOT="$DOWNLOAD_ROOT" \
    FETCH_WHILE_RUNNING=1 \
    EXTRACT="$EXTRACT" \
    DELETE_ON_COMPLETE=0 \
    DELETE_ON_FAILURE=0 \
      "$REPO_ROOT/scripts/thunder/harvest_face_paper_run_metadata.sh" 2>&1 || true
  )"
  printf '%s\n' "$metadata_out" > "$DOWNLOAD_ROOT/preserve_metadata.latest.log"
fi

checkpoint_manifest="$DOWNLOAD_ROOT/checkpoints/manifest.jsonl"
fetched_checkpoint_count=0
if [ "$FETCH_CHECKPOINTS" = "1" ]; then
  while IFS=$'\t' read -r marker remote_path remote_size remote_mtime; do
    [ "$marker" = "CLEARMESH_CHECKPOINT" ] || continue
    [ -n "$remote_path" ] || continue
    safe_run="$(basename "$REMOTE_LAB_ROOT")"
    safe_name="$(basename "$remote_path")"
    local_path="$DOWNLOAD_ROOT/checkpoints/${safe_run}.${safe_name}"
    existing_key=""
    if [ -f "$checkpoint_manifest" ]; then
      existing_key="$(python3 - "$checkpoint_manifest" "$remote_path" "$remote_size" "$remote_mtime" <<'PY'
import json, sys
manifest, remote_path, remote_size, remote_mtime = sys.argv[1:]
for line in open(manifest, encoding="utf-8"):
    if not line.strip():
        continue
    row = json.loads(line)
    if (
        row.get("remote_path") == remote_path
        and str(row.get("remote_size")) == str(remote_size)
        and str(row.get("remote_mtime")) == str(remote_mtime)
        and row.get("local_path")
    ):
        print(row["local_path"])
        break
PY
)"
    fi
    if [ -n "$existing_key" ] && [ -f "$existing_key" ]; then
      continue
    fi
    tmp_path="$local_path.tmp"
    rm -f "$tmp_path"
    "$TNR_BIN" scp "$INSTANCE_ID:$remote_path" "$tmp_path"
    mv "$tmp_path" "$local_path"
    fetched_checkpoint_count=$((fetched_checkpoint_count + 1))
    python3 - "$checkpoint_manifest" "$remote_path" "$remote_size" "$remote_mtime" "$local_path" <<'PY'
import json, sys, time
manifest, remote_path, remote_size, remote_mtime, local_path = sys.argv[1:]
row = {
    "time": time.time(),
    "remote_path": remote_path,
    "remote_size": int(float(remote_size)),
    "remote_mtime": remote_mtime,
    "local_path": local_path,
}
with open(manifest, "a", encoding="utf-8") as handle:
    handle.write(json.dumps(row, sort_keys=True) + "\n")
PY
  done < <(python3 - "$probe_output" <<'PY'
import re, sys
ansi = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
for raw in sys.argv[1].splitlines():
    clean = ansi.sub("", raw).replace("\r", "").strip()
    if clean.startswith("CLEARMESH_CHECKPOINT\t"):
        print(clean)
PY
)
fi

python3 - "$DOWNLOAD_ROOT" "$INSTANCE_ID" "$REMOTE_LAB_ROOT" "$status" "$fetched_checkpoint_count" <<'PY'
import json, sys, time
payload = {
    "time": time.time(),
    "download_root": sys.argv[1],
    "instance_id": sys.argv[2],
    "remote_lab_root": sys.argv[3],
    "status": sys.argv[4],
    "fetched_checkpoint_count": int(sys.argv[5]),
}
print(json.dumps(payload, indent=2, sort_keys=True))
PY
