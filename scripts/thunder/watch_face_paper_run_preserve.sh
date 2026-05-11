#!/usr/bin/env bash
# Periodically preserve metadata and latest checkpoints for a FACE Thunder run.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_INFO="${RUN_INFO:-${1:-}}"
INTERVAL_SEC="${INTERVAL_SEC:-900}"
MAX_LOOPS="${MAX_LOOPS:-0}"
LOG="${LOG:-}"

if [ -z "$RUN_INFO" ]; then
  echo "Usage: RUN_INFO=/path/run_info.json scripts/thunder/watch_face_paper_run_preserve.sh" >&2
  exit 2
fi
if [ -z "$LOG" ]; then
  run_dir="$(dirname "$RUN_INFO")"
  LOG="$run_dir/preserve_watch.log"
fi

mkdir -p "$(dirname "$LOG")"
loop=0
while true; do
  loop=$((loop + 1))
  {
    echo "===== $(date -u '+%Y-%m-%dT%H:%M:%SZ') preserve loop $loop ====="
    RUN_INFO="$RUN_INFO" "$REPO_ROOT/scripts/thunder/preserve_face_paper_run_artifacts.sh" || true
  } >> "$LOG" 2>&1

  status="$(python3 - "$LOG" <<'PY'
import json, sys
text = open(sys.argv[1], encoding="utf-8", errors="ignore").read()
status = ""
for idx, ch in enumerate(text):
    if ch != "{":
        continue
    try:
        payload, _ = json.JSONDecoder().raw_decode(text[idx:])
    except json.JSONDecodeError:
        continue
    if isinstance(payload, dict) and "status" in payload:
        status = str(payload["status"])
print(status)
PY
)"
  if [ "$status" != "running" ] && [ -n "$status" ]; then
    echo "===== preserve watch exiting status=$status at $(date -u '+%Y-%m-%dT%H:%M:%SZ') =====" >> "$LOG"
    break
  fi
  if [ "$MAX_LOOPS" -gt 0 ] && [ "$loop" -ge "$MAX_LOOPS" ]; then
    echo "===== preserve watch exiting max_loops=$MAX_LOOPS at $(date -u '+%Y-%m-%dT%H:%M:%SZ') =====" >> "$LOG"
    break
  fi
  sleep "$INTERVAL_SEC"
done
