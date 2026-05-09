#!/usr/bin/env bash
# Safely harvest a FACE paper Thunder run, then inspect the fetched lab root.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_INFO="${RUN_INFO:-${1:-}}"
DELETE_ON_COMPLETE="${DELETE_ON_COMPLETE:-1}"
DELETE_ON_FAILURE="${DELETE_ON_FAILURE:-1}"
EXTRACT="${EXTRACT:-1}"

if [ -z "$RUN_INFO" ]; then
  echo "Usage: RUN_INFO=/path/run_info.json scripts/thunder/harvest_and_inspect_face_paper_run.sh" >&2
  exit 2
fi

extract_last_json_object() {
  python3 -c '
import json
import sys

text = sys.stdin.read()
decoder = json.JSONDecoder()
last = None
last_end = -1

for start, char in enumerate(text):
    if char != "{":
        continue
    try:
        payload, end = decoder.raw_decode(text[start:])
    except json.JSONDecodeError:
        continue
    if isinstance(payload, dict) and start + end > last_end:
        last = payload
        last_end = start + end

if last is None:
    print("No JSON object found in harvest output.", file=sys.stderr)
    raise SystemExit(1)

print(json.dumps(last))
'
}

HARVEST_OUTPUT="$(
  RUN_INFO="$RUN_INFO" \
  DELETE_ON_COMPLETE="$DELETE_ON_COMPLETE" \
  DELETE_ON_FAILURE="$DELETE_ON_FAILURE" \
  EXTRACT="$EXTRACT" \
  "$REPO_ROOT/scripts/thunder/harvest_face_paper_run.sh"
)"
echo "$HARVEST_OUTPUT"

HARVEST_JSON="$(printf '%s' "$HARVEST_OUTPUT" | extract_last_json_object)"

status="$(python3 - "$HARVEST_JSON" <<'PY'
import json, sys
payload = json.loads(sys.argv[1])
print(payload.get("status") or "")
PY
)"

download_root="$(python3 - "$HARVEST_JSON" <<'PY'
import json, sys
payload = json.loads(sys.argv[1])
print(payload.get("download_root") or "")
PY
)"

if [ "$status" != "completed" ] && ! python3 - "$HARVEST_JSON" <<'PY'
import json, sys
payload = json.loads(sys.argv[1])
raise SystemExit(0 if payload.get("archive") else 1)
PY
then
  exit 0
fi

if [ -z "$download_root" ] || [ ! -d "$download_root" ]; then
  echo "download_root is missing after harvest: $download_root" >&2
  exit 3
fi

lab_root="$(find "$download_root" -mindepth 1 -maxdepth 1 -type d -name 'clearmesh_face_*' -print -quit)"
if [ -z "$lab_root" ]; then
  echo "No extracted FACE lab root found in $download_root" >&2
  exit 4
fi

inspect_out="$download_root/face_gate_inspection.json"
python3 "$REPO_ROOT/scripts/research/inspect_face_paper_gate.py" "$lab_root" --output "$inspect_out"
echo "inspection=$inspect_out"

run_dir="$(python3 - "$inspect_out" <<'PY'
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload.get("run_dir") or "")
PY
)"

if [ -n "$run_dir" ] && [ -d "$run_dir/eval" ]; then
  analysis_out="$download_root/face_ar_failure_analysis.json"
  python3 "$REPO_ROOT/scripts/research/analyze_face_paper_ar_failures.py" "$run_dir" --output "$analysis_out" >/dev/null
  echo "ar_failure_analysis=$analysis_out"
fi
