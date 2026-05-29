#!/usr/bin/env bash
# Safely delete idle Thunder FACE corpus workers after their shard outputs are
# durable in B2. Defaults to dry-run; pass --delete to actually delete.
set -euo pipefail

TNR_BIN="${TNR_BIN:-/Users/Ashar/.tnr/bin/tnr}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
B2_BUCKET="${B2_BUCKET:-clearmesh-pairs}"
B2_ENV_FILE="${B2_ENV_FILE:-$REPO_ROOT/.codex_secrets/b2.env}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/.codex_outputs/thunder_idle_reaper_$(date -u +%Y%m%dT%H%M%SZ)}"
DELETE=0
ALL_A6000=0
ALLOW_PRODUCTION="${ALLOW_PRODUCTION:-0}"
DELETE_WITHOUT_PREFIXES="${DELETE_WITHOUT_PREFIXES:-0}"
DELETE_FAILED="${DELETE_FAILED:-0}"
INSTANCE_IDS=()

usage() {
  cat <<'USAGE'
Usage:
  scripts/thunder/reap_idle_face_corpus_workers.sh --ids "1 2 5" [--delete]
  scripts/thunder/reap_idle_face_corpus_workers.sh 1 2 5 --delete
  scripts/thunder/reap_idle_face_corpus_workers.sh --all-a6000 --delete

Safety behavior:
  - Dry-run by default. Use --delete to call `tnr delete`.
  - Skips production instances unless ALLOW_PRODUCTION=1 is set.
  - Keeps any instance with live queue/download/tokenize/package/upload/dedupe processes.
  - Deletes only when every observed shard B2 prefix has both:
      lean_face_corpus.tar.gz
      corpus/split_pass/split_summary.json
  - Refuses to delete failed/no-status workers unless explicitly overridden.

Useful env:
  B2_ENV_FILE=.codex_secrets/b2.env
  B2_BUCKET=clearmesh-pairs
  ALLOW_PRODUCTION=1
  DELETE_FAILED=1
  DELETE_WITHOUT_PREFIXES=1
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ids)
      shift
      [[ $# -gt 0 ]] || { echo "--ids requires a value" >&2; exit 2; }
      # shellcheck disable=SC2206
      INSTANCE_IDS+=($1)
      ;;
    --delete)
      DELETE=1
      ;;
    --all-a6000)
      ALL_A6000=1
      ;;
    --dry-run)
      DELETE=0
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      INSTANCE_IDS+=("$1")
      ;;
  esac
  shift
done

if [[ -z "${THUNDER_TOKEN:-}" ]]; then
  echo "THUNDER_TOKEN is not set." >&2
  exit 2
fi
if [[ ! -x "$TNR_BIN" ]]; then
  echo "tnr binary not found or not executable: $TNR_BIN" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"

configure_b2env() {
  if [[ -f "$B2_ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$B2_ENV_FILE"
  fi
  local key_id="${B2_KEY_ID:-${B2_KEYID:-${B2_APPLICATION_KEY_ID:-${BACKBLAZE_B2_KEY_ID:-}}}}"
  local app_key="${B2_APP_KEY:-${B2_APPKEY:-${B2_APPLICATION_KEY:-${BACKBLAZE_B2_APPLICATION_KEY:-${BACKBLAZE_B2_APP_KEY:-}}}}}"
  if [[ -z "$key_id" || -z "$app_key" ]] && [[ -n "${B2_TOKEN:-}" ]]; then
    local parsed
    parsed="$(
      python3 - <<'PY'
import json
import os
token = os.environ.get("B2_TOKEN", "").strip()
key_id = app_key = ""
if token:
    if token.startswith("{"):
        payload = json.loads(token)
        key_id = payload.get("keyId") or payload.get("applicationKeyId") or payload.get("key_id") or ""
        app_key = payload.get("applicationKey") or payload.get("application_key") or payload.get("appKey") or ""
    elif ":" in token:
        key_id, app_key = token.split(":", 1)
if key_id and app_key:
    print(key_id)
    print(app_key)
PY
    )"
    if [[ -n "$parsed" ]]; then
      key_id="${key_id:-$(printf '%s\n' "$parsed" | sed -n '1p')}"
      app_key="${app_key:-$(printf '%s\n' "$parsed" | sed -n '2p')}"
    fi
  fi
  if [[ -n "$key_id" && -z "$app_key" && -n "${B2_TOKEN:-}" ]]; then
    case "$B2_TOKEN" in
      \{*|*:*) ;;
      *) app_key="$B2_TOKEN" ;;
    esac
  fi
  if [[ -z "$key_id" || -z "$app_key" ]]; then
    echo "B2 credentials unavailable; set B2_ENV_FILE or B2_* env vars." >&2
    exit 2
  fi
  export RCLONE_CONFIG_B2ENV_TYPE=b2
  export RCLONE_CONFIG_B2ENV_ACCOUNT="$key_id"
  export RCLONE_CONFIG_B2ENV_KEY="$app_key"
}

status_json="$OUT_DIR/tnr_status.json"
"$TNR_BIN" status --json > "$status_json"

if [[ "$ALL_A6000" = "1" ]]; then
  while IFS= read -r discovered_id; do
    [[ -n "$discovered_id" ]] && INSTANCE_IDS+=("$discovered_id")
  done < <(python3 - "$status_json" <<'PY'
import json
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(errors="ignore")
start = text.find("[")
data = json.loads(text[start:]) if start >= 0 else []
for item in data:
    gpu = str(item.get("gpu") or item.get("gpuType") or "")
    if item.get("status") == "RUNNING" and gpu.lower().endswith("a6000"):
        print(item.get("id"))
PY
)
fi
if [[ "${#INSTANCE_IDS[@]}" -eq 0 ]]; then
  echo "Provide instance ids with --ids/positional args, or use --all-a6000." >&2
  usage >&2
  exit 2
fi

configure_b2env

probe_script="$OUT_DIR/remote_idle_probe.py"
cat > "$probe_script" <<'PY'
#!/usr/bin/env python3
import glob
import json
import os
import re
import subprocess
from pathlib import Path

LIVE_PATTERNS = (
    "face_corpus_shard_queue_worker.sh",
    "face_objaversepp_corpus_pilot.sh",
    "prepare_face_strict_targets",
    "build_face_token_dataset.py",
    "package_face_corpus.py",
    "b2_continuous_upload.sh",
    "rclone copy",
    "rclone copyto",
    "tar -czf",
    "merge_face_corpus_shards.py",
    "dedupe_face_token_split.py",
    "check_face_token_leakage.py",
    "clearmesh_faceq_rolling_corpus_prep",
)

def run(cmd):
    return subprocess.run(cmd, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout

ps = run("ps -eo pid,ppid,stat,etimes,args")
live = []
for line in ps.splitlines()[1:]:
    if "remote_idle_probe.py" in line or "pgrep" in line:
        continue
    if any(pattern in line for pattern in LIVE_PATTERNS):
        live.append(line.strip())

status_files = sorted(set(glob.glob("/tmp/clearmesh*_logs/status.jsonl") + glob.glob("/tmp/clearmesh*logs/status.jsonl")))
prefixes = set()
failures = []
queue_completes = []
latest_events = []
for status_file in status_files:
    path = Path(status_file)
    try:
        lines = path.read_text(errors="ignore").splitlines()
    except OSError:
        continue
    for raw in lines:
        try:
            row = json.loads(raw)
        except Exception:
            continue
        detail = str(row.get("detail") or "")
        match = re.search(r"\bb2_prefix=([^ ]+)", detail)
        if match:
            prefixes.add(match.group(1).rstrip("/"))
        state = str(row.get("state") or row.get("event") or "")
        step = str(row.get("step") or "")
        if state == "complete" and step == "queue":
            queue_completes.append({"file": status_file, "detail": detail, "time": row.get("time")})
        if state == "failed":
            failures.append({"file": status_file, "step": step, "detail": detail, "time": row.get("time")})
        latest_events.append({"file": status_file, "state": state, "step": step, "detail": detail, "time": row.get("time")})

for claim in glob.glob("/tmp/clearmesh*/queue_claim.json"):
    try:
        payload = json.loads(Path(claim).read_text())
    except Exception:
        continue
    prefix = str(payload.get("b2_prefix") or "").rstrip("/")
    if prefix:
        prefixes.add(prefix)

disk = run("df -h / /tmp /ephemeral 2>/dev/null | tail -n +2").strip().splitlines()
print(json.dumps({
    "live_count": len(live),
    "live": live[:20],
    "status_file_count": len(status_files),
    "prefixes": sorted(prefixes),
    "failure_count": len(failures),
    "recent_failures": failures[-8:],
    "queue_complete_count": len(queue_completes),
    "recent_queue_completes": queue_completes[-8:],
    "recent_events": latest_events[-12:],
    "disk": disk,
}, sort_keys=True))
PY

get_status_field() {
  local instance_id="$1" field="$2"
  python3 - "$status_json" "$instance_id" "$field" <<'PY'
import json
import sys
from pathlib import Path
path, target, field = sys.argv[1:]
text = Path(path).read_text(errors="ignore")
start = text.find("[")
data = json.loads(text[start:]) if start >= 0 else []
for item in data:
    if str(item.get("id")) == str(target):
        value = item.get(field)
        if value is None and field == "gpu":
            value = item.get("gpuType")
        print("" if value is None else value)
        raise SystemExit(0)
raise SystemExit(1)
PY
}

b2_has_required_outputs() {
  local prefix="$1"
  rclone lsf "b2env:$B2_BUCKET/$prefix" --files-only 2>/dev/null \
    | grep -Fxq "lean_face_corpus.tar.gz" || return 1
  rclone lsf "b2env:$B2_BUCKET/$prefix/corpus/split_pass" --files-only 2>/dev/null \
    | grep -Fxq "split_summary.json" || return 1
}

probe_instance() {
  local instance_id="$1"
  local remote_probe="/tmp/clearmesh_idle_reaper_probe.py"
  local raw
  "$TNR_BIN" scp "$probe_script" "$instance_id:$remote_probe" >/dev/null
  raw="$({
    printf 'python3 %q\n' "$remote_probe"
    printf 'exit\n'
  } | "$TNR_BIN" connect "$instance_id" 2>/dev/null || true)"
  PROBE_RAW="$raw" python3 - <<'PY'
import json
import os
import re
text = os.environ.get("PROBE_RAW", "")
text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
for line in reversed(text.splitlines()):
    candidate = line.strip()
    if not candidate.startswith("{"):
        continue
    try:
        json.loads(candidate)
    except Exception:
        continue
    print(candidate)
    raise SystemExit(0)
raise SystemExit(1)
PY
}

report="$OUT_DIR/reap_report.jsonl"
: > "$report"

for id in "${INSTANCE_IDS[@]}"; do
  status="$(get_status_field "$id" status 2>/dev/null || true)"
  gpu="$(get_status_field "$id" gpu 2>/dev/null || true)"
  mode="$(get_status_field "$id" mode 2>/dev/null || true)"
  uuid="$(get_status_field "$id" uuid 2>/dev/null || true)"
  decision="keep"
  reason=""
  probe_json=""

  if [[ -z "$status" ]]; then
    decision="skip"
    reason="not_found_in_tnr_status"
  elif [[ "$status" != "RUNNING" ]]; then
    decision="skip"
    reason="status=$status"
  elif [[ "$mode" = "production" && "$ALLOW_PRODUCTION" != "1" ]]; then
    decision="keep"
    reason="production_instance_skipped"
  else
    if ! probe_json="$(probe_instance "$id" 2>/dev/null)"; then
      decision="keep"
      reason="probe_failed"
    else
      live_count="$(python3 -c 'import json,sys; print(json.load(sys.stdin).get("live_count", 0))' <<<"$probe_json")"
      prefix_count="$(python3 -c 'import json,sys; print(len(json.load(sys.stdin).get("prefixes", [])))' <<<"$probe_json")"
      failure_count="$(python3 -c 'import json,sys; print(json.load(sys.stdin).get("failure_count", 0))' <<<"$probe_json")"
      if [[ "$live_count" -gt 0 ]]; then
        decision="keep"
        reason="live_processes=$live_count"
      elif [[ "$prefix_count" -eq 0 && "$DELETE_WITHOUT_PREFIXES" != "1" ]]; then
        decision="keep"
        reason="idle_but_no_b2_prefix_evidence"
      elif [[ "$failure_count" -gt 0 && "$DELETE_FAILED" != "1" ]]; then
        decision="keep"
        reason="idle_with_failures_requires_DELETE_FAILED=1"
      else
        prefixes=()
        while IFS= read -r prefix; do
          [[ -n "$prefix" ]] && prefixes+=("$prefix")
        done < <(python3 -c 'import json,sys; [print(p) for p in json.load(sys.stdin).get("prefixes", [])]' <<<"$probe_json")
        missing=()
        for prefix in "${prefixes[@]}"; do
          if ! b2_has_required_outputs "$prefix"; then
            missing+=("$prefix")
          fi
        done
        if [[ "${#missing[@]}" -gt 0 ]]; then
          decision="keep"
          reason="missing_b2_outputs=${missing[*]}"
        else
          decision="delete"
          reason="idle_and_b2_durable_prefixes=$prefix_count"
        fi
      fi
    fi
  fi

  if [[ "$decision" = "delete" && "$DELETE" = "1" ]]; then
    "$TNR_BIN" delete "$id" --yes
    action="deleted"
  elif [[ "$decision" = "delete" ]]; then
    action="would_delete"
  else
    action="kept"
  fi

  python3 - "$report" "$id" "$uuid" "$status" "$mode" "$gpu" "$decision" "$action" "$reason" "$probe_json" <<'PY'
import json
import sys
from datetime import datetime, timezone
report, iid, uuid, status, mode, gpu, decision, action, reason, probe_json = sys.argv[1:]
probe = {}
if probe_json:
    try:
        probe = json.loads(probe_json)
    except Exception:
        probe = {"parse_error": True}
row = {
    "time": datetime.now(timezone.utc).isoformat(),
    "instance_id": iid,
    "uuid": uuid,
    "status": status,
    "mode": mode,
    "gpu": gpu,
    "decision": decision,
    "action": action,
    "reason": reason,
    "probe": probe,
}
with open(report, "a", encoding="utf-8") as handle:
    handle.write(json.dumps(row, sort_keys=True) + "\n")
print(json.dumps({k: row[k] for k in ("instance_id", "uuid", "action", "reason")}, sort_keys=True))
PY
done

echo "Reaper report: $report"
