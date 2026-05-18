#!/usr/bin/env bash
# Launch an incremental FACE-Q corpus merge/dedupe prep worker on Thunder.
#
# The worker watches B2 shard prefixes, downloads only completed lean shard
# archives, extracts strict token-pass corpora, builds a global deduped
# train/test split, leakage-checks it, packages it, uploads it back to B2, and
# repeats on an interval. This keeps data collection and production-scale
# training prep overlapping without using local laptop disk.
set -euo pipefail

TNR_BIN="${TNR_BIN:-/Users/Ashar/.tnr/bin/tnr}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

INSTANCE_ID="${THUNDER_INSTANCE_ID:-}"
CREATE_INSTANCE="${CREATE_INSTANCE:-1}"
GPU="${GPU:-a6000}"
NUM_GPUS="${NUM_GPUS:-1}"
MODE="${MODE:-prototyping}"
VCPUS="${VCPUS:-16}"
PRIMARY_DISK="${PRIMARY_DISK:-1000}"
TEMPLATE="${TEMPLATE:-base}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%d_%H%M%S)_faceq_rolling_corpus_prep}"
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-$REPO_ROOT/.codex_outputs/faceq_rolling_corpus_prep_$RUN_STAMP}"

REMOTE_REPO="${REMOTE_REPO:-/home/ubuntu/clearmesh}"
REMOTE_VENV="${REMOTE_VENV:-/home/ubuntu/clearmesh-data-venv}"
REMOTE_ROOT="${REMOTE_ROOT:-/tmp/clearmesh_faceq_rolling_corpus_prep_$RUN_STAMP}"
REMOTE_LOG="${REMOTE_LOG:-/tmp/clearmesh_faceq_rolling_corpus_prep.nohup.log}"
REMOTE_PID="${REMOTE_PID:-/tmp/clearmesh_faceq_rolling_corpus_prep.pid}"
REMOTE_B2_ENV="${REMOTE_B2_ENV:-$REMOTE_ROOT/.clearmesh_b2.env}"

B2_BUCKET="${B2_BUCKET:-clearmesh-pairs}"
B2_PREFIXES="${B2_PREFIXES:-face-corpora/poolA-shards face-corpora/texverse-small1500-shards}"
B2_OUTPUT_PREFIX="${B2_OUTPUT_PREFIX:-face-corpora/merged/faceq_rolling_400k_total_380k_train}"
MIN_UNIQUE_FOR_ARCHIVE="${MIN_UNIQUE_FOR_ARCHIVE:-50000}"
MIN_NEW_ARCHIVES="${MIN_NEW_ARCHIVES:-3}"
TARGET_UNIQUE="${TARGET_UNIQUE:-400000}"
TEST_RATIO="${TEST_RATIO:-0.02}"
SEED="${SEED:-303}"
POLL_INTERVAL_SECONDS="${POLL_INTERVAL_SECONDS:-300}"
RUN_ONCE="${RUN_ONCE:-0}"
COPY_MODE="${COPY_MODE:-hardlink}"
MAX_ARCHIVES="${MAX_ARCHIVES:-0}"
PACKAGE_FULL_ARCHIVE="${PACKAGE_FULL_ARCHIVE:-1}"
FAST_SHARD_ARCHIVE_LIST="${FAST_SHARD_ARCHIVE_LIST:-1}"

WAIT_INTERVAL_SEC="${WAIT_INTERVAL_SEC:-10}"
WAIT_TIMEOUT_SEC="${WAIT_TIMEOUT_SEC:-1800}"

if [[ -z "${THUNDER_TOKEN:-}" ]]; then
  echo "THUNDER_TOKEN is not set." >&2
  exit 1
fi
mkdir -p "$DOWNLOAD_ROOT"

parse_create_id() {
  CREATE_OUTPUT="$1" python3 - <<'PY'
import json
import os
import re

text = os.environ.get("CREATE_OUTPUT", "")
decoder = json.JSONDecoder()
for match in re.finditer(r"[\[{]", text):
    try:
        payload, _ = decoder.raw_decode(text[match.start():])
    except json.JSONDecodeError:
        continue
    items = payload if isinstance(payload, list) else [payload]
    for item in items:
        if isinstance(item, dict):
            for key in ("id", "identifier", "instance_id", "instanceId", "uuid"):
                if item.get(key) is not None:
                    print(item[key])
                    raise SystemExit(0)
raise SystemExit(1)
PY
}

redact_create_output() {
  CREATE_OUTPUT="$1" python3 - <<'PY'
import json
import os
import re
import sys

text = os.environ.get("CREATE_OUTPUT", "")
decoder = json.JSONDecoder()
for match in re.finditer(r"[\[{]", text):
    try:
        payload, _ = decoder.raw_decode(text[match.start():])
    except json.JSONDecodeError:
        continue
    items = payload if isinstance(payload, list) else [payload]
    for item in items:
        if isinstance(item, dict) and "key" in item:
            item["key"] = "[redacted]"
    json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    raise SystemExit(0)
sys.stdout.write(text)
PY
}

if [[ "$CREATE_INSTANCE" = "1" ]]; then
  create_args=(create --gpu "$GPU" --mode "$MODE" --num-gpus "$NUM_GPUS" --primary-disk "$PRIMARY_DISK" --template "$TEMPLATE" --yes --json)
  if [[ "$MODE" = "prototyping" ]]; then
    create_args+=(--vcpus "$VCPUS")
  fi
  create_output="$("$TNR_BIN" "${create_args[@]}")"
  redact_create_output "$create_output" > "$DOWNLOAD_ROOT/create.json"
  INSTANCE_ID="${INSTANCE_ID:-$(parse_create_id "$create_output")}"
  echo "Created Thunder rolling prep instance $INSTANCE_ID."
fi
if [[ -z "$INSTANCE_ID" ]]; then
  echo "Set THUNDER_INSTANCE_ID or CREATE_INSTANCE=1." >&2
  exit 2
fi

echo "Waiting for Thunder instance $INSTANCE_ID to RUNNING..."
deadline=$(( $(date +%s) + WAIT_TIMEOUT_SEC ))
while [[ "$(date +%s)" -lt "$deadline" ]]; do
  status_json="$("$TNR_BIN" status --json || true)"
  printf '%s\n' "$status_json" > "$DOWNLOAD_ROOT/status.latest.json"
  if python3 - "$INSTANCE_ID" "$DOWNLOAD_ROOT/status.latest.json" <<'PY'
import json
import sys
from pathlib import Path

target = str(sys.argv[1])
text = Path(sys.argv[2]).read_text(errors="ignore")
start = text.find("[")
data = json.loads(text[start:]) if start >= 0 else []
for item in data:
    if str(item.get("id")) == target and item.get("status") == "RUNNING":
        raise SystemExit(0)
raise SystemExit(1)
PY
  then
    break
  fi
  sleep "$WAIT_INTERVAL_SEC"
done

echo "Syncing repo to rolling prep worker..."
THUNDER_INSTANCE_ID="$INSTANCE_ID" "$REPO_ROOT/scripts/thunder/sync_repo.sh" "$INSTANCE_ID"

bootstrap_log="$DOWNLOAD_ROOT/bootstrap.log"
bootstrap_script="$DOWNLOAD_ROOT/remote_bootstrap.sh"
cat > "$bootstrap_script" <<REMOTE_BOOTSTRAP
#!/usr/bin/env bash
set -euo pipefail
cd $(printf '%q' "$REMOTE_REPO")
if ! command -v rclone >/dev/null 2>&1; then
  tmp=\$(mktemp -d)
  cd "\$tmp"
  curl -fsSLO https://downloads.rclone.org/rclone-current-linux-amd64.zip
  python3 - <<'PY'
import zipfile
zipfile.ZipFile("rclone-current-linux-amd64.zip").extractall(".")
PY
  sudo install -m 755 rclone-*-linux-amd64/rclone /usr/local/bin/rclone
  cd /
  rm -rf "\$tmp"
fi
if [ ! -x $(printf '%q' "$REMOTE_VENV/bin/python") ]; then
  python3 -m venv $(printf '%q' "$REMOTE_VENV") || (sudo apt-get update && sudo apt-get install -y python3-venv && python3 -m venv $(printf '%q' "$REMOTE_VENV"))
fi
source $(printf '%q' "$REMOTE_VENV/bin/activate")
python -m pip install -U pip setuptools wheel
python -m pip install -q numpy trimesh
echo CLEARMESH_ROLLING_PREP_BOOTSTRAP_OK
REMOTE_BOOTSTRAP
"$TNR_BIN" scp "$bootstrap_script" "$INSTANCE_ID:/tmp/clearmesh_faceq_rolling_corpus_prep_bootstrap.sh"
printf 'bash /tmp/clearmesh_faceq_rolling_corpus_prep_bootstrap.sh\nexit\n' \
  | "$TNR_BIN" connect "$INSTANCE_ID" 2>&1 | tee "$bootstrap_log"
grep -q CLEARMESH_ROLLING_PREP_BOOTSTRAP_OK "$bootstrap_log"

b2_env_file="$(mktemp "$DOWNLOAD_ROOT/b2_env.XXXXXX")"
{
  printf 'export B2_KEY_ID=%q\n' "${B2_KEY_ID:-${B2_KEYID:-${B2_APPLICATION_KEY_ID:-${BACKBLAZE_B2_KEY_ID:-}}}}"
  printf 'export B2_APP_KEY=%q\n' "${B2_APP_KEY:-${B2_APPKEY:-${B2_APPLICATION_KEY:-${BACKBLAZE_B2_APPLICATION_KEY:-${BACKBLAZE_B2_APP_KEY:-}}}}}"
  printf 'export B2_TOKEN=%q\n' "${B2_TOKEN:-}"
} > "$b2_env_file"
chmod 600 "$b2_env_file"
printf 'mkdir -p %q\nexit\n' "$REMOTE_ROOT" | "$TNR_BIN" connect "$INSTANCE_ID" >/dev/null
"$TNR_BIN" scp "$b2_env_file" "$INSTANCE_ID:$REMOTE_B2_ENV"
rm -f "$b2_env_file"

remote_script="$DOWNLOAD_ROOT/remote_faceq_rolling_corpus_prep.sh"
cat > "$remote_script" <<'REMOTE'
#!/usr/bin/env bash
set -euo pipefail

LOG_TIME() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }
status() {
  local event="$1" detail="${2:-}"
  python3 - "$STATUS_JSONL" "$event" "$detail" <<'PY'
import json
import sys
import time

path, event, detail = sys.argv[1:4]
with open(path, "a", encoding="utf-8") as handle:
    handle.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "event": event, "detail": detail}, sort_keys=True) + "\n")
PY
}

resolve_b2_env() {
  source "$B2_ENV"
  if [[ -z "${B2_KEY_ID:-}" || -z "${B2_APP_KEY:-}" ]] && [[ -n "${B2_TOKEN:-}" ]]; then
    parsed_b2="$(python3 - <<'PY'
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
    if [[ -n "$parsed_b2" ]]; then
      export B2_KEY_ID="${B2_KEY_ID:-$(printf '%s\n' "$parsed_b2" | sed -n '1p')}"
      export B2_APP_KEY="${B2_APP_KEY:-$(printf '%s\n' "$parsed_b2" | sed -n '2p')}"
    fi
  fi
  if [[ -n "${B2_KEY_ID:-}" && -z "${B2_APP_KEY:-}" && -n "${B2_TOKEN:-}" ]]; then
    case "$B2_TOKEN" in
      \{*|*:*) ;;
      *) export B2_APP_KEY="$B2_TOKEN" ;;
    esac
  fi
  export RCLONE_CONFIG_B2ENV_TYPE=b2
  export RCLONE_CONFIG_B2ENV_ACCOUNT="$B2_KEY_ID"
  export RCLONE_CONFIG_B2ENV_KEY="$B2_APP_KEY"
}

list_completed_archives() {
  : > "$ARCHIVE_LIST_TMP"
  for prefix in $B2_PREFIXES; do
    if [[ "$FAST_SHARD_ARCHIVE_LIST" = "1" ]]; then
      # Shard outputs use a regular layout:
      #   <prefix>/shardXXXX/lean_face_corpus.tar.gz
      # Listing only first-level shard directories is much faster than a
      # recursive walk over every uploaded report/mesh/metadata file.
      rclone lsf "b2env:$B2_BUCKET/$prefix" --dirs-only 2>/dev/null \
        | awk '/^shard[0-9]+\/$/ {print p "/" $0 "lean_face_corpus.tar.gz"}' p="$prefix" \
        >> "$ARCHIVE_LIST_TMP" || true
    else
      rclone lsf "b2env:$B2_BUCKET/$prefix" --recursive --files-only 2>/dev/null \
        | awk '/lean_face_corpus\.tar\.gz$/ {print p "/" $0}' p="$prefix" >> "$ARCHIVE_LIST_TMP" || true
    fi
  done
  sort -u "$ARCHIVE_LIST_TMP" > "$ARCHIVE_LIST"
}

download_new_archives() {
  local new_count=0
  mkdir -p "$ARCHIVES_DIR" "$EXTRACT_DIR" "$MANIFEST_INPUTS_DIR"
  while IFS= read -r rel; do
    [[ -n "$rel" ]] || continue
    local safe archive_path marker extract_path
    safe="$(printf '%s' "$rel" | sed -E 's#[^A-Za-z0-9_.-]+#__#g')"
    archive_path="$ARCHIVES_DIR/$safe"
    marker="$EXTRACT_DIR/$safe.extracted"
    extract_path="$EXTRACT_DIR/$safe"
    if [[ ! -f "$marker" ]]; then
      status download_archive "$rel"
      if ! rclone copyto "b2env:$B2_BUCKET/$rel" "$archive_path" --stats 30s; then
        status archive_copy_failed "$rel"
        rm -f "$archive_path"
        continue
      fi
      if [[ ! -s "$archive_path" ]]; then
        status archive_missing_or_empty "$rel"
        rm -f "$archive_path"
        continue
      fi
      rm -rf "$extract_path"
      mkdir -p "$extract_path"
      if ! tar -xzf "$archive_path" -C "$extract_path"; then
        status archive_extract_failed "$rel"
        rm -rf "$extract_path" "$archive_path"
        continue
      fi
      touch "$marker"
      new_count=$((new_count + 1))
    fi
  done < "$ARCHIVE_LIST"
  echo "$new_count"
}

build_input_list() {
  : > "$INPUT_LIST"
  find "$EXTRACT_DIR" -type f \( \
      -path '*/split_pass/train/manifest.jsonl' \
      -o -path '*/split_pass/test/manifest.jsonl' \
    \) -print | sort > "$INPUT_LIST"
}

build_snapshot() {
  local manifest_count source_archive_count unique_count snapshot_name snapshot_dir dedupe_dir archive output_prefix
  manifest_count="$(wc -l < "$INPUT_LIST" | tr -d ' ')"
  source_archive_count="$(find "$EXTRACT_DIR" -name '*.extracted' -type f | wc -l | tr -d ' ')"
  if [[ "$manifest_count" -eq 0 ]]; then
    status snapshot_skipped "no_extracted_archives"
    return 0
  fi
  snapshot_name="snapshot_$(date -u +%Y%m%dT%H%M%SZ)_archives${source_archive_count}_manifests${manifest_count}"
  snapshot_dir="$SNAPSHOTS_DIR/$snapshot_name/merged"
  dedupe_dir="$SNAPSHOTS_DIR/$snapshot_name/split_dedup_tokenhash"
  mkdir -p "$SNAPSHOTS_DIR/$snapshot_name"
  status merge_started "snapshot=$snapshot_name archives=$source_archive_count manifests=$manifest_count"
  python scripts/research/merge_face_corpus_shards.py \
    --input-list "$INPUT_LIST" \
    --output-dir "$snapshot_dir" \
    --source manifest \
    --copy-mode "$COPY_MODE" \
    --no-dedupe \
    --test-ratio "$TEST_RATIO" \
    --seed "$SEED" \
    > "$SNAPSHOTS_DIR/$snapshot_name/merge.log" 2>&1
  python scripts/research/dedupe_face_token_split.py \
    --manifest "$snapshot_dir/tokens_pass/manifest.jsonl" \
    --output-dir "$dedupe_dir" \
    --test-ratio "$TEST_RATIO" \
    --copy-mode "$COPY_MODE" \
    --seed "$SEED" \
    > "$SNAPSHOTS_DIR/$snapshot_name/dedupe.log" 2>&1
  python scripts/research/check_face_token_leakage.py \
    --train-manifest "$dedupe_dir/train/manifest.jsonl" \
    --test-manifest "$dedupe_dir/test/manifest.jsonl" \
    --identity-limit 64 \
    --output "$SNAPSHOTS_DIR/$snapshot_name/leakage_check.json" \
    > "$SNAPSHOTS_DIR/$snapshot_name/leakage.log" 2>&1
  unique_count="$(python3 - "$dedupe_dir/split_summary.json" <<'PY'
import json
import sys
from pathlib import Path
summary = json.loads(Path(sys.argv[1]).read_text())
print(summary["unique_token_hashes"])
PY
)"
  python3 - "$ARCHIVE_LIST" "$INPUT_LIST" "$SNAPSHOTS_DIR/$snapshot_name" "$unique_count" "$TARGET_UNIQUE" <<'PY'
import json
import sys
from pathlib import Path

archive_list, input_list, snapshot_dir, unique_count, target_unique = sys.argv[1:]
summary = {
    "archive_count": sum(1 for line in Path(archive_list).read_text().splitlines() if line.strip()),
    "extracted_input_count": sum(1 for line in Path(input_list).read_text().splitlines() if line.strip()),
    "unique_token_hashes": int(unique_count),
    "target_unique": int(target_unique),
    "target_reached": int(unique_count) >= int(target_unique),
}
Path(snapshot_dir, "rolling_prep_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
PY
  output_prefix="$B2_OUTPUT_PREFIX/$snapshot_name"
  status upload_started "$output_prefix unique=$unique_count"
  rclone copy "$SNAPSHOTS_DIR/$snapshot_name" "b2env:$B2_BUCKET/$output_prefix" \
    --include 'rolling_prep_summary.json' \
    --include 'leakage_check.json' \
    --include '*.log' \
    --include 'split_dedup_tokenhash/split_summary.json' \
    --include 'split_dedup_tokenhash/train/manifest.jsonl' \
    --include 'split_dedup_tokenhash/test/manifest.jsonl' \
    --include 'merged/merge_summary.json' \
    --exclude '*' \
    --stats 30s || true
  if [[ "$PACKAGE_FULL_ARCHIVE" = "1" ]]; then
    archive="$SNAPSHOTS_DIR/${snapshot_name}_faceq_dedup_split.tar.gz"
    status package_started "$archive"
    tar -C "$SNAPSHOTS_DIR/$snapshot_name" -czf "$archive" \
      split_dedup_tokenhash \
      leakage_check.json \
      rolling_prep_summary.json \
      merge.log \
      dedupe.log \
      leakage.log
    rclone copyto "$archive" "b2env:$B2_BUCKET/$output_prefix/faceq_merged_dedup_split.tar.gz" --stats 30s
  else
    status package_skipped "PACKAGE_FULL_ARCHIVE=0 snapshot=$snapshot_name"
  fi
  rclone copyto "$SNAPSHOTS_DIR/$snapshot_name/rolling_prep_summary.json" "b2env:$B2_BUCKET/$B2_OUTPUT_PREFIX/latest_rolling_prep_summary.json" --stats 30s
  rclone copyto "$SNAPSHOTS_DIR/$snapshot_name/leakage_check.json" "b2env:$B2_BUCKET/$B2_OUTPUT_PREFIX/latest_leakage_check.json" --stats 30s
  printf '%s\n' "$output_prefix" > "$ROOT/latest_snapshot_prefix.txt"
  status snapshot_complete "snapshot=$snapshot_name unique=$unique_count package_full_archive=$PACKAGE_FULL_ARCHIVE output=$output_prefix"
}

main_loop() {
  resolve_b2_env
  rclone lsf "b2env:$B2_BUCKET" >/dev/null
  source "$VENV/bin/activate"
  cd "$REPO"
  mkdir -p "$ROOT" "$ARCHIVES_DIR" "$EXTRACT_DIR" "$SNAPSHOTS_DIR"
  status worker_started "prefixes=$B2_PREFIXES target_unique=$TARGET_UNIQUE"
  while true; do
    list_completed_archives
    local archive_count new_count should_build last_archive_count
    archive_count="$(wc -l < "$ARCHIVE_LIST" | tr -d ' ')"
    new_count="$(download_new_archives)"
    build_input_list
    last_archive_count="$(cat "$ROOT/last_snapshot_archive_count" 2>/dev/null || echo 0)"
    should_build=0
    if [[ "$archive_count" -ge "$MAX_ARCHIVES" && "$MAX_ARCHIVES" -gt 0 ]]; then
      should_build=1
    fi
    if [[ "$archive_count" -ge "$MIN_NEW_ARCHIVES" && $(( archive_count - last_archive_count )) -ge "$MIN_NEW_ARCHIVES" ]]; then
      should_build=1
    fi
    status poll "archives=$archive_count new_downloads=$new_count last_snapshot_archives=$last_archive_count should_build=$should_build"
    if [[ "$should_build" = "1" ]]; then
      build_snapshot
      printf '%s\n' "$archive_count" > "$ROOT/last_snapshot_archive_count"
    fi
    if [[ "$RUN_ONCE" = "1" ]]; then
      break
    fi
    sleep "$POLL_INTERVAL_SECONDS"
  done
}

main_loop
REMOTE

python3 - "$remote_script" \
  "$REMOTE_ROOT" "$REMOTE_REPO" "$REMOTE_VENV" "$REMOTE_B2_ENV" \
  "$B2_BUCKET" "$B2_PREFIXES" "$B2_OUTPUT_PREFIX" "$MIN_UNIQUE_FOR_ARCHIVE" \
  "$MIN_NEW_ARCHIVES" "$TARGET_UNIQUE" "$TEST_RATIO" "$SEED" \
  "$POLL_INTERVAL_SECONDS" "$RUN_ONCE" "$COPY_MODE" "$MAX_ARCHIVES" \
  "$PACKAGE_FULL_ARCHIVE" "$FAST_SHARD_ARCHIVE_LIST" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
values = {
    "ROOT": sys.argv[2],
    "REPO": sys.argv[3],
    "VENV": sys.argv[4],
    "B2_ENV": sys.argv[5],
    "B2_BUCKET": sys.argv[6],
    "B2_PREFIXES": sys.argv[7],
    "B2_OUTPUT_PREFIX": sys.argv[8],
    "MIN_UNIQUE_FOR_ARCHIVE": sys.argv[9],
    "MIN_NEW_ARCHIVES": sys.argv[10],
    "TARGET_UNIQUE": sys.argv[11],
    "TEST_RATIO": sys.argv[12],
    "SEED": sys.argv[13],
    "POLL_INTERVAL_SECONDS": sys.argv[14],
    "RUN_ONCE": sys.argv[15],
    "COPY_MODE": sys.argv[16],
    "MAX_ARCHIVES": sys.argv[17],
    "PACKAGE_FULL_ARCHIVE": sys.argv[18],
    "FAST_SHARD_ARCHIVE_LIST": sys.argv[19],
}
text = path.read_text()
prefix = "\n".join(f"{key}={value!r}" for key, value in values.items())
prefix += "\nSTATUS_JSONL=\"$ROOT/status.jsonl\"\nARCHIVES_DIR=\"$ROOT/archives\"\nEXTRACT_DIR=\"$ROOT/extracted\"\nSNAPSHOTS_DIR=\"$ROOT/snapshots\"\nARCHIVE_LIST=\"$ROOT/completed_archives.txt\"\nARCHIVE_LIST_TMP=\"$ROOT/completed_archives.tmp\"\nINPUT_LIST=\"$ROOT/merge_inputs.txt\"\nMANIFEST_INPUTS_DIR=\"$ROOT/manifest_inputs\"\n"
if text.startswith("#!"):
    first, rest = text.split("\n", 1)
    text = first + "\n" + prefix + rest
else:
    text = "#!/usr/bin/env bash\n" + prefix + text
path.write_text(text)
PY

launch_script="$DOWNLOAD_ROOT/remote_launch.sh"
cat > "$launch_script" <<REMOTE_LAUNCH
#!/usr/bin/env bash
set -euo pipefail
chmod +x /tmp/clearmesh_faceq_rolling_corpus_prep.sh
nohup /tmp/clearmesh_faceq_rolling_corpus_prep.sh > $(printf '%q' "$REMOTE_LOG") 2>&1 &
echo \$! > $(printf '%q' "$REMOTE_PID")
echo CLEARMESH_FACEQ_ROLLING_CORPUS_PREP_LAUNCHED pid=\$(cat $(printf '%q' "$REMOTE_PID")) root=$(printf '%q' "$REMOTE_ROOT") log=$(printf '%q' "$REMOTE_LOG")
REMOTE_LAUNCH
"$TNR_BIN" scp "$remote_script" "$INSTANCE_ID:/tmp/clearmesh_faceq_rolling_corpus_prep.sh"
"$TNR_BIN" scp "$launch_script" "$INSTANCE_ID:/tmp/clearmesh_faceq_rolling_corpus_prep_launch.sh"
printf 'bash /tmp/clearmesh_faceq_rolling_corpus_prep_launch.sh\nexit\n' \
  | "$TNR_BIN" connect "$INSTANCE_ID" | tee "$DOWNLOAD_ROOT/remote_launch.log"

cat > "$DOWNLOAD_ROOT/run_info.json" <<JSON
{
  "instance_id": "$INSTANCE_ID",
  "gpu": "$GPU",
  "num_gpus": $NUM_GPUS,
  "mode": "$MODE",
  "remote_root": "$REMOTE_ROOT",
  "remote_log": "$REMOTE_LOG",
  "remote_pid": "$REMOTE_PID",
  "b2_bucket": "$B2_BUCKET",
  "b2_prefixes": "$B2_PREFIXES",
  "b2_output_prefix": "$B2_OUTPUT_PREFIX",
  "target_unique": $TARGET_UNIQUE,
  "min_new_archives": $MIN_NEW_ARCHIVES,
  "poll_interval_seconds": $POLL_INTERVAL_SECONDS,
  "copy_mode": "$COPY_MODE",
  "primary_disk": $PRIMARY_DISK
}
JSON

echo "FACE-Q rolling corpus prep launched."
echo "Run info: $DOWNLOAD_ROOT/run_info.json"
