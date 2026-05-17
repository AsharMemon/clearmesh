#!/usr/bin/env bash
# Launch one Thunder worker that turns a local Objaverse++ annotation shard into
# a lean FACE strict-token corpus archive. Use many copies of this worker for
# embarrassingly parallel 500k-scale corpus prep.
set -euo pipefail

TNR_BIN="${TNR_BIN:-/Users/Ashar/.tnr/bin/tnr}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

INSTANCE_ID="${1:-${THUNDER_INSTANCE_ID:-}}"
CREATE_INSTANCE="${CREATE_INSTANCE:-1}"
DELETE_ON_FAILURE="${DELETE_ON_FAILURE:-1}"
CREATED_INSTANCE=0

LOCAL_ANNOTATIONS_FILE="${LOCAL_ANNOTATIONS_FILE:-}"
if [ -z "$LOCAL_ANNOTATIONS_FILE" ]; then
  echo "Set LOCAL_ANNOTATIONS_FILE to a local shard JSONL." >&2
  exit 2
fi
if [ ! -f "$LOCAL_ANNOTATIONS_FILE" ]; then
  echo "Annotation shard not found: $LOCAL_ANNOTATIONS_FILE" >&2
  exit 2
fi

GPU="${GPU:-a6000}"
MODE="${MODE:-prototyping}"
VCPUS="${VCPUS:-8}"
PRIMARY_DISK="${PRIMARY_DISK:-200}"
TEMPLATE="${TEMPLATE:-base}"
WAIT_INTERVAL_SEC="${WAIT_INTERVAL_SEC:-10}"
WAIT_TIMEOUT_SEC="${WAIT_TIMEOUT_SEC:-1800}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%d_%H%M%S)_$(basename "$LOCAL_ANNOTATIONS_FILE" .jsonl)}"
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-$REPO_ROOT/.codex_outputs/face_corpus_shard_setup_$RUN_STAMP}"
REMOTE_REPO="${REMOTE_REPO:-/home/ubuntu/clearmesh}"
REMOTE_VENV="${REMOTE_VENV:-/home/ubuntu/clearmesh-data-venv}"
REMOTE_LOG="${REMOTE_LOG:-/tmp/clearmesh_face_corpus_shard.nohup.log}"
REMOTE_PID="${REMOTE_PID:-/tmp/clearmesh_face_corpus_shard.pid}"
REMOTE_LAB_ROOT="${REMOTE_LAB_ROOT:-/tmp/clearmesh_face_corpus_shard_$RUN_STAMP}"
REMOTE_ANNOTATIONS="${REMOTE_ANNOTATIONS:-$REMOTE_LAB_ROOT/source_annotations.jsonl}"
SYNC_HF_TOKEN="${SYNC_HF_TOKEN:-1}"

line_count="$(grep -cve '^\s*$' "$LOCAL_ANNOTATIONS_FILE" || true)"
SELECT_TARGET="${SELECT_TARGET:-$line_count}"
SCAN_LIMIT="${SCAN_LIMIT:-0}"
CURATION_TARGET="${CURATION_TARGET:-$SELECT_TARGET}"
MIN_QUALITY="${MIN_QUALITY:-2}"
OVERSAMPLE_FACTOR="${OVERSAMPLE_FACTOR:-1}"
SHUFFLE="${SHUFFLE:-0}"
SEED="${SEED:-303}"
DOWNLOAD_PROCESSES="${DOWNLOAD_PROCESSES:-16}"
DOWNLOAD_FALLBACK_PROCESSES="${DOWNLOAD_FALLBACK_PROCESSES:-1}"
DOWNLOAD_BATCH_SIZE="${DOWNLOAD_BATCH_SIZE:-50}"
DOWNLOAD_BATCH_TIMEOUT_SECONDS="${DOWNLOAD_BATCH_TIMEOUT_SECONDS:-600}"
DOWNLOAD_BATCH_RETRIES="${DOWNLOAD_BATCH_RETRIES:-2}"
DOWNLOAD_RETRY_SLEEP_SECONDS="${DOWNLOAD_RETRY_SLEEP_SECONDS:-15}"
DOWNLOAD_RATE_LIMIT_SLEEP_SECONDS="${DOWNLOAD_RATE_LIMIT_SLEEP_SECONDS:-600}"
TEXVERSE_DOWNLOAD_WORKERS="${TEXVERSE_DOWNLOAD_WORKERS:-$DOWNLOAD_PROCESSES}"
TEXVERSE_MAX_SIZE_MB="${TEXVERSE_MAX_SIZE_MB:-0}"
TEXVERSE_CLEANUP_CACHE_EACH="${TEXVERSE_CLEANUP_CACHE_EACH:-0}"
SOURCE_MIN_FACES="${SOURCE_MIN_FACES:-64}"
SOURCE_MAX_FACES="${SOURCE_MAX_FACES:-250000}"
TARGET_FACES="${TARGET_FACES:-512}"
TOKEN_MAX_FACES="${TOKEN_MAX_FACES:-512}"
POINT_SAMPLES="${POINT_SAMPLES:-8192}"
NUM_BINS="${NUM_BINS:-128}"
PAPER_WITHIN_FACE_ORDER="${PAPER_WITHIN_FACE_ORDER:-rotate_min_zyx}"
STRICT_ENGINE="${STRICT_ENGINE:-voxel_shell}"
FALLBACK="${FALLBACK:-convex_hull}"
VOXEL_RESOLUTION="${VOXEL_RESOLUTION:-64}"
MESH_VOXEL_MAX_FACES="${MESH_VOXEL_MAX_FACES:-5000}"
STRICT_TARGET_PROGRESS_EVERY="${STRICT_TARGET_PROGRESS_EVERY:-100}"
TEST_RATIO="${TEST_RATIO:-0.02}"
LEAN_ARCHIVE_PATH="${LEAN_ARCHIVE_PATH:-$REMOTE_LAB_ROOT/lean_face_corpus.tar.gz}"
START_B2_UPLOAD="${START_B2_UPLOAD:-0}"
B2_BUCKET="${B2_BUCKET:-clearmesh-pairs}"
B2_PREFIX="${B2_PREFIX:-face-corpora/poolA-shards/$(basename "$REMOTE_LAB_ROOT")}"
B2_UPLOAD_INTERVAL_SECONDS="${B2_UPLOAD_INTERVAL_SECONDS:-600}"
B2_UPLOAD_PID="${B2_UPLOAD_PID:-$REMOTE_LAB_ROOT/b2_upload.pid}"
B2_UPLOAD_LOG="${B2_UPLOAD_LOG:-$REMOTE_LAB_ROOT/b2_upload.log}"
REMOTE_B2_ENV="${REMOTE_B2_ENV:-$REMOTE_LAB_ROOT/.clearmesh_b2.env}"
UPLOAD_B2_ENV="${UPLOAD_B2_ENV:-1}"

if [ -z "${THUNDER_TOKEN:-}" ]; then
  echo "THUNDER_TOKEN is not set." >&2
  exit 1
fi
if [ ! -x "$TNR_BIN" ]; then
  echo "tnr binary not found or not executable: $TNR_BIN" >&2
  exit 1
fi
case "$MODE:$GPU" in
  production:a100|production:h100|prototyping:a6000|prototyping:a100|prototyping:h100)
    ;;
  *)
    echo "Unsupported Thunder mode/GPU '$MODE:$GPU'." >&2
    exit 5
    ;;
esac
mkdir -p "$DOWNLOAD_ROOT"

cleanup_instance() {
  local exit_code=$?
  if [ "$exit_code" -ne 0 ] && [ "$CREATED_INSTANCE" = "1" ] && [ "$DELETE_ON_FAILURE" = "1" ] && [ -n "$INSTANCE_ID" ]; then
    echo "Shard launcher failed with status $exit_code; deleting created Thunder instance $INSTANCE_ID." >&2
    "$TNR_BIN" delete "$INSTANCE_ID" --yes || true
  fi
}
trap cleanup_instance EXIT INT TERM HUP

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

if [ "$CREATE_INSTANCE" = "1" ]; then
  create_args=(create --gpu "$GPU" --mode "$MODE" --num-gpus 1 --primary-disk "$PRIMARY_DISK" --template "$TEMPLATE")
  if [ "$MODE" = "prototyping" ]; then
    create_args+=(--vcpus "$VCPUS")
  fi
  create_args+=(--yes --json)
  create_output="$($TNR_BIN "${create_args[@]}")" || {
    echo "Thunder create failed." >&2
    exit 3
  }
  redact_create_output "$create_output" > "$DOWNLOAD_ROOT/create.json"
  CREATED_INSTANCE=1
  if [ -z "$INSTANCE_ID" ]; then
    INSTANCE_ID="$(parse_create_id "$create_output")" || INSTANCE_ID=""
  fi
  echo "Created Thunder shard instance $INSTANCE_ID."
fi
if [ -z "$INSTANCE_ID" ]; then
  echo "Set THUNDER_INSTANCE_ID or CREATE_INSTANCE=1." >&2
  exit 4
fi

echo "Waiting for Thunder instance $INSTANCE_ID to RUNNING..."
running_seen=0
wait_deadline=$(( $(date +%s) + WAIT_TIMEOUT_SEC ))
while [ "$(date +%s)" -lt "$wait_deadline" ]; do
  status_json="$($TNR_BIN status --json || true)"
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
    running_seen=1
    break
  fi
  sleep "$WAIT_INTERVAL_SEC"
done
if [ "$running_seen" != "1" ]; then
  echo "Thunder instance $INSTANCE_ID did not reach RUNNING within ${WAIT_TIMEOUT_SEC}s." >&2
  exit 21
fi

echo "Syncing repo to shard worker $INSTANCE_ID..."
THUNDER_INSTANCE_ID="$INSTANCE_ID" "$REPO_ROOT/scripts/thunder/sync_repo.sh" "$INSTANCE_ID"

if [ "$SYNC_HF_TOKEN" = "1" ]; then
  if [ -n "${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}" ]; then
    echo "Syncing Hugging Face token to shard worker $INSTANCE_ID..."
    THUNDER_INSTANCE_ID="$INSTANCE_ID" "$REPO_ROOT/scripts/thunder/sync_hf_token.sh" "$INSTANCE_ID"
  else
    echo "HF_TOKEN/HUGGINGFACE_HUB_TOKEN not set locally; remote downloads may be rate limited." >&2
  fi
fi

printf 'mkdir -p %q\nexit\n' "$REMOTE_LAB_ROOT" | "$TNR_BIN" connect "$INSTANCE_ID"
"$TNR_BIN" scp "$LOCAL_ANNOTATIONS_FILE" "$INSTANCE_ID:$REMOTE_ANNOTATIONS"

if [ "$START_B2_UPLOAD" = "1" ] && [ "$UPLOAD_B2_ENV" = "1" ]; then
  b2_env_file="$(mktemp "$DOWNLOAD_ROOT/b2_env.XXXXXX")"
  {
    printf 'export B2_KEY_ID=%q\n' "${B2_KEY_ID:-${B2_KEYID:-${B2_APPLICATION_KEY_ID:-${BACKBLAZE_B2_KEY_ID:-}}}}"
    printf 'export B2_APP_KEY=%q\n' "${B2_APP_KEY:-${B2_APPKEY:-${B2_APPLICATION_KEY:-${BACKBLAZE_B2_APPLICATION_KEY:-${BACKBLAZE_B2_APP_KEY:-}}}}}"
    printf 'export B2_TOKEN=%q\n' "${B2_TOKEN:-}"
  } > "$b2_env_file"
  chmod 600 "$b2_env_file"
  "$TNR_BIN" scp "$b2_env_file" "$INSTANCE_ID:$REMOTE_B2_ENV"
  rm -f "$b2_env_file"
elif [ "$START_B2_UPLOAD" = "1" ]; then
  echo "UPLOAD_B2_ENV=0; using existing remote B2 env at $REMOTE_B2_ENV."
fi

remote_setup_log="$DOWNLOAD_ROOT/remote_setup_and_launch.log"
setup_status=0
cat <<REMOTE_SETUP | "$TNR_BIN" connect "$INSTANCE_ID" 2>&1 | tee "$remote_setup_log" || setup_status=$?
set -euo pipefail
REMOTE_REPO=$(printf '%q' "$REMOTE_REPO")
REMOTE_VENV=$(printf '%q' "$REMOTE_VENV")
REMOTE_LOG=$(printf '%q' "$REMOTE_LOG")
REMOTE_PID=$(printf '%q' "$REMOTE_PID")
REMOTE_LAB_ROOT=$(printf '%q' "$REMOTE_LAB_ROOT")
REMOTE_ANNOTATIONS=$(printf '%q' "$REMOTE_ANNOTATIONS")
cd "\$REMOTE_REPO"
if [ -f /home/ubuntu/.clearmesh_hf.env ]; then
  # shellcheck disable=SC1091
  source /home/ubuntu/.clearmesh_hf.env
fi
if [ ! -x "\$REMOTE_VENV/bin/python" ]; then
  python3 -m venv "\$REMOTE_VENV" || (sudo apt-get update && sudo apt-get install -y python3-venv && python3 -m venv "\$REMOTE_VENV")
fi
# shellcheck disable=SC1091
source "\$REMOTE_VENV/bin/activate"
python -m pip install -U pip setuptools wheel
python -m pip install -q -r requirements-data.txt pillow scipy
python - <<'PY'
import fast_simplification  # noqa: F401
import networkx  # noqa: F401
import objaverse  # noqa: F401
import scipy  # noqa: F401
import skimage  # noqa: F401
import trimesh  # noqa: F401
print('face_corpus_shard_python_deps_ok')
PY

export RUN_DIR="\$REMOTE_LAB_ROOT/corpus"
export ANNOTATIONS="\$REMOTE_ANNOTATIONS"
export SPLIT=train
export SELECT_TARGET=$(printf '%q' "$SELECT_TARGET")
export SCAN_LIMIT=$(printf '%q' "$SCAN_LIMIT")
export CURATION_TARGET=$(printf '%q' "$CURATION_TARGET")
export MIN_QUALITY=$(printf '%q' "$MIN_QUALITY")
export OVERSAMPLE_FACTOR=$(printf '%q' "$OVERSAMPLE_FACTOR")
export SHUFFLE=$(printf '%q' "$SHUFFLE")
export SEED=$(printf '%q' "$SEED")
export DOWNLOAD_PROCESSES=$(printf '%q' "$DOWNLOAD_PROCESSES")
export DOWNLOAD_FALLBACK_PROCESSES=$(printf '%q' "$DOWNLOAD_FALLBACK_PROCESSES")
export DOWNLOAD_BATCH_SIZE=$(printf '%q' "$DOWNLOAD_BATCH_SIZE")
export DOWNLOAD_BATCH_TIMEOUT_SECONDS=$(printf '%q' "$DOWNLOAD_BATCH_TIMEOUT_SECONDS")
export DOWNLOAD_BATCH_RETRIES=$(printf '%q' "$DOWNLOAD_BATCH_RETRIES")
export DOWNLOAD_RETRY_SLEEP_SECONDS=$(printf '%q' "$DOWNLOAD_RETRY_SLEEP_SECONDS")
export DOWNLOAD_RATE_LIMIT_SLEEP_SECONDS=$(printf '%q' "$DOWNLOAD_RATE_LIMIT_SLEEP_SECONDS")
export TEXVERSE_DOWNLOAD_WORKERS=$(printf '%q' "$TEXVERSE_DOWNLOAD_WORKERS")
export TEXVERSE_MAX_SIZE_MB=$(printf '%q' "$TEXVERSE_MAX_SIZE_MB")
export TEXVERSE_CLEANUP_CACHE_EACH=$(printf '%q' "$TEXVERSE_CLEANUP_CACHE_EACH")
export SOURCE_MIN_FACES=$(printf '%q' "$SOURCE_MIN_FACES")
export SOURCE_MAX_FACES=$(printf '%q' "$SOURCE_MAX_FACES")
export TARGET_FACES=$(printf '%q' "$TARGET_FACES")
export TOKEN_MAX_FACES=$(printf '%q' "$TOKEN_MAX_FACES")
export POINT_SAMPLES=$(printf '%q' "$POINT_SAMPLES")
export NUM_BINS=$(printf '%q' "$NUM_BINS")
export PAPER_WITHIN_FACE_ORDER=$(printf '%q' "$PAPER_WITHIN_FACE_ORDER")
export STRICT_ENGINE=$(printf '%q' "$STRICT_ENGINE")
export FALLBACK=$(printf '%q' "$FALLBACK")
export VOXEL_RESOLUTION=$(printf '%q' "$VOXEL_RESOLUTION")
export MESH_VOXEL_MAX_FACES=$(printf '%q' "$MESH_VOXEL_MAX_FACES")
export STRICT_TARGET_PROGRESS_EVERY=$(printf '%q' "$STRICT_TARGET_PROGRESS_EVERY")
export TEST_RATIO=$(printf '%q' "$TEST_RATIO")
export LEAN_ARCHIVE_PATH=$(printf '%q' "$LEAN_ARCHIVE_PATH")
export ARCHIVE_PATH=""
export START_B2_UPLOAD=$(printf '%q' "$START_B2_UPLOAD")
export B2_BUCKET=$(printf '%q' "$B2_BUCKET")
export B2_PREFIX=$(printf '%q' "$B2_PREFIX")
export B2_UPLOAD_INTERVAL_SECONDS=$(printf '%q' "$B2_UPLOAD_INTERVAL_SECONDS")
export B2_UPLOAD_PID=$(printf '%q' "$B2_UPLOAD_PID")
export B2_UPLOAD_LOG=$(printf '%q' "$B2_UPLOAD_LOG")
export B2_ENV_FILE=$(printf '%q' "$REMOTE_B2_ENV")

nohup bash scripts/thunder/face_objaversepp_corpus_pilot.sh > "\$REMOTE_LOG" 2>&1 &
echo \$! > "\$REMOTE_PID"
if [ "\$START_B2_UPLOAD" = "1" ]; then
  if [ -f "\$B2_ENV_FILE" ]; then
    # shellcheck disable=SC1090
    source "\$B2_ENV_FILE"
  else
    echo "b2_continuous_upload_skipped=missing_b2_env_file"
    START_B2_UPLOAD=0
  fi
fi
if [ "\$START_B2_UPLOAD" = "1" ]; then
  B2_KEY_ID="\${B2_KEY_ID:-}"
  B2_APP_KEY="\${B2_APP_KEY:-}"
  B2_TOKEN="\${B2_TOKEN:-}"
  if command -v rclone >/dev/null 2>&1 || (command -v curl >/dev/null 2>&1 && curl -fsSL https://rclone.org/install.sh | sudo bash >/dev/null 2>&1); then
    MODE=face_shard \
    LOCAL_ROOT="\$REMOTE_LAB_ROOT" \
    B2_BUCKET="\$B2_BUCKET" \
    B2_PREFIX="\$B2_PREFIX" \
    INTERVAL_SECONDS="\$B2_UPLOAD_INTERVAL_SECONDS" \
    B2_KEY_ID="\$B2_KEY_ID" \
    B2_APP_KEY="\$B2_APP_KEY" \
    B2_TOKEN="\$B2_TOKEN" \
      nohup bash scripts/thunder/b2_continuous_upload.sh > "\$B2_UPLOAD_LOG" 2>&1 &
    echo \$! > "\$B2_UPLOAD_PID"
  else
    echo "b2_continuous_upload_skipped=rclone_unavailable"
  fi
fi
echo "face_corpus_shard_pid=\$(cat "\$REMOTE_PID")"
echo "face_corpus_shard_log=\$REMOTE_LOG"
echo "face_corpus_shard_lab_root=\$REMOTE_LAB_ROOT"
if [ "\$START_B2_UPLOAD" = "1" ] && [ -f "\$B2_UPLOAD_PID" ]; then
  echo "face_corpus_shard_b2_upload_pid=\$(cat "\$B2_UPLOAD_PID")"
  echo "face_corpus_shard_b2_upload_log=\$B2_UPLOAD_LOG"
fi
echo CLEARMESH_FACE_CORPUS_SHARD_LAUNCHED
exit
REMOTE_SETUP

if [ "$setup_status" -ne 0 ]; then
  if ! grep -q 'CLEARMESH_FACE_CORPUS_SHARD_LAUNCHED' "$remote_setup_log"; then
    echo "Remote shard setup failed before launch marker." >&2
    exit "$setup_status"
  fi
fi
if ! grep -q 'CLEARMESH_FACE_CORPUS_SHARD_LAUNCHED' "$remote_setup_log"; then
  echo "Remote shard setup completed without launch marker; treating as failure." >&2
  exit 22
fi

cat > "$DOWNLOAD_ROOT/run_info.json" <<JSON
{
  "instance_id": "$INSTANCE_ID",
  "created_instance": $CREATED_INSTANCE,
  "gpu": "$GPU",
  "mode": "$MODE",
  "local_annotations_file": "$LOCAL_ANNOTATIONS_FILE",
  "remote_annotations": "$REMOTE_ANNOTATIONS",
  "remote_log": "$REMOTE_LOG",
  "remote_pid": "$REMOTE_PID",
  "remote_lab_root": "$REMOTE_LAB_ROOT",
  "download_root": "$DOWNLOAD_ROOT",
  "lean_archive_path": "$LEAN_ARCHIVE_PATH",
  "b2_upload_started": "$START_B2_UPLOAD",
  "b2_bucket": "$B2_BUCKET",
  "b2_prefix": "$B2_PREFIX",
  "b2_upload_log": "$B2_UPLOAD_LOG",
  "remote_b2_env": "$REMOTE_B2_ENV",
  "upload_b2_env": "$UPLOAD_B2_ENV",
  "select_target": $SELECT_TARGET,
  "curation_target": $CURATION_TARGET,
  "min_quality": $MIN_QUALITY,
  "download_processes": $DOWNLOAD_PROCESSES,
  "download_fallback_processes": $DOWNLOAD_FALLBACK_PROCESSES,
  "download_batch_size": $DOWNLOAD_BATCH_SIZE,
  "download_batch_timeout_seconds": $DOWNLOAD_BATCH_TIMEOUT_SECONDS,
  "download_batch_retries": $DOWNLOAD_BATCH_RETRIES,
  "download_retry_sleep_seconds": $DOWNLOAD_RETRY_SLEEP_SECONDS,
  "download_rate_limit_sleep_seconds": $DOWNLOAD_RATE_LIMIT_SLEEP_SECONDS,
  "texverse_download_workers": $TEXVERSE_DOWNLOAD_WORKERS,
  "texverse_max_size_mb": $TEXVERSE_MAX_SIZE_MB,
  "texverse_cleanup_cache_each": "$TEXVERSE_CLEANUP_CACHE_EACH",
  "source_min_faces": $SOURCE_MIN_FACES,
  "source_max_faces": $SOURCE_MAX_FACES,
  "target_faces": $TARGET_FACES,
  "token_max_faces": $TOKEN_MAX_FACES,
  "num_bins": $NUM_BINS,
  "point_samples": $POINT_SAMPLES,
  "paper_within_face_order": "$PAPER_WITHIN_FACE_ORDER",
  "strict_engine": "$STRICT_ENGINE",
  "fallback": "$FALLBACK",
  "test_ratio": $TEST_RATIO
}
JSON

echo "FACE corpus shard launched on Thunder instance $INSTANCE_ID."
echo "Local setup logs: $DOWNLOAD_ROOT"
echo "Remote nohup log: $REMOTE_LOG"
echo "Remote lab root: $REMOTE_LAB_ROOT"
echo "Lean archive: $LEAN_ARCHIVE_PATH"
