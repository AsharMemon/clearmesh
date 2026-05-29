#!/usr/bin/env bash
# Run a queue of FACE corpus shards on one already-provisioned Thunder worker.
#
# This script is meant to live on the GPU instance. It keeps the worker busy
# without a Codex-side automation: wait for any current shard to finish, preserve
# it to B2, clean only the heavy local payload, then process queued shards one by
# one using the same strict FACE corpus pipeline.
set -euo pipefail

QUEUE_SHARD_IDS="${QUEUE_SHARD_IDS:?QUEUE_SHARD_IDS is required, e.g. '0033 0038 0043'}"
QUEUE_SOURCE_DIR="${QUEUE_SOURCE_DIR:-/tmp/clearmesh_face_corpus_queue}"
RUN_STAMP_PREFIX="${RUN_STAMP_PREFIX:-$(date -u +%Y%m%d)_poolA_queue}"
LAB_ROOT_PREFIX="${LAB_ROOT_PREFIX:-/tmp/clearmesh_face_corpus_shard}"
QUEUE_LOG_ROOT="${QUEUE_LOG_ROOT:-/tmp/clearmesh_face_corpus_queue_logs}"
QUEUE_STATUS="${QUEUE_STATUS:-$QUEUE_LOG_ROOT/status.jsonl}"
CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-0}"
CLEANUP_FAILED_ROOT_ON_FAILURE="${CLEANUP_FAILED_ROOT_ON_FAILURE:-0}"
DATA_LANE="${DATA_LANE:-unspecified}"
SOURCE_POOL_NAME="${SOURCE_POOL_NAME:-}"
REMOTE_VENV="${REMOTE_VENV:-/home/ubuntu/clearmesh-data-venv}"
HF_ENV_FILE="${HF_ENV_FILE:-/home/ubuntu/.clearmesh_hf.env}"

WAIT_FOR_PID_FILE="${WAIT_FOR_PID_FILE:-}"
WAIT_POLL_SECONDS="${WAIT_POLL_SECONDS:-60}"
PREQUEUE_COMPLETED_ROOT="${PREQUEUE_COMPLETED_ROOT:-}"
PREQUEUE_B2_PREFIX="${PREQUEUE_B2_PREFIX:-}"
PREQUEUE_B2_UPLOAD_PID_FILE="${PREQUEUE_B2_UPLOAD_PID_FILE:-}"

B2_ENV_FILE="${B2_ENV_FILE:-}"
START_B2_UPLOAD="${START_B2_UPLOAD:-1}"
B2_BUCKET="${B2_BUCKET:-clearmesh-pairs}"
B2_PREFIX_ROOT="${B2_PREFIX_ROOT:-face-corpora/poolA-shards}"
B2_UPLOAD_INTERVAL_SECONDS="${B2_UPLOAD_INTERVAL_SECONDS:-600}"

SELECT_TARGET="${SELECT_TARGET:-auto}"
SOURCE_KIND="${SOURCE_KIND:-objaversepp}"
SCAN_LIMIT="${SCAN_LIMIT:-0}"
CURATION_TARGET="${CURATION_TARGET:-auto}"
MIN_QUALITY="${MIN_QUALITY:-2}"
OVERSAMPLE_FACTOR="${OVERSAMPLE_FACTOR:-1}"
SHUFFLE="${SHUFFLE:-0}"
SEED="${SEED:-303}"
DOWNLOAD_PROCESSES="${DOWNLOAD_PROCESSES:-8}"
DOWNLOAD_FALLBACK_PROCESSES="${DOWNLOAD_FALLBACK_PROCESSES:-1}"
DOWNLOAD_BATCH_SIZE="${DOWNLOAD_BATCH_SIZE:-50}"
DOWNLOAD_BATCH_TIMEOUT_SECONDS="${DOWNLOAD_BATCH_TIMEOUT_SECONDS:-600}"
DOWNLOAD_BATCH_RETRIES="${DOWNLOAD_BATCH_RETRIES:-2}"
DOWNLOAD_RETRY_SLEEP_SECONDS="${DOWNLOAD_RETRY_SLEEP_SECONDS:-15}"
DOWNLOAD_RATE_LIMIT_SLEEP_SECONDS="${DOWNLOAD_RATE_LIMIT_SLEEP_SECONDS:-600}"
TEXVERSE_DOWNLOAD_WORKERS="${TEXVERSE_DOWNLOAD_WORKERS:-$DOWNLOAD_PROCESSES}"
TEXVERSE_MAX_SIZE_MB="${TEXVERSE_MAX_SIZE_MB:-0}"
TEXVERSE_CLEANUP_CACHE_EACH="${TEXVERSE_CLEANUP_CACHE_EACH:-0}"
OBJAVERSEXL_INCLUDE_SOURCES="${OBJAVERSEXL_INCLUDE_SOURCES:-}"
OBJAVERSEXL_EXCLUDE_SOURCES="${OBJAVERSEXL_EXCLUDE_SOURCES:-}"
OBJAVERSEXL_SAVE_REPO_FORMAT="${OBJAVERSEXL_SAVE_REPO_FORMAT:-zip}"
OBJAVERSEXL_MAX_DOWNLOAD_DIR_GB="${OBJAVERSEXL_MAX_DOWNLOAD_DIR_GB:-0}"
EXCLUDE_SOURCE_IDS="${EXCLUDE_SOURCE_IDS:-}"
B2_QUEUE_CLAIMS="${B2_QUEUE_CLAIMS:-1}"
B2_QUEUE_CLAIM_TTL_SECONDS="${B2_QUEUE_CLAIM_TTL_SECONDS:-21600}"
SOURCE_MIN_FACES="${SOURCE_MIN_FACES:-64}"
SOURCE_MAX_FACES="${SOURCE_MAX_FACES:-250000}"
MAX_FILE_MB="${MAX_FILE_MB:-256}"
MAX_COMPONENTS="${MAX_COMPONENTS:-48}"
MIN_LARGEST_COMPONENT_AREA_RATIO="${MIN_LARGEST_COMPONENT_AREA_RATIO:-0.60}"
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
STRICT_TARGET_WORKERS="${STRICT_TARGET_WORKERS:-2}"
TOKEN_BUILD_WORKERS="${TOKEN_BUILD_WORKERS:-1}"
FORCE_MIN_POSTPROCESS_WORKERS="${FORCE_MIN_POSTPROCESS_WORKERS:-0}"
TARGET_MAX_OUTPUT_COMPONENTS="${TARGET_MAX_OUTPUT_COMPONENTS:-1}"
TARGET_MAX_BOUNDARY_LOOPS="${TARGET_MAX_BOUNDARY_LOOPS:-0}"
TARGET_MAX_NONMANIFOLD_EDGES="${TARGET_MAX_NONMANIFOLD_EDGES:-0}"
TARGET_REQUIRE_WATERTIGHT="${TARGET_REQUIRE_WATERTIGHT:-1}"
GATE_PROFILE="${GATE_PROFILE:-strict}"
GATE_MAX_BOUNDARY_EDGES="${GATE_MAX_BOUNDARY_EDGES:-}"
GATE_MAX_NONMANIFOLD_EDGES="${GATE_MAX_NONMANIFOLD_EDGES:-}"
GATE_MIN_EDGE_PAIRING_RATIO="${GATE_MIN_EDGE_PAIRING_RATIO:-}"
TEST_RATIO="${TEST_RATIO:-0.02}"

cd /home/ubuntu/clearmesh
mkdir -p "$QUEUE_SOURCE_DIR" "$QUEUE_LOG_ROOT"

if [[ -x "$REMOTE_VENV/bin/python" ]]; then
  # Match the one-shot shard launcher environment; the base image does not
  # necessarily have objaverse/trimesh/skimage installed in system Python.
  # shellcheck disable=SC1091
  source "$REMOTE_VENV/bin/activate"
else
  echo "Missing FACE corpus venv at $REMOTE_VENV" >&2
  exit 12
fi
if [[ -f "$HF_ENV_FILE" ]]; then
  # Hugging Face Pro limits only help if the long-lived queue worker exports
  # the token before spawning shard download jobs.
  # shellcheck disable=SC1090
  source "$HF_ENV_FILE"
fi

log_event() {
  local state="$1" step="$2" detail="${3:-}"
  python3 - "$QUEUE_STATUS" "$state" "$step" "$detail" <<'PY'
import json
import sys
from datetime import datetime, timezone

path, state, step, detail = sys.argv[1:]
row = {
    "time": datetime.now(timezone.utc).isoformat(),
    "state": state,
    "step": step,
    "detail": detail,
}
with open(path, "a", encoding="utf-8") as handle:
    handle.write(json.dumps(row, sort_keys=True) + "\n")
PY
}

source_b2_env() {
  if [[ -n "$B2_ENV_FILE" && -f "$B2_ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$B2_ENV_FILE"
  fi
}

configure_b2env_rclone() {
  source_b2_env
  local key_id="${B2_KEY_ID:-${B2_KEYID:-}}"
  local app_key="${B2_APP_KEY:-${B2_APPKEY:-}}"
  if [[ -z "$key_id" || -z "$app_key" ]] && [[ -n "${B2_TOKEN:-}" ]]; then
    local parsed_b2
    parsed_b2="$(
      python3 - <<'PY'
import json
import os
import sys

token = os.environ.get("B2_TOKEN", "").strip()
key_id = app_key = ""
if token:
    if token.startswith("{"):
        payload = json.loads(token)
        key_id = (
            payload.get("keyId")
            or payload.get("keyID")
            or payload.get("applicationKeyId")
            or payload.get("applicationKeyID")
            or payload.get("key_id")
            or payload.get("application_key_id")
            or payload.get("accountId")
            or payload.get("accountID")
            or ""
        )
        app_key = (
            payload.get("applicationKey")
            or payload.get("application_key")
            or payload.get("appKey")
            or payload.get("app_key")
            or payload.get("key")
            or ""
        )
    elif ":" in token:
        key_id, app_key = token.split(":", 1)
if key_id and app_key:
    sys.stdout.write(key_id + "\n" + app_key)
PY
    )"
    if [[ -n "$parsed_b2" ]]; then
      key_id="${key_id:-$(printf '%s\n' "$parsed_b2" | sed -n '1p')}"
      app_key="${app_key:-$(printf '%s\n' "$parsed_b2" | sed -n '2p')}"
    fi
  fi
  if [[ -n "$key_id" && -z "$app_key" && -n "${B2_TOKEN:-}" ]]; then
    case "$B2_TOKEN" in
      \{*|*:*) ;;
      *) app_key="$B2_TOKEN" ;;
    esac
  fi
  if [[ -n "$key_id" && -n "$app_key" ]]; then
    export RCLONE_CONFIG_B2ENV_TYPE=b2
    export RCLONE_CONFIG_B2ENV_ACCOUNT="$key_id"
    export RCLONE_CONFIG_B2ENV_KEY="$app_key"
    return 0
  fi
  return 1
}

run_b2_once() {
  local root="$1" prefix="$2"
  [[ "$START_B2_UPLOAD" = "1" ]] || return 0
  source_b2_env
  MODE=face_shard \
    LOCAL_ROOT="$root" \
    B2_BUCKET="$B2_BUCKET" \
    B2_PREFIX="$prefix" \
    RUN_ONCE=1 \
    STABILITY_SECONDS=0 \
    B2_KEY_ID="${B2_KEY_ID:-}" \
    B2_APP_KEY="${B2_APP_KEY:-}" \
    B2_TOKEN="${B2_TOKEN:-}" \
    bash scripts/thunder/b2_continuous_upload.sh
}

b2_shard_archive_exists() {
  local prefix="$1"
  local remote_prefix="${prefix%/}/"
  [[ "$START_B2_UPLOAD" = "1" ]] || return 1
  configure_b2env_rclone || return 1

  rclone lsf "b2env:$B2_BUCKET/$remote_prefix" --files-only 2>/dev/null \
    | grep -Fxq "lean_face_corpus.tar.gz"
}

b2_shard_claim_exists() {
  local prefix="$1"
  local remote_prefix="${prefix%/}/"
  [[ "$START_B2_UPLOAD" = "1" && "$B2_QUEUE_CLAIMS" = "1" ]] || return 1
  configure_b2env_rclone || return 1
  if rclone lsf "b2env:$B2_BUCKET/$remote_prefix" --files-only 2>/dev/null \
    | grep -Fxq "lean_face_corpus.tar.gz"; then
    return 0
  fi
  if ! rclone lsf "b2env:$B2_BUCKET/$remote_prefix" --files-only 2>/dev/null \
    | grep -Fxq "queue_claim.json"; then
    return 1
  fi
  if [[ "$B2_QUEUE_CLAIM_TTL_SECONDS" = "0" ]]; then
    return 0
  fi
  local claim_tmp
  claim_tmp="$(mktemp)"
  if ! rclone copyto "b2env:$B2_BUCKET/$remote_prefix/queue_claim.json" "$claim_tmp" >/dev/null 2>&1; then
    rm -f "$claim_tmp"
    return 1
  fi
  local fresh
  if python3 - "$claim_tmp" "$B2_QUEUE_CLAIM_TTL_SECONDS" <<'PY'
import json
import sys
from datetime import datetime, timezone

path, ttl_s = sys.argv[1], int(sys.argv[2])
try:
    payload = json.load(open(path, "r", encoding="utf-8"))
    stamp = str(payload.get("time") or "")
    claimed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    if claimed.tzinfo is None:
        claimed = claimed.replace(tzinfo=timezone.utc)
except Exception:
    raise SystemExit(1)
age = (datetime.now(timezone.utc) - claimed).total_seconds()
raise SystemExit(0 if age <= ttl_s else 1)
PY
  then
    fresh=0
  else
    fresh=1
  fi
  rm -f "$claim_tmp"
  return "$fresh"
}

write_b2_shard_claim() {
  local prefix="$1" shard_id="$2" lab_root="$3"
  [[ "$START_B2_UPLOAD" = "1" && "$B2_QUEUE_CLAIMS" = "1" ]] || return 0
  configure_b2env_rclone || return 0
  mkdir -p "$lab_root"
  local claim="$lab_root/queue_claim.json"
  python3 - "$claim" "$shard_id" "$prefix" "$lab_root" <<'PY'
import json
import os
import socket
import sys
from datetime import datetime, timezone

path, shard_id, prefix, lab_root = sys.argv[1:]
payload = {
    "time": datetime.now(timezone.utc).isoformat(),
    "shard_id": shard_id,
    "b2_prefix": prefix,
    "lab_root": lab_root,
    "hostname": socket.gethostname(),
    "pid": os.getpid(),
    "source_kind": os.environ.get("SOURCE_KIND", ""),
    "data_lane": os.environ.get("DATA_LANE", ""),
    "source_pool_name": os.environ.get("SOURCE_POOL_NAME", ""),
}
with open(path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
  rclone copyto "$claim" "b2env:$B2_BUCKET/$prefix/queue_claim.json" >/dev/null 2>&1 || true
}

cleanup_heavy_payload() {
  local root="$1"
  [[ -d "$root" ]] || return 0
  for f in pilot_summary.json strict_gate.json train_strict_gate.json test_strict_gate.json; do
    if [[ -f "$root/corpus/$f" && ! -f "$root/$f" ]]; then
      cp "$root/corpus/$f" "$root/$f" || true
    fi
  done
  # TexVerse/HF downloads can leave a large local cache even when the shard
  # fails before corpus packaging. Remove it so queue workers can safely move
  # on to the next shard instead of slowly filling the 200GB Thunder disk.
  rm -rf "$root/corpus" "$root/.hf_cache" "$root/hf_cache" "$root/lean_face_corpus.tar.gz"
}

stop_pid_file() {
  local pid_file="$1"
  [[ -n "$pid_file" && -f "$pid_file" ]] || return 0
  local pid
  pid="$(cat "$pid_file" 2>/dev/null || true)"
  if [[ -n "$pid" ]]; then
    kill "$pid" 2>/dev/null || true
  fi
}

wait_for_existing_shard() {
  [[ -n "$WAIT_FOR_PID_FILE" ]] || return 0
  if [[ ! -f "$WAIT_FOR_PID_FILE" ]]; then
    log_event skipped wait "missing_pid_file=$WAIT_FOR_PID_FILE"
    return 0
  fi
  local pid
  pid="$(cat "$WAIT_FOR_PID_FILE" 2>/dev/null || true)"
  if [[ -z "$pid" ]]; then
    log_event skipped wait "empty_pid_file=$WAIT_FOR_PID_FILE"
    return 0
  fi
  log_event started wait "pid=$pid"
  while kill -0 "$pid" 2>/dev/null; do
    sleep "$WAIT_POLL_SECONDS"
  done
  log_event complete wait "pid=$pid"

  if [[ -n "$PREQUEUE_COMPLETED_ROOT" ]]; then
    if [[ ! -f "$PREQUEUE_COMPLETED_ROOT/lean_face_corpus.tar.gz" ]]; then
      log_event failed prequeue_cleanup "missing_lean_archive=$PREQUEUE_COMPLETED_ROOT/lean_face_corpus.tar.gz"
      return 20
    fi
    if [[ -n "$PREQUEUE_B2_PREFIX" ]]; then
      run_b2_once "$PREQUEUE_COMPLETED_ROOT" "$PREQUEUE_B2_PREFIX"
    fi
    stop_pid_file "$PREQUEUE_B2_UPLOAD_PID_FILE"
    cleanup_heavy_payload "$PREQUEUE_COMPLETED_ROOT"
    log_event complete prequeue_cleanup "root=$PREQUEUE_COMPLETED_ROOT"
  fi
}

run_one_shard() {
  local shard_id="$1"
  local shard_file="$QUEUE_SOURCE_DIR/shard_${shard_id}.jsonl"
  local lab_root="${LAB_ROOT_PREFIX}_${RUN_STAMP_PREFIX}_shard${shard_id}"
  local run_dir="$lab_root/corpus"
  local b2_prefix="$B2_PREFIX_ROOT/shard${shard_id}"
  local b2_pid_file="$lab_root/b2_upload.pid"
  local b2_log="$lab_root/b2_upload.log"
  local shard_log="$lab_root/corpus_pilot.log"

  if [[ ! -f "$shard_file" ]]; then
    log_event failed shard "shard=$shard_id missing_annotations=$shard_file"
    return 30
  fi
  if [[ -f "$lab_root/.queue_complete" ]]; then
    log_event skipped shard "shard=$shard_id already_complete"
    return 0
  fi
  if b2_shard_archive_exists "$b2_prefix"; then
    log_event skipped shard "shard=$shard_id already_complete_in_b2 prefix=$b2_prefix"
    mkdir -p "$lab_root"
    touch "$lab_root/.queue_complete"
    return 0
  fi
  if b2_shard_claim_exists "$b2_prefix"; then
    log_event skipped shard "shard=$shard_id already_claimed_or_complete_in_b2 prefix=$b2_prefix"
    mkdir -p "$lab_root"
    touch "$lab_root/.queue_skipped_claimed"
    return 0
  fi
  write_b2_shard_claim "$b2_prefix" "$shard_id" "$lab_root"

  local line_count select_target curation_target
  line_count="$(grep -cve '^\s*$' "$shard_file" || true)"
  select_target="$SELECT_TARGET"
  curation_target="$CURATION_TARGET"
  [[ "$select_target" = "auto" ]] && select_target="$line_count"
  [[ "$curation_target" = "auto" ]] && curation_target="$select_target"

  mkdir -p "$lab_root"
  cp "$shard_file" "$lab_root/source_annotations.jsonl"
  log_event started shard "shard=$shard_id root=$lab_root b2_prefix=$b2_prefix"

  if [[ "$START_B2_UPLOAD" = "1" ]]; then
    source_b2_env
    MODE=face_shard \
      LOCAL_ROOT="$lab_root" \
      B2_BUCKET="$B2_BUCKET" \
      B2_PREFIX="$b2_prefix" \
      INTERVAL_SECONDS="$B2_UPLOAD_INTERVAL_SECONDS" \
      B2_KEY_ID="${B2_KEY_ID:-}" \
      B2_APP_KEY="${B2_APP_KEY:-}" \
      B2_TOKEN="${B2_TOKEN:-}" \
      nohup bash scripts/thunder/b2_continuous_upload.sh > "$b2_log" 2>&1 &
    echo $! > "$b2_pid_file"
  fi

  set +e
  RUN_DIR="$run_dir" \
    DATA_LANE="$DATA_LANE" \
    SOURCE_POOL_NAME="$SOURCE_POOL_NAME" \
    SOURCE_SHARD_ID="$shard_id" \
    SOURCE_KIND="$SOURCE_KIND" \
    ANNOTATIONS="$lab_root/source_annotations.jsonl" \
    SPLIT=train \
    SELECT_TARGET="$select_target" \
    SCAN_LIMIT="$SCAN_LIMIT" \
    CURATION_TARGET="$curation_target" \
    MIN_QUALITY="$MIN_QUALITY" \
    OVERSAMPLE_FACTOR="$OVERSAMPLE_FACTOR" \
    SHUFFLE="$SHUFFLE" \
    SEED="$SEED" \
    DOWNLOAD_PROCESSES="$DOWNLOAD_PROCESSES" \
    DOWNLOAD_FALLBACK_PROCESSES="$DOWNLOAD_FALLBACK_PROCESSES" \
    DOWNLOAD_BATCH_SIZE="$DOWNLOAD_BATCH_SIZE" \
    DOWNLOAD_BATCH_TIMEOUT_SECONDS="$DOWNLOAD_BATCH_TIMEOUT_SECONDS" \
    DOWNLOAD_BATCH_RETRIES="$DOWNLOAD_BATCH_RETRIES" \
    DOWNLOAD_RETRY_SLEEP_SECONDS="$DOWNLOAD_RETRY_SLEEP_SECONDS" \
    DOWNLOAD_RATE_LIMIT_SLEEP_SECONDS="$DOWNLOAD_RATE_LIMIT_SLEEP_SECONDS" \
    TEXVERSE_DOWNLOAD_WORKERS="$TEXVERSE_DOWNLOAD_WORKERS" \
    TEXVERSE_MAX_SIZE_MB="$TEXVERSE_MAX_SIZE_MB" \
    TEXVERSE_CLEANUP_CACHE_EACH="$TEXVERSE_CLEANUP_CACHE_EACH" \
    OBJAVERSEXL_INCLUDE_SOURCES="$OBJAVERSEXL_INCLUDE_SOURCES" \
    OBJAVERSEXL_EXCLUDE_SOURCES="$OBJAVERSEXL_EXCLUDE_SOURCES" \
    OBJAVERSEXL_SAVE_REPO_FORMAT="$OBJAVERSEXL_SAVE_REPO_FORMAT" \
    OBJAVERSEXL_MAX_DOWNLOAD_DIR_GB="$OBJAVERSEXL_MAX_DOWNLOAD_DIR_GB" \
    EXCLUDE_SOURCE_IDS="$EXCLUDE_SOURCE_IDS" \
    SOURCE_MIN_FACES="$SOURCE_MIN_FACES" \
    SOURCE_MAX_FACES="$SOURCE_MAX_FACES" \
    MAX_FILE_MB="$MAX_FILE_MB" \
    MAX_COMPONENTS="$MAX_COMPONENTS" \
    MIN_LARGEST_COMPONENT_AREA_RATIO="$MIN_LARGEST_COMPONENT_AREA_RATIO" \
    TARGET_FACES="$TARGET_FACES" \
    TOKEN_MAX_FACES="$TOKEN_MAX_FACES" \
    POINT_SAMPLES="$POINT_SAMPLES" \
    NUM_BINS="$NUM_BINS" \
    PAPER_WITHIN_FACE_ORDER="$PAPER_WITHIN_FACE_ORDER" \
    STRICT_ENGINE="$STRICT_ENGINE" \
    FALLBACK="$FALLBACK" \
    VOXEL_RESOLUTION="$VOXEL_RESOLUTION" \
    MESH_VOXEL_MAX_FACES="$MESH_VOXEL_MAX_FACES" \
    STRICT_TARGET_PROGRESS_EVERY="$STRICT_TARGET_PROGRESS_EVERY" \
    STRICT_TARGET_WORKERS="$STRICT_TARGET_WORKERS" \
    TOKEN_BUILD_WORKERS="$TOKEN_BUILD_WORKERS" \
    FORCE_MIN_POSTPROCESS_WORKERS="$FORCE_MIN_POSTPROCESS_WORKERS" \
    TARGET_MAX_OUTPUT_COMPONENTS="$TARGET_MAX_OUTPUT_COMPONENTS" \
    TARGET_MAX_BOUNDARY_LOOPS="$TARGET_MAX_BOUNDARY_LOOPS" \
    TARGET_MAX_NONMANIFOLD_EDGES="$TARGET_MAX_NONMANIFOLD_EDGES" \
    TARGET_REQUIRE_WATERTIGHT="$TARGET_REQUIRE_WATERTIGHT" \
    GATE_PROFILE="$GATE_PROFILE" \
    GATE_MAX_BOUNDARY_EDGES="$GATE_MAX_BOUNDARY_EDGES" \
    GATE_MAX_NONMANIFOLD_EDGES="$GATE_MAX_NONMANIFOLD_EDGES" \
    GATE_MIN_EDGE_PAIRING_RATIO="$GATE_MIN_EDGE_PAIRING_RATIO" \
    TEST_RATIO="$TEST_RATIO" \
    LEAN_ARCHIVE_PATH="$lab_root/lean_face_corpus.tar.gz" \
    ARCHIVE_PATH="" \
    bash scripts/thunder/face_objaversepp_corpus_pilot.sh > "$shard_log" 2>&1
  local code=$?
  set -e

  if [[ "$code" -ne 0 ]]; then
    log_event failed shard "shard=$shard_id exit=$code root=$lab_root"
    run_b2_once "$lab_root" "$b2_prefix" || true
    stop_pid_file "$b2_pid_file"
    if [[ "$CLEANUP_FAILED_ROOT_ON_FAILURE" = "1" ]]; then
      cleanup_heavy_payload "$lab_root"
    fi
    return "$code"
  fi

  if [[ ! -f "$lab_root/lean_face_corpus.tar.gz" ]]; then
    log_event failed shard "shard=$shard_id missing_lean_archive"
    stop_pid_file "$b2_pid_file"
    return 31
  fi
  run_b2_once "$lab_root" "$b2_prefix"
  stop_pid_file "$b2_pid_file"
  touch "$lab_root/.queue_complete"
  cleanup_heavy_payload "$lab_root"
  log_event complete shard "shard=$shard_id root=$lab_root"
}

wait_for_existing_shard

log_event started queue "shards=$QUEUE_SHARD_IDS"
for shard_id in $QUEUE_SHARD_IDS; do
  if run_one_shard "$shard_id"; then
    continue
  else
    code=$?
    if [[ "$CONTINUE_ON_FAILURE" = "1" ]]; then
      log_event continuing queue "shard=$shard_id failed exit=$code"
      continue
    fi
    log_event failed queue "shard=$shard_id exit=$code"
    exit "$code"
  fi
done
log_event complete queue "shards=$QUEUE_SHARD_IDS"
