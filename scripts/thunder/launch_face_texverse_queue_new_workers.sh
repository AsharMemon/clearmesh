#!/usr/bin/env bash
# Create additional Thunder A6000 workers and start TexVerse FACE queues on them.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TNR_BIN="${TNR_BIN:-/Users/Ashar/.tnr/bin/tnr}"
POOL_DIR="${POOL_DIR:-$REPO_ROOT/.codex_outputs/face_source_pool_texverse_highquality_675k_20260516}"
SHARD_DIR="${SHARD_DIR:-$POOL_DIR/shards}"
WORKER_COUNT="${WORKER_COUNT:-5}"
SHARD_OFFSET="${SHARD_OFFSET:-31}"
MAX_SHARDS="${MAX_SHARDS:-31}"
GPU="${GPU:-a6000}"
MODE="${MODE:-prototyping}"
VCPUS="${VCPUS:-8}"
PRIMARY_DISK="${PRIMARY_DISK:-200}"
TEMPLATE="${TEMPLATE:-base}"
REMOTE_B2_ENV="${REMOTE_B2_ENV:-/home/ubuntu/.clearmesh_b2.env}"
LOCAL_B2_ENV_FILE="${LOCAL_B2_ENV_FILE:-}"
B2_SEED_INSTANCE="${B2_SEED_INSTANCE:-1}"
RUN_STAMP_PREFIX_BASE="${RUN_STAMP_PREFIX_BASE:-$(date -u +%Y%m%d)_texverse_expand}"
B2_PREFIX_ROOT="${B2_PREFIX_ROOT:-face-corpora/texverse-shards}"
LAB_ROOT_PREFIX="${LAB_ROOT_PREFIX:-/tmp/clearmesh_face_corpus_texverse_shard}"
REMOTE_QUEUE_SOURCE_DIR="${REMOTE_QUEUE_SOURCE_DIR:-/tmp/clearmesh_face_corpus_queue_texverse}"
REMOTE_QUEUE_LOG_ROOT="${REMOTE_QUEUE_LOG_ROOT:-/tmp/clearmesh_face_corpus_queue_texverse_logs}"
WAIT_FOR_PID_FILE="${WAIT_FOR_PID_FILE:-}"

if [[ ! -d "$SHARD_DIR" ]]; then
  echo "Missing TexVerse shard dir: $SHARD_DIR" >&2
  exit 2
fi
if [[ -z "$LOCAL_B2_ENV_FILE" ]]; then
  tmp_env="$(mktemp "$REPO_ROOT/.codex_outputs/texverse_b2_env.XXXXXX")"
  chmod 600 "$tmp_env"
  "$TNR_BIN" scp "$B2_SEED_INSTANCE:$REMOTE_B2_ENV" "$tmp_env" >/dev/null
  LOCAL_B2_ENV_FILE="$tmp_env"
  cleanup_tmp_env=1
else
  cleanup_tmp_env=0
fi
cleanup() {
  if [[ "${cleanup_tmp_env:-0}" = "1" && -n "${LOCAL_B2_ENV_FILE:-}" ]]; then
    rm -f "$LOCAL_B2_ENV_FILE"
  fi
}
trap cleanup EXIT INT TERM HUP

shard_paths=()
while IFS= read -r shard_path; do
  shard_paths+=("$shard_path")
done < <(find "$SHARD_DIR" -maxdepth 1 -type f -name 'shard_*.jsonl' | sort)
shard_paths=("${shard_paths[@]:$SHARD_OFFSET}")
if [[ "$MAX_SHARDS" -gt 0 && "${#shard_paths[@]}" -gt "$MAX_SHARDS" ]]; then
  shard_paths=("${shard_paths[@]:0:$MAX_SHARDS}")
fi
if [[ "${#shard_paths[@]}" -eq 0 ]]; then
  echo "No shards selected after SHARD_OFFSET=$SHARD_OFFSET MAX_SHARDS=$MAX_SHARDS" >&2
  exit 2
fi

launch_root="$REPO_ROOT/.codex_outputs/texverse_new_worker_launch_${RUN_STAMP_PREFIX_BASE}"
mkdir -p "$launch_root"
launch_pids=()
for worker_index in $(seq 0 $((WORKER_COUNT - 1))); do
  queue_ids=()
  for shard_index in "${!shard_paths[@]}"; do
    if (( shard_index % WORKER_COUNT == worker_index )); then
      base="$(basename "${shard_paths[$shard_index]}")"
      shard_id="${base#shard_}"
      shard_id="${shard_id%.jsonl}"
      queue_ids+=("$shard_id")
    fi
  done
  [[ "${#queue_ids[@]}" -gt 0 ]] || continue
  queue_string="${queue_ids[*]}"
  log="$launch_root/worker_${worker_index}.launch.log"
  echo "Creating TexVerse worker $worker_index with ${#queue_ids[@]} shards" | tee "$log"
  (
    cd "$REPO_ROOT"
    CREATE_INSTANCE=1 \
      GPU="$GPU" MODE="$MODE" VCPUS="$VCPUS" PRIMARY_DISK="$PRIMARY_DISK" TEMPLATE="$TEMPLATE" \
      BOOTSTRAP_REMOTE=1 SYNC_HF_TOKEN=1 \
      SOURCE_KIND=texverse \
      QUEUE_SHARD_IDS="$queue_string" \
      POOL_DIR="$POOL_DIR" SHARD_DIR="$SHARD_DIR" \
      REMOTE_QUEUE_SOURCE_DIR="$REMOTE_QUEUE_SOURCE_DIR" \
      REMOTE_QUEUE_LOG_ROOT="$REMOTE_QUEUE_LOG_ROOT" \
      RUN_STAMP_PREFIX="${RUN_STAMP_PREFIX_BASE}_w${worker_index}" \
      LAB_ROOT_PREFIX="$LAB_ROOT_PREFIX" \
      B2_PREFIX_ROOT="$B2_PREFIX_ROOT" \
      REMOTE_B2_ENV="$REMOTE_B2_ENV" LOCAL_B2_ENV_FILE="$LOCAL_B2_ENV_FILE" \
      START_B2_UPLOAD=1 WAIT_FOR_PID_FILE="$WAIT_FOR_PID_FILE" CONTINUE_ON_FAILURE=1 \
      CLEANUP_FAILED_ROOT_ON_FAILURE="${CLEANUP_FAILED_ROOT_ON_FAILURE:-1}" \
      DATA_LANE="${DATA_LANE:-unspecified}" SOURCE_POOL_NAME="${SOURCE_POOL_NAME:-texverse}" \
      SELECT_TARGET="${SELECT_TARGET:-auto}" CURATION_TARGET="${CURATION_TARGET:-auto}" SCAN_LIMIT="${SCAN_LIMIT:-0}" \
      MIN_QUALITY="${MIN_QUALITY:-2}" DOWNLOAD_PROCESSES="${DOWNLOAD_PROCESSES:-4}" TEXVERSE_DOWNLOAD_WORKERS="${TEXVERSE_DOWNLOAD_WORKERS:-4}" \
      TEXVERSE_MAX_SIZE_MB="${TEXVERSE_MAX_SIZE_MB:-0}" TEXVERSE_CLEANUP_CACHE_EACH="${TEXVERSE_CLEANUP_CACHE_EACH:-0}" \
      DOWNLOAD_BATCH_RETRIES="${DOWNLOAD_BATCH_RETRIES:-4}" DOWNLOAD_RETRY_SLEEP_SECONDS="${DOWNLOAD_RETRY_SLEEP_SECONDS:-30}" \
      SOURCE_MIN_FACES="${SOURCE_MIN_FACES:-64}" SOURCE_MAX_FACES="${SOURCE_MAX_FACES:-250000}" \
      MAX_FILE_MB="${MAX_FILE_MB:-256}" MAX_COMPONENTS="${MAX_COMPONENTS:-48}" \
      TARGET_FACES="${TARGET_FACES:-512}" TOKEN_MAX_FACES="${TOKEN_MAX_FACES:-512}" POINT_SAMPLES="${POINT_SAMPLES:-8192}" NUM_BINS="${NUM_BINS:-128}" \
      PAPER_WITHIN_FACE_ORDER="${PAPER_WITHIN_FACE_ORDER:-rotate_min_zyx}" STRICT_ENGINE="${STRICT_ENGINE:-voxel_shell}" FALLBACK="${FALLBACK:-convex_hull}" \
      VOXEL_RESOLUTION="${VOXEL_RESOLUTION:-64}" MESH_VOXEL_MAX_FACES="${MESH_VOXEL_MAX_FACES:-5000}" STRICT_TARGET_PROGRESS_EVERY="${STRICT_TARGET_PROGRESS_EVERY:-100}" \
      TARGET_MAX_OUTPUT_COMPONENTS="${TARGET_MAX_OUTPUT_COMPONENTS:-1}" TARGET_MAX_BOUNDARY_LOOPS="${TARGET_MAX_BOUNDARY_LOOPS:-0}" \
      TARGET_MAX_NONMANIFOLD_EDGES="${TARGET_MAX_NONMANIFOLD_EDGES:-0}" TARGET_REQUIRE_WATERTIGHT="${TARGET_REQUIRE_WATERTIGHT:-1}" \
      GATE_PROFILE="${GATE_PROFILE:-strict}" GATE_MAX_BOUNDARY_EDGES="${GATE_MAX_BOUNDARY_EDGES:-}" \
      GATE_MAX_NONMANIFOLD_EDGES="${GATE_MAX_NONMANIFOLD_EDGES:-}" GATE_MIN_EDGE_PAIRING_RATIO="${GATE_MIN_EDGE_PAIRING_RATIO:-}" \
      TEST_RATIO="${TEST_RATIO:-0.02}" \
      scripts/thunder/launch_face_corpus_shard_queue_on_instance.sh
  ) >> "$log" 2>&1 &
  pid=$!
  launch_pids+=("$pid")
  echo "$pid" > "$launch_root/worker_${worker_index}.launch.pid"
done

echo "Launched create flows. Logs: $launch_root"
exit_code=0
for pid in "${launch_pids[@]}"; do
  if ! wait "$pid"; then
    exit_code=1
  fi
done
exit "$exit_code"
