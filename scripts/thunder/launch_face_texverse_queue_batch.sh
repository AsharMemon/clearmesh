#!/usr/bin/env bash
# Attach a TexVerse FACE corpus queue to existing Thunder workers.
#
# This is intentionally a local orchestration helper. It assumes source shards
# have already been built by scripts/data/build_texverse_face_source_pool.py and
# that the target workers already have a safe remote B2 env file.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
POOL_DIR="${POOL_DIR:-$REPO_ROOT/.codex_outputs/face_source_pool_texverse_highquality_675k_20260516}"
SHARD_DIR="${SHARD_DIR:-$POOL_DIR/shards}"
INSTANCE_IDS="${INSTANCE_IDS:-1 2 3 4 5}"
WAIT_FOR_PID_FILE="${WAIT_FOR_PID_FILE:-/tmp/clearmesh_face_corpus_queue_poolB_medium_logs/queue_worker.pid}"
REMOTE_B2_ENV="${REMOTE_B2_ENV:-/home/ubuntu/.clearmesh_b2.env}"
REMOTE_QUEUE_SOURCE_DIR="${REMOTE_QUEUE_SOURCE_DIR:-/tmp/clearmesh_face_corpus_queue_texverse}"
REMOTE_QUEUE_LOG_ROOT="${REMOTE_QUEUE_LOG_ROOT:-/tmp/clearmesh_face_corpus_queue_texverse_logs}"
RUN_STAMP_PREFIX_BASE="${RUN_STAMP_PREFIX_BASE:-$(date -u +%Y%m%d)_texverse_after_poolB}"
LAB_ROOT_PREFIX="${LAB_ROOT_PREFIX:-/tmp/clearmesh_face_corpus_texverse_shard}"
B2_PREFIX_ROOT="${B2_PREFIX_ROOT:-face-corpora/texverse-shards}"
CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-1}"
MAX_SHARDS="${MAX_SHARDS:-0}"
SHARD_OFFSET="${SHARD_OFFSET:-0}"

if [[ ! -d "$SHARD_DIR" ]]; then
  echo "Missing TexVerse shard dir: $SHARD_DIR" >&2
  exit 2
fi
shard_paths=()
while IFS= read -r shard_path; do
  shard_paths+=("$shard_path")
done < <(find "$SHARD_DIR" -maxdepth 1 -type f -name 'shard_*.jsonl' | sort)
if [[ "${#shard_paths[@]}" -eq 0 ]]; then
  echo "No TexVerse shard_*.jsonl files under $SHARD_DIR" >&2
  exit 2
fi
if [[ "$SHARD_OFFSET" -gt 0 ]]; then
  shard_paths=("${shard_paths[@]:$SHARD_OFFSET}")
fi
if [[ "$MAX_SHARDS" -gt 0 && "${#shard_paths[@]}" -gt "$MAX_SHARDS" ]]; then
  shard_paths=("${shard_paths[@]:0:$MAX_SHARDS}")
fi
read -r -a instances <<< "$INSTANCE_IDS"
if [[ "${#instances[@]}" -eq 0 ]]; then
  echo "INSTANCE_IDS is empty" >&2
  exit 2
fi

launch_root="$REPO_ROOT/.codex_outputs/texverse_queue_attach_${RUN_STAMP_PREFIX_BASE}"
mkdir -p "$launch_root"

for index in "${!instances[@]}"; do
  instance_id="${instances[$index]}"
  queue_ids=()
  for shard_index in "${!shard_paths[@]}"; do
    if (( shard_index % ${#instances[@]} == index )); then
      base="$(basename "${shard_paths[$shard_index]}")"
      shard_id="${base#shard_}"
      shard_id="${shard_id%.jsonl}"
      queue_ids+=("$shard_id")
    fi
  done
  if [[ "${#queue_ids[@]}" -eq 0 ]]; then
    continue
  fi
  queue_string="${queue_ids[*]}"
  log="$launch_root/instance_${instance_id}.launch.log"
  echo "Launching TexVerse queue on instance $instance_id: ${#queue_ids[@]} shards" | tee "$log"
  (
    cd "$REPO_ROOT"
    SOURCE_KIND=texverse \
      QUEUE_SHARD_IDS="$queue_string" \
      POOL_DIR="$POOL_DIR" \
      SHARD_DIR="$SHARD_DIR" \
      REMOTE_QUEUE_SOURCE_DIR="$REMOTE_QUEUE_SOURCE_DIR" \
      REMOTE_QUEUE_LOG_ROOT="$REMOTE_QUEUE_LOG_ROOT" \
      RUN_STAMP_PREFIX="${RUN_STAMP_PREFIX_BASE}_i${instance_id}" \
      LAB_ROOT_PREFIX="$LAB_ROOT_PREFIX" \
      B2_PREFIX_ROOT="$B2_PREFIX_ROOT" \
      REMOTE_B2_ENV="$REMOTE_B2_ENV" \
      LOCAL_B2_ENV_FILE="${LOCAL_B2_ENV_FILE:-}" \
      START_B2_UPLOAD=1 \
      WAIT_FOR_PID_FILE="$WAIT_FOR_PID_FILE" \
      CONTINUE_ON_FAILURE="$CONTINUE_ON_FAILURE" \
      MIN_QUALITY="${MIN_QUALITY:-2}" \
      DOWNLOAD_PROCESSES="${DOWNLOAD_PROCESSES:-4}" \
      TEXVERSE_DOWNLOAD_WORKERS="${TEXVERSE_DOWNLOAD_WORKERS:-4}" \
      DOWNLOAD_BATCH_RETRIES="${DOWNLOAD_BATCH_RETRIES:-4}" \
      DOWNLOAD_RETRY_SLEEP_SECONDS="${DOWNLOAD_RETRY_SLEEP_SECONDS:-30}" \
      DOWNLOAD_RATE_LIMIT_SLEEP_SECONDS="${DOWNLOAD_RATE_LIMIT_SLEEP_SECONDS:-600}" \
      SOURCE_MIN_FACES="${SOURCE_MIN_FACES:-64}" \
      SOURCE_MAX_FACES="${SOURCE_MAX_FACES:-250000}" \
      TARGET_FACES="${TARGET_FACES:-512}" \
      TOKEN_MAX_FACES="${TOKEN_MAX_FACES:-512}" \
      POINT_SAMPLES="${POINT_SAMPLES:-8192}" \
      NUM_BINS="${NUM_BINS:-128}" \
      PAPER_WITHIN_FACE_ORDER="${PAPER_WITHIN_FACE_ORDER:-rotate_min_zyx}" \
      STRICT_ENGINE="${STRICT_ENGINE:-voxel_shell}" \
      FALLBACK="${FALLBACK:-convex_hull}" \
      TEST_RATIO="${TEST_RATIO:-0.02}" \
      SYNC_QUEUE_FILES=1 \
      scripts/thunder/launch_face_corpus_shard_queue_on_instance.sh "$instance_id"
  ) >> "$log" 2>&1 &
  echo $! > "$launch_root/instance_${instance_id}.launch.pid"
done

echo "TexVerse queue attach launched. Logs: $launch_root"
