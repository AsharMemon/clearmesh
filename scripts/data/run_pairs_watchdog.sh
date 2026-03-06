#!/usr/bin/env bash
set -uo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <shard_id> <num_shards> [extra generate_pairs args...]" >&2
  exit 2
fi

SHARD_ID="$1"
NUM_SHARDS="$2"
shift 2

INPUT_JSON="${INPUT_JSON:-/workspace/data/filtered/valid_models.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/data/training_pairs}"
TRELLIS2_DIR="${TRELLIS2_DIR:-/workspace/TRELLIS.2}"
MODEL_DIR="${MODEL_DIR:-/workspace/models/trellis2-4b}"
PIPELINE_TYPE="${PIPELINE_TYPE:-512}"
RENDER_SIZE="${RENDER_SIZE:-1024}"
NUM_VIEWS="${NUM_VIEWS:-8}"
LOG_RSS_EVERY="${LOG_RSS_EVERY:-10}"
RSS_HARD_LIMIT_GB="${RSS_HARD_LIMIT_GB:-46}"
RSS_EMERGENCY_LIMIT_GB="${RSS_EMERGENCY_LIMIT_GB:-$RSS_HARD_LIMIT_GB}"
RSS_WATCH_INTERVAL_SEC="${RSS_WATCH_INTERVAL_SEC:-0.25}"
MAX_MODELS_PER_RUN="${MAX_MODELS_PER_RUN:-120}"
MAX_EMERGENCY_RECYCLES_PER_UID="${MAX_EMERGENCY_RECYCLES_PER_UID:-3}"
RECYCLE_EXIT_CODE="${RECYCLE_EXIT_CODE:-75}"
RESTART_SLEEP_SEC="${RESTART_SLEEP_SEC:-8}"
ATTN_BACKEND="${ATTN_BACKEND:-flash_attn}"
SPARSE_ATTN_BACKEND="${SPARSE_ATTN_BACKEND:-$ATTN_BACKEND}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.4}"
HF_HOME="${HF_HOME:-/workspace/.cache/hf}"

attempt=0
while true; do
  attempt=$((attempt + 1))
  ts="$(date -Iseconds)"

  # Prefer flash-attn when installed; auto-fallback to sdpa if unavailable.
  if ! python - <<'PY' >/dev/null 2>&1
import flash_attn  # noqa: F401
PY
  then
    ATTN_BACKEND="sdpa"
    SPARSE_ATTN_BACKEND="sdpa"
  fi

  # Ensure OpenGL libraries are present (lost on container restart).
  if ! ldconfig -p 2>/dev/null | grep -q libGLU; then
    echo "[$ts] shard=${SHARD_ID} installing libGLU..."
    apt-get update -qq && apt-get install -y -qq libglu1-mesa 2>/dev/null || true
  fi

  # Ensure pyglet<2 is installed (required by trimesh for OpenGL rendering).
  if ! python -c "import pyglet" >/dev/null 2>&1; then
    echo "[$ts] shard=${SHARD_ID} installing pyglet..."
    pip install -q "pyglet<2" --no-cache-dir 2>/dev/null || true
  fi

  # Keep root disk stable across long restart loops.
  rm -rf "/tmp/shard_${SHARD_ID}_cache" "/tmp/xvfb-run."* 2>/dev/null || true
  mkdir -p "${HF_HOME}"

  echo "[$ts] shard=${SHARD_ID} attempt=${attempt} starting"
  echo "[$ts] shard=${SHARD_ID} attention backends: dense=${ATTN_BACKEND} sparse=${SPARSE_ATTN_BACKEND}"

  MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}" \
  ATTN_BACKEND="${ATTN_BACKEND}" SPARSE_ATTN_BACKEND="${SPARSE_ATTN_BACKEND}" \
  CUDA_HOME="${CUDA_HOME}" PATH="${CUDA_HOME}/bin:${PATH}" \
  LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}" \
  HF_HOME="${HF_HOME}" \
  xvfb-run -a python -u scripts/data/generate_pairs.py \
    --input_json "${INPUT_JSON}" \
    --output_dir "${OUTPUT_DIR}" \
    --pipeline_type "${PIPELINE_TYPE}" \
    --render_size "${RENDER_SIZE}" \
    --num_views "${NUM_VIEWS}" \
    --trellis2_dir "${TRELLIS2_DIR}" \
    --model_dir "${MODEL_DIR}" \
    --shard_id "${SHARD_ID}" \
    --num_shards "${NUM_SHARDS}" \
    --gpu 0 \
    --geometry_only \
    --disable_rembg_model \
    --log_rss_every "${LOG_RSS_EVERY}" \
    --rss_hard_limit_gb "${RSS_HARD_LIMIT_GB}" \
    --rss_emergency_limit_gb "${RSS_EMERGENCY_LIMIT_GB}" \
    --rss_watch_interval_sec "${RSS_WATCH_INTERVAL_SEC}" \
    --max_models_per_run "${MAX_MODELS_PER_RUN}" \
    --max_emergency_recycles_per_uid "${MAX_EMERGENCY_RECYCLES_PER_UID}" \
    --recycle_exit_code "${RECYCLE_EXIT_CODE}" \
    "$@"
  code=$?

  ts="$(date -Iseconds)"
  if [[ $code -eq 0 ]]; then
    echo "[$ts] shard=${SHARD_ID} completed"
    break
  fi

  if [[ $code -eq ${RECYCLE_EXIT_CODE} ]]; then
    echo "[$ts] shard=${SHARD_ID} recycle requested (code=${code}), restarting in ${RESTART_SLEEP_SEC}s"
    sleep "${RESTART_SLEEP_SEC}"
    continue
  fi

  echo "[$ts] shard=${SHARD_ID} worker exited code=${code}, restarting in ${RESTART_SLEEP_SEC}s"
  sleep "${RESTART_SLEEP_SEC}"
done
