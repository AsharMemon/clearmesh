#!/usr/bin/env bash
# Launch a bounded batch of Pool A FACE corpus shard workers.
#
# This is intentionally a corpus-prep launcher, not a training launcher. It
# turns local Objaverse++ annotation shards into lean strict-token archives that
# can later be merged into a 130k+ FACE training split.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
POOL_DIR="${POOL_DIR:-$REPO_ROOT/.codex_outputs/face_source_pool_objpp600k_20260508_061413_managed}"
SHARD_DIR="${SHARD_DIR:-$POOL_DIR/shards}"
START_SHARD="${START_SHARD:-0}"
NUM_SHARDS="${NUM_SHARDS:-8}"
RUN_STAMP_PREFIX="${RUN_STAMP_PREFIX:-$(date -u +%Y%m%d_%H%M%S)_poolA}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/.codex_outputs/face_poolA_shard_batch_$RUN_STAMP_PREFIX}"

GPU="${GPU:-a6000}"
MODE="${MODE:-prototyping}"
VCPUS="${VCPUS:-8}"
PRIMARY_DISK="${PRIMARY_DISK:-200}"
START_B2_UPLOAD="${START_B2_UPLOAD:-0}"
B2_BUCKET="${B2_BUCKET:-clearmesh-pairs}"
B2_PREFIX_ROOT="${B2_PREFIX_ROOT:-face-corpora/poolA-shards}"

mkdir -p "$LOG_DIR"

if [ ! -d "$SHARD_DIR" ]; then
  echo "Shard directory not found: $SHARD_DIR" >&2
  exit 2
fi

launched_manifest="$LOG_DIR/launched_shards.jsonl"
touch "$launched_manifest"

for shard_id in $(seq "$START_SHARD" $((START_SHARD + NUM_SHARDS - 1))); do
  shard_file="$SHARD_DIR/shard_$(printf '%04d' "$shard_id").jsonl"
  if [ ! -f "$shard_file" ]; then
    echo "Skipping missing shard $shard_id: $shard_file" >&2
    continue
  fi
  run_stamp="${RUN_STAMP_PREFIX}_shard$(printf '%04d' "$shard_id")"
  log_file="$LOG_DIR/shard_$(printf '%04d' "$shard_id").launch.log"
  b2_prefix="$B2_PREFIX_ROOT/shard$(printf '%04d' "$shard_id")"
  echo "Launching Pool A shard $shard_id -> $log_file"
  nohup env \
    LOCAL_ANNOTATIONS_FILE="$shard_file" \
    RUN_STAMP="$run_stamp" \
    GPU="$GPU" \
    MODE="$MODE" \
    VCPUS="$VCPUS" \
    PRIMARY_DISK="$PRIMARY_DISK" \
    START_B2_UPLOAD="$START_B2_UPLOAD" \
    B2_BUCKET="$B2_BUCKET" \
    B2_PREFIX="$b2_prefix" \
    SELECT_TARGET="$(grep -cve '^\s*$' "$shard_file" || true)" \
    CURATION_TARGET="$(grep -cve '^\s*$' "$shard_file" || true)" \
    MIN_QUALITY="${MIN_QUALITY:-2}" \
    TARGET_FACES="${TARGET_FACES:-512}" \
    TOKEN_MAX_FACES="${TOKEN_MAX_FACES:-512}" \
    NUM_BINS="${NUM_BINS:-128}" \
    POINT_SAMPLES="${POINT_SAMPLES:-8192}" \
    PAPER_WITHIN_FACE_ORDER="${PAPER_WITHIN_FACE_ORDER:-rotate_min_zyx}" \
    TEST_RATIO="${TEST_RATIO:-0.02}" \
    "$REPO_ROOT/scripts/thunder/launch_face_corpus_shard_instance.sh" \
      > "$log_file" 2>&1 &
  pid=$!
  python3 - "$launched_manifest" "$shard_id" "$shard_file" "$run_stamp" "$log_file" "$pid" "$b2_prefix" <<'PY'
import json
import sys

manifest, shard_id, shard_file, run_stamp, log_file, pid, b2_prefix = sys.argv[1:]
row = {
    "shard_id": int(shard_id),
    "shard_file": shard_file,
    "run_stamp": run_stamp,
    "launch_log": log_file,
    "launcher_pid": int(pid),
    "b2_prefix": b2_prefix,
}
with open(manifest, "a", encoding="utf-8") as handle:
    handle.write(json.dumps(row, sort_keys=True) + "\n")
PY
done

cat > "$LOG_DIR/batch_info.json" <<JSON
{
  "pool_dir": "$POOL_DIR",
  "shard_dir": "$SHARD_DIR",
  "start_shard": $START_SHARD,
  "num_shards": $NUM_SHARDS,
  "gpu": "$GPU",
  "mode": "$MODE",
  "primary_disk": $PRIMARY_DISK,
  "start_b2_upload": "$START_B2_UPLOAD",
  "b2_bucket": "$B2_BUCKET",
  "b2_prefix_root": "$B2_PREFIX_ROOT",
  "launched_manifest": "$launched_manifest"
}
JSON

echo "Launched shard batch. Logs: $LOG_DIR"
