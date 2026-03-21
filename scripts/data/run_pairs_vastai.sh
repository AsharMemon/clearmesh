#!/usr/bin/env bash
# ============================================================================
# Run pair generation on a Vast.ai pod.
#
# This is the main entry point after setup_vastai_pod.sh completes.
# It runs the full pipeline: pair gen → SDF conversion → upload results to B2.
#
# Environment variables:
#   SHARD_ID       - Which shard this pod processes
#   NUM_SHARDS     - Total number of shards
#   B2_BUCKET      - B2 bucket name (default: clearmesh-pairs)
#
# Usage:
#   bash scripts/data/run_pairs_vastai.sh
# ============================================================================
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
CLEARMESH_DIR="${CLEARMESH_DIR:-$WORKSPACE/clearmesh}"
DATA_DIR="${DATA_DIR:-$WORKSPACE/data}"
SHARD_DIR="${SHARD_DIR:-$DATA_DIR/shards}"
PAIRS_DIR="${PAIRS_DIR:-$DATA_DIR/training_pairs}"
SHARD_ID="${SHARD_ID:?SHARD_ID must be set}"
NUM_SHARDS="${NUM_SHARDS:-8}"
B2_BUCKET="${B2_BUCKET:-clearmesh-pairs}"

# Pair generation settings
PIPELINE_TYPE="${PIPELINE_TYPE:-512}"
NUM_VIEWS="${NUM_VIEWS:-6}"
RENDER_SIZE="${RENDER_SIZE:-1024}"
MAX_MODELS_PER_RUN="${MAX_MODELS_PER_RUN:-150}"
RSS_HARD_LIMIT_GB="${RSS_HARD_LIMIT_GB:-200}"

# SDF settings
SDF_RESOLUTION="${SDF_RESOLUTION:-32}"
SDF_WORKERS="${SDF_WORKERS:-4}"

# Cache dirs
export HF_HOME="$WORKSPACE/.hf_cache"
export TORCH_HOME="$WORKSPACE/.torch_cache"
export TRITON_CACHE_DIR="$WORKSPACE/.triton_cache"

# TRELLIS.2 backend config
export SPARSE_CONV_BACKEND="${SPARSE_CONV_BACKEND:-spconv}"
export ATTN_BACKEND="${ATTN_BACKEND:-flash_attn}"
export PYTHONPATH="${WORKSPACE}/TRELLIS.2:${PYTHONPATH:-}"

INPUT_JSON="$SHARD_DIR/shard_${SHARD_ID}.json"

echo "============================================================"
echo "ClearMesh Vast.ai Pair Generation"
echo "============================================================"
echo "Shard:  $SHARD_ID / $NUM_SHARDS"
echo "Input:  $INPUT_JSON"
echo "Output: $PAIRS_DIR"
echo "============================================================"

# Verify input exists
if [ ! -f "$INPUT_JSON" ]; then
    echo "ERROR: Shard JSON not found: $INPUT_JSON"
    echo "Did setup_vastai_pod.sh complete successfully?"
    exit 1
fi

MODEL_COUNT=$(python3 -c "import json; print(len(json.load(open('$INPUT_JSON'))))" 2>/dev/null)
echo "Models to process: $MODEL_COUNT"
echo ""

cd "$CLEARMESH_DIR"

# ---- Phase 1: Pair Generation (via watchdog) ----
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Phase 1: Pair Generation                               ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Export env vars for the watchdog
export INPUT_JSON
export OUTPUT_DIR="$PAIRS_DIR"
export PIPELINE_TYPE
export NUM_VIEWS
export RENDER_SIZE
export MAX_MODELS_PER_RUN
export RSS_HARD_LIMIT_GB

# Start B2 progress sync in background
if [ -f scripts/data/sync_progress_b2.sh ]; then
    echo "Starting B2 progress sync daemon..."
    bash scripts/data/sync_progress_b2.sh --daemon &
    SYNC_PID=$!
    echo "  Sync PID: $SYNC_PID"
fi

# Run watchdog (will auto-restart on crash/OOM)
# Use shard_id=0, num_shards=1 because data is already pre-sharded.
# The shard JSON only contains this pod's models.
echo "Starting watchdog..."
bash scripts/data/run_pairs_watchdog.sh 0 1 2>&1 | \
    tee "$PAIRS_DIR/shard_${SHARD_ID}.log"

echo ""
echo "Pair generation complete!"

# ---- Phase 2: SDF Conversion ----
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Phase 2: SDF Conversion                                ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

PAIRS_DONE=$(find "$PAIRS_DIR" -name "coarse_voxels.npy" 2>/dev/null | wc -l)
SDF_DONE=$(find "$PAIRS_DIR" -name "fine_sdf.npy" 2>/dev/null | wc -l)
echo "Completed pairs: $PAIRS_DONE"
echo "Already SDF-converted: $SDF_DONE"

if [ "$PAIRS_DONE" -gt "$SDF_DONE" ]; then
    python3 scripts/data/convert_pairs_to_sdf.py \
        --pairs_dir "$PAIRS_DIR" \
        --resolution "$SDF_RESOLUTION" \
        --num_workers "$SDF_WORKERS"
    echo "SDF conversion complete!"
else
    echo "All pairs already converted."
fi

# ---- Phase 3: Upload Results to B2 ----
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Phase 3: Upload Results to B2                          ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

if [ -f scripts/data/upload_results_b2.sh ]; then
    bash scripts/data/upload_results_b2.sh
else
    echo "Uploading training files to B2..."
    # Upload only training-relevant files (skip raw GLBs to save bandwidth)
    rclone copy "$PAIRS_DIR/" "b2:${B2_BUCKET}/pairs/" \
        --include "*/coarse_voxels.npy" \
        --include "*/fine_sdf.npy" \
        --include "*/positions.npy" \
        --include "*/cond_features.npy" \
        --include "*/rendered.png" \
        --include "*/meta.json" \
        --include "*/progress.json" \
        --include "*/failed.json" \
        --progress

    echo "Upload complete!"
fi

# ---- Final stats ----
echo ""
echo "============================================================"
FINAL_PAIRS=$(find "$PAIRS_DIR" -name "coarse_voxels.npy" 2>/dev/null | wc -l)
FINAL_SDF=$(find "$PAIRS_DIR" -name "fine_sdf.npy" 2>/dev/null | wc -l)
FINAL_FAILED=$(find "$PAIRS_DIR" -name "failed.json" -exec python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" {} \; 2>/dev/null | paste -sd+ | bc 2>/dev/null || echo 0)
echo "Shard $SHARD_ID Complete!"
echo "  Total pairs:    $FINAL_PAIRS"
echo "  SDF converted:  $FINAL_SDF"
echo "  Failed:         $FINAL_FAILED"
echo "  Input models:   $MODEL_COUNT"
echo "============================================================"

# Kill sync daemon
if [ -n "${SYNC_PID:-}" ]; then
    kill "$SYNC_PID" 2>/dev/null || true
fi

# Final sync
if [ -f scripts/data/sync_progress_b2.sh ]; then
    bash scripts/data/sync_progress_b2.sh
fi
