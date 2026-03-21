#!/usr/bin/env bash
# ============================================================================
# Launch 500K pair generation across multiple RunPod GPU pods.
#
# This script manages the full pipeline:
#   Phase 1: Download TRELLIS-500K model files (CPU-bound, runs on any pod)
#   Phase 2: Launch pair generation across N GPU pods with sharding
#   Phase 3: Post-process: SDF conversion + manifest building
#
# Architecture:
#   - All pods share a RunPod network volume at /workspace
#   - Each pod runs one shard of the pair generation
#   - Cross-shard dedup prevents duplicate work
#   - Watchdog auto-restarts crashed workers
#
# Usage:
#   # On any pod connected to the shared network volume:
#   bash scripts/data/launch_500k_pairgen.sh --phase download
#   bash scripts/data/launch_500k_pairgen.sh --phase generate --shard_id 0 --num_shards 8
#   bash scripts/data/launch_500k_pairgen.sh --phase postprocess
#
#   # Or run all phases (single pod, for testing):
#   bash scripts/data/launch_500k_pairgen.sh --phase all --num_shards 1
#
# Multi-pod deployment (8 A100 pods):
#   Pod 0: bash launch_500k_pairgen.sh --phase generate --shard_id 0 --num_shards 8
#   Pod 1: bash launch_500k_pairgen.sh --phase generate --shard_id 1 --num_shards 8
#   ...
#   Pod 7: bash launch_500k_pairgen.sh --phase generate --shard_id 7 --num_shards 8
# ============================================================================
set -euo pipefail

# ---- Configuration ----
WORKSPACE="${WORKSPACE:-/workspace}"
CLEARMESH_DIR="${CLEARMESH_DIR:-$WORKSPACE/clearmesh}"
DATA_DIR="${DATA_DIR:-$WORKSPACE/data}"
TRELLIS500K_DIR="${TRELLIS500K_DIR:-$DATA_DIR/trellis500k}"
PAIRS_DIR="${PAIRS_DIR:-$DATA_DIR/training_pairs_500k}"
INPUT_JSON="${INPUT_JSON:-$TRELLIS500K_DIR/valid_models.json}"

# Download settings
DOWNLOAD_SOURCES="${DOWNLOAD_SOURCES:-github sketchfab}"  # Skip ABO/3D-FUTURE/HSSD for now
DOWNLOAD_PROCESSES="${DOWNLOAD_PROCESSES:-16}"

# Generation settings
NUM_SHARDS="${NUM_SHARDS:-8}"
SHARD_ID="${SHARD_ID:-0}"
PIPELINE_TYPE="${PIPELINE_TYPE:-512}"
NUM_VIEWS="${NUM_VIEWS:-6}"
RENDER_SIZE="${RENDER_SIZE:-1024}"
MAX_MODELS_PER_RUN="${MAX_MODELS_PER_RUN:-150}"
RSS_HARD_LIMIT_GB="${RSS_HARD_LIMIT_GB:-200}"

# Post-processing settings
SDF_WORKERS="${SDF_WORKERS:-8}"
SDF_RESOLUTION="${SDF_RESOLUTION:-32}"

# Cache dirs (persistent across pod restarts)
export HF_HOME="$WORKSPACE/.hf_cache"
export TORCH_HOME="$WORKSPACE/.torch_cache"
export TRITON_CACHE_DIR="$WORKSPACE/.triton_cache"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR"

# ---- Parse arguments ----
PHASE="all"
while [[ $# -gt 0 ]]; do
    case $1 in
        --phase)       PHASE="$2"; shift 2 ;;
        --shard_id)    SHARD_ID="$2"; shift 2 ;;
        --num_shards)  NUM_SHARDS="$2"; shift 2 ;;
        --input_json)  INPUT_JSON="$2"; shift 2 ;;
        --output_dir)  PAIRS_DIR="$2"; shift 2 ;;
        --sources)     DOWNLOAD_SOURCES="$2"; shift 2 ;;
        *)             echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "============================================================"
echo "ClearMesh 500K Pair Generation"
echo "============================================================"
echo "Phase:      $PHASE"
echo "Shard:      $SHARD_ID / $NUM_SHARDS"
echo "Input:      $INPUT_JSON"
echo "Output:     $PAIRS_DIR"
echo "Workspace:  $WORKSPACE"
echo "============================================================"

# ---- Phase 1: Download ----
run_download() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║  PHASE 1: Download TRELLIS-500K Models                  ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo ""

    cd "$CLEARMESH_DIR"

    # Check if valid_models.json already exists with enough entries
    if [ -f "$INPUT_JSON" ]; then
        COUNT=$(python3 -c "import json; print(len(json.load(open('$INPUT_JSON'))))" 2>/dev/null || echo 0)
        echo "Existing valid_models.json has $COUNT entries"
        if [ "$COUNT" -gt 100000 ]; then
            echo "Skipping download — already have $COUNT models"
            return 0
        fi
    fi

    python3 scripts/data/download_trellis500k.py \
        --output_dir "$TRELLIS500K_DIR" \
        --sources $DOWNLOAD_SOURCES \
        --processes "$DOWNLOAD_PROCESSES" \
        --filter

    echo ""
    echo "Download complete!"
    echo "Models at: $TRELLIS500K_DIR"
    echo "Input JSON: $INPUT_JSON"
}

# ---- Phase 2: Generate pairs ----
run_generate() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║  PHASE 2: Generate Pairs (shard $SHARD_ID/$NUM_SHARDS)                ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo ""

    # Verify input exists
    if [ ! -f "$INPUT_JSON" ]; then
        echo "ERROR: Input JSON not found: $INPUT_JSON"
        echo "Run with --phase download first"
        exit 1
    fi

    COUNT=$(python3 -c "import json; print(len(json.load(open('$INPUT_JSON'))))" 2>/dev/null || echo 0)
    SHARD_SIZE=$((COUNT / NUM_SHARDS))
    echo "Total models: $COUNT"
    echo "Shard size: ~$SHARD_SIZE models"
    echo ""

    cd "$CLEARMESH_DIR"

    # Set environment for the watchdog
    export INPUT_JSON="$INPUT_JSON"
    export OUTPUT_DIR="$PAIRS_DIR"
    export PIPELINE_TYPE="$PIPELINE_TYPE"
    export NUM_VIEWS="$NUM_VIEWS"
    export RENDER_SIZE="$RENDER_SIZE"
    export MAX_MODELS_PER_RUN="$MAX_MODELS_PER_RUN"
    export RSS_HARD_LIMIT_GB="$RSS_HARD_LIMIT_GB"

    # Launch via watchdog (handles restarts, OOM, etc.)
    echo "Launching watchdog for shard $SHARD_ID..."
    echo "Log: $PAIRS_DIR/shard_${SHARD_ID}.log"
    echo ""

    # Run in foreground (use tmux/screen for background)
    bash scripts/data/run_pairs_watchdog.sh "$SHARD_ID" "$NUM_SHARDS" 2>&1 | \
        tee "$PAIRS_DIR/shard_${SHARD_ID}.log"
}

# ---- Phase 3: Post-process ----
run_postprocess() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║  PHASE 3: Post-processing (SDF + Manifest)              ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo ""

    cd "$CLEARMESH_DIR"

    # Count completed pairs
    PAIRS=$(find "$PAIRS_DIR" -name "coarse_voxels.npy" 2>/dev/null | wc -l)
    SDF_DONE=$(find "$PAIRS_DIR" -name "fine_sdf.npy" 2>/dev/null | wc -l)
    echo "Completed pairs: $PAIRS"
    echo "SDF converted:   $SDF_DONE"
    echo ""

    # Step 3a: Convert to SDF
    echo "--- SDF Conversion ---"
    python3 scripts/data/convert_pairs_to_sdf.py \
        --pairs_dir "$PAIRS_DIR" \
        --resolution "$SDF_RESOLUTION" \
        --num_workers "$SDF_WORKERS"

    # Step 3b: Build manifest
    echo ""
    echo "--- Build Manifest ---"
    python3 scripts/data/build_manifest.py \
        --pairs_dir "$PAIRS_DIR" \
        --output "$PAIRS_DIR/manifest_train.json"

    echo ""
    echo "Post-processing complete!"
    FINAL_SDF=$(find "$PAIRS_DIR" -name "fine_sdf.npy" 2>/dev/null | wc -l)
    echo "Trainable pairs: $FINAL_SDF"
}

# ---- Progress monitor ----
run_monitor() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║  Progress Monitor                                       ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo ""

    while true; do
        TOTAL_PAIRS=$(find "$PAIRS_DIR" -name "coarse_voxels.npy" 2>/dev/null | wc -l)
        TOTAL_SDF=$(find "$PAIRS_DIR" -name "fine_sdf.npy" 2>/dev/null | wc -l)
        TOTAL_FAILED=0

        echo "$(date '+%Y-%m-%d %H:%M:%S') | Pairs: $TOTAL_PAIRS | SDF: $TOTAL_SDF"

        for s in $(seq 0 $((NUM_SHARDS - 1))); do
            SHARD_DIR="$PAIRS_DIR/shard_$s"
            if [ -d "$SHARD_DIR" ]; then
                S_DONE=$(find "$SHARD_DIR" -name "coarse_voxels.npy" 2>/dev/null | wc -l)
                S_PROG=$(cat "$SHARD_DIR/progress.json" 2>/dev/null | python3 -c "import json,sys; print(len(json.load(sys.stdin)))" 2>/dev/null || echo 0)
                S_FAIL=$(cat "$SHARD_DIR/failed.json" 2>/dev/null | python3 -c "import json,sys; print(len(json.load(sys.stdin)))" 2>/dev/null || echo 0)
                echo "  Shard $s: done=$S_DONE attempted=$S_PROG failed=$S_FAIL"
                TOTAL_FAILED=$((TOTAL_FAILED + S_FAIL))
            fi
        done

        INPUT_COUNT=$(python3 -c "import json; print(len(json.load(open('$INPUT_JSON'))))" 2>/dev/null || echo "?")
        TOTAL_ATTEMPTED=$((TOTAL_PAIRS + TOTAL_FAILED))
        echo "  Total: $TOTAL_ATTEMPTED / $INPUT_COUNT attempted | $TOTAL_PAIRS succeeded | $TOTAL_FAILED failed"

        if [ "$TOTAL_ATTEMPTED" -gt 0 ] 2>/dev/null; then
            SUCCESS_RATE=$(python3 -c "print(f'{$TOTAL_PAIRS/$TOTAL_ATTEMPTED*100:.1f}%')" 2>/dev/null || echo "?")
            echo "  Success rate: $SUCCESS_RATE"
        fi

        echo ""
        sleep 300  # Check every 5 minutes
    done
}

# ---- Main ----
case "$PHASE" in
    download)
        run_download
        ;;
    generate)
        run_generate
        ;;
    postprocess)
        run_postprocess
        ;;
    monitor)
        run_monitor
        ;;
    all)
        run_download
        run_generate
        run_postprocess
        ;;
    *)
        echo "Unknown phase: $PHASE"
        echo "Valid phases: download, generate, postprocess, monitor, all"
        exit 1
        ;;
esac
