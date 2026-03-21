#!/usr/bin/env bash
# ============================================================================
# Sync pair generation progress to/from Backblaze B2.
#
# Uploads this shard's progress.json and downloads other shards' progress
# for cross-shard dedup. Designed to run as a cron job every 5 minutes
# or as a background daemon.
#
# Usage:
#   # Run once
#   SHARD_ID=0 PAIRS_DIR=/workspace/data/training_pairs bash sync_progress_b2.sh
#
#   # Run as daemon (sync every 5 min)
#   SHARD_ID=0 PAIRS_DIR=/workspace/data/training_pairs bash sync_progress_b2.sh --daemon
# ============================================================================
set -uo pipefail

SHARD_ID="${SHARD_ID:?SHARD_ID must be set}"
PAIRS_DIR="${PAIRS_DIR:-/workspace/data/training_pairs}"
B2_BUCKET="${B2_BUCKET:-clearmesh-pairs}"
SYNC_INTERVAL="${SYNC_INTERVAL:-300}"  # 5 minutes

do_sync() {
    local ts
    ts=$(date '+%Y-%m-%d %H:%M:%S')

    # Upload this shard's progress
    LOCAL_PROGRESS="$PAIRS_DIR/shard_${SHARD_ID}/progress.json"
    LOCAL_FAILED="$PAIRS_DIR/shard_${SHARD_ID}/failed.json"

    if [ -f "$LOCAL_PROGRESS" ]; then
        rclone copy "$LOCAL_PROGRESS" "b2:${B2_BUCKET}/progress/shard_${SHARD_ID}/" 2>/dev/null
    fi
    if [ -f "$LOCAL_FAILED" ]; then
        rclone copy "$LOCAL_FAILED" "b2:${B2_BUCKET}/progress/shard_${SHARD_ID}/" 2>/dev/null
    fi

    # Download other shards' progress (for cross-shard dedup)
    for s in $(rclone lsd "b2:${B2_BUCKET}/progress/" 2>/dev/null | awk '{print $NF}'); do
        if [ "$s" != "shard_${SHARD_ID}" ]; then
            mkdir -p "$PAIRS_DIR/$s"
            rclone copy "b2:${B2_BUCKET}/progress/$s/progress.json" "$PAIRS_DIR/$s/" 2>/dev/null || true
        fi
    done

    # Log counts
    local my_count=0
    if [ -f "$LOCAL_PROGRESS" ]; then
        my_count=$(python3 -c "import json; print(len(json.load(open('$LOCAL_PROGRESS'))))" 2>/dev/null || echo 0)
    fi
    echo "[$ts] Shard $SHARD_ID: $my_count completed, synced to B2"
}

# Main
if [ "${1:-}" = "--daemon" ]; then
    echo "Starting B2 sync daemon (interval: ${SYNC_INTERVAL}s)..."
    while true; do
        do_sync
        sleep "$SYNC_INTERVAL"
    done
else
    do_sync
fi
