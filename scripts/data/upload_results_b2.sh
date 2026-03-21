#!/usr/bin/env bash
# ============================================================================
# Upload pair generation results to Backblaze B2.
#
# Uploads only training-relevant files (skips raw GLBs to save bandwidth):
#   - coarse_voxels.npy  (~200KB per pair)
#   - fine_sdf.npy       (~130KB per pair)
#   - positions.npy      (~25KB per pair)
#   - cond_features.npy  (~2MB per pair)
#   - rendered.png       (~500KB per pair)
#   - meta.json          (~1KB per pair)
#
# Total per pair: ~3MB (vs ~25MB with GLBs)
#
# Usage:
#   SHARD_ID=0 PAIRS_DIR=/workspace/data/training_pairs bash upload_results_b2.sh
# ============================================================================
set -uo pipefail

SHARD_ID="${SHARD_ID:?SHARD_ID must be set}"
PAIRS_DIR="${PAIRS_DIR:-/workspace/data/training_pairs}"
B2_BUCKET="${B2_BUCKET:-clearmesh-pairs}"

echo "Uploading shard $SHARD_ID results to B2..."

# Count files to upload
PAIRS=$(find "$PAIRS_DIR/shard_${SHARD_ID}" -name "coarse_voxels.npy" 2>/dev/null | wc -l)
SDF=$(find "$PAIRS_DIR/shard_${SHARD_ID}" -name "fine_sdf.npy" 2>/dev/null | wc -l)
echo "  Pairs to upload: $PAIRS"
echo "  SDF converted:   $SDF"

# Upload training files only
rclone copy "$PAIRS_DIR/shard_${SHARD_ID}/" "b2:${B2_BUCKET}/pairs/shard_${SHARD_ID}/" \
    --include "*/coarse_voxels.npy" \
    --include "*/fine_sdf.npy" \
    --include "*/positions.npy" \
    --include "*/cond_features.npy" \
    --include "*/rendered.png" \
    --include "*/meta.json" \
    --include "progress.json" \
    --include "failed.json" \
    --include "failure_details.json" \
    --transfers 16 \
    --progress

echo ""
echo "Upload complete!"

# Verify
REMOTE_COUNT=$(rclone ls "b2:${B2_BUCKET}/pairs/shard_${SHARD_ID}/" --include "*/coarse_voxels.npy" 2>/dev/null | wc -l)
echo "  Remote pair count: $REMOTE_COUNT"
echo "  Local pair count:  $PAIRS"
