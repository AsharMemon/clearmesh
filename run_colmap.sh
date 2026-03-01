#!/bin/bash
# ============================================================
# run_colmap.sh — Process phone photos through COLMAP
# Usage: bash scripts/run_colmap.sh data/test_mug
# ============================================================

INPUT_DIR="$1"

if [ -z "$INPUT_DIR" ]; then
    echo "Usage: bash scripts/run_colmap.sh <photo_folder>"
    echo "Example: bash scripts/run_colmap.sh data/test_mug"
    exit 1
fi

# Resolve to absolute path
INPUT_DIR=$(realpath "$INPUT_DIR")
WORKSPACE="$INPUT_DIR/colmap_workspace"
DATABASE="$WORKSPACE/database.db"
SPARSE="$WORKSPACE/sparse"
DENSE="$WORKSPACE/dense"

echo "============================================"
echo "  COLMAP Processing"
echo "  Photos: $INPUT_DIR"
echo "  Workspace: $WORKSPACE"
echo "============================================"

# Count photos
NUM_PHOTOS=$(ls "$INPUT_DIR"/*.{jpg,jpeg,png,JPG,JPEG,PNG} 2>/dev/null | wc -l)
echo "  Found $NUM_PHOTOS photos"

if [ "$NUM_PHOTOS" -lt 10 ]; then
    echo "  WARNING: Fewer than 10 photos. You need 20-60 for good results."
fi

mkdir -p "$SPARSE" "$DENSE"

# --------------------------------------------------
# Step 1: Feature Extraction
# --------------------------------------------------
echo ""
echo ">>> Step 1/6: Extracting features..."
colmap feature_extractor \
    --database_path "$DATABASE" \
    --image_path "$INPUT_DIR" \
    --ImageReader.camera_model OPENCV \
    --ImageReader.single_camera 1 \
    --SiftExtraction.use_gpu 1

echo "  Features extracted."

# --------------------------------------------------
# Step 2: Feature Matching
# --------------------------------------------------
echo ""
echo ">>> Step 2/6: Matching features..."

if [ "$NUM_PHOTOS" -le 100 ]; then
    echo "  Using exhaustive matching ($NUM_PHOTOS photos)..."
    colmap exhaustive_matcher \
        --database_path "$DATABASE" \
        --SiftMatching.use_gpu 1
else
    echo "  Using sequential matching ($NUM_PHOTOS photos, too many for exhaustive)..."
    colmap sequential_matcher \
        --database_path "$DATABASE" \
        --SiftMatching.use_gpu 1
fi

echo "  Matching done."

# --------------------------------------------------
# Step 3: Sparse Reconstruction (Structure-from-Motion)
# --------------------------------------------------
echo ""
echo ">>> Step 3/6: Running sparse reconstruction (SfM)..."
colmap mapper \
    --database_path "$DATABASE" \
    --image_path "$INPUT_DIR" \
    --output_path "$SPARSE"

# Check if reconstruction succeeded
if [ ! -d "$SPARSE/0" ]; then
    echo "  ERROR: Sparse reconstruction failed."
    echo "  Common causes:"
    echo "    - Too few photos or not enough overlap"
    echo "    - Blurry photos"
    echo "    - Textureless/reflective object"
    echo "  Try taking more photos with 70%+ overlap."
    exit 1
fi

echo "  Sparse reconstruction succeeded."

# Print stats
colmap model_analyzer \
    --path "$SPARSE/0" 2>/dev/null || true

# --------------------------------------------------
# Step 4: Undistort Images
# --------------------------------------------------
echo ""
echo ">>> Step 4/6: Undistorting images..."
colmap image_undistorter \
    --image_path "$INPUT_DIR" \
    --input_path "$SPARSE/0" \
    --output_path "$DENSE" \
    --output_type COLMAP

echo "  Images undistorted."

# --------------------------------------------------
# Step 5: Dense Reconstruction (Patch Match Stereo)
# --------------------------------------------------
echo ""
echo ">>> Step 5/6: Running dense reconstruction (this takes a few minutes)..."
colmap patch_match_stereo \
    --workspace_path "$DENSE" \
    --workspace_format COLMAP \
    --PatchMatchStereo.geom_consistency true

echo "  Dense reconstruction done."

# --------------------------------------------------
# Step 6: Fuse into Dense Point Cloud
# --------------------------------------------------
echo ""
echo ">>> Step 6/6: Fusing dense point cloud..."
colmap stereo_fusion \
    --workspace_path "$DENSE" \
    --workspace_format COLMAP \
    --output_path "$DENSE/fused.ply"

echo ""
echo "============================================"
echo "  COLMAP COMPLETE"
echo "============================================"
echo ""
echo "  Output files:"
echo "    Sparse model:      $SPARSE/0/"
echo "    Dense point cloud: $DENSE/fused.ply"
echo "    Undistorted imgs:  $DENSE/images/"
echo "    Camera params:     $DENSE/sparse/"
echo ""
echo "  Next step: Run Instant-NSR-PL"
echo "    cd ~/meshclean/repos/instant-nsr-pl"
echo "    python launch.py --config configs/neus-colmap.yaml \\"
echo "      --train --gpu 0 \\"
echo "      dataset.root_dir=$DENSE \\"
echo "      dataset.img_wh=[800,600]"
echo ""
echo "  Or run the full pipeline:"
echo "    python ~/meshclean/scripts/pipeline.py $INPUT_DIR"
echo ""
echo "============================================"
