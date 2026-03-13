#!/bin/bash
# Setup script for UltraShape refinement experiment
# Run this on a GPU machine (A100/4090, 16-32GB VRAM)
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ULTRASHAPE_DIR="$SCRIPT_DIR/UltraShape-1.0"
CHECKPOINT_DIR="$SCRIPT_DIR/checkpoints"

echo "=== UltraShape Refinement Experiment Setup ==="

# 1. Clone UltraShape repo
if [ ! -d "$ULTRASHAPE_DIR" ]; then
    echo "[1/4] Cloning UltraShape-1.0..."
    git clone https://github.com/PKU-YuanGroup/UltraShape-1.0.git "$ULTRASHAPE_DIR"
else
    echo "[1/4] UltraShape-1.0 already cloned."
fi

# 2. Install UltraShape dependencies
echo "[2/4] Installing UltraShape dependencies..."
pip install omegaconf einops rembg trimesh diso sageattention 2>/dev/null || true

# flash-attn needs special install
pip install flash-attn --no-build-isolation 2>/dev/null || {
    echo "WARNING: flash-attn install failed. Trying without..."
    echo "You may need: pip install flash-attn --no-build-isolation"
}

# Install cubvh for marching cubes acceleration (optional)
pip install cubvh 2>/dev/null || echo "WARNING: cubvh not installed (optional)"

# diffusers for scheduler
pip install diffusers>=0.30.0 2>/dev/null || true

# 3. Download UltraShape checkpoint from HuggingFace
mkdir -p "$CHECKPOINT_DIR"
if [ ! -f "$CHECKPOINT_DIR/ultrashape_v1.pt" ]; then
    echo "[3/4] Downloading UltraShape checkpoint (~7.4GB)..."
    huggingface-cli download infinith/UltraShape ultrashape_v1.pt \
        --local-dir "$CHECKPOINT_DIR" \
        --local-dir-use-symlinks False
else
    echo "[3/4] Checkpoint already downloaded."
fi

# 4. Download VAE checkpoint
if [ ! -f "$CHECKPOINT_DIR/vae/vae_step=15000.ckpt" ]; then
    echo "[4/4] Downloading VAE checkpoint..."
    mkdir -p "$CHECKPOINT_DIR/vae"
    huggingface-cli download infinith/UltraShape vae/vae_step=15000.ckpt \
        --local-dir "$CHECKPOINT_DIR" \
        --local-dir-use-symlinks False
else
    echo "[4/4] VAE checkpoint already downloaded."
fi

# Create input/output directories
mkdir -p "$SCRIPT_DIR/inputs/images"
mkdir -p "$SCRIPT_DIR/inputs/coarse_meshes"
mkdir -p "$SCRIPT_DIR/outputs"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Place your reference images in:  experiments/ultrashape/inputs/images/"
echo "  2. Place TRELLIS.2 coarse GLBs in:  experiments/ultrashape/inputs/coarse_meshes/"
echo "  3. Run:  python experiments/ultrashape/refine.py \\"
echo "             --image inputs/images/example.png \\"
echo "             --mesh inputs/coarse_meshes/example.glb"
echo ""
echo "Or run the batch experiment:"
echo "  python experiments/ultrashape/run_experiment.py"
