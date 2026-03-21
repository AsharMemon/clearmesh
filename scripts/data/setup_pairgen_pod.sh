#!/usr/bin/env bash
# ============================================================================
# Setup a RunPod A100 pod for ClearMesh pair generation.
#
# This script bootstraps all dependencies needed for generate_pairs.py:
#   - TRELLIS.2 repo + 4B weights
#   - Blender 3.0.1
#   - Python dependencies (spconv, nvdiffrast, flash_attn, etc.)
#   - ClearMesh repo
#
# Usage:
#   # On a fresh RunPod pod with network volume at /workspace:
#   curl -sSL <raw_url>/setup_pairgen_pod.sh | bash
#
#   # Or from the repo:
#   bash scripts/data/setup_pairgen_pod.sh
#
# Prerequisites:
#   - RunPod pod with A100 80GB GPU
#   - Network volume mounted at /workspace
#   - CUDA 12.x + Python 3.10+
# ============================================================================
set -euo pipefail

WORKSPACE="/workspace"
TOOLS_DIR="$WORKSPACE/tools"
TRELLIS_DIR="$WORKSPACE/TRELLIS.2"
MODEL_DIR="$WORKSPACE/models/trellis2-4b"
CLEARMESH_DIR="$WORKSPACE/clearmesh"
BLENDER_VERSION="3.0.1"
BLENDER_DIR="$TOOLS_DIR/blender-${BLENDER_VERSION}-linux-x64"

# Redirect caches to persistent storage
export HF_HOME="$WORKSPACE/.hf_cache"
export TORCH_HOME="$WORKSPACE/.torch_cache"
export TRITON_CACHE_DIR="$WORKSPACE/.triton_cache"
export PIP_CACHE_DIR="$WORKSPACE/.pip_cache"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$PIP_CACHE_DIR"

echo "============================================================"
echo "ClearMesh Pair Generation Pod Setup"
echo "============================================================"
echo "Workspace: $WORKSPACE"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo ""

# ---- Step 1: System dependencies ----
echo "[1/7] System dependencies..."
apt-get update -qq
apt-get install -y -qq libglu1-mesa xvfb git-lfs tmux htop > /dev/null 2>&1
echo "  ✓ System deps installed"

# ---- Step 2: Blender ----
echo "[2/7] Blender ${BLENDER_VERSION}..."
if [ -x "$BLENDER_DIR/blender" ]; then
    echo "  ✓ Blender already installed"
else
    mkdir -p "$TOOLS_DIR"
    cd "$TOOLS_DIR"
    BLENDER_URL="https://download.blender.org/release/Blender${BLENDER_VERSION%.*}/blender-${BLENDER_VERSION}-linux-x64.tar.xz"
    echo "  Downloading from $BLENDER_URL..."
    wget -q "$BLENDER_URL" -O blender.tar.xz
    tar xf blender.tar.xz
    rm blender.tar.xz
    echo "  ✓ Blender installed to $BLENDER_DIR"
fi

# ---- Step 3: TRELLIS.2 repo ----
echo "[3/7] TRELLIS.2 repository..."
if [ -d "$TRELLIS_DIR/.git" ]; then
    echo "  ✓ TRELLIS.2 already cloned"
else
    cd "$WORKSPACE"
    git clone https://github.com/microsoft/TRELLIS.2.git
    echo "  ✓ TRELLIS.2 cloned"
fi

# ---- Step 4: TRELLIS.2 Python deps ----
echo "[4/7] TRELLIS.2 Python dependencies..."
cd "$TRELLIS_DIR"

# Core dependencies
pip install -q torch torchvision --extra-index-url https://download.pytorch.org/whl/cu121 2>/dev/null || true
pip install -q numpy scipy pillow trimesh PyMCubes imageio tqdm pyyaml pandas 2>/dev/null
pip install -q safetensors huggingface_hub transformers einops 2>/dev/null
pip install -q "pyglet<2" xatlas 2>/dev/null

# spconv (needed for sparse structure)
pip install -q spconv-cu120 2>/dev/null || pip install -q spconv-cu121 2>/dev/null || echo "  WARNING: spconv install failed"

# flash-attn (optional but recommended for speed)
pip install -q flash-attn --no-build-isolation 2>/dev/null || echo "  WARNING: flash-attn not installed (will use sdpa fallback)"

# nvdiffrast (needed for FlexiCubes mesh extraction)
pip install -q git+https://github.com/NVlabs/nvdiffrast.git 2>/dev/null || echo "  WARNING: nvdiffrast install failed"

# objaverse (for downloading models)
pip install -q objaverse 2>/dev/null

# Run TRELLIS.2 setup if it has one
if [ -f "setup.py" ] || [ -f "pyproject.toml" ]; then
    pip install -q -e . 2>/dev/null || echo "  WARNING: TRELLIS.2 setup.py install failed"
fi

echo "  ✓ Python dependencies installed"

# ---- Step 5: TRELLIS.2 model weights ----
echo "[5/7] TRELLIS.2 4B model weights..."
if [ -d "$MODEL_DIR/ckpts" ]; then
    echo "  ✓ Model weights already present"
else
    mkdir -p "$MODEL_DIR"
    echo "  Downloading TRELLIS.2-4B weights from HuggingFace..."
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'JeffreyXiang/TRELLIS-image-large',
    local_dir='$MODEL_DIR',
    ignore_patterns=['*.md', '*.txt'],
)
print('  ✓ Weights downloaded')
" 2>/dev/null || echo "  WARNING: Weight download failed — may need manual download"
fi

# ---- Step 6: ClearMesh repo ----
echo "[6/7] ClearMesh repository..."
if [ -d "$CLEARMESH_DIR/.git" ]; then
    echo "  ✓ ClearMesh already cloned, pulling latest..."
    cd "$CLEARMESH_DIR" && git pull --ff-only 2>/dev/null || true
else
    cd "$WORKSPACE"
    # Clone from the user's repo — update URL as needed
    if [ -n "${CLEARMESH_REPO_URL:-}" ]; then
        git clone "$CLEARMESH_REPO_URL" clearmesh
    else
        echo "  Set CLEARMESH_REPO_URL env var to clone automatically"
        echo "  Or: git clone <your-repo-url> $CLEARMESH_DIR"
    fi
fi

if [ -d "$CLEARMESH_DIR" ]; then
    cd "$CLEARMESH_DIR"
    pip install -q -e . 2>/dev/null || true
fi

echo "  ✓ ClearMesh ready"

# ---- Step 7: Verify ----
echo "[7/7] Verification..."
echo -n "  Python: "; python3 --version
echo -n "  CUDA: "; python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "N/A"
echo -n "  GPU: "; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "N/A"
echo -n "  Blender: "; $BLENDER_DIR/blender --version 2>/dev/null | head -1 || echo "N/A"
echo -n "  spconv: "; python3 -c "import spconv; print('OK')" 2>/dev/null || echo "MISSING"
echo -n "  flash_attn: "; python3 -c "import flash_attn; print('OK')" 2>/dev/null || echo "MISSING (will use sdpa)"
echo -n "  trimesh: "; python3 -c "import trimesh; print('OK')" 2>/dev/null || echo "MISSING"
echo -n "  objaverse: "; python3 -c "import objaverse; print('OK')" 2>/dev/null || echo "MISSING"

echo ""
echo "============================================================"
echo "Setup complete!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Download TRELLIS-500K models:"
echo "     python $CLEARMESH_DIR/scripts/data/download_trellis500k.py \\"
echo "         --output_dir /workspace/data/trellis500k --filter"
echo ""
echo "  2. Start pair generation (single shard):"
echo "     cd $CLEARMESH_DIR"
echo "     INPUT_JSON=/workspace/data/trellis500k/valid_models.json \\"
echo "         ./scripts/data/run_pairs_watchdog.sh 0 1"
echo ""
echo "  3. Or with sharding (e.g., shard 0 of 8):"
echo "     INPUT_JSON=/workspace/data/trellis500k/valid_models.json \\"
echo "         ./scripts/data/run_pairs_watchdog.sh 0 8"
