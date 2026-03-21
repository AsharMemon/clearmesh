#!/usr/bin/env bash
# ============================================================================
# Setup a Vast.ai A100 pod for ClearMesh pair generation.
#
# Unlike RunPod, Vast.ai pods don't share a network volume.
# This script downloads model files + weights from Backblaze B2.
#
# Docker image: pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
#   - Already has PyTorch 2.6.0 + CUDA 12.4 (required by TRELLIS.2)
#
# Environment variables (set via Vast.ai --env or onstart script):
#   SHARD_ID       - Which shard this pod processes (0-7)
#   NUM_SHARDS     - Total number of shards
#   B2_KEY_ID      - Backblaze B2 application key ID
#   B2_APP_KEY     - Backblaze B2 application key
#   B2_BUCKET      - B2 bucket name (default: clearmesh-pairs)
#   HF_TOKEN       - HuggingFace token for gated models (DINOv3)
#
# Expects:
#   - ClearMesh repo already at $WORKSPACE/clearmesh (extracted by onstart)
#   - rclone already installed and configured (done by onstart)
# ============================================================================
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
B2_BUCKET="${B2_BUCKET:-clearmesh-pairs}"
SHARD_ID="${SHARD_ID:?SHARD_ID must be set}"
NUM_SHARDS="${NUM_SHARDS:-8}"

TRELLIS_DIR="$WORKSPACE/TRELLIS.2"
MODEL_DIR="$WORKSPACE/models/trellis2-4b"
CLEARMESH_DIR="$WORKSPACE/clearmesh"
TOOLS_DIR="$WORKSPACE/tools"
BLENDER_VERSION="3.0.1"
BLENDER_DIR="$TOOLS_DIR/blender-${BLENDER_VERSION}-linux-x64"
DATA_DIR="$WORKSPACE/data"
SHARD_DIR="$DATA_DIR/shards"
PAIRS_DIR="$DATA_DIR/training_pairs"

# Redirect caches to workspace (persistent on Vast.ai local disk)
export HF_HOME="$WORKSPACE/.hf_cache"
export TORCH_HOME="$WORKSPACE/.torch_cache"
export TRITON_CACHE_DIR="$WORKSPACE/.triton_cache"
export PIP_CACHE_DIR="$WORKSPACE/.pip_cache"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$PIP_CACHE_DIR"
mkdir -p "$SHARD_DIR" "$PAIRS_DIR" "$DATA_DIR/models"

echo "============================================================"
echo "ClearMesh Vast.ai Pod Setup — Shard $SHARD_ID / $NUM_SHARDS"
echo "============================================================"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Disk: $(df -h $WORKSPACE | tail -1 | awk '{print $2 " total, " $4 " free"}')"
echo ""

# ---- Step 1: System deps ----
echo "[1/9] System dependencies..."
apt-get update -qq
apt-get install -y -qq libglu1-mesa xvfb git-lfs tmux htop unzip libjpeg-dev \
    libxi6 libxrandr2 libxfixes3 libxcursor1 libxinerama1 libxxf86vm1 libxrender1 > /dev/null 2>&1
# rclone should already be installed by onstart, but install if missing
if ! command -v rclone &> /dev/null; then
    curl -sSL https://rclone.org/install.sh | bash 2>/dev/null
fi
echo "  ✓ System deps"

# ---- Step 2: TRELLIS.2 repo (from HuggingFace Spaces) ----
# NOTE: TRELLIS 2 is hosted on HuggingFace Spaces, NOT GitHub.
# github.com/microsoft/TRELLIS is v1; huggingface.co/spaces/microsoft/TRELLIS.2 is v2.
echo "[2/9] TRELLIS.2 repository..."
if [ -d "$TRELLIS_DIR/trellis2" ]; then
    echo "  ✓ TRELLIS.2 already cloned"
else
    cd "$WORKSPACE"
    rm -rf "$TRELLIS_DIR"
    GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/spaces/microsoft/TRELLIS.2 TRELLIS.2
    echo "  ✓ TRELLIS.2 cloned from HuggingFace Spaces"
fi

# ---- Step 3: TRELLIS.2 dependencies ----
# TRELLIS.2 from HF Spaces has custom CUDA wheels (cumesh, flex_gemm, o_voxel,
# nvdiffrast, nvdiffrec_render) that are cp310-only. Instead of building from
# source, we use spconv as the sparse conv backend and create stubs for packages
# only needed for final mesh decoding (not for intermediate extraction).
echo "[3/9] TRELLIS.2 Python dependencies..."
cd "$TRELLIS_DIR"

# Install basic Python deps
pip install -q easydict kornia timm pillow-heif einops imageio \
    imageio-ffmpeg rembg onnxruntime transformers accelerate \
    safetensors huggingface_hub 2>/dev/null

# Install utils3d from PyPI
pip install -q utils3d 2>/dev/null

# spconv as sparse conv backend (works across Python versions)
pip install -q spconv-cu120 2>/dev/null || pip install -q spconv-cu124 2>/dev/null || {
    echo "  WARNING: spconv failed, trying spconv-cu126..."
    pip install -q spconv-cu126 2>/dev/null || echo "  ERROR: No spconv wheel found"
}

# flash-attn for attention backend
pip install -q flash-attn 2>/dev/null || {
    echo "  WARNING: flash-attn failed, trying xformers..."
    pip install -q xformers 2>/dev/null || true
}

# Create stub modules for packages only needed for mesh decoding/rendering.
# Pair generation only needs intermediates (coarse voxels, positions, cond features),
# not the final mesh decode step which requires these CUDA extensions.
echo "  Creating stub modules for optional CUDA extensions..."
for pkg in cumesh o_voxel flex_gemm nvdiffrast nvdiffrec_render; do
    PKG_DIR="$TRELLIS_DIR/$pkg"
    if [ ! -d "$PKG_DIR" ] || [ ! -f "$PKG_DIR/__init__.py" ]; then
        mkdir -p "$PKG_DIR"
        cat > "$PKG_DIR/__init__.py" << 'STUBEOF'
"""Stub module — the real package requires cp310-only CUDA wheels.
Only needed for final mesh decoding, not for intermediate extraction."""
class _Stub:
    def __getattr__(self, name): return _Stub()
    def __call__(self, *a, **kw): return _Stub()
def __getattr__(name): return _Stub()
STUBEOF
    fi
done

# Extra submodule stubs
mkdir -p "$TRELLIS_DIR/o_voxel"
cat > "$TRELLIS_DIR/o_voxel/convert.py" << 'STUBEOF'
def flexible_dual_grid_to_mesh(*args, **kwargs):
    raise NotImplementedError("o_voxel.convert requires cp310-only CUDA wheel")
STUBEOF

mkdir -p "$TRELLIS_DIR/flex_gemm/ops" "$TRELLIS_DIR/flex_gemm/kernels"
touch "$TRELLIS_DIR/flex_gemm/ops/__init__.py"
touch "$TRELLIS_DIR/flex_gemm/kernels/__init__.py"
cat > "$TRELLIS_DIR/flex_gemm/ops/spconv.py" << 'STUBEOF'
def __getattr__(name):
    class _Stub:
        def __getattr__(self, n): return _Stub()
        def __call__(self, *a, **kw): return _Stub()
    return _Stub()
STUBEOF
cp "$TRELLIS_DIR/flex_gemm/ops/spconv.py" "$TRELLIS_DIR/flex_gemm/ops/grid_sample.py"
cat > "$TRELLIS_DIR/flex_gemm/kernels/cuda.py" << 'STUBEOF'
def __getattr__(name):
    class _Stub:
        def __getattr__(self, n): return _Stub()
        def __call__(self, *a, **kw): return _Stub()
    return _Stub()
STUBEOF

mkdir -p "$TRELLIS_DIR/nvdiffrast"
cat > "$TRELLIS_DIR/nvdiffrast/torch.py" << 'STUBEOF'
def __getattr__(name):
    class _Stub:
        def __getattr__(self, n): return _Stub()
        def __call__(self, *a, **kw): return _Stub()
    return _Stub()
STUBEOF

mkdir -p "$TRELLIS_DIR/nvdiffrec_render"
cat > "$TRELLIS_DIR/nvdiffrec_render/light.py" << 'STUBEOF'
def __getattr__(name):
    class _Stub:
        def __getattr__(self, n): return _Stub()
        def __call__(self, *a, **kw): return _Stub()
    return _Stub()
STUBEOF

# Set environment variables for spconv backend
export SPARSE_CONV_BACKEND=spconv
export ATTN_BACKEND=flash_attn

echo "  ✓ TRELLIS.2 deps installed (spconv backend + stub modules)"

# Patch BiRefNet.py for dtype mismatch (model loads in fp16, input is fp32)
BIREFNET_PATH="$TRELLIS_DIR/trellis2/pipelines/rembg/BiRefNet.py"
if [ -f "$BIREFNET_PATH" ]; then
    if grep -q 'to("cuda")' "$BIREFNET_PATH" 2>/dev/null; then
        sed -i 's/\.to("cuda")/\.to(device=next(self.model.parameters()).device, dtype=next(self.model.parameters()).dtype)/g' "$BIREFNET_PATH"
        echo "  ✓ BiRefNet.py patched for dtype compatibility"
    fi
fi

# Configure HuggingFace token for gated models (DINOv3)
HF_TOKEN="${HF_TOKEN:-}"
if [ -n "$HF_TOKEN" ]; then
    python3 -c "from huggingface_hub import login; login(token='$HF_TOKEN')" 2>/dev/null
    echo "  ✓ HuggingFace token configured"
fi

# Verify TRELLIS.2 import
python3 -c "
import sys; sys.path.insert(0, '$TRELLIS_DIR')
import os; os.environ['SPARSE_CONV_BACKEND']='spconv'; os.environ['ATTN_BACKEND']='flash_attn'
from trellis2.pipelines import Trellis2ImageTo3DPipeline
print('  ✓ TRELLIS.2 import verified')
" 2>/dev/null || echo "  WARNING: TRELLIS.2 import check failed"

# ---- Step 4: Additional Python deps ----
echo "[4/9] Additional Python dependencies..."
# objaverse (for model downloading)
pip install -q objaverse pandas 2>/dev/null

# Install ClearMesh
if [ -d "$CLEARMESH_DIR" ]; then
    cd "$CLEARMESH_DIR"
    pip install -q -e . 2>/dev/null || true
fi

echo "  ✓ Additional deps installed"

# ---- Step 5: Blender ----
echo "[5/9] Blender ${BLENDER_VERSION}..."
if [ -x "$BLENDER_DIR/blender" ]; then
    echo "  ✓ Blender already installed"
else
    mkdir -p "$TOOLS_DIR"
    cd "$TOOLS_DIR"
    BLENDER_URL="https://download.blender.org/release/Blender${BLENDER_VERSION%.*}/blender-${BLENDER_VERSION}-linux-x64.tar.xz"
    wget -q "$BLENDER_URL" -O blender.tar.xz
    tar xf blender.tar.xz
    rm blender.tar.xz
    echo "  ✓ Blender installed"
fi

# Check GPU driver compatibility with Blender 3.0.1 Cycles
# Drivers < 565 crash Blender 3.0.1's Cycles GPU rendering on various GPUs
DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo "0")
DRIVER_MAJOR=$(echo "$DRIVER_VER" | cut -d. -f1)
if [ "$DRIVER_MAJOR" -lt 565 ] 2>/dev/null; then
    echo "  ⚠ Driver $DRIVER_VER < 565 — forcing CPU rendering for Blender"
    export BLENDER_FORCE_CPU=1
fi

# ---- Step 6a: TRELLIS.2 weights (from B2, ~16GB) ----
echo "[6/9] TRELLIS.2 model weights..."
if [ -d "$MODEL_DIR/ckpts" ]; then
    echo "  ✓ Weights already present"
else
    if rclone ls "b2:${B2_BUCKET}/models/trellis2-4b/ckpts/" > /dev/null 2>&1; then
        echo "  Downloading weights from B2..."
        mkdir -p "$MODEL_DIR"
        rclone copy "b2:${B2_BUCKET}/models/trellis2-4b/" "$MODEL_DIR/" --progress --transfers 8
        echo "  ✓ Weights downloaded from B2"
    else
        echo "  Downloading weights from HuggingFace..."
        mkdir -p "$MODEL_DIR"
        python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'JeffreyXiang/TRELLIS-image-large',
    local_dir='$MODEL_DIR',
    ignore_patterns=['*.md', '*.txt'],
)
print('  ✓ Weights downloaded from HF')
"
    fi
fi

# ---- Step 6b: DINOv3 weights (gated on HF, use B2 or ungated mirror) ----
DINOV3_DIR="$WORKSPACE/models/dinov3-vitl16-local"
echo "  DINOv3 image encoder..."
if [ -f "$DINOV3_DIR/model.safetensors" ]; then
    echo "  ✓ DINOv3 already present"
else
    mkdir -p "$DINOV3_DIR"
    if rclone ls "b2:${B2_BUCKET}/models/dinov3-vitl16-local/model.safetensors" > /dev/null 2>&1; then
        echo "  Downloading DINOv3 from B2..."
        rclone copy "b2:${B2_BUCKET}/models/dinov3-vitl16-local/" "$DINOV3_DIR/" --progress
        echo "  ✓ DINOv3 downloaded from B2"
    else
        echo "  Downloading DINOv3 from HF mirror..."
        python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'tao-hunter/dinov3-vitl16-pretrain-lvd1689m',
    local_dir='$DINOV3_DIR',
    ignore_patterns=['*.md', '*.txt', 'LICENSE*'],
)
print('  ✓ DINOv3 downloaded from HF mirror')
"
    fi
fi

# Patch pipeline.json to use local DINOv3 path
python3 -c "
import json
path = '$MODEL_DIR/pipeline.json'
with open(path) as f:
    pipeline = json.load(f)
pipeline['args']['image_cond_model']['args']['model_name'] = '$DINOV3_DIR'
with open(path, 'w') as f:
    json.dump(pipeline, f, indent=2)
print('  ✓ pipeline.json patched for local DINOv3')
"

# ---- Step 7: Download shard JSON from B2 ----
echo "[7/9] Downloading shard $SHARD_ID JSON..."
rclone copy "b2:${B2_BUCKET}/shards/shard_${SHARD_ID}.json" "$SHARD_DIR/" --progress
echo "  ✓ Shard JSON downloaded"

# ---- Step 8: Download model files from ObjaverseXL ----
echo "[8/9] Downloading model files from ObjaverseXL (~64GB)..."
cd "$CLEARMESH_DIR"
python3 scripts/data/download_shard_models.py \
    --shard_json "$SHARD_DIR/shard_${SHARD_ID}.json" \
    --download_dir "$DATA_DIR/models" \
    --processes 16

# Download cross-shard progress for dedup
echo "  Downloading cross-shard progress files..."
for s in $(seq 0 $((NUM_SHARDS - 1))); do
    if [ "$s" != "$SHARD_ID" ]; then
        rclone copy "b2:${B2_BUCKET}/progress/shard_${s}/progress.json" \
            "$PAIRS_DIR/shard_${s}/" 2>/dev/null || true
    fi
done
echo "  ✓ Shard data ready"

# ---- Step 9: Setup progress sync ----
echo "[9/9] Setting up progress sync..."
SYNC_SCRIPT="$CLEARMESH_DIR/scripts/data/sync_progress_b2.sh"
if [ -f "$SYNC_SCRIPT" ]; then
    chmod +x "$SYNC_SCRIPT"
    # Install cron if missing (some Docker images don't have it)
    if ! command -v crontab &> /dev/null; then
        apt-get install -y -qq cron > /dev/null 2>&1 || true
    fi
    if command -v crontab &> /dev/null; then
        # Use explicit true fallback to avoid pipefail issue with empty crontab
        ( (crontab -l 2>/dev/null || true); echo "*/5 * * * * SHARD_ID=$SHARD_ID B2_BUCKET=$B2_BUCKET PAIRS_DIR=$PAIRS_DIR $SYNC_SCRIPT >> /var/log/b2_sync.log 2>&1") | crontab - || true
        echo "  ✓ Progress sync cron installed"
    else
        echo "  WARNING: crontab not available, using daemon mode for sync"
    fi
fi

# ---- Verify ----
echo ""
echo "============================================================"
echo "Verification"
echo "============================================================"
echo -n "  Python: "; python3 --version
echo -n "  PyTorch: "; python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "N/A"
echo -n "  CUDA: "; python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "N/A"
echo -n "  GPU: "; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "N/A"
echo -n "  TRELLIS import: "; cd "$TRELLIS_DIR" && SPARSE_CONV_BACKEND=spconv ATTN_BACKEND=flash_attn python3 -c "import sys; sys.path.insert(0,'.'); from trellis2.pipelines import Trellis2ImageTo3DPipeline; print('OK')" 2>/dev/null || echo "FAILED"
echo -n "  Blender: "; $BLENDER_DIR/blender --version 2>/dev/null | head -1 || echo "N/A"
echo -n "  Shard JSON: "; [ -f "$SHARD_DIR/shard_${SHARD_ID}.json" ] && echo "OK ($(python3 -c "import json; print(len(json.load(open('$SHARD_DIR/shard_${SHARD_ID}.json'))))" 2>/dev/null) models)" || echo "MISSING"
echo -n "  Disk free: "; df -h $WORKSPACE | tail -1 | awk '{print $4}'

MODEL_COUNT=$(find "$DATA_DIR/models" -name "*.glb" -o -name "*.obj" -o -name "*.fbx" 2>/dev/null | wc -l)
echo "  Model files: $MODEL_COUNT"

echo ""
echo "============================================================"
echo "Setup complete! Ready for pair generation."
echo "============================================================"
echo ""
echo "To start pair generation:"
echo "  cd $CLEARMESH_DIR"
echo "  bash scripts/data/run_pairs_vastai.sh"
