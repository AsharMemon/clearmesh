#!/usr/bin/env bash
# Install UltraShape reference code and download Stage 2 checkpoint.
# UltraShape's Stage 2 training code is adapted for ClearMesh's refinement model.
#
# Usage:
#   source /home/ubuntu/clearmesh-venv/bin/activate
#   scripts/setup/install_ultrashape.sh /workspace

set -euo pipefail

INSTALL_ROOT="${1:-/workspace}"
ULTRASHAPE_DIR="${ULTRASHAPE_DIR:-$INSTALL_ROOT/UltraShape-1.0}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$INSTALL_ROOT/checkpoints}"

echo "=== Installing UltraShape ==="

# Clone repository
mkdir -p "${INSTALL_ROOT}"
if [ ! -d "${ULTRASHAPE_DIR}" ]; then
    git clone https://github.com/PKU-YuanGroup/UltraShape-1.0.git "${ULTRASHAPE_DIR}"
fi
cd "${ULTRASHAPE_DIR}"

# UltraShape compiles CUDA extensions such as diso/flash-attn. Thunder images can
# expose multiple CUDA toolkits, so pin builds to the CUDA version used by Torch.
if [ -z "${CUDA_HOME:-}" ] && [ -d /usr/local/cuda-12.4 ]; then
    export CUDA_HOME=/usr/local/cuda-12.4
fi
if [ -n "${CUDA_HOME:-}" ]; then
    export PATH="${CUDA_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi

# Install dependencies
python -m pip install --upgrade pip setuptools wheel packaging ninja psutil
python -m pip install --no-build-isolation -r requirements.txt
python -m pip install --force-reinstall --no-deps --no-build-isolation flash-attn==2.7.3
python -m pip install --no-build-isolation "git+https://github.com/ashawkey/cubvh"
python -m pip install omegaconf einops rembg trimesh sageattention diffusers "huggingface_hub[cli]"
python -m pip install "setuptools<81"

# Download UltraShape Stage 2 checkpoint (for reference/comparison)
mkdir -p "${CHECKPOINT_DIR}"
if [ ! -f "${CHECKPOINT_DIR}/ultrashape_v1.pt" ]; then
    echo "Downloading UltraShape checkpoint..."
    huggingface-cli download infinith/UltraShape ultrashape_v1.pt \
        --local-dir "${CHECKPOINT_DIR}" \
        --local-dir-use-symlinks False
else
    echo "UltraShape checkpoint already exists: ${CHECKPOINT_DIR}/ultrashape_v1.pt"
fi

echo ""
echo "=== UltraShape installed ==="
echo "UltraShape repo: ${ULTRASHAPE_DIR}"
echo "Checkpoint:      ${CHECKPOINT_DIR}/ultrashape_v1.pt"
