#!/usr/bin/env bash
# One-shot setup for ClearMesh on a RunPod A100 80GB pod.
#
# RunPod specifics:
#   - Network volume mounts at /workspace (persists across pod restarts)
#   - PyTorch template has CUDA + drivers pre-installed
#   - Spot pods receive SIGTERM before termination (no 30s grace like GCP)
#
# Usage (run inside RunPod terminal or via SSH):
#   git clone https://github.com/AsharMemon/clearmesh.git /workspace/clearmesh
#   cd /workspace/clearmesh
#   chmod +x scripts/**/*.sh
#   ./scripts/runpod/setup_runpod.sh [--skip-rigging] [--skip-optional]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SETUP_DIR="${PROJECT_DIR}/scripts/setup"

# RunPod network volume = persistent storage
DATA_DIR="/workspace"

# Parse flags
SKIP_RIGGING=false
SKIP_OPTIONAL=false
for arg in "$@"; do
    case "$arg" in
        --skip-rigging) SKIP_RIGGING=true ;;
        --skip-optional) SKIP_OPTIONAL=true ;;
    esac
done

echo "============================================"
echo "  ClearMesh — RunPod Setup"
echo "============================================"
echo "Data directory: ${DATA_DIR}"
echo "Skip rigging:   ${SKIP_RIGGING}"
echo "Skip optional:  ${SKIP_OPTIONAL}"
echo ""

# Verify GPU
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Verify we're on RunPod
if [ -z "${RUNPOD_POD_ID:-}" ]; then
    echo "WARNING: RUNPOD_POD_ID not set. Are you on a RunPod pod?"
    echo "Continuing anyway..."
    echo ""
fi

# System dependencies
echo "=== Installing system dependencies ==="
apt-get update -qq && apt-get install -y -qq \
    build-essential git wget curl unzip \
    libgl1-mesa-glx libglib2.0-0 \
    2>/dev/null || true
echo ""

# Step 1: Conda environments
echo "=== Step 1/8: Setting up conda environments ==="
source "${SETUP_DIR}/setup_envs.sh"

# Ensure conda is available in this shell after setup_envs.sh
eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"

conda activate clearmesh

# Step 2: TRELLIS.2 (Stage 1 coarse generation)
echo ""
echo "=== Step 2/8: Installing TRELLIS.2 ==="
bash "${SETUP_DIR}/install_trellis2.sh" "${DATA_DIR}"

# Step 3: UltraShape (Stage 2 reference code)
echo ""
echo "=== Step 3/8: Installing UltraShape ==="
bash "${SETUP_DIR}/install_ultrashape.sh" "${DATA_DIR}"

# Step 4: NDC + FlexiCubes
echo ""
echo "=== Step 4/8: Installing NDC + FlexiCubes ==="
bash "${SETUP_DIR}/install_ndc.sh" "${DATA_DIR}"

# Step 5: PartCrafter (optional)
if [ "${SKIP_OPTIONAL}" = false ]; then
    echo ""
    echo "=== Step 5/8: Installing PartCrafter ==="
    bash "${SETUP_DIR}/install_partcrafter.sh" "${DATA_DIR}"
else
    echo ""
    echo "=== Step 5/8: Skipping PartCrafter (--skip-optional) ==="
fi

# Step 6: SuperCarver / CraftsMan3D (optional)
if [ "${SKIP_OPTIONAL}" = false ]; then
    echo ""
    echo "=== Step 6/8: Installing CraftsMan3D ==="
    bash "${SETUP_DIR}/install_supercarver.sh" "${DATA_DIR}"
else
    echo ""
    echo "=== Step 6/8: Skipping CraftsMan3D (--skip-optional) ==="
fi

# Step 7: BPT Retopology (optional)
if [ "${SKIP_OPTIONAL}" = false ]; then
    echo ""
    echo "=== Step 7/8: Installing BPT Retopology ==="
    bash "${SETUP_DIR}/install_retopo.sh" "${DATA_DIR}"
else
    echo ""
    echo "=== Step 7/8: Skipping BPT Retopology (--skip-optional) ==="
fi

conda deactivate

# Step 8: Auto-rigging (optional)
if [ "${SKIP_RIGGING}" = false ]; then
    echo ""
    echo "=== Step 8/8: Installing Puppeteer + UniRig ==="
    conda activate rigging
    bash "${SETUP_DIR}/install_rigging.sh" "${DATA_DIR}"
    conda deactivate
else
    echo ""
    echo "=== Step 8/8: Skipping rigging (--skip-rigging) ==="
fi

# Install ClearMesh package itself
echo ""
echo "=== Installing ClearMesh package ==="
conda activate clearmesh
cd "${PROJECT_DIR}"
pip install -r requirements.txt
pip install -e . 2>/dev/null || pip install -r requirements.txt

echo ""
echo "============================================"
echo "  ClearMesh RunPod Setup Complete!"
echo "============================================"
echo ""
echo "All data stored under: ${DATA_DIR}/"
echo "  (Network volume — persists across pod restarts)"
echo ""
echo "Next steps:"
echo "  conda activate clearmesh"
echo "  python -m clearmesh.pipeline --input photo.png --output model.glb"
echo ""
echo "For training:"
echo "  python -m clearmesh.stage2.train --config configs/train_stage2_flexicubes_runpod.yaml"
