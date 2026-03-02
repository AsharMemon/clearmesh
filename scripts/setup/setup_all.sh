#!/usr/bin/env bash
# Master setup script for ClearMesh on a fresh GCP VM.
# Installs all components in the correct order.
#
# Usage: Run on GCP VM after SSH:
#   git clone https://github.com/AsharMemon/clearmesh.git
#   cd clearmesh
#   chmod +x scripts/setup/*.sh scripts/gcp/*.sh
#   ./scripts/setup/setup_all.sh [DATA_DIR] [--skip-rigging] [--skip-optional]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${1:-/mnt/data}"

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
echo "  ClearMesh — Full Environment Setup"
echo "============================================"
echo "Data directory: ${DATA_DIR}"
echo "Skip rigging:   ${SKIP_RIGGING}"
echo "Skip optional:  ${SKIP_OPTIONAL}"
echo ""

# Verify GPU is available
if command -v nvidia-smi &>/dev/null; then
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
else
    echo "WARNING: No GPU detected. Some components won't build."
    echo ""
fi

# Ensure data directory exists
mkdir -p "${DATA_DIR}"

# Step 1: Conda environments
echo "=== Step 1/8: Creating conda environments ==="
bash "${SCRIPT_DIR}/setup_envs.sh"

eval "$(conda shell.bash hook)"
conda activate clearmesh

# Step 2: TRELLIS.2 (Stage 1 coarse generation)
echo ""
echo "=== Step 2/8: Installing TRELLIS.2 ==="
bash "${SCRIPT_DIR}/install_trellis2.sh" "${DATA_DIR}"

# Step 3: UltraShape (Stage 2 reference code)
echo ""
echo "=== Step 3/8: Installing UltraShape ==="
bash "${SCRIPT_DIR}/install_ultrashape.sh" "${DATA_DIR}"

# Step 4: NDC + FlexiCubes
echo ""
echo "=== Step 4/8: Installing NDC + FlexiCubes ==="
bash "${SCRIPT_DIR}/install_ndc.sh" "${DATA_DIR}"

# Step 5: PartCrafter (optional — part decomposition for kitbashing)
if [ "${SKIP_OPTIONAL}" = false ]; then
    echo ""
    echo "=== Step 5/8: Installing PartCrafter ==="
    bash "${SCRIPT_DIR}/install_partcrafter.sh" "${DATA_DIR}"
else
    echo ""
    echo "=== Step 5/8: Skipping PartCrafter (--skip-optional) ==="
fi

# Step 6: SuperCarver / CraftsMan3D (optional — geometry super-resolution)
if [ "${SKIP_OPTIONAL}" = false ]; then
    echo ""
    echo "=== Step 6/8: Installing CraftsMan3D (geometry super-resolution) ==="
    bash "${SCRIPT_DIR}/install_supercarver.sh" "${DATA_DIR}"
else
    echo ""
    echo "=== Step 6/8: Skipping CraftsMan3D (--skip-optional) ==="
fi

# Step 7: BPT Retopology (optional — clean topology)
if [ "${SKIP_OPTIONAL}" = false ]; then
    echo ""
    echo "=== Step 7/8: Installing BPT Retopology ==="
    bash "${SCRIPT_DIR}/install_retopo.sh" "${DATA_DIR}"
else
    echo ""
    echo "=== Step 7/8: Skipping BPT Retopology (--skip-optional) ==="
fi

conda deactivate

# Step 8: Auto-rigging (optional — separate environment)
if [ "${SKIP_RIGGING}" = false ]; then
    echo ""
    echo "=== Step 8/8: Installing Puppeteer + UniRig ==="
    conda activate rigging
    bash "${SCRIPT_DIR}/install_rigging.sh" "${DATA_DIR}"
    conda deactivate
else
    echo ""
    echo "=== Step 8/8: Skipping rigging (--skip-rigging) ==="
fi

echo ""
echo "============================================"
echo "  Setup Complete!"
echo "============================================"
echo ""
echo "Directory layout:"
echo "  ${DATA_DIR}/TRELLIS.2/       - Stage 1 coarse generation"
echo "  ${DATA_DIR}/UltraShape-1.0/  - Stage 2 reference code"
echo "  ${DATA_DIR}/NDC/             - Neural Dual Contouring"
if [ "${SKIP_OPTIONAL}" = false ]; then
echo "  ${DATA_DIR}/PartCrafter/     - Part decomposition (optional)"
echo "  ${DATA_DIR}/CraftsMan3D/     - Geometry super-resolution (optional)"
echo "  ${DATA_DIR}/BPT/             - Retopology (optional)"
fi
if [ "${SKIP_RIGGING}" = false ]; then
echo "  ${DATA_DIR}/Puppeteer/       - Auto-rigging (optional)"
echo "  ${DATA_DIR}/UniRig/          - Auto-rigging fallback (optional)"
fi
echo ""
echo "Next: Run data preparation scripts in scripts/data/"
