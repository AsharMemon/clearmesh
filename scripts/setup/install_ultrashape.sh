#!/usr/bin/env bash
# Install UltraShape reference code and download Stage 2 checkpoint.
# UltraShape's Stage 2 training code is adapted for ClearMesh's refinement model.
#
# Usage: conda activate clearmesh && ./install_ultrashape.sh

set -euo pipefail

DATA_DIR="${1:-/mnt/data}"

echo "=== Installing UltraShape ==="

cd "${DATA_DIR}"

# Clone repository
if [ ! -d "UltraShape-1.0" ]; then
    git clone https://github.com/PKU-YuanGroup/UltraShape-1.0.git
fi
cd UltraShape-1.0

# Install dependencies
pip install -r requirements.txt

# Download UltraShape Stage 2 checkpoint (for reference/comparison)
echo "Downloading UltraShape checkpoint..."
python -c "
from huggingface_hub import snapshot_download
snapshot_download('infinith/UltraShape', local_dir='checkpoints/')
print('UltraShape checkpoint downloaded')
"

echo ""
echo "=== UltraShape installed ==="
echo "Key files to study:"
echo "  scripts/sampling.py    - Data preparation (coarse/fine pairs)"
echo "  configs/               - Training configurations"
echo "  The refinement DiT uses RoPE encoding anchored to coarse geometry"
