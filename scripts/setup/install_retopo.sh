#!/usr/bin/env bash
# Install BPT retopology (Tencent, CVPR 2025).
# Optional: only needed for game-ready/digital output (not for 3D printing).
#
# Usage: conda activate clearmesh && ./install_retopo.sh

set -euo pipefail

DATA_DIR="${1:-/mnt/data}"

echo "=== Installing BPT (retopology) ==="

cd "${DATA_DIR}"

if [ ! -d "bpt" ]; then
    git clone https://github.com/whaohan/bpt.git
fi
cd bpt

pip install -r requirements.txt 2>/dev/null || echo "Install BPT dependencies manually"

# Download BPT checkpoints
echo "Downloading BPT checkpoints..."
python -c "
from huggingface_hub import snapshot_download
snapshot_download('whaohan/bpt', local_dir='checkpoints/')
print('BPT checkpoints downloaded')
" 2>/dev/null || echo "Download BPT checkpoints manually per their README"

echo ""
echo "=== BPT retopology installed ==="
echo "Path: ${DATA_DIR}/bpt/"
echo ""
echo "Generates meshes up to 8,000 clean faces from high-poly input."
echo "Optional for print-only output; recommended for digital/game-ready."
