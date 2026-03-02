#!/usr/bin/env bash
# Install PartCrafter for part decomposition (optional, kitbashing mode).
# Requires A100 80GB (~48GB VRAM).
#
# Usage: conda activate clearmesh && ./install_partcrafter.sh

set -euo pipefail

DATA_DIR="${1:-/mnt/data}"

echo "=== Installing PartCrafter ==="

cd "${DATA_DIR}"

if [ ! -d "PartCrafter" ]; then
    git clone https://github.com/VAST-AI-Research/PartCrafter.git
fi
cd PartCrafter

pip install -r requirements.txt

# Download PartCrafter checkpoints
echo "Downloading PartCrafter checkpoints..."
python -c "
from huggingface_hub import snapshot_download
snapshot_download('VAST-AI/PartCrafter', local_dir='checkpoints/')
print('PartCrafter checkpoints downloaded')
" 2>/dev/null || echo "Download PartCrafter checkpoints manually per their README"

echo ""
echo "=== PartCrafter installed ==="
echo "Path: ${DATA_DIR}/PartCrafter/"
echo "VRAM requirement: ~48GB+"
echo ""
echo "Test: python run.py --input image.png --output_dir parts/"
