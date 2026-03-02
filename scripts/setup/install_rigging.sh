#!/usr/bin/env bash
# Install auto-rigging tools: Puppeteer (primary) and UniRig (fallback).
# These use a separate conda environment due to PyTorch version conflicts.
#
# Usage: conda activate rigging && ./install_rigging.sh

set -euo pipefail

DATA_DIR="${1:-/mnt/data}"

echo "=== Installing Puppeteer ==="

cd "${DATA_DIR}"

# Puppeteer (NeurIPS 2025 Spotlight — skeleton + skinning + animation)
if [ ! -d "Puppeteer" ]; then
    git clone https://github.com/Seed3D/Puppeteer.git --recursive
fi
cd Puppeteer
pip install -r requirements.txt

# Download Puppeteer checkpoints
echo "Downloading Puppeteer checkpoints..."
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Seed3D/Puppeteer', local_dir='checkpoints/')
print('Puppeteer checkpoints downloaded')
" 2>/dev/null || echo "Download Puppeteer checkpoints manually per their README"

echo ""
echo "=== Installing UniRig ==="

cd "${DATA_DIR}"

# UniRig (SIGGRAPH 2025 — generalist rigging)
if [ ! -d "UniRig" ]; then
    git clone https://github.com/VAST-AI-Research/UniRig.git
fi
cd UniRig
pip install -r requirements.txt

# Download UniRig checkpoints
echo "Downloading UniRig checkpoints..."
python -c "
from huggingface_hub import snapshot_download
snapshot_download('VAST-AI/UniRig', local_dir='checkpoints/')
print('UniRig checkpoints downloaded')
" 2>/dev/null || echo "Download UniRig checkpoints manually per their README"

echo ""
echo "=== Auto-rigging tools installed ==="
echo "Puppeteer: ${DATA_DIR}/Puppeteer/"
echo "UniRig:    ${DATA_DIR}/UniRig/"
echo ""
echo "Quick test:"
echo "  cd ${DATA_DIR}/Puppeteer"
echo "  python run_skeleton.py --input mesh.obj --output skeleton.txt"
