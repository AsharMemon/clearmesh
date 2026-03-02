#!/usr/bin/env bash
# Install geometry super-resolution: CraftsMan3D (available now).
# SuperCarver code release expected Q1-Q2 2026 — will be added when available.
#
# Usage: conda activate clearmesh && ./install_supercarver.sh

set -euo pipefail

DATA_DIR="${1:-/mnt/data}"

echo "=== Installing CraftsMan3D (geometry super-resolution fallback) ==="

cd "${DATA_DIR}"

# CraftsMan3D (CVPR 2025) — available now as SuperCarver fallback
if [ ! -d "CraftsMan3D" ]; then
    git clone https://github.com/wyysf-98/CraftsMan3D.git
fi
cd CraftsMan3D

pip install -r requirements.txt 2>/dev/null || echo "Install CraftsMan3D dependencies manually"

# Download checkpoints
echo "Downloading CraftsMan3D checkpoints..."
python -c "
from huggingface_hub import snapshot_download
snapshot_download('wyysf/CraftsMan3D', local_dir='checkpoints/')
print('CraftsMan3D checkpoints downloaded')
" 2>/dev/null || echo "Download checkpoints manually per their README"

echo ""
echo "=== CraftsMan3D installed ==="
echo "Path: ${DATA_DIR}/CraftsMan3D/"
echo ""
echo "Note: SuperCarver (2025) code release expected Q1-Q2 2026."
echo "When available, clone to ${DATA_DIR}/SuperCarver/ and it will"
echo "be auto-detected by the ClearMesh pipeline."
