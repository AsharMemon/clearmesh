#!/usr/bin/env bash
# Install neural retopology backends (all optional — only needed for
# digital/game-ready output, not for 3D printing).
#
#   - BPT      (Tencent, CVPR 2025): triangle meshes up to ~8K faces
#   - QuadGPT  (arxiv:2509.21420):  native quad meshes up to ~20K faces
#                                   (installs once the public repo lands)
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
echo "Generates triangle meshes up to 8,000 faces from high-poly input."

# === QuadGPT (optional — quad retopology) ===
# Official code is expected at https://github.com/<tba>/QuadGPT once the
# authors release it (paper commits to "a public API and Code"). Pinned
# env var so we can swap URLs without editing the script.
QUADGPT_REPO="${QUADGPT_REPO:-https://github.com/liu-jian-21/QuadGPT.git}"

cd "${DATA_DIR}"
echo ""
echo "=== Installing QuadGPT (quad retopology, optional) ==="
if git ls-remote "${QUADGPT_REPO}" &>/dev/null; then
    if [ ! -d "QuadGPT" ]; then
        git clone "${QUADGPT_REPO}" QuadGPT || echo "QuadGPT clone failed — repo may not be public yet."
    fi
    if [ -d "QuadGPT" ]; then
        cd QuadGPT
        if [ -f requirements.txt ]; then
            pip install -r requirements.txt 2>/dev/null || echo "Install QuadGPT dependencies manually"
        fi
        echo "QuadGPT installed at ${DATA_DIR}/QuadGPT (generates quad meshes up to ~20K faces)."
    fi
else
    echo "QuadGPT public repo not reachable at ${QUADGPT_REPO}."
    echo "Skipping — set QUADGPT_REPO=<url> and re-run once code is released."
fi

echo ""
echo "=== Retopology setup complete ==="
echo "Optional for print-only output; recommended for digital/game-ready."
