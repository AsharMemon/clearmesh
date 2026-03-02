#!/usr/bin/env bash
# Create conda environments for ClearMesh.
#
# Two environments are needed due to dependency conflicts:
#   - clearmesh: PyTorch 2.6 + CUDA 12.4 (TRELLIS.2, UltraShape, training)
#   - rigging:   PyTorch 2.1.1 + CUDA 11.8 (UniRig, Puppeteer)
#
# Run on GCP VM after initial SSH.

set -euo pipefail

echo "=== Setting up conda environments ==="

# Ensure conda is available
if ! command -v conda &>/dev/null; then
    echo "Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
    eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"
    conda init bash
    rm /tmp/miniconda.sh
fi

# Environment 1: clearmesh (main training + inference)
echo ""
echo "--- Creating 'clearmesh' environment (PyTorch 2.6, CUDA 12.4) ---"
conda create -n clearmesh python=3.10 -y
conda activate clearmesh

pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install trimesh pymeshfix open3d Pillow rembg opencv-python-headless
pip install objaverse huggingface_hub transformers accelerate wandb pyyaml tqdm matplotlib pygltflib
pip install pytorch-lightning nerfacc

conda deactivate

# Environment 2: rigging (UniRig + Puppeteer)
echo ""
echo "--- Creating 'rigging' environment (PyTorch 2.1.1, CUDA 11.8) ---"
conda create -n rigging python=3.10 -y
conda activate rigging

pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu118
pip install trimesh open3d pygltflib tqdm
pip install spconv-cu118

conda deactivate

echo ""
echo "=== Environments created ==="
echo "Use: conda activate clearmesh"
echo "Use: conda activate rigging"
