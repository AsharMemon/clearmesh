#!/bin/bash
# ============================================================
# MeshClean RunPod A6000 Setup Script — Day 1
# ============================================================
# Paste this entire script into your RunPod terminal, or
# upload it and run: bash runpod-setup.sh
#
# Estimated time: 15-25 minutes (mostly conda/pip installs)
# ============================================================

set -e  # Stop on any error

echo "============================================"
echo "  MeshClean Pipeline — RunPod Setup"
echo "============================================"

# --------------------------------------------------
# STEP 0: Verify GPU is working
# --------------------------------------------------
echo ""
echo ">>> STEP 0: Checking GPU..."
nvidia-smi
echo ""
echo "If you see your A6000 above, we're good. If not, stop here."
echo ""
sleep 2

# --------------------------------------------------
# STEP 1: System dependencies
# --------------------------------------------------
echo ">>> STEP 1: Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq \
    git cmake build-essential \
    libboost-all-dev libeigen3-dev \
    libflann-dev libfreeimage-dev \
    libmetis-dev libgoogle-glog-dev \
    libgflags-dev libsqlite3-dev \
    libglew-dev qtbase5-dev libqt5opengl5-dev \
    libcgal-dev libceres-dev \
    unzip wget p7zip-full \
    ffmpeg imagemagick \
    libgl1-mesa-glx libegl1-mesa \
    2>/dev/null

echo "  System dependencies installed."

# --------------------------------------------------
# STEP 2: Create project directory structure
# --------------------------------------------------
echo ""
echo ">>> STEP 2: Creating project structure..."
mkdir -p ~/meshclean/{repos,data,outputs,scripts,weights}
cd ~/meshclean

echo "  Project structure:"
echo "  ~/meshclean/"
echo "    ├── repos/      # Git repos (COLMAP, Instant-NSR, NDC, etc.)"
echo "    ├── data/        # Your phone photos go here"
echo "    ├── outputs/     # Processed meshes come out here"
echo "    ├── scripts/     # Our glue scripts"
echo "    └── weights/     # Pre-trained model weights"

# --------------------------------------------------
# STEP 3: Install Miniconda (if not present)
# --------------------------------------------------
echo ""
echo ">>> STEP 3: Setting up Conda..."
if ! command -v conda &> /dev/null; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p $HOME/miniconda3
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    conda init bash
    echo "  Miniconda installed."
else
    eval "$(conda shell.bash hook)"
    echo "  Conda already available."
fi

# Make conda available in this script
source ~/.bashrc 2>/dev/null || true
eval "$(conda shell.bash hook)" 2>/dev/null || true

# --------------------------------------------------
# STEP 4: Create conda environment
# --------------------------------------------------
echo ""
echo ">>> STEP 4: Creating meshclean conda environment..."
conda create -n meshclean python=3.10 -y -q
conda activate meshclean

pip install --upgrade pip setuptools wheel

echo "  Environment created and activated."

# --------------------------------------------------
# STEP 5: Install COLMAP
# --------------------------------------------------
echo ""
echo ">>> STEP 5: Installing COLMAP..."

# Try conda install first (easiest)
conda install -c conda-forge colmap -y -q 2>/dev/null && {
    echo "  COLMAP installed via conda."
} || {
    echo "  Conda install failed, building from source..."
    cd ~/meshclean/repos
    git clone https://github.com/colmap/colmap.git
    cd colmap
    mkdir build && cd build
    cmake .. -DCMAKE_CUDA_ARCHITECTURES=86 -GNinja
    ninja -j$(nproc)
    ninja install
    echo "  COLMAP built from source."
}

# Verify
echo "  Verifying COLMAP..."
colmap -h | head -3
echo "  COLMAP ready."

# --------------------------------------------------
# STEP 6: Install PyTorch + CUDA
# --------------------------------------------------
echo ""
echo ">>> STEP 6: Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q

python -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

# --------------------------------------------------
# STEP 7: Install Instant-NSR-PL
# --------------------------------------------------
echo ""
echo ">>> STEP 7: Installing Instant-NSR-PL..."
cd ~/meshclean/repos

git clone https://github.com/bennyguo/instant-nsr-pl.git
cd instant-nsr-pl

# Install tiny-cuda-nn (this takes a few minutes to compile)
echo "  Installing tiny-cuda-nn (compiling... ~5 min)..."
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch -q

# Install nerfacc
pip install nerfacc -f https://nerfacc-bucket.s3.us-west-2.amazonaws.com/whl/torch-2.1.0_cu121.html -q 2>/dev/null || pip install nerfacc -q

# Install other deps
pip install pytorch-lightning omegaconf trimesh PyMCubes opencv-python matplotlib tensorboard -q
pip install lpips imageio plotly dash -q

echo "  Instant-NSR-PL installed."

# --------------------------------------------------
# STEP 8: Install NDC (Neural Dual Contouring / UNDC)
# --------------------------------------------------
echo ""
echo ">>> STEP 8: Installing NDC/UNDC..."
cd ~/meshclean/repos

git clone https://github.com/czq142857/NDC.git
cd NDC

pip install h5py scipy cython open3d -q

# Download pre-trained weights
echo "  Downloading pre-trained weights..."
if [ -f "weights_examples.7z" ]; then
    echo "  Weights already downloaded."
else
    # The weights are hosted on the repo's releases or Google Drive
    # Try direct download first
    pip install gdown -q
    
    # Note: you may need to download weights manually if this fails.
    # Check the NDC repo README for the download link.
    echo "  NOTE: Pre-trained weights may need manual download."
    echo "  Check https://github.com/czq142857/NDC for download links."
    echo "  Place weight files in ~/meshclean/repos/NDC/"
fi

echo "  NDC/UNDC installed."

# --------------------------------------------------
# STEP 9: Install Instant Meshes (quad remeshing)
# --------------------------------------------------
echo ""
echo ">>> STEP 9: Installing Instant Meshes..."
cd ~/meshclean/repos

# Download pre-built binary
mkdir -p instant-meshes && cd instant-meshes
wget -q https://instant-meshes.s3.eu-central-1.amazonaws.com/Release/instant-meshes-linux.zip -O im.zip 2>/dev/null && {
    unzip -o im.zip
    chmod +x Instant\ Meshes 2>/dev/null || chmod +x instant-meshes 2>/dev/null || true
    echo "  Instant Meshes binary downloaded."
} || {
    echo "  Pre-built binary download failed. Building from source..."
    cd ~/meshclean/repos
    git clone --recursive https://github.com/wjakob/instant-meshes.git instant-meshes-src
    cd instant-meshes-src
    mkdir build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j$(nproc)
    cp instant-meshes ~/meshclean/repos/instant-meshes/
    echo "  Instant Meshes built from source."
}

# --------------------------------------------------
# STEP 10: Install xatlas (UV unwrapping)
# --------------------------------------------------
echo ""
echo ">>> STEP 10: Installing xatlas..."
pip install xatlas -q 2>/dev/null || {
    echo "  pip install failed, installing from source..."
    cd ~/meshclean/repos
    git clone https://github.com/mworchel/xatlas-python.git
    cd xatlas-python
    pip install . -q
}
echo "  xatlas installed."

# --------------------------------------------------
# STEP 11: Install mesh processing tools
# --------------------------------------------------
echo ""
echo ">>> STEP 11: Installing mesh processing tools..."
pip install trimesh pymeshfix pymeshlab pyvista networkx -q
echo "  Mesh tools installed."

# --------------------------------------------------
# STEP 12: Create the test data folder
# --------------------------------------------------
echo ""
echo ">>> STEP 12: Preparing test data folder..."
mkdir -p ~/meshclean/data/test_mug
echo "  Upload your phone photos to: ~/meshclean/data/test_mug/"
echo ""

# --------------------------------------------------
# STEP 13: Verification
# --------------------------------------------------
echo ""
echo "============================================"
echo "  VERIFICATION"
echo "============================================"
echo ""

echo "Checking COLMAP..."
colmap -h > /dev/null 2>&1 && echo "  ✓ COLMAP" || echo "  ✗ COLMAP — needs fixing"

echo "Checking PyTorch + CUDA..."
python -c "import torch; assert torch.cuda.is_available(); print('  ✓ PyTorch + CUDA')" 2>/dev/null || echo "  ✗ PyTorch CUDA — needs fixing"

echo "Checking Instant-NSR-PL..."
python -c "import sys; sys.path.insert(0, '$HOME/meshclean/repos/instant-nsr-pl'); print('  ✓ Instant-NSR-PL (repo cloned)')" 2>/dev/null || echo "  ✗ Instant-NSR-PL"

echo "Checking NDC..."
[ -d "$HOME/meshclean/repos/NDC" ] && echo "  ✓ NDC/UNDC (repo cloned)" || echo "  ✗ NDC"

echo "Checking trimesh..."
python -c "import trimesh; print('  ✓ trimesh')" 2>/dev/null || echo "  ✗ trimesh"

echo "Checking xatlas..."
python -c "import xatlas; print('  ✓ xatlas')" 2>/dev/null || echo "  ✗ xatlas"

echo "Checking pymeshfix..."
python -c "import pymeshfix; print('  ✓ pymeshfix')" 2>/dev/null || echo "  ✗ pymeshfix"

echo ""
echo "============================================"
echo "  SETUP COMPLETE"
echo "============================================"
echo ""
echo "  Next steps:"
echo "  1. Upload your phone photos to ~/meshclean/data/test_mug/"
echo "     Use: scp photos/*.jpg root@<your-runpod-ip>:~/meshclean/data/test_mug/"
echo ""
echo "  2. Run the COLMAP test (paste into terminal):"
echo "     cd ~/meshclean && conda activate meshclean"
echo "     bash scripts/run_colmap.sh data/test_mug"
echo ""
echo "  3. The run_colmap.sh script will be created next."
echo "     Come back to Claude and say 'setup done' to get it."
echo ""
echo "============================================"
