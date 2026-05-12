#!/usr/bin/env bash
set -euo pipefail

# Venv-based TRELLIS.2 installer for GPU hosts without conda.
# The official repo recommends conda, but setup.sh can install into an active
# Python environment when --new-env is omitted.

TRELLIS2_DIR="${TRELLIS2_DIR:-/home/ubuntu/TRELLIS.2}"
TRELLIS2_VENV="${TRELLIS2_VENV:-/home/ubuntu/trellis2-venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-1}"
INSTALL_EXTENSIONS="${INSTALL_EXTENSIONS:-1}"
INSTALL_CUDA_TOOLKIT="${INSTALL_CUDA_TOOLKIT:-1}"
CUDA_TOOLKIT_HOME="${CUDA_TOOLKIT_HOME:-/usr/local/cuda-12.4}"
CUDA_TOOLKIT_APT_PACKAGE="${CUDA_TOOLKIT_APT_PACKAGE:-cuda-toolkit-12-4}"

export CUDA_HOME

if [ ! -d "$TRELLIS2_DIR/.git" ]; then
  git clone -b main https://github.com/microsoft/TRELLIS.2.git --recursive "$TRELLIS2_DIR"
else
  git -C "$TRELLIS2_DIR" fetch --all --prune
  git -C "$TRELLIS2_DIR" submodule update --init --recursive
fi

"$PYTHON_BIN" -m venv "$TRELLIS2_VENV"
. "$TRELLIS2_VENV/bin/activate"
python -m pip install --upgrade pip wheel setuptools packaging ninja
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url "$PYTORCH_INDEX_URL"

if [ "$INSTALL_CUDA_TOOLKIT" = "1" ] && [ ! -x "$CUDA_TOOLKIT_HOME/bin/nvcc" ]; then
  # Thunder currently exposes CUDA 13 at /usr/local/cuda while TRELLIS.2 pins
  # PyTorch cu124. Native extension builds require a matching CUDA toolkit.
  tmp_keyring="/tmp/cuda-keyring_1.1-1_all.deb"
  curl -fsSL \
    "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb" \
    -o "$tmp_keyring"
  sudo dpkg -i "$tmp_keyring"
  rm -f "$tmp_keyring"
  sudo apt-get update
  sudo apt-get install -y "$CUDA_TOOLKIT_APT_PACKAGE"
fi

if [ -x "$CUDA_TOOLKIT_HOME/bin/nvcc" ]; then
  CUDA_HOME="$CUDA_TOOLKIT_HOME"
  export CUDA_HOME
  export PATH="$CUDA_HOME/bin:$PATH"
fi

export LD_LIBRARY_PATH="$(python - <<'PY'
from pathlib import Path
import site
libs = []
for site_dir in site.getsitepackages():
    nvidia_dir = Path(site_dir) / "nvidia"
    if nvidia_dir.exists():
        libs.extend(str(path) for path in nvidia_dir.glob("*/lib") if path.exists())
print(":".join(libs))
PY
)${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd "$TRELLIS2_DIR"
sudo apt-get update
sudo apt-get install -y \
  libjpeg-dev \
  zlib1g-dev \
  libpng-dev \
  libtiff-dev \
  libwebp-dev \
  libfreetype6-dev \
  liblcms2-dev \
  libgl1 \
  libglib2.0-0 \
  libx11-6 \
  libxext6 \
  libsm6

. ./setup.sh --basic
# TRELLIS.2's image conditioner currently expects the Transformers 4.57 DINOv3
# object layout (`DINOv3ViTModel.layer`). Newer Transformers releases changed
# that shape and fail at runtime during `pipeline.run(...)`.
python -m pip install "transformers==4.57.5" "huggingface_hub<2.0,>=0.33.5"
if [ "$INSTALL_FLASH_ATTN" = "1" ]; then
  python -m pip install psutil
  python -m pip install flash-attn==2.7.3 --no-build-isolation
fi
if [ "$INSTALL_EXTENSIONS" = "1" ]; then
  # The official setup script stages extension repos under /tmp/extensions and
  # plain `git clone` fails if a prior interrupted install left partial clones.
  rm -rf /tmp/extensions
  . ./setup.sh --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm
fi

python - <<'PY'
import importlib
mods = ["torch", "trellis2", "o_voxel", "cv2", "PIL", "trimesh"]
missing = []
for mod in mods:
    try:
        importlib.import_module(mod)
    except Exception as exc:
        missing.append(f"{mod}: {type(exc).__name__}: {exc}")
if missing:
    raise SystemExit("\n".join(missing))
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
PY

echo "TRELLIS.2 venv ready: $TRELLIS2_VENV"
