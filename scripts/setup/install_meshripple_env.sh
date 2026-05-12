#!/usr/bin/env bash
set -euo pipefail

# Venv-based MeshRipple runtime installer for Thunder hosts without conda.
# The public repo asks for Python 3.12, PyTorch 2.8/cu128, and FlashAttention.

MESHRIPPLE_DIR="${MESHRIPPLE_DIR:-/home/ubuntu/mesh-heads/MeshRipple}"
MESHRIPPLE_VENV="${MESHRIPPLE_VENV:-/home/ubuntu/meshripple-venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
FLASH_ATTN_WHEEL_URL="${FLASH_ATTN_WHEEL_URL:-https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-1}"

if [ ! -d "$MESHRIPPLE_DIR" ]; then
  echo "MeshRipple repo not found at $MESHRIPPLE_DIR" >&2
  exit 1
fi

"$PYTHON_BIN" -m venv "$MESHRIPPLE_VENV"
. "$MESHRIPPLE_VENV/bin/activate"
python -m pip install --upgrade pip wheel setuptools packaging
python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url "$PYTORCH_INDEX_URL"
python -m pip install \
  accelerate beartype einops huggingface_hub omegaconf open3d pyaml pyyaml scikit-image scipy timm tqdm transformers trimesh

if [ -s "$MESHRIPPLE_DIR/requirement.txt" ]; then
  python -m pip install -r "$MESHRIPPLE_DIR/requirement.txt"
fi

if [ "$INSTALL_FLASH_ATTN" = "1" ]; then
  tmp_wheel="/tmp/flash_attn-2.7.3+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
  trap 'rm -f "$tmp_wheel"' EXIT
  python - <<'PY' "$FLASH_ATTN_WHEEL_URL" "$tmp_wheel"
from pathlib import Path
import sys
from urllib.request import urlretrieve
url, out = sys.argv[1], sys.argv[2]
print(f"downloading {url}")
urlretrieve(url, out)
print(f"saved {Path(out).stat().st_size} bytes")
PY
  python -m pip install "$tmp_wheel"
fi

python - <<'PY'
import importlib
mods = ['torch', 'accelerate', 'trimesh', 'open3d', 'timm', 'transformers', 'einops', 'omegaconf', 'beartype']
missing = []
for mod in mods:
    try:
        importlib.import_module(mod)
    except Exception as exc:
        missing.append(f"{mod}: {type(exc).__name__}: {exc}")
if missing:
    raise SystemExit('\n'.join(missing))
import torch
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())
PY

echo "MeshRipple venv ready: $MESHRIPPLE_VENV"
