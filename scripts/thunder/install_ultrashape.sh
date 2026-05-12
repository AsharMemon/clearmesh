#!/usr/bin/env bash
# Install UltraShape on the Thunder instance used by ClearMesh smoke runs.
set -euo pipefail

INSTANCE_ID="${1:-${THUNDER_INSTANCE_ID:-0}}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

"$REPO_ROOT/scripts/thunder/run_remote.sh" "$INSTANCE_ID" <<'EOF'
set -euo pipefail
if [ -f /home/ubuntu/.clearmesh_hf.env ]; then
  source /home/ubuntu/.clearmesh_hf.env
fi
sudo apt-get update -y >/tmp/clearmesh_ultrashape_apt_update.log || true
sudo apt-get install -y python3.10-venv build-essential ninja-build git libeigen3-dev >/tmp/clearmesh_ultrashape_apt_install.log
if [ -x /home/ubuntu/ultrashape-venv/bin/python ]; then
  if ! /home/ubuntu/ultrashape-venv/bin/python - <<'PY'
import sys
raise SystemExit(0 if sys.version_info[:2] == (3, 10) else 1)
PY
  then
    rm -rf /home/ubuntu/ultrashape-venv
  fi
fi
if [ ! -x /home/ubuntu/ultrashape-venv/bin/python ]; then
  python3.10 -m venv /home/ubuntu/ultrashape-venv
fi
source /home/ubuntu/ultrashape-venv/bin/activate
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.4}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
python -m pip install --upgrade pip setuptools wheel packaging ninja psutil
python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision
cd /home/ubuntu/clearmesh
sudo mkdir -p /workspace
sudo chown ubuntu:ubuntu /workspace
bash scripts/setup/install_ultrashape.sh /workspace
PYTHONPATH=/home/ubuntu/clearmesh python scripts/product/run_ultrashape_refinement.py --help >/tmp/clearmesh_ultrashape_help.txt
ls -lh /workspace/UltraShape-1.0 /workspace/checkpoints/ultrashape_v1.pt
EOF
