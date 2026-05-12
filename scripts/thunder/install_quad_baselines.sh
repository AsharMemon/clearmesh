#!/usr/bin/env bash
set -euo pipefail

INSTANCE_ID="${1:-${THUNDER_INSTANCE_ID:-0}}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

"$REPO_ROOT/scripts/thunder/run_remote.sh" "$INSTANCE_ID" \
  "source /home/ubuntu/clearmesh-venv/bin/activate; python -m pip install -q pyinstantmeshes; python - <<'PY'
import importlib.util
print('pyinstantmeshes', bool(importlib.util.find_spec('pyinstantmeshes')))
PY
if [[ -x /home/ubuntu/meshripple-venv/bin/python ]]; then
  source /home/ubuntu/meshripple-venv/bin/activate
  python -m pip install -q pyinstantmeshes
fi
if [[ \"\${INSTALL_QUADRIFLOW:-0}\" == \"1\" || \"\${INSTALL_MANIFOLDPLUS:-0}\" == \"1\" ]]; then
  sudo apt-get update
  sudo apt-get install -y cmake build-essential git libeigen3-dev libboost-all-dev libgmp-dev libmpfr-dev
fi
if [[ \"\${INSTALL_QUADRIFLOW:-0}\" == \"1\" ]]; then
  mkdir -p /home/ubuntu/quad-tools
  cd /home/ubuntu/quad-tools
  if [[ ! -d QuadriFlow ]]; then
    git clone --recursive https://github.com/hjwdzh/QuadriFlow.git
  fi
  cd QuadriFlow
  mkdir -p build
  cd build
  cmake ..
  make -j\$(nproc)
  echo quadriflow_path=\$PWD/quadriflow
fi
if [[ \"\${INSTALL_MANIFOLDPLUS:-0}\" == \"1\" ]]; then
  mkdir -p /home/ubuntu/manifold-tools
  cd /home/ubuntu/manifold-tools
  if [[ ! -d ManifoldPlus ]]; then
    git clone --recursive https://github.com/hjwdzh/ManifoldPlus.git
  fi
  cd ManifoldPlus
  git submodule update --init --recursive
  mkdir -p build
  cd build
  cmake .. -DCMAKE_BUILD_TYPE=Release
  make -j\$(nproc)
  echo manifoldplus_path=\$PWD/manifold
fi"
