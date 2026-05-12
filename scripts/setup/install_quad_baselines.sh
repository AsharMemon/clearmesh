#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
QUAD_TOOLS_DIR="${QUAD_TOOLS_DIR:-/tmp/clearmesh-quad-tools}"

"$PYTHON_BIN" -m pip install pyinstantmeshes
"$PYTHON_BIN" - <<'PY'
import importlib.util

if importlib.util.find_spec("pyinstantmeshes") is None:
    raise SystemExit("pyinstantmeshes install did not complete")
print("pyinstantmeshes ready")
PY

if [[ "${INSTALL_QUADRIFLOW:-0}" == "1" ]]; then
  if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y cmake build-essential git libeigen3-dev libboost-all-dev
  fi
  mkdir -p "$QUAD_TOOLS_DIR"
  cd "$QUAD_TOOLS_DIR"
  if [[ ! -d QuadriFlow ]]; then
    git clone --recursive https://github.com/hjwdzh/QuadriFlow.git
  fi
  cd QuadriFlow
  mkdir -p build
  cd build
  cmake ..
  make -j"$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"
  echo "QuadriFlow binary: $PWD/quadriflow"
fi
