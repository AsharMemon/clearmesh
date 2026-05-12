#!/usr/bin/env bash
set -euo pipefail

TOOLS_DIR="${TOOLS_DIR:-/tmp/clearmesh-manifold-tools}"
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y cmake build-essential git libeigen3-dev libboost-all-dev libgmp-dev libmpfr-dev
fi
mkdir -p "$TOOLS_DIR"
cd "$TOOLS_DIR"

if [[ ! -d ManifoldPlus ]]; then
  git clone --recursive https://github.com/hjwdzh/ManifoldPlus.git
fi
cd ManifoldPlus
git submodule update --init --recursive
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j"$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"
echo "ManifoldPlus binary: $PWD/manifold"
