#!/usr/bin/env bash
set -euo pipefail

# Installs/clones the public mesh-head repos used in the ClearMesh bake-off.
# Usage:
#   MESH_HEAD_ROOT=/workspace/mesh-heads INSTALL_ENV=1 bash scripts/setup/install_mesh_heads.sh
#   INSTALL_ENV=0 bash scripts/setup/install_mesh_heads.sh /workspace/mesh-heads

MESH_HEAD_ROOT="${1:-${MESH_HEAD_ROOT:-/workspace/mesh-heads}}"
INSTALL_ENV="${INSTALL_ENV:-1}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
mkdir -p "$MESH_HEAD_ROOT"

clone_or_update() {
  local repo_url="$1"
  local dir_name="$2"
  local target="$MESH_HEAD_ROOT/$dir_name"
  if [ -d "$target/.git" ]; then
    echo "[mesh-heads] $dir_name already exists; fetching latest refs"
    git -C "$target" fetch --all --prune
  else
    echo "[mesh-heads] cloning $repo_url -> $target"
    git clone "$repo_url" "$target"
  fi
}

clone_or_update "https://github.com/MayMhappy/MeshRipple" "MeshRipple"
clone_or_update "https://github.com/gaochao-s/Mesh-Silksong" "Mesh-Silksong"
clone_or_update "https://github.com/zhaorw02/DeepMesh" "DeepMesh"
clone_or_update "https://github.com/sail-sg/TreeMeshGPT" "TreeMeshGPT"
clone_or_update "https://github.com/Xrvitd/MeshMosaic" "MeshMosaic"
clone_or_update "https://github.com/jhkim0759/FastMesh" "FastMesh"

if [ "$INSTALL_ENV" = "1" ]; then
  if command -v conda >/dev/null 2>&1; then
    if ! conda env list | awk '{print $1}' | grep -qx "meshripple"; then
      echo "[mesh-heads] creating conda env meshripple"
      conda create -y -n meshripple python=3.12
    fi
    echo "[mesh-heads] installing MeshRipple Python dependencies"
    conda run -n meshripple python -m pip install \
      torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
      --index-url "$PYTORCH_INDEX_URL"
    if [ -f "$MESH_HEAD_ROOT/MeshRipple/requirements.txt" ]; then
      conda run -n meshripple python -m pip install -r "$MESH_HEAD_ROOT/MeshRipple/requirements.txt"
    fi
  else
    echo "[mesh-heads] conda not found; cloned repos only. Install MeshRipple env manually on the GPU host."
  fi
fi

cat <<EOF

[mesh-heads] next steps
- Put MeshRipple checkpoints in: $MESH_HEAD_ROOT/MeshRipple/ckpt
- Validate demo inference:
  cd $MESH_HEAD_ROOT/MeshRipple
  conda run -n meshripple python main.py --config config_loader/config_20k_nsa.yaml
- Then call ClearMesh adapter:
  python /workspace/clearmesh/scripts/product/run_mesh_head.py \
    --head meshripple \
    --case-id smoke \
    --point-cloud /workspace/clearmesh/pointclouds/smoke_40960.ply \
    --output-dir /workspace/clearmesh/artifacts/smoke/meshripple \
    --config-json /workspace/clearmesh/configs/meshripple.gpu.json
EOF
