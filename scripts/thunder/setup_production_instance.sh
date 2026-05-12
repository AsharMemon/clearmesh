#!/usr/bin/env bash
# Bootstrap a fresh Thunder GPU instance with the dependency fixes ClearMesh
# needs for the TRELLIS.2 -> UltraShape -> mesh-head production path.
set -euo pipefail

INSTANCE_ID="${1:-${THUNDER_INSTANCE_ID:-0}}"
INSTALL_MESHRIPPLE="${INSTALL_MESHRIPPLE:-1}"
INSTALL_QUAD="${INSTALL_QUAD:-1}"
INSTALL_ULTRASHAPE="${INSTALL_ULTRASHAPE:-1}"
INSTALL_TRELLIS="${INSTALL_TRELLIS:-1}"
RUN_PREFLIGHTS="${RUN_PREFLIGHTS:-1}"
SYNC_HF_TOKEN="${SYNC_HF_TOKEN:-1}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

cd "$REPO_ROOT"

if [ -z "${THUNDER_TOKEN:-}" ]; then
  echo "THUNDER_TOKEN is not set. Export it before setting up a Thunder instance." >&2
  exit 1
fi

echo "[clearmesh] syncing repo to Thunder instance ${INSTANCE_ID}"
THUNDER_INSTANCE_ID="$INSTANCE_ID" scripts/thunder/sync_repo.sh

if [ "$SYNC_HF_TOKEN" = "1" ]; then
  if [ -n "${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}" ]; then
    echo "[clearmesh] syncing Hugging Face token"
    THUNDER_INSTANCE_ID="$INSTANCE_ID" scripts/thunder/sync_hf_token.sh
  else
    echo "[clearmesh] HF_TOKEN/HUGGINGFACE_HUB_TOKEN not set; skipping token sync"
  fi
fi

echo "[clearmesh] bootstrapping lightweight product worker env"
INSTALL_MESH_HEAD_ENVS=0 THUNDER_INSTANCE_ID="$INSTANCE_ID" scripts/thunder/bootstrap_remote.sh

if [ "$INSTALL_TRELLIS" = "1" ]; then
  echo "[clearmesh] installing TRELLIS.2 runtime"
  THUNDER_INSTANCE_ID="$INSTANCE_ID" scripts/thunder/install_trellis2_env.sh
fi

if [ "$INSTALL_ULTRASHAPE" = "1" ]; then
  echo "[clearmesh] installing UltraShape runtime and checkpoint"
  THUNDER_INSTANCE_ID="$INSTANCE_ID" scripts/thunder/install_ultrashape.sh
fi

if [ "$INSTALL_MESHRIPPLE" = "1" ]; then
  echo "[clearmesh] cloning mesh-head repos"
  THUNDER_INSTANCE_ID="$INSTANCE_ID" scripts/thunder/run_remote.sh "$INSTANCE_ID" <<'EOF'
set -euo pipefail
cd /home/ubuntu/clearmesh
MESH_HEAD_ROOT=/home/ubuntu/mesh-heads INSTALL_ENV=0 bash scripts/setup/install_mesh_heads.sh
EOF
  echo "[clearmesh] installing MeshRipple runtime"
  THUNDER_INSTANCE_ID="$INSTANCE_ID" scripts/thunder/install_meshripple_env.sh
  echo "[clearmesh] downloading MeshRipple checkpoints"
  THUNDER_INSTANCE_ID="$INSTANCE_ID" scripts/thunder/download_meshripple_checkpoints.sh
fi

if [ "$INSTALL_QUAD" = "1" ]; then
  echo "[clearmesh] installing quad remeshing baselines"
  THUNDER_INSTANCE_ID="$INSTANCE_ID" scripts/thunder/install_quad_baselines.sh
fi

if [ "$RUN_PREFLIGHTS" = "1" ]; then
  echo "[clearmesh] running preflights"
  if [ "$INSTALL_TRELLIS" = "1" ]; then
    THUNDER_INSTANCE_ID="$INSTANCE_ID" scripts/thunder/trellis2_preflight.sh
  fi
  if [ "$INSTALL_MESHRIPPLE" = "1" ]; then
    THUNDER_INSTANCE_ID="$INSTANCE_ID" scripts/thunder/meshripple_preflight.sh
  fi
fi

cat <<EOF

[clearmesh] production GPU setup complete for instance ${INSTANCE_ID}

Useful next checks:
  THUNDER_INSTANCE_ID=${INSTANCE_ID} scripts/thunder/trellis2_smoke.sh
  THUNDER_INSTANCE_ID=${INSTANCE_ID} REFERENCE_MODE=ultrashape scripts/thunder/production_path_smoke.sh

Optional flags:
  INSTALL_MESHRIPPLE=0
  INSTALL_QUAD=0
  INSTALL_ULTRASHAPE=0
  INSTALL_TRELLIS=0
  RUN_PREFLIGHTS=0
  SYNC_HF_TOKEN=0
EOF
