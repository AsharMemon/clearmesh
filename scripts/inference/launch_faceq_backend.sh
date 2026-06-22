#!/usr/bin/env bash
# Launch the ClearMesh inference backend with the corrected pure-coordinate FACE
# model as the faceq path. This is the persistent server launcher used on the
# GPU backend (restart-on-crash loop; point the Vast/RunPod onstart at it so the
# server also comes back after a reboot).
#
# Env overrides:
#   CLEARMESH_REPO_ROOT        (default /workspace/clearmesh)
#   CLEARMESH_FACEQ_CHECKPOINT (default /workspace/mesh-heads/faceq/model.avg.pt)
#   CLEARMESH_PIPELINE_PORT    (default 8787)
#   CLEARMESH_PIPELINE_TOKEN   (shared secret the bridge sends; empty => open)
set -u

REPO="${CLEARMESH_REPO_ROOT:-/workspace/clearmesh}"
CKPT="${CLEARMESH_FACEQ_CHECKPOINT:-/workspace/mesh-heads/faceq/model.avg.pt}"
PORT="${CLEARMESH_PIPELINE_PORT:-8787}"

export CLEARMESH_FACEQ_REPRESENTATION="${CLEARMESH_FACEQ_REPRESENTATION:-paper}"
export CLEARMESH_FACEQ_DEVICE="${CLEARMESH_FACEQ_DEVICE:-cuda}"
export CLEARMESH_FACEQ_POINT_SAMPLES="${CLEARMESH_FACEQ_POINT_SAMPLES:-16384}"

cd "$REPO" || { echo "[launch] repo not found at $REPO"; exit 1; }
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

while true; do
  echo "[launch] starting pipeline server $(date -u) ckpt=$CKPT repr=$CLEARMESH_FACEQ_REPRESENTATION device=$CLEARMESH_FACEQ_DEVICE"
  python scripts/inference/clearmesh_pipeline_server.py \
    --host 0.0.0.0 --port "$PORT" --checkpoint "$CKPT"
  rc=$?
  echo "[launch] server exited rc=$rc — restarting in 5s"
  sleep 5
done
