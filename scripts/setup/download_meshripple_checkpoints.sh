#!/usr/bin/env bash
set -euo pipefail

# Download the public MeshRipple checkpoints into the layout expected by the
# official repo and ClearMesh adapter.

MESHRIPPLE_DIR="${MESHRIPPLE_DIR:-/home/ubuntu/mesh-heads/MeshRipple}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$MESHRIPPLE_DIR/ckpt}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GDRIVE_FOLDER_URL="${GDRIVE_FOLDER_URL:-https://drive.google.com/drive/folders/1qex2gbIoxh4-qRbAUYxIF5b_OwvLhOxq}"

mkdir -p "$CHECKPOINT_DIR"

if ! "$PYTHON_BIN" -m gdown --version >/dev/null 2>&1; then
  "$PYTHON_BIN" -m pip install --upgrade gdown
fi

"$PYTHON_BIN" -m gdown --folder "$GDRIVE_FOLDER_URL" -O "$CHECKPOINT_DIR"

found=0
for checkpoint in meshRipple_10k.pth meshRipple_nsa.pth; do
  if [ -s "$CHECKPOINT_DIR/$checkpoint" ]; then
    echo "found $CHECKPOINT_DIR/$checkpoint"
    found=1
  fi
done

if [ "$found" != "1" ]; then
  echo "No MeshRipple checkpoints were downloaded into $CHECKPOINT_DIR" >&2
  echo "Expected at least one of: meshRipple_10k.pth, meshRipple_nsa.pth" >&2
  exit 1
fi
