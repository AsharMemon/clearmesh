#!/usr/bin/env bash
set -euo pipefail

INSTANCE_ID="${1:-${THUNDER_INSTANCE_ID:-0}}"
REMOTE_DIR="${REMOTE_DIR:-/home/ubuntu/clearmesh}"
REMOTE_VENV="${REMOTE_VENV:-/home/ubuntu/clearmesh-venv}"
TNR_BIN="${TNR_BIN:-/Users/Ashar/.tnr/bin/tnr}"
DATASET_DIR="${DATASET_DIR:-/tmp/clearmesh_face_conditioned_smoke/dataset}"
OUTPUT="${OUTPUT:-/tmp/clearmesh_face_conditioned_smoke/face_conditioned_tiny.pt}"
STEPS="${STEPS:-80}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

cat <<EOF | "$TNR_BIN" connect "$INSTANCE_ID"
set -euo pipefail
cd "$REMOTE_DIR"
. "$REMOTE_VENV/bin/activate"
python - <<'PY' || python -m pip install torch==2.8.0 --index-url "$PYTORCH_INDEX_URL"
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
PY
rm -rf "$DATASET_DIR"
python scripts/research/build_face_token_dataset.py \
  --output-dir "$DATASET_DIR" \
  --synthetic-count 16 \
  --max-faces 64 \
  --point-samples 256
python scripts/research/train_face_conditioned_tiny.py \
  --dataset-dir "$DATASET_DIR" \
  --steps "$STEPS" \
  --batch-size 4 \
  --point-samples 128 \
  --hidden-size 128 \
  --layers 2 \
  --heads 4 \
  --condition-tokens 4 \
  --output "$OUTPUT"
ls -lh "$OUTPUT"
exit
EOF
