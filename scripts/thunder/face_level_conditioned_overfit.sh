#!/usr/bin/env bash
set -euo pipefail

INSTANCE_ID="${1:-${THUNDER_INSTANCE_ID:-0}}"
REMOTE_DIR="${REMOTE_DIR:-/home/ubuntu/clearmesh}"
REMOTE_VENV="${REMOTE_VENV:-/home/ubuntu/clearmesh-venv}"
TNR_BIN="${TNR_BIN:-/Users/Ashar/.tnr/bin/tnr}"
RUN_DIR="${RUN_DIR:-/tmp/clearmesh_face_level_conditioned_overfit}"
STEPS="${STEPS:-400}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

cat <<EOF | "$TNR_BIN" connect "$INSTANCE_ID"
set -euo pipefail
cd "$REMOTE_DIR"
. "$REMOTE_VENV/bin/activate"
python - <<'PY' || python -m pip install torch==2.8.0 --index-url "$PYTORCH_INDEX_URL"
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
PY
rm -rf "$RUN_DIR"
python scripts/research/build_face_token_dataset.py \
  --output-dir "$RUN_DIR/dataset" \
  --synthetic-count 1 \
  --max-faces 64 \
  --point-samples 512
python scripts/research/train_face_level_conditioned_tiny.py \
  --dataset-dir "$RUN_DIR/dataset" \
  --steps "$STEPS" \
  --batch-size 4 \
  --point-samples 256 \
  --hidden-size 128 \
  --layers 2 \
  --heads 4 \
  --condition-tokens 4 \
  --lr 5e-4 \
  --output "$RUN_DIR/face_level_conditioned_overfit.pt"
python scripts/research/sample_face_level_conditioned_tiny.py \
  --checkpoint "$RUN_DIR/face_level_conditioned_overfit.pt" \
  --sample "$RUN_DIR/dataset/0000000_synthetic_box_00000.npz" \
  --output "$RUN_DIR/generated.glb" \
  --teacher-output "$RUN_DIR/teacher.glb" \
  --point-samples 256
ls -lh "$RUN_DIR"/*.glb "$RUN_DIR"/*.pt
exit
EOF
