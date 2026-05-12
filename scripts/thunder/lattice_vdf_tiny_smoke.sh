#!/usr/bin/env bash
set -euo pipefail

INSTANCE_ID="${1:-${THUNDER_INSTANCE_ID:-0}}"
REMOTE_DIR="${REMOTE_DIR:-/home/ubuntu/clearmesh}"
REMOTE_VENV="${REMOTE_VENV:-/home/ubuntu/clearmesh-venv}"
TNR_BIN="${TNR_BIN:-/Users/Ashar/.tnr/bin/tnr}"
RUN_DIR="${RUN_DIR:-/tmp/clearmesh_lattice_vdf_smoke}"
STEPS="${STEPS:-1000}"
WELD_DIGITS="${WELD_DIGITS:-3}"
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
python scripts/research/build_lattice_sample_dataset.py \
  --output-dir "$RUN_DIR/dataset" \
  --synthetic-count 1 \
  --resolution 16 \
  --surface-samples 128 \
  --vdf-samples 64 \
  --patch-surface-samples 64 \
  --anchor-count 8 \
  --patch-size 8 \
  --negative-edges 8
python scripts/research/train_lattice_vdf_tiny.py \
  --dataset-dir "$RUN_DIR/dataset" \
  --use-face-vdf \
  --steps "$STEPS" \
  --batch-points 12 \
  --hidden-size 128 \
  --layers 4 \
  --lr 1e-3 \
  --weld-digits "$WELD_DIGITS" \
  --export-predicted "$RUN_DIR/predicted.glb" \
  --export-target "$RUN_DIR/target.glb" \
  --output "$RUN_DIR/lattice_vdf_tiny.pt"
ls -lh "$RUN_DIR"/*.glb "$RUN_DIR"/*.pt
exit
EOF
