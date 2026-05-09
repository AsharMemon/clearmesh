#!/usr/bin/env bash
set -euo pipefail

INSTANCE_ID="${1:-${THUNDER_INSTANCE_ID:-0}}"
REMOTE_DIR="${REMOTE_DIR:-/home/ubuntu/clearmesh}"
REMOTE_VENV="${REMOTE_VENV:-/home/ubuntu/clearmesh-venv}"
TNR_BIN="${TNR_BIN:-/Users/Ashar/.tnr/bin/tnr}"
RUN_DIR="${RUN_DIR:-/tmp/clearmesh_face_paper_faithful_smoke}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-}"
MESH_DIR="${MESH_DIR:-}"
SYNTHETIC_COUNT="${SYNTHETIC_COUNT:-12}"
SYNTHETIC_KIND="${SYNTHETIC_KIND:-mixed_cycle}"
MAX_FACES="${MAX_FACES:-256}"
MODEL_MAX_FACES="${MODEL_MAX_FACES:-0}"
POINT_SAMPLES="${POINT_SAMPLES:-1024}"
TRAIN_POINT_SAMPLES="${TRAIN_POINT_SAMPLES:-1024}"
STEPS="${STEPS:-500}"
BATCH_SIZE="${BATCH_SIZE:-2}"
HIDDEN_SIZE="${HIDDEN_SIZE:-128}"
ENCODER_HIDDEN_SIZE="${ENCODER_HIDDEN_SIZE:-128}"
ENCODER_LAYERS="${ENCODER_LAYERS:-2}"
DECODER_LAYERS="${DECODER_LAYERS:-2}"
HEADS="${HEADS:-4}"
VECSET_TOKENS="${VECSET_TOKENS:-64}"
LATENT_DIM="${LATENT_DIM:-64}"
ENCODER_BACKEND="${ENCODER_BACKEND:-shape2vecset}"
CAUSAL_MLP_VARIANT="${CAUSAL_MLP_VARIANT:-legacy_concat}"
DISABLE_EOS_HEAD="${DISABLE_EOS_HEAD:-0}"
EOS_LOSS_WEIGHT="${EOS_LOSS_WEIGHT:-0.05}"
DECODE_HEAD="${DECODE_HEAD:-causal}"
OPTIMIZER="${OPTIMIZER:-muon}"
LR="${LR:-6e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
DISABLE_AUGMENT="${DISABLE_AUGMENT:-0}"
AUGMENT_ROTATION="${AUGMENT_ROTATION:-so3}"
AUGMENT_SCALE_MIN="${AUGMENT_SCALE_MIN:-0.75}"
AUGMENT_SCALE_MAX="${AUGMENT_SCALE_MAX:-1.25}"
AUGMENT_FLIP_PROB="${AUGMENT_FLIP_PROB:-0.5}"
LOG_EVERY="${LOG_EVERY:-0}"
SELECTION_EVAL_EVERY="${SELECTION_EVAL_EVERY:-0}"
SELECTION_EVAL_BATCH_SIZE="${SELECTION_EVAL_BATCH_SIZE:-1}"
EVAL_LIMIT="${EVAL_LIMIT:-6}"
PAIR_SAMPLES="${PAIR_SAMPLES:-300}"
GENERATION_MODE="${GENERATION_MODE:-autoregressive}"
FACE_COUNT_MODE="${FACE_COUNT_MODE:-gt}"
EOS_THRESHOLD="${EOS_THRESHOLD:-0.5}"
MIN_GENERATED_FACES="${MIN_GENERATED_FACES:-1}"
GENERATION_FACE_LIMIT="${GENERATION_FACE_LIMIT:-0}"
DISABLE_INCREMENTAL_GENERATION="${DISABLE_INCREMENTAL_GENERATION:-0}"
EVAL_LOG_EVERY="${EVAL_LOG_EVERY:-1}"
SEED="${SEED:-41}"
DATASET_GATE_PROFILE="${DATASET_GATE_PROFILE:-}"
DATASET_GATE_TOKEN_FAMILY="${DATASET_GATE_TOKEN_FAMILY:-paper}"
FAIL_ON_DATASET_GATE="${FAIL_ON_DATASET_GATE:-1}"
if [ -z "${THUNDER_TOKEN:-}" ]; then
  echo "THUNDER_TOKEN is not set. Export it before running Thunder commands." >&2
  exit 1
fi

TRAIN_AUGMENT_ARGS_STR=""
if [ "$DISABLE_AUGMENT" = "1" ]; then
  TRAIN_AUGMENT_ARGS_STR=" --disable-augment"
fi
TRAIN_EOS_ARGS_STR=""
if [ "$DISABLE_EOS_HEAD" = "1" ]; then
  TRAIN_EOS_ARGS_STR=" --disable-eos-head"
fi
BUILD_SOURCE_ARGS_STR=""
if [ -n "$MESH_DIR" ]; then
  BUILD_SOURCE_ARGS_STR+=" --mesh-dir $(printf "%q" "$MESH_DIR")"
fi
if [ "$SYNTHETIC_COUNT" != "0" ]; then
  BUILD_SOURCE_ARGS_STR+=" --synthetic-count $(printf "%q" "$SYNTHETIC_COUNT") --synthetic-kind $(printf "%q" "$SYNTHETIC_KIND")"
fi
EVAL_INCREMENTAL_ARGS_STR=""
if [ "$DISABLE_INCREMENTAL_GENERATION" = "1" ]; then
  EVAL_INCREMENTAL_ARGS_STR=" --disable-incremental-generation"
fi
DATASET_GATE_FAIL_ARGS_STR=""
if [ "$FAIL_ON_DATASET_GATE" = "1" ]; then
  DATASET_GATE_FAIL_ARGS_STR=" --fail-on-violations"
fi

cat <<EOF | "$TNR_BIN" connect "$INSTANCE_ID"
set -euo pipefail
cd "$REMOTE_DIR"
. "$REMOTE_VENV/bin/activate"
python - <<'PY'
import torch
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available(), 'muon', hasattr(torch.optim, 'Muon'))
if not hasattr(torch.optim, 'Muon'):
    print('native torch.optim.Muon missing; using ClearMesh Muon fallback for 2D matrix parameters')
PY
rm -rf "$RUN_DIR"
mkdir -p "$RUN_DIR"
python scripts/research/build_face_token_dataset.py \
  --output-dir "$RUN_DIR/dataset" \
  --max-faces "$MAX_FACES" \
  --point-samples "$POINT_SAMPLES" \
  --seed "$SEED" \
  $BUILD_SOURCE_ARGS_STR
if [ -n "$DATASET_GATE_PROFILE" ]; then
  python scripts/research/check_face_dataset_targets.py \
    --manifest "$RUN_DIR/dataset/manifest.jsonl" \
    --output "$RUN_DIR/dataset_gate.json" \
    --profile "$DATASET_GATE_PROFILE" \
    --token-family "$DATASET_GATE_TOKEN_FAMILY" \
    $DATASET_GATE_FAIL_ARGS_STR
fi
python scripts/research/train_face_paper_faithful.py \
  --dataset-dir "$RUN_DIR/dataset" \
  --output "$RUN_DIR/face_paper_faithful.pt" \
  --steps "$STEPS" \
  --batch-size "$BATCH_SIZE" \
  --point-samples "$TRAIN_POINT_SAMPLES" \
  --model-max-faces "$MODEL_MAX_FACES" \
  --hidden-size "$HIDDEN_SIZE" \
  --encoder-hidden-size "$ENCODER_HIDDEN_SIZE" \
  --encoder-layers "$ENCODER_LAYERS" \
  --decoder-layers "$DECODER_LAYERS" \
  --heads "$HEADS" \
  --vecset-tokens "$VECSET_TOKENS" \
  --latent-dim "$LATENT_DIM" \
  --encoder-backend "$ENCODER_BACKEND" \
  --causal-mlp-variant "$CAUSAL_MLP_VARIANT" \
  --eos-loss-weight "$EOS_LOSS_WEIGHT" \
  --decode-head "$DECODE_HEAD" \
  --optimizer "$OPTIMIZER" \
  --lr "$LR" \
  --weight-decay "$WEIGHT_DECAY" \
  --seed "$SEED" \
  --augment-rotation "$AUGMENT_ROTATION" \
  --augment-scale-min "$AUGMENT_SCALE_MIN" \
  --augment-scale-max "$AUGMENT_SCALE_MAX" \
  --augment-flip-prob "$AUGMENT_FLIP_PROB" \
  --log-every "$LOG_EVERY" \
  --selection-eval-every "$SELECTION_EVAL_EVERY" \
  --selection-eval-batch-size "$SELECTION_EVAL_BATCH_SIZE" \
  $TRAIN_EOS_ARGS_STR \
  $TRAIN_AUGMENT_ARGS_STR
python scripts/research/eval_face_paper_faithful.py \
  --checkpoint "$RUN_DIR/face_paper_faithful.pt" \
  --dataset-dir "$RUN_DIR/dataset" \
  --output "$RUN_DIR/eval_report.json" \
  --export-dir "$RUN_DIR/meshes" \
  --limit "$EVAL_LIMIT" \
  --point-samples "$TRAIN_POINT_SAMPLES" \
  --face-count-mode "$FACE_COUNT_MODE" \
  --generation-mode "$GENERATION_MODE" \
  --eos-threshold "$EOS_THRESHOLD" \
  --min-generated-faces "$MIN_GENERATED_FACES" \
  --generation-face-limit "$GENERATION_FACE_LIMIT" \
  --pair-samples "$PAIR_SAMPLES" \
  --log-every "$EVAL_LOG_EVERY" \
  $EVAL_INCREMENTAL_ARGS_STR
python - <<'PY'
import json, torch
from pathlib import Path
run = Path('$RUN_DIR')
report = json.loads((run / 'eval_report.json').read_text())
ckpt = torch.load(run / 'face_paper_faithful.pt', map_location='cpu', weights_only=False)
print(json.dumps({
    'best_loss': ckpt.get('best_loss'),
    'best_step': ckpt.get('best_step'),
    'decode_head': ckpt.get('decode_head'),
    'summary': report['summary'],
}, indent=2, sort_keys=True))
PY
tar -czf /tmp/$(basename "$RUN_DIR").tar.gz -C "$RUN_DIR" .
ls -lh /tmp/$(basename "$RUN_DIR").tar.gz
exit
EOF

if [ -n "$DOWNLOAD_DIR" ]; then
  mkdir -p "$DOWNLOAD_DIR"
  archive="/tmp/$(basename "$RUN_DIR").tar.gz"
  "$TNR_BIN" scp "$INSTANCE_ID:$archive" "$DOWNLOAD_DIR/$(basename "$archive")"
  tar -xzf "$DOWNLOAD_DIR/$(basename "$archive")" -C "$DOWNLOAD_DIR"
fi
