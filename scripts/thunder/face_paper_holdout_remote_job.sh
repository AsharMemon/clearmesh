#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${RUN_DIR:-/tmp/clearmesh_face_paper_holdout_smoke}"
MESH_DIR="${MESH_DIR:-}"
SYNTHETIC_COUNT="${SYNTHETIC_COUNT:-24}"
SYNTHETIC_KIND="${SYNTHETIC_KIND:-mixed_cycle}"
MAX_FACES="${MAX_FACES:-4096}"
MODEL_MAX_FACES="${MODEL_MAX_FACES:-$MAX_FACES}"
POINT_SAMPLES="${POINT_SAMPLES:-2048}"
TRAIN_POINT_SAMPLES="${TRAIN_POINT_SAMPLES:-2048}"
TEST_COUNT="${TEST_COUNT:-4}"
SPLIT_SEED="${SPLIT_SEED:-73}"
SPLIT_SHUFFLE="${SPLIT_SHUFFLE:-1}"
SPLIT_GROUP_FIELD="${SPLIT_GROUP_FIELD:-}"
SPLIT_GROUP_REGEX="${SPLIT_GROUP_REGEX:-}"
STEPS="${STEPS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-1}"
HIDDEN_SIZE="${HIDDEN_SIZE:-192}"
ENCODER_HIDDEN_SIZE="${ENCODER_HIDDEN_SIZE:-192}"
ENCODER_LAYERS="${ENCODER_LAYERS:-2}"
DECODER_LAYERS="${DECODER_LAYERS:-3}"
HEADS="${HEADS:-4}"
VECSET_TOKENS="${VECSET_TOKENS:-128}"
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
LOG_EVERY="${LOG_EVERY:-200}"
SELECTION_EVAL_EVERY="${SELECTION_EVAL_EVERY:-200}"
SELECTION_EVAL_BATCH_SIZE="${SELECTION_EVAL_BATCH_SIZE:-1}"
DATASET_GATE_PROFILE="${DATASET_GATE_PROFILE:-strict}"
DATASET_GATE_TOKEN_FAMILY="${DATASET_GATE_TOKEN_FAMILY:-paper}"
FAIL_ON_DATASET_GATE="${FAIL_ON_DATASET_GATE:-1}"
TRAIN_EVAL_MODE="${TRAIN_EVAL_MODE:-autoregressive}"
TEST_EVAL_MODE="${TEST_EVAL_MODE:-autoregressive}"
TRAIN_FACE_COUNT_MODE="${TRAIN_FACE_COUNT_MODE:-gt}"
TEST_FACE_COUNT_MODE="${TEST_FACE_COUNT_MODE:-gt}"
EOS_THRESHOLD="${EOS_THRESHOLD:-0.5}"
MIN_GENERATED_FACES="${MIN_GENERATED_FACES:-1}"
EVAL_LIMIT="${EVAL_LIMIT:-0}"
TRAIN_EVAL_LIMIT="${TRAIN_EVAL_LIMIT:-$EVAL_LIMIT}"
TEST_EVAL_LIMIT="${TEST_EVAL_LIMIT:-$EVAL_LIMIT}"
GENERATION_FACE_LIMIT="${GENERATION_FACE_LIMIT:-0}"
PAIR_SAMPLES="${PAIR_SAMPLES:-500}"
EVAL_LOG_EVERY="${EVAL_LOG_EVERY:-1}"
SEED="${SEED:-73}"
ARCHIVE_PATH="${ARCHIVE_PATH:-/tmp/$(basename "$RUN_DIR").tar.gz}"

python - <<'PY'
import torch

print(
    "torch",
    torch.__version__,
    "cuda",
    torch.version.cuda,
    "available",
    torch.cuda.is_available(),
    "muon",
    hasattr(torch.optim, "Muon"),
)
if not hasattr(torch.optim, "Muon"):
    print("native torch.optim.Muon missing; using ClearMesh Muon fallback for 2D matrix parameters")
PY

rm -rf "$RUN_DIR"
mkdir -p "$RUN_DIR"

BUILD_SOURCE_ARGS=()
if [ -n "$MESH_DIR" ]; then
  BUILD_SOURCE_ARGS+=(--mesh-dir "$MESH_DIR")
fi
if [ "$SYNTHETIC_COUNT" != "0" ]; then
  BUILD_SOURCE_ARGS+=(--synthetic-count "$SYNTHETIC_COUNT" --synthetic-kind "$SYNTHETIC_KIND")
fi

python scripts/research/build_face_token_dataset.py \
  --output-dir "$RUN_DIR/dataset_all" \
  --max-faces "$MAX_FACES" \
  --point-samples "$POINT_SAMPLES" \
  --seed "$SEED" \
  "${BUILD_SOURCE_ARGS[@]}"

SPLIT_ARGS=()
if [ "$SPLIT_SHUFFLE" = "1" ]; then
  SPLIT_ARGS+=(--shuffle)
fi
if [ -n "$SPLIT_GROUP_FIELD" ]; then
  SPLIT_ARGS+=(--group-field "$SPLIT_GROUP_FIELD")
fi
if [ -n "$SPLIT_GROUP_REGEX" ]; then
  SPLIT_ARGS+=(--group-regex "$SPLIT_GROUP_REGEX")
fi

python scripts/research/split_face_token_dataset.py \
  --dataset-dir "$RUN_DIR/dataset_all" \
  --output-dir "$RUN_DIR/split" \
  --test-count "$TEST_COUNT" \
  --seed "$SPLIT_SEED" \
  "${SPLIT_ARGS[@]}"

DATASET_GATE_FAIL_ARGS=()
if [ "$FAIL_ON_DATASET_GATE" = "1" ]; then
  DATASET_GATE_FAIL_ARGS+=(--fail-on-violations)
fi
if [ -n "$DATASET_GATE_PROFILE" ]; then
  python scripts/research/check_face_dataset_targets.py \
    --manifest "$RUN_DIR/split/train/manifest.jsonl" \
    --output "$RUN_DIR/train_dataset_gate.json" \
    --profile "$DATASET_GATE_PROFILE" \
    --token-family "$DATASET_GATE_TOKEN_FAMILY" \
    "${DATASET_GATE_FAIL_ARGS[@]}"
  python scripts/research/check_face_dataset_targets.py \
    --manifest "$RUN_DIR/split/test/manifest.jsonl" \
    --output "$RUN_DIR/test_dataset_gate.json" \
    --profile "$DATASET_GATE_PROFILE" \
    --token-family "$DATASET_GATE_TOKEN_FAMILY" \
    "${DATASET_GATE_FAIL_ARGS[@]}"
fi

TRAIN_AUGMENT_ARGS=()
if [ "$DISABLE_AUGMENT" = "1" ]; then
  TRAIN_AUGMENT_ARGS+=(--disable-augment)
fi
TRAIN_EOS_ARGS=()
if [ "$DISABLE_EOS_HEAD" = "1" ]; then
  TRAIN_EOS_ARGS+=(--disable-eos-head)
fi

python scripts/research/train_face_paper_faithful.py \
  --dataset-dir "$RUN_DIR/split/train" \
  --output "$RUN_DIR/face_paper_holdout.pt" \
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
  "${TRAIN_EOS_ARGS[@]}" \
  "${TRAIN_AUGMENT_ARGS[@]}"

python scripts/research/eval_face_paper_faithful.py \
  --checkpoint "$RUN_DIR/face_paper_holdout.pt" \
  --dataset-dir "$RUN_DIR/split/train" \
  --output "$RUN_DIR/train_eval_report.json" \
  --export-dir "$RUN_DIR/train_meshes" \
  --limit "$TRAIN_EVAL_LIMIT" \
  --point-samples "$TRAIN_POINT_SAMPLES" \
  --face-count-mode "$TRAIN_FACE_COUNT_MODE" \
  --generation-mode "$TRAIN_EVAL_MODE" \
  --eos-threshold "$EOS_THRESHOLD" \
  --min-generated-faces "$MIN_GENERATED_FACES" \
  --generation-face-limit "$GENERATION_FACE_LIMIT" \
  --pair-samples "$PAIR_SAMPLES" \
  --log-every "$EVAL_LOG_EVERY"

python scripts/research/eval_face_paper_faithful.py \
  --checkpoint "$RUN_DIR/face_paper_holdout.pt" \
  --dataset-dir "$RUN_DIR/split/test" \
  --output "$RUN_DIR/test_eval_report.json" \
  --export-dir "$RUN_DIR/test_meshes" \
  --limit "$TEST_EVAL_LIMIT" \
  --point-samples "$TRAIN_POINT_SAMPLES" \
  --face-count-mode "$TEST_FACE_COUNT_MODE" \
  --generation-mode "$TEST_EVAL_MODE" \
  --eos-threshold "$EOS_THRESHOLD" \
  --min-generated-faces "$MIN_GENERATED_FACES" \
  --generation-face-limit "$GENERATION_FACE_LIMIT" \
  --pair-samples "$PAIR_SAMPLES" \
  --log-every "$EVAL_LOG_EVERY"

python - <<PY
import json
from pathlib import Path

import torch

run = Path("$RUN_DIR")
ckpt = torch.load(run / "face_paper_holdout.pt", map_location="cpu", weights_only=False)
summary = {
    "checkpoint": {
        "best_loss": ckpt.get("best_loss"),
        "best_step": ckpt.get("best_step"),
        "decode_head": ckpt.get("decode_head"),
    },
    "split": json.loads((run / "split" / "split_summary.json").read_text()),
    "train": json.loads((run / "train_eval_report.json").read_text())["summary"],
    "test": json.loads((run / "test_eval_report.json").read_text())["summary"],
}
(run / "holdout_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
print(json.dumps(summary, indent=2, sort_keys=True))
PY

tar -czf "$ARCHIVE_PATH" -C "$RUN_DIR" .
ls -lh "$ARCHIVE_PATH"
