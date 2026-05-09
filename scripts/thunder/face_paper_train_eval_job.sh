#!/usr/bin/env bash
# Train and evaluate a paper-faithful FACE ARAE checkpoint on an existing
# FACE-token split. Intended to run on a Thunder instance from /home/ubuntu/clearmesh.
set -euo pipefail

cd "${REMOTE_REPO:-/home/ubuntu/clearmesh}"

DATA_RUN="${DATA_RUN:-$(cat /tmp/clearmesh_latest_face_objpp_run.txt 2>/dev/null || true)}"
if [ -z "$DATA_RUN" ]; then
  echo "DATA_RUN is not set and /tmp/clearmesh_latest_face_objpp_run.txt is missing." >&2
  exit 1
fi

SPLIT_DIR="${SPLIT_DIR:-$DATA_RUN/split_pass}"
TRAIN_DIR="${TRAIN_DIR:-$SPLIT_DIR/train}"
TEST_DIR="${TEST_DIR:-$SPLIT_DIR/test}"
RUN_NAME="${RUN_NAME:-face_paper_train_eval_$(date -u +%Y%m%d_%H%M%S)}"
RUN_DIR="${TRAIN_RUN_DIR:-${RUN_DIR:-$DATA_RUN/$RUN_NAME}}"

STEPS="${STEPS:-3000}"
BATCH_SIZE="${BATCH_SIZE:-2}"
POINT_SAMPLES="${POINT_SAMPLES:-1024}"
MODEL_MAX_FACES="${MODEL_MAX_FACES:-512}"
HIDDEN_SIZE="${HIDDEN_SIZE:-128}"
ENCODER_HIDDEN_SIZE="${ENCODER_HIDDEN_SIZE:-128}"
ENCODER_LAYERS="${ENCODER_LAYERS:-2}"
DECODER_LAYERS="${DECODER_LAYERS:-2}"
HEADS="${HEADS:-4}"
VECSET_TOKENS="${VECSET_TOKENS:-64}"
LATENT_DIM="${LATENT_DIM:-64}"
ENCODER_BACKEND="${ENCODER_BACKEND:-shape2vecset}"
CAUSAL_MLP_VARIANT="${CAUSAL_MLP_VARIANT:-legacy_concat}"
FACE_EMBEDDING_VARIANT="${FACE_EMBEDDING_VARIANT:-token_concat_project}"
ALLOW_DEPRECATED_FACE_EMBEDDING="${ALLOW_DEPRECATED_FACE_EMBEDDING:-0}"
DECODE_HEAD="${DECODE_HEAD:-causal}"
STRICT_FACE_PAPER_GATE="${STRICT_FACE_PAPER_GATE:-1}"
OPTIMIZER="${OPTIMIZER:-adamw}"
LR="${LR:-0.0006}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
EOS_LOSS_WEIGHT="${EOS_LOSS_WEIGHT:-0.05}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-auto}"
PRECISION="${PRECISION:-fp32}"
LOG_EVERY="${LOG_EVERY:-250}"
SELECTION_EVAL_EVERY="${SELECTION_EVAL_EVERY:-500}"
SELECTION_EVAL_BATCH_SIZE="${SELECTION_EVAL_BATCH_SIZE:-1}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-0}"
SAVE_CURRENT_CHECKPOINT="${SAVE_CURRENT_CHECKPOINT:-0}"
PREFETCH_BATCHES="${PREFETCH_BATCHES:-0}"
CACHE_FPS_INDICES="${CACHE_FPS_INDICES:-0}"
DISABLE_AUGMENT="${DISABLE_AUGMENT:-0}"
AUGMENT_ROTATION="${AUGMENT_ROTATION:-so3}"
AUGMENT_SCALE_MIN="${AUGMENT_SCALE_MIN:-0.75}"
AUGMENT_SCALE_MAX="${AUGMENT_SCALE_MAX:-1.25}"
AUGMENT_FLIP_PROB="${AUGMENT_FLIP_PROB:-0.5}"
AUGMENT_DIAGNOSTICS="${AUGMENT_DIAGNOSTICS:-0}"
DISABLE_EOS_HEAD="${DISABLE_EOS_HEAD:-0}"
TRAIN_LIMIT="${TRAIN_LIMIT:-0}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"
SPLIT_INTEGRITY_IDENTITY_LIMIT="${SPLIT_INTEGRITY_IDENTITY_LIMIT:-64}"

EVAL_LIMIT="${EVAL_LIMIT:-0}"
AR_LIMIT="${AR_LIMIT:-5}"
AR_FACE_LIMIT="${AR_FACE_LIMIT:-0}"
TEACHER_PREFIX_LIMIT="${TEACHER_PREFIX_LIMIT:-16}"
TEACHER_PREFIX_FACE_COUNTS="${TEACHER_PREFIX_FACE_COUNTS:-1 4 16}"
PREDICTED_LIMIT="${PREDICTED_LIMIT:-5}"
PREDICTED_FACE_LIMIT="${PREDICTED_FACE_LIMIT:-0}"
TEACHER_FORCED_LIMIT="${TEACHER_FORCED_LIMIT:-0}"
PAIR_SAMPLES="${PAIR_SAMPLES:-500}"
EXPORT_AR_MESHES="${EXPORT_AR_MESHES:-1}"
RUN_SCALE_READINESS="${RUN_SCALE_READINESS:-1}"
FAIL_ON_SCALE_NOT_READY="${FAIL_ON_SCALE_NOT_READY:-0}"
MIN_SCALE_DATASET_SAMPLES="${MIN_SCALE_DATASET_SAMPLES:-64}"
RELAX_PAPER_KNOBS="${RELAX_PAPER_KNOBS:-0}"
ARCHIVE_PATH="${ARCHIVE_PATH:-}"

mkdir -p "$RUN_DIR"/{eval,exports,logs}

if [ "$STRICT_FACE_PAPER_GATE" = "1" ]; then
  if [ "$FACE_EMBEDDING_VARIANT" != "token_concat_project" ]; then
    echo "Strict FACE paper train/eval requires FACE_EMBEDDING_VARIANT=token_concat_project; got '$FACE_EMBEDDING_VARIANT'." >&2
    echo "Set STRICT_FACE_PAPER_GATE=0 only for an explicitly named compatibility ablation." >&2
    exit 7
  fi
  if [ "$ALLOW_DEPRECATED_FACE_EMBEDDING" = "1" ]; then
    echo "Strict FACE paper train/eval must not set ALLOW_DEPRECATED_FACE_EMBEDDING=1." >&2
    echo "Set STRICT_FACE_PAPER_GATE=0 only for an explicitly named compatibility ablation." >&2
    exit 7
  fi
fi

CHECKPOINT="$RUN_DIR/checkpoint.pt"
TRAIN_LOG="$RUN_DIR/logs/train.log"

TRAIN_ARGS=()
if [ "$DISABLE_AUGMENT" = "1" ]; then
  TRAIN_ARGS+=(--disable-augment)
fi
if [ "$DISABLE_EOS_HEAD" = "1" ]; then
  TRAIN_ARGS+=(--disable-eos-head)
fi
if [ "$AUGMENT_DIAGNOSTICS" = "1" ]; then
  TRAIN_ARGS+=(--augment-diagnostics)
fi
if [ "$CACHE_FPS_INDICES" = "1" ]; then
  TRAIN_ARGS+=(--cache-fps-indices)
fi
if [ "$SAVE_CURRENT_CHECKPOINT" = "1" ]; then
  TRAIN_ARGS+=(--save-current-checkpoint)
fi
if [ -n "$INIT_CHECKPOINT" ]; then
  TRAIN_ARGS+=(--init-checkpoint "$INIT_CHECKPOINT")
fi
if [ "$ALLOW_DEPRECATED_FACE_EMBEDDING" = "1" ]; then
  TRAIN_ARGS+=(--allow-deprecated-face-embedding)
fi

echo "[clearmesh] FACE train/eval job"
echo "  data_run=$DATA_RUN"
echo "  train_dir=$TRAIN_DIR"
echo "  test_dir=$TEST_DIR"
echo "  run_dir=$RUN_DIR"
echo "  steps=$STEPS model_max_faces=$MODEL_MAX_FACES hidden=$HIDDEN_SIZE decoder_layers=$DECODER_LAYERS"

python scripts/research/check_face_token_leakage.py \
  --train-dir "$TRAIN_DIR" \
  --test-dir "$TEST_DIR" \
  --identity-limit "$SPLIT_INTEGRITY_IDENTITY_LIMIT" \
  --output "$RUN_DIR/split_integrity.json"

python scripts/research/train_face_paper_faithful.py \
  --dataset-dir "$TRAIN_DIR" \
  --output "$CHECKPOINT" \
  --steps "$STEPS" \
  --batch-size "$BATCH_SIZE" \
  --limit "$TRAIN_LIMIT" \
  --point-samples "$POINT_SAMPLES" \
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
  --face-embedding-variant "$FACE_EMBEDDING_VARIANT" \
  --decode-head "$DECODE_HEAD" \
  --optimizer "$OPTIMIZER" \
  --lr "$LR" \
  --weight-decay "$WEIGHT_DECAY" \
  --eos-loss-weight "$EOS_LOSS_WEIGHT" \
  --seed "$SEED" \
  --device "$DEVICE" \
  --precision "$PRECISION" \
  --log-every "$LOG_EVERY" \
  --selection-eval-every "$SELECTION_EVAL_EVERY" \
  --selection-eval-batch-size "$SELECTION_EVAL_BATCH_SIZE" \
  --prefetch-batches "$PREFETCH_BATCHES" \
  --checkpoint-every "$CHECKPOINT_EVERY" \
  --augment-rotation "$AUGMENT_ROTATION" \
  --augment-scale-min "$AUGMENT_SCALE_MIN" \
  --augment-scale-max "$AUGMENT_SCALE_MAX" \
  --augment-flip-prob "$AUGMENT_FLIP_PROB" \
  "${TRAIN_ARGS[@]}" | tee "$TRAIN_LOG"

eval_dataset() {
  local split_name="$1"
  local dataset_dir="$2"
  local generation_mode="$3"
  local limit="$4"
  local face_limit="$5"
  local export_dir="$6"
  local face_count_mode="${7:-gt}"
  local teacher_prefix_faces="${8:-0}"
  local output_name="${split_name}_${generation_mode}"
  if [ "$face_count_mode" != "gt" ]; then
    output_name="${output_name}_${face_count_mode}_count"
  fi
  if [ "$teacher_prefix_faces" -gt 0 ]; then
    output_name="${output_name}_prefix${teacher_prefix_faces}"
  fi
  local output="$RUN_DIR/eval/${output_name}.json"
  local export_args=()
  local prefix_args=()
  if [ -n "$export_dir" ]; then
    mkdir -p "$export_dir"
    export_args+=(--export-dir "$export_dir")
  fi
  if [ "$teacher_prefix_faces" -gt 0 ]; then
    prefix_args+=(--teacher-prefix-faces "$teacher_prefix_faces")
  fi
  python scripts/research/eval_face_paper_faithful.py \
    --checkpoint "$CHECKPOINT" \
    --dataset-dir "$dataset_dir" \
    --output "$output" \
    --limit "$limit" \
    --point-samples "$POINT_SAMPLES" \
    --face-count-mode "$face_count_mode" \
    --generation-mode "$generation_mode" \
    --generation-face-limit "$face_limit" \
    --pair-samples "$PAIR_SAMPLES" \
    --device "$DEVICE" \
    --log-every 1 \
    "${prefix_args[@]}" \
    "${export_args[@]}"
}

eval_dataset train "$TRAIN_DIR" teacher_forced "$TEACHER_FORCED_LIMIT" 0 "" gt
eval_dataset test "$TEST_DIR" teacher_forced "$TEACHER_FORCED_LIMIT" 0 "" gt

if [ "$AR_LIMIT" -gt 0 ]; then
  train_export=""
  test_export=""
  if [ "$EXPORT_AR_MESHES" = "1" ]; then
    train_export="$RUN_DIR/exports/train_ar"
    test_export="$RUN_DIR/exports/test_ar"
  fi
  eval_dataset train "$TRAIN_DIR" autoregressive "$AR_LIMIT" "$AR_FACE_LIMIT" "$train_export" gt
  eval_dataset test "$TEST_DIR" autoregressive "$AR_LIMIT" "$AR_FACE_LIMIT" "$test_export" gt
fi

if [ "$TEACHER_PREFIX_LIMIT" -gt 0 ]; then
  for prefix_faces in $TEACHER_PREFIX_FACE_COUNTS; do
    eval_dataset train "$TRAIN_DIR" autoregressive "$TEACHER_PREFIX_LIMIT" "$AR_FACE_LIMIT" "" gt "$prefix_faces"
    eval_dataset test "$TEST_DIR" autoregressive "$TEACHER_PREFIX_LIMIT" "$AR_FACE_LIMIT" "" gt "$prefix_faces"
  done
fi

if [ "$PREDICTED_LIMIT" -gt 0 ]; then
  eval_dataset train "$TRAIN_DIR" autoregressive "$PREDICTED_LIMIT" "$PREDICTED_FACE_LIMIT" "" predicted
  eval_dataset test "$TEST_DIR" autoregressive "$PREDICTED_LIMIT" "$PREDICTED_FACE_LIMIT" "" predicted
fi

python - <<PY
import json
from pathlib import Path

run = Path("$RUN_DIR")

def read(path):
    return json.loads(path.read_text()) if path.exists() else None

summary = {
    "run_dir": str(run),
    "data_run": "$DATA_RUN",
    "split_dir": "$SPLIT_DIR",
    "train_dir": "$TRAIN_DIR",
    "test_dir": "$TEST_DIR",
    "checkpoint": "$CHECKPOINT",
    "settings": {
        "steps": int("$STEPS"),
        "batch_size": int("$BATCH_SIZE"),
        "point_samples": int("$POINT_SAMPLES"),
        "model_max_faces": int("$MODEL_MAX_FACES"),
        "hidden_size": int("$HIDDEN_SIZE"),
        "encoder_hidden_size": int("$ENCODER_HIDDEN_SIZE"),
        "encoder_layers": int("$ENCODER_LAYERS"),
        "decoder_layers": int("$DECODER_LAYERS"),
        "heads": int("$HEADS"),
        "vecset_tokens": int("$VECSET_TOKENS"),
        "latent_dim": int("$LATENT_DIM"),
        "encoder_backend": "$ENCODER_BACKEND",
        "causal_mlp_variant": "$CAUSAL_MLP_VARIANT",
        "face_embedding_variant": "$FACE_EMBEDDING_VARIANT",
        "allow_deprecated_face_embedding": "$ALLOW_DEPRECATED_FACE_EMBEDDING" == "1",
        "strict_face_paper_gate": "$STRICT_FACE_PAPER_GATE" == "1",
        "decode_head": "$DECODE_HEAD",
        "optimizer": "$OPTIMIZER",
        "lr": float("$LR"),
        "weight_decay": float("$WEIGHT_DECAY"),
        "seed": int("$SEED"),
        "precision": "$PRECISION",
        "checkpoint_every": int("$CHECKPOINT_EVERY"),
        "prefetch_batches": int("$PREFETCH_BATCHES"),
        "cache_fps_indices": "$CACHE_FPS_INDICES" == "1",
        "disable_augment": "$DISABLE_AUGMENT" == "1",
        "augment_rotation": "$AUGMENT_ROTATION",
        "augment_scale_min": float("$AUGMENT_SCALE_MIN"),
        "augment_scale_max": float("$AUGMENT_SCALE_MAX"),
        "augment_flip_prob": float("$AUGMENT_FLIP_PROB"),
        "augment_diagnostics": "$AUGMENT_DIAGNOSTICS" == "1",
        "train_limit": int("$TRAIN_LIMIT"),
        "ar_limit": int("$AR_LIMIT"),
        "ar_face_limit": int("$AR_FACE_LIMIT"),
        "teacher_prefix_limit": int("$TEACHER_PREFIX_LIMIT"),
        "teacher_prefix_face_counts": "$TEACHER_PREFIX_FACE_COUNTS",
        "predicted_limit": int("$PREDICTED_LIMIT"),
        "predicted_face_limit": int("$PREDICTED_FACE_LIMIT"),
        "split_integrity_identity_limit": int("$SPLIT_INTEGRITY_IDENTITY_LIMIT"),
    },
    "split_integrity": read(run / "split_integrity.json"),
    "eval": {
        "train_teacher_forced": (read(run / "eval" / "train_teacher_forced.json") or {}).get("summary"),
        "test_teacher_forced": (read(run / "eval" / "test_teacher_forced.json") or {}).get("summary"),
        "train_autoregressive": (read(run / "eval" / "train_autoregressive.json") or {}).get("summary"),
        "test_autoregressive": (read(run / "eval" / "test_autoregressive.json") or {}).get("summary"),
        "train_autoregressive_predicted_count": (read(run / "eval" / "train_autoregressive_predicted_count.json") or {}).get("summary"),
        "test_autoregressive_predicted_count": (read(run / "eval" / "test_autoregressive_predicted_count.json") or {}).get("summary"),
        "teacher_prefix": {
            path.stem: (read(path) or {}).get("summary")
            for path in sorted((run / "eval").glob("*_autoregressive_prefix*.json"))
        },
    },
}
(run / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
PY

if [ "$RUN_SCALE_READINESS" = "1" ]; then
  readiness_args=()
  if [ "$FAIL_ON_SCALE_NOT_READY" = "1" ]; then
    readiness_args+=(--fail-on-not-ready)
  fi
  if [ "$RELAX_PAPER_KNOBS" = "1" ]; then
    readiness_args+=(--relax-paper-knobs)
  fi
  python scripts/research/assess_face_paper_scale_readiness.py \
    --run-dir "$RUN_DIR" \
    --output "$RUN_DIR/scale_readiness.json" \
    --min-dataset-samples "$MIN_SCALE_DATASET_SAMPLES" \
    "${readiness_args[@]}"
fi

if [ -n "$ARCHIVE_PATH" ]; then
  tar -czf "$ARCHIVE_PATH" -C "$(dirname "$RUN_DIR")" "$(basename "$RUN_DIR")"
  echo "[clearmesh] archived $RUN_DIR to $ARCHIVE_PATH"
fi
