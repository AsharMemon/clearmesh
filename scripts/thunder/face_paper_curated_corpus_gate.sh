#!/usr/bin/env bash
# Build a curated strict FACE paper-token corpus on a Thunder box, train the
# paper-faithful FACE lane, evaluate full-face AR, gate scale readiness, and
# archive artifacts.
set -euo pipefail

cd "${REMOTE_REPO:-/home/ubuntu/clearmesh}"

LAB_ROOT="${LAB_ROOT:-/tmp/clearmesh_face_paper_corpus_gate_$(date -u +%Y%m%d_%H%M%S)}"
CORPUS_DIR="${CORPUS_DIR:-$LAB_ROOT/corpus}"
RUN_LABEL="${RUN_LABEL:-paper_corpus_gate_128_vec2048_muon_aug}"
RUN_DIR="${RUN_DIR:-$LAB_ROOT/runs/$RUN_LABEL}"
LOG_DIR="$LAB_ROOT/logs"
STATUS_FILE="$LAB_ROOT/status.jsonl"
mkdir -p "$LOG_DIR" "$RUN_DIR"
echo "$LAB_ROOT" > /tmp/clearmesh_latest_face_paper_corpus_gate_run.txt

log_status() {
  local step="$1"
  local status="$2"
  local detail="${3:-}"
  STATUS_FILE="$STATUS_FILE" STEP="$step" STATUS="$status" DETAIL="$detail" python - <<'PY' >&2
import json
import os
import time
from pathlib import Path
row = {"time": time.time(), "step": os.environ["STEP"], "status": os.environ["STATUS"], "detail": os.environ.get("DETAIL", "")}
path = Path(os.environ["STATUS_FILE"])
path.parent.mkdir(parents=True, exist_ok=True)
with path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(row, sort_keys=True) + "\n")
print(json.dumps(row, sort_keys=True), flush=True)
PY
}

run_step() {
  local step="$1"
  local log="$LOG_DIR/$step.log"
  shift
  log_status "$step" started "log=$log"
  if "$@" >"$log" 2>&1; then
    log_status "$step" completed "log=$log"
  else
    local code=$?
    log_status "$step" failed "exit=$code log=$log"
    return "$code"
  fi
}

# Corpus knobs. Defaults are a bounded paper-lane rung, not the final 130k/4000-face run.
ANNOTATIONS="${ANNOTATIONS:-cindyxl/ObjaversePlusPlus}"
SPLIT="${SPLIT:-train}"
SELECT_TARGET="${SELECT_TARGET:-1000}"
SCAN_LIMIT="${SCAN_LIMIT:-50000}"
CURATION_TARGET="${CURATION_TARGET:-$SELECT_TARGET}"
MIN_QUALITY="${MIN_QUALITY:-2}"
OVERSAMPLE_FACTOR="${OVERSAMPLE_FACTOR:-5}"
DOWNLOAD_PROCESSES="${DOWNLOAD_PROCESSES:-8}"
DOWNLOAD_BATCH_SIZE="${DOWNLOAD_BATCH_SIZE:-25}"
TARGET_FACES="${TARGET_FACES:-512}"
TOKEN_MAX_FACES="${TOKEN_MAX_FACES:-512}"
MODEL_MAX_FACES="${MODEL_MAX_FACES:-512}"
NUM_BINS="${NUM_BINS:-128}"
PAPER_WITHIN_FACE_ORDER="${PAPER_WITHIN_FACE_ORDER:-preserve}"
POINT_SAMPLES="${POINT_SAMPLES:-8192}"
VOXEL_RESOLUTION="${VOXEL_RESOLUTION:-64}"
MESH_VOXEL_MAX_FACES="${MESH_VOXEL_MAX_FACES:-5000}"
STRICT_ENGINE="${STRICT_ENGINE:-voxel_shell}"
FALLBACK="${FALLBACK:-convex_hull}"
TEST_RATIO="${TEST_RATIO:-0.2}"
SEED="${SEED:-303}"

# Paper training knobs.
STEPS="${STEPS:-30000}"
BATCH_SIZE="${BATCH_SIZE:-1}"
VECSET_TOKENS="${VECSET_TOKENS:-2048}"
LATENT_DIM="${LATENT_DIM:-64}"
HIDDEN_SIZE="${HIDDEN_SIZE:-384}"
ENCODER_HIDDEN_SIZE="${ENCODER_HIDDEN_SIZE:-384}"
ENCODER_LAYERS="${ENCODER_LAYERS:-4}"
DECODER_LAYERS="${DECODER_LAYERS:-8}"
HEADS="${HEADS:-8}"
PRECISION="${PRECISION:-bf16}"
LOG_EVERY="${LOG_EVERY:-50}"
SELECTION_EVAL_EVERY="${SELECTION_EVAL_EVERY:-1000}"
SELECTION_EVAL_BATCH_SIZE="${SELECTION_EVAL_BATCH_SIZE:-1}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1000}"
SAVE_CURRENT_CHECKPOINT="${SAVE_CURRENT_CHECKPOINT:-1}"
PREFETCH_BATCHES="${PREFETCH_BATCHES:-0}"
CACHE_FPS_INDICES="${CACHE_FPS_INDICES:-0}"
TEACHER_FORCED_LIMIT="${TEACHER_FORCED_LIMIT:-0}"
AR_LIMIT="${AR_LIMIT:-20}"
AR_FACE_LIMIT="${AR_FACE_LIMIT:-0}"
TEACHER_PREFIX_LIMIT="${TEACHER_PREFIX_LIMIT:-0}"
TEACHER_PREFIX_FACE_COUNTS="${TEACHER_PREFIX_FACE_COUNTS:-1 4 16}"
PREDICTED_LIMIT="${PREDICTED_LIMIT:-5}"
PREDICTED_FACE_LIMIT="${PREDICTED_FACE_LIMIT:-0}"
PAIR_SAMPLES="${PAIR_SAMPLES:-500}"
MIN_SCALE_DATASET_SAMPLES="${MIN_SCALE_DATASET_SAMPLES:-256}"
FAIL_ON_SCALE_NOT_READY="${FAIL_ON_SCALE_NOT_READY:-0}"
ARCHIVE_PATH="${ARCHIVE_PATH:-$LAB_ROOT.tar.gz}"
CAUSAL_MLP_VARIANT="${CAUSAL_MLP_VARIANT:-legacy_concat}"
FACE_EMBEDDING_VARIANT="${FACE_EMBEDDING_VARIANT:-token_concat_project}"
DECODE_HEAD="${DECODE_HEAD:-causal}"
ALLOW_DEPRECATED_FACE_EMBEDDING="${ALLOW_DEPRECATED_FACE_EMBEDDING:-0}"
STRICT_FACE_PAPER_GATE="${STRICT_FACE_PAPER_GATE:-1}"

if [ "$STRICT_FACE_PAPER_GATE" = "1" ]; then
  if [ "$FACE_EMBEDDING_VARIANT" != "token_concat_project" ]; then
    echo "Strict FACE paper corpus gate requires FACE_EMBEDDING_VARIANT=token_concat_project; got '$FACE_EMBEDDING_VARIANT'." >&2
    echo "Set STRICT_FACE_PAPER_GATE=0 only for an explicitly named compatibility ablation." >&2
    exit 7
  fi
  if [ "$ALLOW_DEPRECATED_FACE_EMBEDDING" = "1" ]; then
    echo "Strict FACE paper corpus gate must not set ALLOW_DEPRECATED_FACE_EMBEDDING=1." >&2
    echo "Set STRICT_FACE_PAPER_GATE=0 only for an explicitly named compatibility ablation." >&2
    exit 7
  fi
fi

archive_on_failure() {
  local code=$?
  if [ "$code" -eq 0 ]; then
    return 0
  fi
  if [ -d "$LAB_ROOT" ]; then
    tar -czf "$ARCHIVE_PATH" -C "$(dirname "$LAB_ROOT")" "$(basename "$LAB_ROOT")" \
      && log_status corpus_gate archived_after_failure "exit=$code archive=$ARCHIVE_PATH" \
      || true
  fi
}
trap archive_on_failure EXIT

log_status corpus_gate started "lab_root=$LAB_ROOT select_target=$SELECT_TARGET num_bins=$NUM_BINS steps=$STEPS"

run_step env_probe python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0), "bf16", torch.cuda.is_bf16_supported())
print("native_muon", hasattr(torch.optim, "Muon"))
PY

run_step build_curated_corpus env \
  RUN_DIR="$CORPUS_DIR" \
  ANNOTATIONS="$ANNOTATIONS" \
  SPLIT="$SPLIT" \
  SELECT_TARGET="$SELECT_TARGET" \
  SCAN_LIMIT="$SCAN_LIMIT" \
  CURATION_TARGET="$CURATION_TARGET" \
  MIN_QUALITY="$MIN_QUALITY" \
  OVERSAMPLE_FACTOR="$OVERSAMPLE_FACTOR" \
  DOWNLOAD_PROCESSES="$DOWNLOAD_PROCESSES" \
  DOWNLOAD_BATCH_SIZE="$DOWNLOAD_BATCH_SIZE" \
  STRICT_ENGINE="$STRICT_ENGINE" \
  TARGET_FACES="$TARGET_FACES" \
  TOKEN_MAX_FACES="$TOKEN_MAX_FACES" \
  NUM_BINS="$NUM_BINS" \
  PAPER_WITHIN_FACE_ORDER="$PAPER_WITHIN_FACE_ORDER" \
  POINT_SAMPLES="$POINT_SAMPLES" \
  VOXEL_RESOLUTION="$VOXEL_RESOLUTION" \
  MESH_VOXEL_MAX_FACES="$MESH_VOXEL_MAX_FACES" \
  FALLBACK="$FALLBACK" \
  TEST_RATIO="$TEST_RATIO" \
  SEED="$SEED" \
  FAIL_ON_GATE=0 \
  PROMOTE_PASSING=1 \
  SPLIT_PASSING=1 \
  ARCHIVE_PATH= \
  scripts/thunder/face_objaversepp_corpus_pilot.sh

if [ ! -d "$CORPUS_DIR/split_pass/train" ] || [ ! -d "$CORPUS_DIR/split_pass/test" ]; then
  log_status corpus_gate failed "missing split_pass under $CORPUS_DIR"
  exit 10
fi

run_step train_eval env \
  DATA_RUN="$CORPUS_DIR" \
  SPLIT_DIR="$CORPUS_DIR/split_pass" \
  TRAIN_DIR="$CORPUS_DIR/split_pass/train" \
  TEST_DIR="$CORPUS_DIR/split_pass/test" \
  RUN_NAME="$RUN_LABEL" \
  RUN_DIR="$RUN_DIR" \
  STEPS="$STEPS" \
  BATCH_SIZE="$BATCH_SIZE" \
  POINT_SAMPLES="$POINT_SAMPLES" \
  MODEL_MAX_FACES="$MODEL_MAX_FACES" \
  HIDDEN_SIZE="$HIDDEN_SIZE" \
  ENCODER_HIDDEN_SIZE="$ENCODER_HIDDEN_SIZE" \
  ENCODER_LAYERS="$ENCODER_LAYERS" \
  DECODER_LAYERS="$DECODER_LAYERS" \
  HEADS="$HEADS" \
  VECSET_TOKENS="$VECSET_TOKENS" \
  LATENT_DIM="$LATENT_DIM" \
  PRECISION="$PRECISION" \
  ENCODER_BACKEND=shape2vecset \
  CAUSAL_MLP_VARIANT="$CAUSAL_MLP_VARIANT" \
  FACE_EMBEDDING_VARIANT="$FACE_EMBEDDING_VARIANT" \
  DECODE_HEAD="$DECODE_HEAD" \
  ALLOW_DEPRECATED_FACE_EMBEDDING="$ALLOW_DEPRECATED_FACE_EMBEDDING" \
  STRICT_FACE_PAPER_GATE="$STRICT_FACE_PAPER_GATE" \
  OPTIMIZER=muon \
  LR=0.0006 \
  WEIGHT_DECAY=0.1 \
  EOS_LOSS_WEIGHT=0.05 \
  DISABLE_AUGMENT=0 \
  AUGMENT_ROTATION=so3 \
  AUGMENT_SCALE_MIN=0.75 \
  AUGMENT_SCALE_MAX=1.25 \
  AUGMENT_FLIP_PROB=0.5 \
  SEED="$SEED" \
  LOG_EVERY="$LOG_EVERY" \
  SELECTION_EVAL_EVERY="$SELECTION_EVAL_EVERY" \
  SELECTION_EVAL_BATCH_SIZE="$SELECTION_EVAL_BATCH_SIZE" \
  CHECKPOINT_EVERY="$CHECKPOINT_EVERY" \
  SAVE_CURRENT_CHECKPOINT="$SAVE_CURRENT_CHECKPOINT" \
  PREFETCH_BATCHES="$PREFETCH_BATCHES" \
  CACHE_FPS_INDICES="$CACHE_FPS_INDICES" \
  TEACHER_FORCED_LIMIT="$TEACHER_FORCED_LIMIT" \
  AR_LIMIT="$AR_LIMIT" \
  AR_FACE_LIMIT="$AR_FACE_LIMIT" \
  TEACHER_PREFIX_LIMIT="$TEACHER_PREFIX_LIMIT" \
  TEACHER_PREFIX_FACE_COUNTS="$TEACHER_PREFIX_FACE_COUNTS" \
  PREDICTED_LIMIT="$PREDICTED_LIMIT" \
  PREDICTED_FACE_LIMIT="$PREDICTED_FACE_LIMIT" \
  PAIR_SAMPLES="$PAIR_SAMPLES" \
  EXPORT_AR_MESHES=1 \
  RUN_SCALE_READINESS=1 \
  FAIL_ON_SCALE_NOT_READY="$FAIL_ON_SCALE_NOT_READY" \
  MIN_SCALE_DATASET_SAMPLES="$MIN_SCALE_DATASET_SAMPLES" \
  scripts/thunder/face_paper_train_eval_job.sh

for split in train test; do
  mesh_dir="$RUN_DIR/exports/${split}_ar"
  if [ -d "$mesh_dir" ]; then
    if ! run_step "gallery_${split}_ar" python scripts/eval/make_mesh_contact_sheet.py \
      --mesh-dir "$mesh_dir" \
      --report "$RUN_DIR/eval/${split}_autoregressive.json" \
      --output "$LAB_ROOT/${split}_ar_contact_sheet.png" \
      --title "FACE $RUN_LABEL $split full-face AR"; then
      log_status "gallery_${split}_ar" nonfatal "continuing_to_summary_and_archive"
    fi
  fi
done

run_step summary python - "$LAB_ROOT" "$CORPUS_DIR" "$RUN_DIR" <<'PY'
import json
import sys
from pathlib import Path
lab = Path(sys.argv[1])
corpus = Path(sys.argv[2])
run = Path(sys.argv[3])

def read(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
summary = {
    "lab_root": str(lab),
    "corpus_dir": str(corpus),
    "run_dir": str(run),
    "corpus_summary": read(corpus / "pilot_summary.json"),
    "run_summary": read(run / "summary.json"),
    "scale_readiness": read(run / "scale_readiness.json"),
    "galleries": sorted(str(path) for path in lab.glob("*_contact_sheet.png")),
}
(lab / "paper_corpus_gate_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps({"lab_root": str(lab), "scale_ready": (summary.get("scale_readiness") or {}).get("scale_ready")}, indent=2, sort_keys=True))
PY

tar -czf "$ARCHIVE_PATH" -C "$(dirname "$LAB_ROOT")" "$(basename "$LAB_ROOT")"
log_status corpus_gate completed "archive=$ARCHIVE_PATH"
trap - EXIT
