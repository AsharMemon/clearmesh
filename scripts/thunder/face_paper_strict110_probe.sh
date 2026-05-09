#!/usr/bin/env bash
# Strict FACE paper-lane probe on the current 110-sample pass-only split.
#
# This is the next rung after memorization: keep paper knobs and online
# augmentation, but still avoid production scale until full-face AR metrics look
# healthy on a bounded corpus.
set -euo pipefail

cd "${REMOTE_REPO:-/home/ubuntu/clearmesh}"

SOURCE_LAB="${SOURCE_LAB:-$(cat /tmp/clearmesh_latest_face_a100_probe_run.txt 2>/dev/null || true)}"
if [ -z "$SOURCE_LAB" ] || [ ! -d "$SOURCE_LAB" ]; then
  echo "SOURCE_LAB missing or not found: $SOURCE_LAB" >&2
  exit 1
fi
SPLIT_DIR="${SPLIT_DIR:-$SOURCE_LAB/tokens_128_preserve_paper_knobs/pass_only/split_pass}"
if [ ! -d "$SPLIT_DIR/train" ] || [ ! -d "$SPLIT_DIR/test" ]; then
  echo "split train/test not found under: $SPLIT_DIR" >&2
  exit 1
fi

LAB_ROOT="${LAB_ROOT:-/tmp/clearmesh_face_strict110_probe_$(date -u +%Y%m%d_%H%M%S)}"
RUN_LABEL="${RUN_LABEL:-strict110_paperaug_vec2048_muon}"
RUN_DIR="$LAB_ROOT/runs/$RUN_LABEL"
LOG_DIR="$LAB_ROOT/logs"
STATUS_FILE="$LAB_ROOT/status.jsonl"
mkdir -p "$LOG_DIR" "$RUN_DIR"
echo "$LAB_ROOT" > /tmp/clearmesh_latest_face_strict110_probe_run.txt

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

STEPS="${STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-1}"
POINT_SAMPLES="${POINT_SAMPLES:-8192}"
MODEL_MAX_FACES="${MODEL_MAX_FACES:-512}"
VECSET_TOKENS="${VECSET_TOKENS:-2048}"
LATENT_DIM="${LATENT_DIM:-64}"
HIDDEN_SIZE="${HIDDEN_SIZE:-384}"
ENCODER_HIDDEN_SIZE="${ENCODER_HIDDEN_SIZE:-384}"
ENCODER_LAYERS="${ENCODER_LAYERS:-4}"
DECODER_LAYERS="${DECODER_LAYERS:-8}"
HEADS="${HEADS:-8}"
PRECISION="${PRECISION:-bf16}"
LOG_EVERY="${LOG_EVERY:-500}"
SELECTION_EVAL_EVERY="${SELECTION_EVAL_EVERY:-1000}"
SELECTION_EVAL_BATCH_SIZE="${SELECTION_EVAL_BATCH_SIZE:-1}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-2000}"
AR_LIMIT="${AR_LIMIT:-10}"
AR_FACE_LIMIT="${AR_FACE_LIMIT:-0}"
PAIR_SAMPLES="${PAIR_SAMPLES:-500}"
TRAIN_LIMIT="${TRAIN_LIMIT:-0}"
TEACHER_FORCED_LIMIT="${TEACHER_FORCED_LIMIT:-0}"
DISABLE_AUGMENT="${DISABLE_AUGMENT:-0}"

log_status "strict110_probe" "started" "lab_root=$LAB_ROOT source_lab=$SOURCE_LAB split_dir=$SPLIT_DIR steps=$STEPS ar_face_limit=$AR_FACE_LIMIT disable_augment=$DISABLE_AUGMENT"

if env \
  DATA_RUN="$(cat /tmp/clearmesh_latest_face_objpp_run.txt 2>/dev/null || echo "$SOURCE_LAB")" \
  SPLIT_DIR="$SPLIT_DIR" \
  TRAIN_DIR="$SPLIT_DIR/train" \
  TEST_DIR="$SPLIT_DIR/test" \
  RUN_NAME="$RUN_LABEL" \
  RUN_DIR="$RUN_DIR" \
  STEPS="$STEPS" \
  TRAIN_LIMIT="$TRAIN_LIMIT" \
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
  ENCODER_BACKEND="shape2vecset" \
  CAUSAL_MLP_VARIANT="legacy_concat" \
  DECODE_HEAD="causal" \
  OPTIMIZER="muon" \
  LR="0.0006" \
  WEIGHT_DECAY="0.1" \
  EOS_LOSS_WEIGHT="0.05" \
  DISABLE_AUGMENT="$DISABLE_AUGMENT" \
  AUGMENT_ROTATION="so3" \
  AUGMENT_SCALE_MIN="0.75" \
  AUGMENT_SCALE_MAX="1.25" \
  AUGMENT_FLIP_PROB="0.5" \
  SEED="303" \
  LOG_EVERY="$LOG_EVERY" \
  SELECTION_EVAL_EVERY="$SELECTION_EVAL_EVERY" \
  SELECTION_EVAL_BATCH_SIZE="$SELECTION_EVAL_BATCH_SIZE" \
  CHECKPOINT_EVERY="$CHECKPOINT_EVERY" \
  TEACHER_FORCED_LIMIT="$TEACHER_FORCED_LIMIT" \
  AR_LIMIT="$AR_LIMIT" \
  AR_FACE_LIMIT="$AR_FACE_LIMIT" \
  PAIR_SAMPLES="$PAIR_SAMPLES" \
  EXPORT_AR_MESHES="1" \
  scripts/thunder/face_paper_train_eval_job.sh >"$LOG_DIR/train_eval.log" 2>&1; then
  log_status "train_eval" "completed" "run_dir=$RUN_DIR"
else
  code=$?
  log_status "train_eval" "failed" "exit=$code log=$LOG_DIR/train_eval.log"
  exit "$code"
fi

for split in train test; do
  mesh_dir="$RUN_DIR/exports/${split}_ar"
  if [ -d "$mesh_dir" ]; then
    log_status "gallery_${split}_ar" "started" "mesh_dir=$mesh_dir"
    python scripts/eval/make_mesh_contact_sheet.py \
      --mesh-dir "$mesh_dir" \
      --report "$RUN_DIR/eval/${split}_autoregressive.json" \
      --output "$LAB_ROOT/${split}_ar_contact_sheet.png" \
      --title "FACE ${RUN_LABEL} ${split} full-face AR" >"$LOG_DIR/gallery_${split}_ar.log" 2>&1 || {
        code=$?
        log_status "gallery_${split}_ar" "failed" "exit=$code log=$LOG_DIR/gallery_${split}_ar.log"
        exit "$code"
      }
    log_status "gallery_${split}_ar" "completed" "output=$LAB_ROOT/${split}_ar_contact_sheet.png"
  fi
done

python - "$LAB_ROOT" "$RUN_DIR" <<'PY' >"$LOG_DIR/summary.log" 2>&1
import json
import sys
from pathlib import Path

lab = Path(sys.argv[1])
run = Path(sys.argv[2])
summary = {
    "lab_root": str(lab),
    "run_dir": str(run),
    "run_summary": None,
    "scale_readiness": None,
    "galleries": sorted(str(path) for path in lab.glob("*_contact_sheet.png")),
}
summary_path = run / "summary.json"
if summary_path.exists():
    summary["run_summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
readiness_path = run / "scale_readiness.json"
if readiness_path.exists():
    summary["scale_readiness"] = json.loads(readiness_path.read_text(encoding="utf-8"))
(lab / "strict110_probe_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
log_status "summary" "completed" "log=$LOG_DIR/summary.log"

tar -czf "$LAB_ROOT.tar.gz" -C "$(dirname "$LAB_ROOT")" "$(basename "$LAB_ROOT")"
log_status "strict110_probe" "completed" "archive=$LAB_ROOT.tar.gz"
