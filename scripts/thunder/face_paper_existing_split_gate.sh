#!/usr/bin/env bash
# Train/evaluate the paper-faithful FACE lane from an already-built strict split.
# This avoids spending A100 time on Objaverse download, target conversion, and
# tokenization when a corpus has already been prepared on cheaper hardware.
set -euo pipefail

cd "${REMOTE_REPO:-/home/ubuntu/clearmesh}"

DATA_RUN="${DATA_RUN:-}"
if [ -z "$DATA_RUN" ]; then
  DATA_RUN="$(cat /tmp/clearmesh_latest_face_objpp_run.txt 2>/dev/null || true)"
fi
if [ -z "$DATA_RUN" ]; then
  echo "DATA_RUN is required, or /tmp/clearmesh_latest_face_objpp_run.txt must exist." >&2
  exit 2
fi

if [ -z "${SPLIT_DIR:-}" ]; then
  if [ -d "$DATA_RUN/split_pass" ]; then
    SPLIT_DIR="$DATA_RUN/split_pass"
  elif [ -d "$DATA_RUN/split" ]; then
    SPLIT_DIR="$DATA_RUN/split"
  else
    SPLIT_DIR="$DATA_RUN/split_pass"
  fi
fi
TRAIN_DIR="${TRAIN_DIR:-$SPLIT_DIR/train}"
TEST_DIR="${TEST_DIR:-$SPLIT_DIR/test}"

# Fail paper-faithfulness issues before any data-path or GPU work so stale
# compatibility overrides cannot be hidden behind an unrelated setup error.
FACE_EMBEDDING_VARIANT="${FACE_EMBEDDING_VARIANT:-token_concat_project}"
ALLOW_DEPRECATED_FACE_EMBEDDING="${ALLOW_DEPRECATED_FACE_EMBEDDING:-0}"
STRICT_FACE_PAPER_GATE="${STRICT_FACE_PAPER_GATE:-1}"
if [ "$STRICT_FACE_PAPER_GATE" = "1" ]; then
  if [ "$FACE_EMBEDDING_VARIANT" != "token_concat_project" ]; then
    echo "Strict FACE existing-split gate requires FACE_EMBEDDING_VARIANT=token_concat_project; got '$FACE_EMBEDDING_VARIANT'." >&2
    echo "Set STRICT_FACE_PAPER_GATE=0 only for an explicitly named compatibility ablation." >&2
    exit 7
  fi
  if [ "$ALLOW_DEPRECATED_FACE_EMBEDDING" = "1" ]; then
    echo "Strict FACE existing-split gate must not set ALLOW_DEPRECATED_FACE_EMBEDDING=1." >&2
    echo "Set STRICT_FACE_PAPER_GATE=0 only for an explicitly named compatibility ablation." >&2
    exit 7
  fi
fi

if [ ! -d "$TRAIN_DIR" ] || [ ! -d "$TEST_DIR" ]; then
  echo "Expected split directories under SPLIT_DIR=$SPLIT_DIR" >&2
  echo "  missing train? $TRAIN_DIR" >&2
  echo "  missing test?  $TEST_DIR" >&2
  exit 3
fi

LAB_ROOT="${LAB_ROOT:-/tmp/clearmesh_face_paper_existing_split_gate_$(date -u +%Y%m%d_%H%M%S)}"
RUN_LABEL="${RUN_LABEL:-paper_existing_split_128_vec2048_muon_aug}"
RUN_DIR="${RUN_DIR:-$LAB_ROOT/runs/$RUN_LABEL}"
LOG_DIR="$LAB_ROOT/logs"
STATUS_FILE="$LAB_ROOT/status.jsonl"
mkdir -p "$LOG_DIR" "$RUN_DIR"
echo "$LAB_ROOT" > /tmp/clearmesh_latest_face_paper_existing_split_gate_run.txt

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

# Paper training knobs. Defaults match the bounded paper lane; override only by
# explicit environment for bigger rungs.
STEPS="${STEPS:-30000}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_BINS="${NUM_BINS:-128}"
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
LOG_EVERY="${LOG_EVERY:-50}"
SELECTION_EVAL_EVERY="${SELECTION_EVAL_EVERY:-1000}"
SELECTION_EVAL_BATCH_SIZE="${SELECTION_EVAL_BATCH_SIZE:-1}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-2000}"
SAVE_CURRENT_CHECKPOINT="${SAVE_CURRENT_CHECKPOINT:-0}"
PREFETCH_BATCHES="${PREFETCH_BATCHES:-1}"
CACHE_FPS_INDICES="${CACHE_FPS_INDICES:-0}"
TRAIN_LIMIT="${TRAIN_LIMIT:-0}"
TEACHER_FORCED_LIMIT="${TEACHER_FORCED_LIMIT:-0}"
AR_LIMIT="${AR_LIMIT:-20}"
AR_FACE_LIMIT="${AR_FACE_LIMIT:-0}"
PREDICTED_LIMIT="${PREDICTED_LIMIT:-5}"
PREDICTED_FACE_LIMIT="${PREDICTED_FACE_LIMIT:-0}"
PAIR_SAMPLES="${PAIR_SAMPLES:-500}"
MIN_SCALE_DATASET_SAMPLES="${MIN_SCALE_DATASET_SAMPLES:-256}"
FAIL_ON_SCALE_NOT_READY="${FAIL_ON_SCALE_NOT_READY:-0}"
RELAX_PAPER_KNOBS="${RELAX_PAPER_KNOBS:-0}"
SEED="${SEED:-303}"
ARCHIVE_PATH="${ARCHIVE_PATH:-$LAB_ROOT.tar.gz}"
OPTIMIZER="${OPTIMIZER:-muon}"
LR="${LR:-0.0006}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
EOS_LOSS_WEIGHT="${EOS_LOSS_WEIGHT:-0.05}"
DISABLE_AUGMENT="${DISABLE_AUGMENT:-0}"
AUGMENT_ROTATION="${AUGMENT_ROTATION:-so3}"
AUGMENT_SCALE_MIN="${AUGMENT_SCALE_MIN:-0.75}"
AUGMENT_SCALE_MAX="${AUGMENT_SCALE_MAX:-1.25}"
AUGMENT_FLIP_PROB="${AUGMENT_FLIP_PROB:-0.5}"
AUGMENT_DIAGNOSTICS="${AUGMENT_DIAGNOSTICS:-0}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"
CAUSAL_MLP_VARIANT="${CAUSAL_MLP_VARIANT:-legacy_concat}"
FACE_EMBEDDING_VARIANT="${FACE_EMBEDDING_VARIANT:-token_concat_project}"
ALLOW_DEPRECATED_FACE_EMBEDDING="${ALLOW_DEPRECATED_FACE_EMBEDDING:-0}"
DECODE_HEAD="${DECODE_HEAD:-causal}"
STRICT_FACE_PAPER_GATE="${STRICT_FACE_PAPER_GATE:-1}"

if [ "$STRICT_FACE_PAPER_GATE" = "1" ]; then
  if [ "$FACE_EMBEDDING_VARIANT" != "token_concat_project" ]; then
    echo "Strict FACE existing-split gate requires FACE_EMBEDDING_VARIANT=token_concat_project; got '$FACE_EMBEDDING_VARIANT'." >&2
    echo "Set STRICT_FACE_PAPER_GATE=0 only for an explicitly named compatibility ablation." >&2
    exit 7
  fi
  if [ "$ALLOW_DEPRECATED_FACE_EMBEDDING" = "1" ]; then
    echo "Strict FACE existing-split gate must not set ALLOW_DEPRECATED_FACE_EMBEDDING=1." >&2
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
      && log_status existing_split_gate archived_after_failure "exit=$code archive=$ARCHIVE_PATH" \
      || true
  fi
}
trap archive_on_failure EXIT

log_status existing_split_gate started "lab_root=$LAB_ROOT data_run=$DATA_RUN split_dir=$SPLIT_DIR steps=$STEPS"

run_step env_probe python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0), "bf16", torch.cuda.is_bf16_supported())
print("native_muon", hasattr(torch.optim, "Muon"))
PY

run_step dataset_probe python - "$DATA_RUN" "$SPLIT_DIR" "$TRAIN_DIR" "$TEST_DIR" <<'PY'
import json
import sys
from pathlib import Path
data_run, split_dir, train_dir, test_dir = map(Path, sys.argv[1:])
def count_manifest(path: Path) -> int:
    manifest = path / "manifest.jsonl"
    if not manifest.exists():
        return 0
    return sum(1 for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip())
summary = {
    "data_run": str(data_run),
    "split_dir": str(split_dir),
    "train_dir": str(train_dir),
    "test_dir": str(test_dir),
    "train_count": count_manifest(train_dir),
    "test_count": count_manifest(test_dir),
    "has_train_manifest": (train_dir / "manifest.jsonl").exists(),
    "has_test_manifest": (test_dir / "manifest.jsonl").exists(),
}
print(json.dumps(summary, indent=2, sort_keys=True))
if summary["train_count"] <= 0 or summary["test_count"] <= 0:
    raise SystemExit("empty train/test split")
PY

run_step train_eval env \
  DATA_RUN="$DATA_RUN" \
  SPLIT_DIR="$SPLIT_DIR" \
  TRAIN_DIR="$TRAIN_DIR" \
  TEST_DIR="$TEST_DIR" \
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
  OPTIMIZER="$OPTIMIZER" \
  LR="$LR" \
  WEIGHT_DECAY="$WEIGHT_DECAY" \
  EOS_LOSS_WEIGHT="$EOS_LOSS_WEIGHT" \
  DISABLE_AUGMENT="$DISABLE_AUGMENT" \
  AUGMENT_ROTATION="$AUGMENT_ROTATION" \
  AUGMENT_SCALE_MIN="$AUGMENT_SCALE_MIN" \
  AUGMENT_SCALE_MAX="$AUGMENT_SCALE_MAX" \
  AUGMENT_FLIP_PROB="$AUGMENT_FLIP_PROB" \
  AUGMENT_DIAGNOSTICS="$AUGMENT_DIAGNOSTICS" \
  INIT_CHECKPOINT="$INIT_CHECKPOINT" \
  SEED="$SEED" \
  ALLOW_DEPRECATED_FACE_EMBEDDING="$ALLOW_DEPRECATED_FACE_EMBEDDING" \
  STRICT_FACE_PAPER_GATE="$STRICT_FACE_PAPER_GATE" \
  LOG_EVERY="$LOG_EVERY" \
  SELECTION_EVAL_EVERY="$SELECTION_EVAL_EVERY" \
  SELECTION_EVAL_BATCH_SIZE="$SELECTION_EVAL_BATCH_SIZE" \
  CHECKPOINT_EVERY="$CHECKPOINT_EVERY" \
  SAVE_CURRENT_CHECKPOINT="$SAVE_CURRENT_CHECKPOINT" \
  PREFETCH_BATCHES="$PREFETCH_BATCHES" \
  CACHE_FPS_INDICES="$CACHE_FPS_INDICES" \
  TRAIN_LIMIT="$TRAIN_LIMIT" \
  TEACHER_FORCED_LIMIT="$TEACHER_FORCED_LIMIT" \
  AR_LIMIT="$AR_LIMIT" \
  AR_FACE_LIMIT="$AR_FACE_LIMIT" \
  TEACHER_PREFIX_LIMIT="${TEACHER_PREFIX_LIMIT:-16}" \
  TEACHER_PREFIX_FACE_COUNTS="${TEACHER_PREFIX_FACE_COUNTS:-1 4 16}" \
  PREDICTED_LIMIT="$PREDICTED_LIMIT" \
  PREDICTED_FACE_LIMIT="$PREDICTED_FACE_LIMIT" \
  PAIR_SAMPLES="$PAIR_SAMPLES" \
  EXPORT_AR_MESHES=1 \
  RUN_SCALE_READINESS=1 \
  FAIL_ON_SCALE_NOT_READY="$FAIL_ON_SCALE_NOT_READY" \
  MIN_SCALE_DATASET_SAMPLES="$MIN_SCALE_DATASET_SAMPLES" \
  RELAX_PAPER_KNOBS="$RELAX_PAPER_KNOBS" \
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

run_step summary python - "$LAB_ROOT" "$DATA_RUN" "$SPLIT_DIR" "$RUN_DIR" <<'PY'
import json
import sys
from pathlib import Path
lab = Path(sys.argv[1])
data_run = Path(sys.argv[2])
split = Path(sys.argv[3])
run = Path(sys.argv[4])

def read(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

def count(path):
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())

summary = {
    "lab_root": str(lab),
    "data_run": str(data_run),
    "split_dir": str(split),
    "run_dir": str(run),
    "dataset": {
        "train": count(split / "train" / "manifest.jsonl"),
        "test": count(split / "test" / "manifest.jsonl"),
    },
    "run_summary": read(run / "summary.json"),
    "scale_readiness": read(run / "scale_readiness.json"),
    "galleries": sorted(str(path) for path in lab.glob("*_contact_sheet.png")),
}
(lab / "paper_existing_split_gate_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps({"lab_root": str(lab), "scale_ready": (summary.get("scale_readiness") or {}).get("scale_ready")}, indent=2, sort_keys=True))
PY

tar -czf "$ARCHIVE_PATH" -C "$(dirname "$LAB_ROOT")" "$(basename "$LAB_ROOT")"
log_status existing_split_gate completed "archive=$ARCHIVE_PATH"
trap - EXIT
