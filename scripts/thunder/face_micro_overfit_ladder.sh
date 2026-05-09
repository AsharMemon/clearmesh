#!/usr/bin/env bash
# Tiny FACE AR-overfit trust ladder.
#
# Run this on a Thunder GPU instance from /home/ubuntu/clearmesh. It is designed
# to answer one question quickly: can a configuration autoregressively overfit a
# single known-good watertight token sample? If not, larger runs are premature.
set -euo pipefail

cd "${REMOTE_REPO:-/home/ubuntu/clearmesh}"

DATASET_DIR="${DATASET_DIR:-}"
SAMPLE_NPZ="${SAMPLE_NPZ:-}"
LAB_ROOT="${LAB_ROOT:-/tmp/clearmesh_face_micro_overfit_$(date -u +%Y%m%d_%H%M%S)}"
STEPS="${STEPS:-3000}"
POINT_SAMPLES="${POINT_SAMPLES:-2048}"
BATCH_SIZE="${BATCH_SIZE:-1}"
HIDDEN_SIZE="${HIDDEN_SIZE:-192}"
ENCODER_HIDDEN_SIZE="${ENCODER_HIDDEN_SIZE:-192}"
ENCODER_LAYERS="${ENCODER_LAYERS:-2}"
DECODER_LAYERS="${DECODER_LAYERS:-3}"
HEADS="${HEADS:-4}"
VECSET_TOKENS="${VECSET_TOKENS:-128}"
LATENT_DIM="${LATENT_DIM:-64}"
PRECISION="${PRECISION:-bf16}"
DEVICE="${DEVICE:-auto}"
STATUS_FILE="$LAB_ROOT/status.jsonl"
LOG_DIR="$LAB_ROOT/logs"
SPLIT_DIR="$LAB_ROOT/split_one"
mkdir -p "$LOG_DIR" "$SPLIT_DIR/train" "$SPLIT_DIR/test"
echo "$LAB_ROOT" > /tmp/clearmesh_latest_face_micro_overfit_run.txt

log_status() {
  local step="$1"
  local status="$2"
  local detail="${3:-}"
  STATUS_FILE="$STATUS_FILE" STEP="$step" STATUS="$status" DETAIL="$detail" python - <<'PY' >&2
import json
import os
import time
from pathlib import Path

row = {
    "time": time.time(),
    "step": os.environ["STEP"],
    "status": os.environ["STATUS"],
    "detail": os.environ.get("DETAIL", ""),
}
path = Path(os.environ["STATUS_FILE"])
path.parent.mkdir(parents=True, exist_ok=True)
with path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(row, sort_keys=True) + "\n")
print(json.dumps(row, sort_keys=True), flush=True)
PY
}

if [ -z "$SAMPLE_NPZ" ]; then
  if [ -z "$DATASET_DIR" ]; then
    DATASET_DIR="$(cat /tmp/clearmesh_latest_face_a100_probe_run.txt 2>/dev/null || true)"
    if [ -n "$DATASET_DIR" ]; then
      DATASET_DIR="$DATASET_DIR/tokens_128_preserve_paper_knobs/pass_only"
    fi
  fi
  if [ -z "$DATASET_DIR" ] || [ ! -d "$DATASET_DIR" ]; then
    echo "Set DATASET_DIR to a FACE NPZ directory or SAMPLE_NPZ to one shard." >&2
    exit 2
  fi
  SAMPLE_NPZ="$(find "$DATASET_DIR" -maxdepth 1 -type f -name '*.npz' | sort | head -n 1)"
fi
if [ -z "$SAMPLE_NPZ" ] || [ ! -f "$SAMPLE_NPZ" ]; then
  echo "sample NPZ not found: $SAMPLE_NPZ" >&2
  exit 2
fi

ln -sf "$SAMPLE_NPZ" "$SPLIT_DIR/train/$(basename "$SAMPLE_NPZ")"
ln -sf "$SAMPLE_NPZ" "$SPLIT_DIR/test/$(basename "$SAMPLE_NPZ")"
printf '{"path": "%s", "split": "train"}\n' "$SPLIT_DIR/train/$(basename "$SAMPLE_NPZ")" > "$SPLIT_DIR/train/manifest.jsonl"
printf '{"path": "%s", "split": "test"}\n' "$SPLIT_DIR/test/$(basename "$SAMPLE_NPZ")" > "$SPLIT_DIR/test/manifest.jsonl"

MODEL_MAX_FACES="${MODEL_MAX_FACES:-$(python - "$SAMPLE_NPZ" <<'PY'
import sys
import numpy as np
with np.load(sys.argv[1], allow_pickle=False) as data:
    print(int(data["paper_tokens"].shape[0]))
PY
)}"

log_status "micro_ladder" "started" "lab_root=$LAB_ROOT sample=$SAMPLE_NPZ model_max_faces=$MODEL_MAX_FACES"

run_config() {
  local name="$1"
  local causal="$2"
  local optimizer="$3"
  local disable_augment="$4"
  local log="$LOG_DIR/$name.log"
  local run_dir="$LAB_ROOT/runs/$name"

  mkdir -p "$run_dir"
  log_status "$name" "started" "causal=$causal optimizer=$optimizer disable_augment=$disable_augment"
  if env \
    DATA_RUN="$LAB_ROOT" \
    SPLIT_DIR="$SPLIT_DIR" \
    TRAIN_DIR="$SPLIT_DIR/train" \
    TEST_DIR="$SPLIT_DIR/test" \
    RUN_NAME="$name" \
    RUN_DIR="$run_dir" \
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
    ENCODER_BACKEND="shape2vecset" \
    CAUSAL_MLP_VARIANT="$causal" \
    DECODE_HEAD="causal" \
    OPTIMIZER="$optimizer" \
    LR="0.0006" \
    WEIGHT_DECAY="0.1" \
    EOS_LOSS_WEIGHT="0.05" \
    DISABLE_AUGMENT="$disable_augment" \
    SEED="311" \
    DEVICE="$DEVICE" \
    PRECISION="$PRECISION" \
    LOG_EVERY="100" \
    SELECTION_EVAL_EVERY="500" \
    SELECTION_EVAL_BATCH_SIZE="1" \
    TEACHER_FORCED_LIMIT="1" \
    AR_LIMIT="1" \
    AR_FACE_LIMIT="0" \
    PAIR_SAMPLES="500" \
    EXPORT_AR_MESHES="1" \
    scripts/thunder/face_paper_train_eval_job.sh >"$log" 2>&1; then
    log_status "$name" "completed" "run_dir=$run_dir log=$log"
  else
    local code=$?
    log_status "$name" "failed" "exit=$code run_dir=$run_dir log=$log"
    return "$code"
  fi
}

run_config "legacy_adamw_noaug" "legacy_concat" "adamw" "1" || true
run_config "paper_chain_muon_noaug" "paper_chain" "muon" "1" || true
run_config "paper_chain_muon_aug" "paper_chain" "muon" "0" || true

python - "$LAB_ROOT" <<'PY'
import json
import sys
from pathlib import Path

lab = Path(sys.argv[1])
runs = {}
for path in sorted((lab / "runs").glob("*/summary.json")):
    runs[path.parent.name] = json.loads(path.read_text(encoding="utf-8"))
summary = {
    "lab_root": str(lab),
    "runs": runs,
}
(lab / "micro_overfit_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps({"lab_root": str(lab), "runs": sorted(runs)}, indent=2, sort_keys=True))
PY

tar -czf "$LAB_ROOT.tar.gz" -C "$(dirname "$LAB_ROOT")" "$(basename "$LAB_ROOT")"
log_status "micro_ladder" "completed" "archive=$LAB_ROOT.tar.gz"
