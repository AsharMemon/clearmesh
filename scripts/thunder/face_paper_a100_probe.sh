#!/usr/bin/env bash
# Launch a FACE paper-knob probe on an A100-class Thunder instance.
#
# This is intentionally narrower than the A6000 ladder: it restores the paper's
# most important implementation details for the next validation gate:
#   - 128 coordinate bins
#   - 8192 sampled points + normals
#   - 2048 VecSet tokens
#   - latent bottleneck dimension 64
#   - Muon optimizer, lr=6e-4, wd=0.1
#   - online rotation/flip/per-axis scaling augmentation enabled
#
# The current strict-target pilot was capped at 512 faces, so MODEL_MAX_FACES
# defaults to 512. For a true paper-scale reproduction, rebuild the corpus with
# TOKEN_MAX_FACES=4000 and set MODEL_MAX_FACES=4000.
set -euo pipefail

cd "${REMOTE_REPO:-/home/ubuntu/clearmesh}"

DATA_RUN="${DATA_RUN:-$(cat /tmp/clearmesh_latest_face_objpp_run.txt 2>/dev/null || true)}"
if [ -z "$DATA_RUN" ]; then
  DATA_RUN="/tmp/clearmesh_face_objpp200_512_20260504_020301"
fi
if [ ! -d "$DATA_RUN/strict_targets/meshes" ]; then
  echo "strict target meshes not found under DATA_RUN=$DATA_RUN" >&2
  exit 1
fi

LAB_ROOT="${LAB_ROOT:-/tmp/clearmesh_face_a100_probe_$(date -u +%Y%m%d_%H%M%S)}"
LOG_DIR="$LAB_ROOT/logs"
TOKENS_DIR="$LAB_ROOT/tokens_128_preserve_paper_knobs"
PASSED_DIR="$TOKENS_DIR/pass_only"
SPLIT_DIR="$PASSED_DIR/split_pass"
RUN_DIR="$LAB_ROOT/runs/paper_knob_128_vec2048_muon_aug"
STATUS_FILE="$LAB_ROOT/status.jsonl"
mkdir -p "$LOG_DIR" "$RUN_DIR"
echo "$LAB_ROOT" > /tmp/clearmesh_latest_face_a100_probe_run.txt

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

run_cmd() {
  local step="$1"
  local log="$LOG_DIR/${step}.log"
  local code
  shift
  log_status "$step" "started" ""
  if "$@" >"$log" 2>&1; then
    log_status "$step" "completed" "log=$log"
    return 0
  fi
  code=$?
  log_status "$step" "failed" "exit=$code log=$log"
  return "$code"
}

split_tokens() {
  run_cmd "split_tokens" python - "$PASSED_DIR" "$SPLIT_DIR" <<'PY'
import json
import random
import shutil
import sys
from pathlib import Path

dataset = Path(sys.argv[1])
out = Path(sys.argv[2])
train = out / "train"
test = out / "test"
for split_dir in (train, test):
    split_dir.mkdir(parents=True, exist_ok=True)
    for old in split_dir.glob("*"):
        if old.is_symlink() or old.is_file():
            old.unlink()

paths = sorted(dataset.glob("*.npz"))
if not paths:
    raise SystemExit(f"no NPZ shards found in {dataset}")
rng = random.Random(23)
rng.shuffle(paths)
test_count = max(1, int(round(len(paths) * 0.2))) if len(paths) > 1 else 0
test_paths = set(paths[:test_count])
summary = {"train": 0, "test": 0, "dataset": str(dataset), "split_dir": str(out)}
for split_name, split_dir, selected in (
    ("test", test, [p for p in paths if p in test_paths]),
    ("train", train, [p for p in paths if p not in test_paths]),
):
    with (split_dir / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for source in selected:
            dest = split_dir / source.name
            if dest.exists() or dest.is_symlink():
                dest.unlink()
            try:
                dest.symlink_to(source)
            except OSError:
                shutil.copy2(source, dest)
            handle.write(json.dumps({"path": str(dest), "source": str(source), "split": split_name}, sort_keys=True) + "\n")
            summary[split_name] += 1
(out / "split_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
}

TOKEN_MAX_FACES="${TOKEN_MAX_FACES:-512}"
MODEL_MAX_FACES="${MODEL_MAX_FACES:-512}"
POINT_SAMPLES="${POINT_SAMPLES:-8192}"
VECSET_TOKENS="${VECSET_TOKENS:-2048}"
LATENT_DIM="${LATENT_DIM:-64}"
CAUSAL_MLP_VARIANT="${CAUSAL_MLP_VARIANT:-legacy_concat}"
HIDDEN_SIZE="${HIDDEN_SIZE:-384}"
ENCODER_HIDDEN_SIZE="${ENCODER_HIDDEN_SIZE:-384}"
ENCODER_LAYERS="${ENCODER_LAYERS:-4}"
DECODER_LAYERS="${DECODER_LAYERS:-8}"
HEADS="${HEADS:-8}"
BATCH_SIZE="${BATCH_SIZE:-1}"
STEPS="${STEPS:-5000}"
LOG_EVERY="${LOG_EVERY:-100}"
SELECTION_EVAL_EVERY="${SELECTION_EVAL_EVERY:-1000}"
CAPACITY_STEPS="${CAPACITY_STEPS:-3}"
PRECISION="${PRECISION:-bf16}"

log_status "probe" "started" "lab_root=$LAB_ROOT data_run=$DATA_RUN token_max_faces=$TOKEN_MAX_FACES model_max_faces=$MODEL_MAX_FACES"

run_cmd "env_probe" python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
print("native_muon", hasattr(torch.optim, "Muon"))
PY

if [ ! -f "$TOKENS_DIR/manifest.jsonl" ]; then
  run_cmd "build_tokens_128_paper" \
    python scripts/research/build_face_token_dataset.py \
      --mesh-dir "$DATA_RUN/strict_targets/meshes" \
      --output-dir "$TOKENS_DIR" \
      --num-bins 128 \
      --max-faces "$TOKEN_MAX_FACES" \
      --point-samples "$POINT_SAMPLES" \
      --paper-within-face-order preserve \
      --seed 11
fi

run_cmd "gate_tokens_strict" \
  python scripts/research/check_face_dataset_targets.py \
    --manifest "$TOKENS_DIR/manifest.jsonl" \
    --output "$TOKENS_DIR/strict_gate.json" \
    --profile strict \
    --token-family paper

run_cmd "filter_tokens_pass_only" \
  python scripts/research/filter_face_dataset_by_gate.py \
    --dataset-dir "$TOKENS_DIR" \
    --gate-report "$TOKENS_DIR/strict_gate.json" \
    --output-dir "$PASSED_DIR" \
    --copy-mode symlink

split_tokens

run_cmd "capacity_train_${CAPACITY_STEPS}_steps" \
  python scripts/research/train_face_paper_faithful.py \
    --dataset-dir "$SPLIT_DIR/train" \
    --output "$LAB_ROOT/capacity_probe.pt" \
    --steps "$CAPACITY_STEPS" \
    --batch-size "$BATCH_SIZE" \
    --point-samples "$POINT_SAMPLES" \
    --model-max-faces "$MODEL_MAX_FACES" \
    --hidden-size "$HIDDEN_SIZE" \
    --encoder-hidden-size "$ENCODER_HIDDEN_SIZE" \
    --encoder-layers "$ENCODER_LAYERS" \
    --decoder-layers "$DECODER_LAYERS" \
    --heads "$HEADS" \
    --vecset-tokens "$VECSET_TOKENS" \
    --latent-dim "$LATENT_DIM" \
    --encoder-backend shape2vecset \
    --causal-mlp-variant "$CAUSAL_MLP_VARIANT" \
    --decode-head causal \
    --optimizer muon \
    --lr 0.0006 \
    --weight-decay 0.1 \
    --eos-loss-weight 0.05 \
    --augment-rotation so3 \
    --augment-scale-min 0.75 \
    --augment-scale-max 1.25 \
    --augment-flip-prob 0.5 \
    --selection-eval-every 0 \
    --log-every 1 \
    --seed 101 \
    --precision "$PRECISION"

log_status "train_eval" "started" "run_dir=$RUN_DIR steps=$STEPS"
if env \
  DATA_RUN="$DATA_RUN" \
  SPLIT_DIR="$SPLIT_DIR" \
  TRAIN_DIR="$SPLIT_DIR/train" \
  TEST_DIR="$SPLIT_DIR/test" \
  RUN_NAME="paper_knob_128_vec2048_muon_aug" \
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
  ENCODER_BACKEND="shape2vecset" \
  CAUSAL_MLP_VARIANT="$CAUSAL_MLP_VARIANT" \
  DECODE_HEAD="causal" \
  OPTIMIZER="muon" \
  LR="0.0006" \
  WEIGHT_DECAY="0.1" \
  EOS_LOSS_WEIGHT="0.05" \
  DISABLE_AUGMENT="0" \
  AUGMENT_ROTATION="so3" \
  AUGMENT_SCALE_MIN="0.75" \
  AUGMENT_SCALE_MAX="1.25" \
  AUGMENT_FLIP_PROB="0.5" \
  SEED="101" \
  LOG_EVERY="$LOG_EVERY" \
  SELECTION_EVAL_EVERY="$SELECTION_EVAL_EVERY" \
  SELECTION_EVAL_BATCH_SIZE="1" \
  EVAL_LIMIT="0" \
  TEACHER_FORCED_LIMIT="0" \
  AR_LIMIT="${AR_LIMIT:-5}" \
  AR_FACE_LIMIT="${AR_FACE_LIMIT:-0}" \
  PAIR_SAMPLES="${PAIR_SAMPLES:-500}" \
  EXPORT_AR_MESHES="1" \
  scripts/thunder/face_paper_train_eval_job.sh >"$LOG_DIR/train_eval.log" 2>&1; then
  log_status "train_eval" "completed" "run_dir=$RUN_DIR"
else
  code=$?
  log_status "train_eval" "failed" "exit=$code log=$LOG_DIR/train_eval.log"
  exit "$code"
fi

run_cmd "gallery_train_ar" \
  python scripts/eval/make_mesh_contact_sheet.py \
    --mesh-dir "$RUN_DIR/exports/train_ar" \
    --report "$RUN_DIR/eval/train_autoregressive.json" \
    --output "$LAB_ROOT/train_ar_contact_sheet.png" \
    --title "A100 FACE paper-knob train AR"

run_cmd "gallery_test_ar" \
  python scripts/eval/make_mesh_contact_sheet.py \
    --mesh-dir "$RUN_DIR/exports/test_ar" \
    --report "$RUN_DIR/eval/test_autoregressive.json" \
    --output "$LAB_ROOT/test_ar_contact_sheet.png" \
    --title "A100 FACE paper-knob test AR"

run_cmd "summary" python - "$LAB_ROOT" <<'PY'
import json
import sys
from pathlib import Path

lab = Path(sys.argv[1])
summary = {
    "lab_root": str(lab),
    "status_file": str(lab / "status.jsonl"),
    "train_summary": None,
    "scale_readiness": None,
    "gate": None,
    "split": None,
    "galleries": sorted(str(path) for path in lab.glob("*_contact_sheet.png")),
}
for key, path in (
    ("train_summary", lab / "runs" / "paper_knob_128_vec2048_muon_aug" / "summary.json"),
    ("scale_readiness", lab / "runs" / "paper_knob_128_vec2048_muon_aug" / "scale_readiness.json"),
    ("gate", lab / "tokens_128_preserve_paper_knobs" / "strict_gate.json"),
    ("split", lab / "tokens_128_preserve_paper_knobs" / "pass_only" / "split_pass" / "split_summary.json"),
):
    if path.exists():
        summary[key] = json.loads(path.read_text(encoding="utf-8"))
(lab / "a100_probe_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps({"lab_root": str(lab), "galleries": len(summary["galleries"])}, indent=2, sort_keys=True))
PY

tar -czf "$LAB_ROOT.tar.gz" -C "$(dirname "$LAB_ROOT")" "$(basename "$LAB_ROOT")"
log_status "probe" "completed" "archive=$LAB_ROOT.tar.gz"
