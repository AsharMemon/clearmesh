#!/usr/bin/env bash
# Run the FACE paper-faithfulness ladder on a Thunder instance.
#
# This script intentionally keeps status logging off stdout inside helper
# functions so command substitution can safely capture only path values.
set -euo pipefail

cd "${REMOTE_REPO:-/home/ubuntu/clearmesh}"

DATA_RUN="${DATA_RUN:-$(cat /tmp/clearmesh_latest_face_objpp_run.txt 2>/dev/null || true)}"
if [ -z "$DATA_RUN" ]; then
  DATA_RUN="/tmp/clearmesh_face_objpp200_512_20260504_020301"
fi
if [ ! -d "$DATA_RUN" ]; then
  echo "DATA_RUN does not exist: $DATA_RUN" >&2
  exit 1
fi

LAB_ROOT="${LAB_ROOT:-/tmp/clearmesh_face_ladder_$(date -u +%Y%m%d_%H%M%S)}"
STATUS_FILE="$LAB_ROOT/status.jsonl"
LOG_DIR="$LAB_ROOT/logs"
GALLERY_DIR="$LAB_ROOT/galleries"
RUNS_DIR="$LAB_ROOT/runs"
mkdir -p "$LOG_DIR" "$GALLERY_DIR" "$RUNS_DIR"
echo "$LAB_ROOT" > /tmp/clearmesh_latest_face_ladder_run.txt

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
  log_status "$step" "failed" "exit=$code; log=$log"
  return "$code"
}

build_tokens() {
  local bins="$1"
  local order="$2"
  local out="$LAB_ROOT/tokens_${bins}_${order}"
  if [ ! -f "$out/manifest.jsonl" ]; then
    run_cmd "build_tokens_${bins}_${order}" \
      python scripts/research/build_face_token_dataset.py \
        --mesh-dir "$DATA_RUN/strict_targets/meshes" \
        --output-dir "$out" \
        --num-bins "$bins" \
        --max-faces 512 \
        --point-samples 2048 \
        --paper-within-face-order "$order" \
        --seed 7
  fi
  printf '%s\n' "$out"
}

gate_tokens() {
  local dataset="$1"
  local bins="$2"
  local order="$3"
  local report="$dataset/strict_gate.json"
  run_cmd "gate_tokens_${bins}_${order}" \
    python scripts/research/check_face_dataset_targets.py \
      --manifest "$dataset/manifest.jsonl" \
      --output "$report" \
      --profile strict \
      --token-family paper
  printf '%s\n' "$report"
}

filter_tokens() {
  local dataset="$1"
  local gate="$2"
  local bins="$3"
  local order="$4"
  local out="$dataset/pass_only"
  run_cmd "filter_tokens_${bins}_${order}" \
    python scripts/research/filter_face_dataset_by_gate.py \
      --dataset-dir "$dataset" \
      --gate-report "$gate" \
      --output-dir "$out" \
      --copy-mode symlink
  printf '%s\n' "$out"
}

split_tokens() {
  local dataset="$1"
  local bins="$2"
  local order="$3"
  local out="$dataset/split_pass"
  run_cmd "split_tokens_${bins}_${order}" \
    python - "$dataset" "$out" <<'PY'
import json
import os
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
rng = random.Random(17)
rng.shuffle(paths)
test_count = max(1, int(round(len(paths) * 0.2))) if len(paths) > 1 else 0
test_set = set(paths[:test_count])
manifest_rows = []
for split_name, split_dir, selected in (
    ("test", test, [p for p in paths if p in test_set]),
    ("train", train, [p for p in paths if p not in test_set]),
):
    rows = []
    for source in selected:
        dest = split_dir / source.name
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        try:
            dest.symlink_to(source)
        except OSError:
            shutil.copy2(source, dest)
        rows.append({"path": str(dest), "source": str(source), "split": split_name})
    with (split_dir / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    manifest_rows.extend(rows)
summary = {
    "dataset": str(dataset),
    "split_dir": str(out),
    "train": len([row for row in manifest_rows if row["split"] == "train"]),
    "test": len([row for row in manifest_rows if row["split"] == "test"]),
}
(out / "split_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
  printf '%s\n' "$out"
}

prepare_tokens() {
  local bins="$1"
  local order="$2"
  local tokens gate passed split
  tokens="$(build_tokens "$bins" "$order")"
  gate="$(gate_tokens "$tokens" "$bins" "$order")"
  passed="$(filter_tokens "$tokens" "$gate" "$bins" "$order")"
  split="$(split_tokens "$passed" "$bins" "$order")"
  printf '%s\n' "$split"
}

make_target_gallery() {
  if [ -d "$DATA_RUN/strict_targets/meshes" ]; then
    run_cmd "target_gallery" \
      python scripts/eval/make_mesh_gallery.py \
        --mesh-dir "$DATA_RUN/strict_targets/meshes" \
        --output "$GALLERY_DIR/strict_target_gallery.png" \
        --title "FACE strict target meshes" \
        --limit 24 \
        --columns 4 \
        --cell-size 280
  else
    log_status "target_gallery" "skipped" "missing $DATA_RUN/strict_targets/meshes"
  fi
}

make_contact_sheet() {
  local name="$1"
  local split="$2"
  local run_dir="$RUNS_DIR/$name"
  local export_dir="$run_dir/exports/${split}_ar"
  local report="$run_dir/eval/${split}_autoregressive.json"
  if [ -d "$export_dir" ]; then
    run_cmd "gallery_${name}_${split}_ar" \
      python scripts/eval/make_mesh_contact_sheet.py \
        --mesh-dir "$export_dir" \
        --report "$report" \
        --output "$GALLERY_DIR/${name}_${split}_ar.png" \
        --title "$name $split autoregressive"
  else
    log_status "gallery_${name}_${split}_ar" "skipped" "missing $export_dir"
  fi
}

train_eval() {
  local name="$1"
  local split="$2"
  local bins="$3"
  local steps="$4"
  local decode_head="$5"
  local order="$6"
  local log="$LOG_DIR/train_${name}.log"
  local run_dir="$RUNS_DIR/$name"
  local train_dir="$split/train"
  local test_dir="$split/test"

  mkdir -p "$run_dir"
  log_status "train_${name}" "started" "bins=$bins steps=$steps head=$decode_head order=$order split=$split"
  if env \
    DATA_RUN="$DATA_RUN" \
    SPLIT_DIR="$split" \
    TRAIN_DIR="$train_dir" \
    TEST_DIR="$test_dir" \
    RUN_NAME="$name" \
    RUN_DIR="$run_dir" \
    STEPS="$steps" \
    BATCH_SIZE="${BATCH_SIZE:-2}" \
    POINT_SAMPLES="${POINT_SAMPLES:-2048}" \
    MODEL_MAX_FACES="${MODEL_MAX_FACES:-512}" \
    HIDDEN_SIZE="${HIDDEN_SIZE:-256}" \
    ENCODER_HIDDEN_SIZE="${ENCODER_HIDDEN_SIZE:-256}" \
    ENCODER_LAYERS="${ENCODER_LAYERS:-3}" \
    DECODER_LAYERS="${DECODER_LAYERS:-6}" \
    HEADS="${HEADS:-8}" \
    VECSET_TOKENS="${VECSET_TOKENS:-256}" \
    LATENT_DIM="${LATENT_DIM:-64}" \
    ENCODER_BACKEND="${ENCODER_BACKEND:-shape2vecset}" \
    CAUSAL_MLP_VARIANT="${CAUSAL_MLP_VARIANT:-legacy_concat}" \
    DECODE_HEAD="$decode_head" \
    OPTIMIZER="${OPTIMIZER:-adamw}" \
    LR="${LR:-0.0006}" \
    WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}" \
    EOS_LOSS_WEIGHT="${EOS_LOSS_WEIGHT:-0.05}" \
    DISABLE_AUGMENT="${DISABLE_AUGMENT:-1}" \
    SEED="${SEED:-0}" \
    LOG_EVERY="${LOG_EVERY:-250}" \
    SELECTION_EVAL_EVERY="${SELECTION_EVAL_EVERY:-1000}" \
    SELECTION_EVAL_BATCH_SIZE="${SELECTION_EVAL_BATCH_SIZE:-1}" \
    EVAL_LIMIT="${EVAL_LIMIT:-0}" \
    TEACHER_FORCED_LIMIT="${TEACHER_FORCED_LIMIT:-0}" \
    AR_LIMIT="${AR_LIMIT:-5}" \
    AR_FACE_LIMIT="${AR_FACE_LIMIT:-0}" \
    PAIR_SAMPLES="${PAIR_SAMPLES:-500}" \
    EXPORT_AR_MESHES="${EXPORT_AR_MESHES:-1}" \
    DEVICE="${DEVICE:-auto}" \
    scripts/thunder/face_paper_train_eval_job.sh >"$log" 2>&1; then
    log_status "train_${name}" "completed" "run_dir=$run_dir log=$log"
    make_contact_sheet "$name" train || true
    make_contact_sheet "$name" test || true
    return 0
  fi
  local code
  code=$?
  log_status "train_${name}" "failed" "exit=$code; run_dir=$run_dir log=$log"
  return "$code"
}

write_summary() {
  python - "$LAB_ROOT" <<'PY'
import json
import sys
from pathlib import Path

lab = Path(sys.argv[1])

def read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"read_error": f"{type(exc).__name__}: {exc}", "path": str(path)}

runs = {}
for summary_path in sorted((lab / "runs").glob("*/summary.json")):
    runs[summary_path.parent.name] = read_json(summary_path)
gates = {}
for gate_path in sorted(lab.glob("tokens_*/strict_gate.json")):
    gates[gate_path.parent.name] = read_json(gate_path)
split_summaries = {}
for split_path in sorted(lab.glob("tokens_*/pass_only/split_pass/split_summary.json")):
    split_summaries[split_path.parents[1].name] = read_json(split_path)
status_rows = []
status_file = lab / "status.jsonl"
if status_file.exists():
    for line in status_file.read_text(encoding="utf-8").splitlines():
        if line.strip():
            status_rows.append(json.loads(line))
summary = {
    "lab_root": str(lab),
    "status_file": str(status_file),
    "galleries": sorted(str(path) for path in (lab / "galleries").glob("*.png")),
    "gates": gates,
    "splits": split_summaries,
    "runs": runs,
    "status_tail": status_rows[-30:],
}
(lab / "ladder_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps({"lab_root": str(lab), "runs": sorted(runs), "galleries": len(summary["galleries"])}, indent=2, sort_keys=True))
PY
}

main() {
  log_status "supervisor" "started" "lab_root=$LAB_ROOT data_run=$DATA_RUN"
  make_target_gallery || true

  local split128 split256 split512 split256_rotate split256_sort
  split128="$(prepare_tokens 128 preserve)"
  split256="$(prepare_tokens 256 preserve)"
  split512="$(prepare_tokens 512 preserve)"
  split256_rotate="$(prepare_tokens 256 rotate_min_zyx)"
  split256_sort="$(prepare_tokens 256 sort_zyx)"

  train_eval "bin128_legacy_causal_3000" "$split128" 128 3000 causal preserve || true
  train_eval "bin256_legacy_causal_3000" "$split256" 256 3000 causal preserve || true
  train_eval "bin512_legacy_causal_3000" "$split512" 512 3000 causal preserve || true
  train_eval "bin128_parallel_3000" "$split128" 128 3000 parallel preserve || true
  train_eval "bin256_rotate_min_1500" "$split256_rotate" 256 1500 causal rotate_min_zyx || true
  train_eval "bin256_sort_zyx_1500" "$split256_sort" 256 1500 causal sort_zyx || true
  train_eval "bin512_legacy_causal_30000" "$split512" 512 "${LONG_STEPS:-30000}" causal preserve || true

  run_cmd "ladder_summary" write_summary
  log_status "supervisor" "completed" "lab_root=$LAB_ROOT"
}

main "$@"
