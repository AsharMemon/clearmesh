#!/usr/bin/env bash
set -euo pipefail

INSTANCE_ID="${1:-${THUNDER_INSTANCE_ID:-0}}"
REMOTE_DIR="${REMOTE_DIR:-/home/ubuntu/clearmesh}"
REMOTE_VENV="${REMOTE_VENV:-/home/ubuntu/clearmesh-venv}"
TNR_BIN="${TNR_BIN:-/Users/Ashar/.tnr/bin/tnr}"
RUN_DIR="${RUN_DIR:-/tmp/clearmesh_face_level_conditioned_holdout}"
TRAIN_COUNT="${TRAIN_COUNT:-48}"
TEST_COUNT="${TEST_COUNT:-24}"
TRAIN_SEED="${TRAIN_SEED:-0}"
TEST_SEED="${TEST_SEED:-999}"
SYNTHETIC_KIND="${SYNTHETIC_KIND:-mixed}"
MAX_FACES="${MAX_FACES:-256}"
POINT_SAMPLES="${POINT_SAMPLES:-512}"
TRAIN_POINT_SAMPLES="${TRAIN_POINT_SAMPLES:-256}"
STEPS="${STEPS:-3000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
HIDDEN_SIZE="${HIDDEN_SIZE:-192}"
LAYERS="${LAYERS:-3}"
HEADS="${HEADS:-6}"
CONDITION_TOKENS="${CONDITION_TOKENS:-8}"
LR="${LR:-5e-4}"
REUSE_VERTEX_LOSS_WEIGHT="${REUSE_VERTEX_LOSS_WEIGHT:-0.0}"
EDGE_CLOSURE_LOSS_WEIGHT="${EDGE_CLOSURE_LOSS_WEIGHT:-0.0}"
TOPOLOGY_AUX_LOSS_WEIGHT="${TOPOLOGY_AUX_LOSS_WEIGHT:-0.0}"
EVAL_LIMIT="${EVAL_LIMIT:-24}"
PAIR_SAMPLES="${PAIR_SAMPLES:-1000}"
CLEANUP_FILL_HOLES="${CLEANUP_FILL_HOLES:-true}"
CLEANUP_MIN_COMPONENT_FACES="${CLEANUP_MIN_COMPONENT_FACES:-1}"
TOKEN_REPAIR_MODE="${TOKEN_REPAIR_MODE:-manifold}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-}"

if [ -z "${THUNDER_TOKEN:-}" ]; then
  echo "THUNDER_TOKEN is not set. Export it before running Thunder commands." >&2
  exit 1
fi

cat <<EOF | "$TNR_BIN" connect "$INSTANCE_ID"
set -euo pipefail
cd "$REMOTE_DIR"
. "$REMOTE_VENV/bin/activate"
python - <<'PY' || python -m pip install torch==2.8.0 --index-url "$PYTORCH_INDEX_URL"
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
PY
rm -rf "$RUN_DIR"
mkdir -p "$RUN_DIR"
python scripts/research/build_face_token_dataset.py \
  --output-dir "$RUN_DIR/train_dataset" \
  --synthetic-count "$TRAIN_COUNT" \
  --synthetic-kind "$SYNTHETIC_KIND" \
  --max-faces "$MAX_FACES" \
  --point-samples "$POINT_SAMPLES" \
  --seed "$TRAIN_SEED"
python scripts/research/build_face_token_dataset.py \
  --output-dir "$RUN_DIR/test_dataset" \
  --synthetic-count "$TEST_COUNT" \
  --synthetic-kind "$SYNTHETIC_KIND" \
  --max-faces "$MAX_FACES" \
  --point-samples "$POINT_SAMPLES" \
  --seed "$TEST_SEED"
python scripts/research/train_face_level_conditioned_tiny.py \
  --dataset-dir "$RUN_DIR/train_dataset" \
  --steps "$STEPS" \
  --batch-size "$BATCH_SIZE" \
  --point-samples "$TRAIN_POINT_SAMPLES" \
  --hidden-size "$HIDDEN_SIZE" \
  --layers "$LAYERS" \
  --heads "$HEADS" \
  --condition-tokens "$CONDITION_TOKENS" \
  --lr "$LR" \
  --reuse-vertex-loss-weight "$REUSE_VERTEX_LOSS_WEIGHT" \
  --edge-closure-loss-weight "$EDGE_CLOSURE_LOSS_WEIGHT" \
  --topology-aux-loss-weight "$TOPOLOGY_AUX_LOSS_WEIGHT" \
  --output "$RUN_DIR/face_level_conditioned_holdout.pt"
python scripts/research/eval_face_level_conditioned_tiny.py \
  --checkpoint "$RUN_DIR/face_level_conditioned_holdout.pt" \
  --dataset-dir "$RUN_DIR/test_dataset" \
  --output "$RUN_DIR/eval_gt_count_report.json" \
  --export-dir "$RUN_DIR/eval_gt_count_meshes" \
  --limit "$EVAL_LIMIT" \
  --point-samples "$TRAIN_POINT_SAMPLES" \
  --pair-samples "$PAIR_SAMPLES" \
  --face-count-mode gt
CLEANUP_ARGS=()
if [ "$CLEANUP_FILL_HOLES" = "true" ]; then
  CLEANUP_ARGS+=(--cleanup-fill-holes)
fi
python scripts/research/eval_face_level_conditioned_tiny.py \
  --checkpoint "$RUN_DIR/face_level_conditioned_holdout.pt" \
  --dataset-dir "$RUN_DIR/test_dataset" \
  --output "$RUN_DIR/eval_predicted_count_report.json" \
  --export-dir "$RUN_DIR/eval_predicted_count_meshes" \
  --cleanup-export-dir "$RUN_DIR/eval_predicted_count_cleanup" \
  --limit "$EVAL_LIMIT" \
  --point-samples "$TRAIN_POINT_SAMPLES" \
  --pair-samples "$PAIR_SAMPLES" \
  --face-count-mode predicted \
  --cleanup-min-component-faces "$CLEANUP_MIN_COMPONENT_FACES" \
  "\${CLEANUP_ARGS[@]}"
python scripts/research/eval_face_level_conditioned_tiny.py \
  --checkpoint "$RUN_DIR/face_level_conditioned_holdout.pt" \
  --dataset-dir "$RUN_DIR/test_dataset" \
  --output "$RUN_DIR/eval_predicted_count_token_repair_report.json" \
  --export-dir "$RUN_DIR/eval_predicted_count_token_repair_meshes" \
  --cleanup-export-dir "$RUN_DIR/eval_predicted_count_token_repair_cleanup" \
  --limit "$EVAL_LIMIT" \
  --point-samples "$TRAIN_POINT_SAMPLES" \
  --pair-samples "$PAIR_SAMPLES" \
  --face-count-mode predicted \
  --token-repair-mode "$TOKEN_REPAIR_MODE" \
  --cleanup-min-component-faces "$CLEANUP_MIN_COMPONENT_FACES" \
  "\${CLEANUP_ARGS[@]}"
python - <<'PY'
import json
from pathlib import Path
run = Path("$RUN_DIR")
predicted = json.loads((run / "eval_predicted_count_report.json").read_text())
token_repair = json.loads((run / "eval_predicted_count_token_repair_report.json").read_text())
summary = {
    "run_dir": str(run),
    "gt_count": json.loads((run / "eval_gt_count_report.json").read_text())["summary"],
    "predicted_count": predicted["summary"],
    "predicted_count_cleanup": predicted.get("cleanup_summary"),
    "predicted_count_token_repair": token_repair["summary"],
    "predicted_count_token_repair_cleanup": token_repair.get("cleanup_summary"),
    "predicted_count_token_repair_topology": token_repair.get("token_summary", {}).get("decoded_generated"),
}
(run / "holdout_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
print(json.dumps(summary, indent=2, sort_keys=True))
PY
ls -lh "$RUN_DIR"/face_level_conditioned_holdout.pt "$RUN_DIR"/*_report.json "$RUN_DIR"/holdout_summary.json
exit
EOF

if [ -n "$DOWNLOAD_DIR" ]; then
  mkdir -p "$DOWNLOAD_DIR"
  REMOTE_ARCHIVE="/tmp/$(basename "$RUN_DIR")_reports.tar.gz"
  cat <<EOF | "$TNR_BIN" connect "$INSTANCE_ID"
set -euo pipefail
tar -czf "$REMOTE_ARCHIVE" -C "$RUN_DIR" \
  face_level_conditioned_holdout.pt \
  holdout_summary.json \
  eval_gt_count_report.json \
  eval_predicted_count_report.json \
  eval_predicted_count_token_repair_report.json \
  eval_gt_count_meshes \
  eval_predicted_count_meshes \
  eval_predicted_count_cleanup \
  eval_predicted_count_token_repair_meshes \
  eval_predicted_count_token_repair_cleanup
ls -lh "$REMOTE_ARCHIVE"
exit
EOF
  "$TNR_BIN" scp "$INSTANCE_ID:$REMOTE_ARCHIVE" "$DOWNLOAD_DIR/$(basename "$REMOTE_ARCHIVE")"
  tar -xzf "$DOWNLOAD_DIR/$(basename "$REMOTE_ARCHIVE")" -C "$DOWNLOAD_DIR"
fi
