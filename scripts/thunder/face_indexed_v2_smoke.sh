#!/usr/bin/env bash
set -euo pipefail

INSTANCE_ID="${1:-${THUNDER_INSTANCE_ID:-0}}"
REMOTE_DIR="${REMOTE_DIR:-/home/ubuntu/clearmesh}"
REMOTE_VENV="${REMOTE_VENV:-/home/ubuntu/clearmesh-venv}"
TNR_BIN="${TNR_BIN:-/Users/Ashar/.tnr/bin/tnr}"
RUN_DIR="${RUN_DIR:-/tmp/clearmesh_face_indexed_v2_smoke}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-}"
LOCAL_DATASET_TAR="${LOCAL_DATASET_TAR:-}"
REMOTE_DATASET_TAR="${REMOTE_DATASET_TAR:-/tmp/clearmesh_face_indexed_v2_dataset.tar.gz}"
LOCAL_CHECKPOINT="${LOCAL_CHECKPOINT:-}"
REMOTE_CHECKPOINT="${REMOTE_CHECKPOINT:-/tmp/clearmesh_face_indexed_v2_checkpoint.pt}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
DATASET_DIR="${DATASET_DIR:-}"
SYNTHETIC_COUNT="${SYNTHETIC_COUNT:-10}"
SYNTHETIC_KIND="${SYNTHETIC_KIND:-mixed_cycle}"
MAX_FACES="${MAX_FACES:-256}"
POINT_SAMPLES="${POINT_SAMPLES:-256}"
TRAIN_POINT_SAMPLES="${TRAIN_POINT_SAMPLES:-128}"
STEPS="${STEPS:-300}"
BATCH_SIZE="${BATCH_SIZE:-2}"
TRAIN_LIMIT="${TRAIN_LIMIT:-0}"
HIDDEN_SIZE="${HIDDEN_SIZE:-96}"
LAYERS="${LAYERS:-2}"
HEADS="${HEADS:-4}"
CONDITION_TOKENS="${CONDITION_TOKENS:-4}"
EDGE_HEAD_MODE="${EDGE_HEAD_MODE:-geometry}"
PAIR_SAMPLES="${PAIR_SAMPLES:-300}"
EVAL_LIMIT="${EVAL_LIMIT:-6}"
EVAL_OFFSET="${EVAL_OFFSET:-0}"
TEACHER_GATE_MODE="${TEACHER_GATE_MODE:-}"
if [ -z "$TEACHER_GATE_MODE" ]; then
  if [ "$EVAL_OFFSET" -gt 0 ]; then
    TEACHER_GATE_MODE="generalization"
  else
    TEACHER_GATE_MODE="memorization"
  fi
fi
AUX_TEACHER_FORCED_EVAL="${AUX_TEACHER_FORCED_EVAL:-0}"
AUX_TEACHER_EVAL_LIMIT="${AUX_TEACHER_EVAL_LIMIT:-$EVAL_LIMIT}"
RUN_SCALE_READINESS="${RUN_SCALE_READINESS:-1}"
FAIL_ON_SCALE_NOT_READY="${FAIL_ON_SCALE_NOT_READY:-0}"
SEED="${SEED:-31}"
CORNER_HEAD="${CORNER_HEAD:-parallel}"
TOPOLOGY_LOSS_WEIGHT="${TOPOLOGY_LOSS_WEIGHT:-0.0}"
EDGE_ACTION_LOSS_WEIGHT="${EDGE_ACTION_LOSS_WEIGHT:-0.0}"
EDGE_CHOICE_LOSS_WEIGHT="${EDGE_CHOICE_LOSS_WEIGHT:-0.0}"
EDGE_CHOICE_CANDIDATES="${EDGE_CHOICE_CANDIDATES:-64}"
SEED_FACE_LOSS_WEIGHT="${SEED_FACE_LOSS_WEIGHT:-0.0}"
SEED_FACE_LOSS_STOP_STEP="${SEED_FACE_LOSS_STOP_STEP:-0}"
EARLY_FACE_COUNT="${EARLY_FACE_COUNT:-0}"
EARLY_FACE_LOSS_WEIGHT="${EARLY_FACE_LOSS_WEIGHT:-1.0}"
DECODE_STRATEGY="${DECODE_STRATEGY:-free_run}"
DECODE_MODE="${DECODE_MODE:-edge_constrained}"
TOKEN_REPAIR_MODE="${TOKEN_REPAIR_MODE:-none}"
BOUNDARY_FILL="${BOUNDARY_FILL:-none}"
BOUNDARY_FILL_MAX_LOOP_EDGES="${BOUNDARY_FILL_MAX_LOOP_EDGES:-128}"
SPLIT_PINCHED_VERTICES="${SPLIT_PINCHED_VERTICES:-0}"
CLEANUP_SPLIT_NONMANIFOLD_VERTICES="${CLEANUP_SPLIT_NONMANIFOLD_VERTICES:-0}"
CONSTRAINT_TOP_K="${CONSTRAINT_TOP_K:-24}"
LOCAL_CANDIDATE_NEIGHBORS="${LOCAL_CANDIDATE_NEIGHBORS:-0}"
CLOSURE_BONUS="${CLOSURE_BONUS:-1.0}"
NEW_EDGE_PENALTY="${NEW_EDGE_PENALTY:-0.1}"
EDGE_LENGTH_PENALTY="${EDGE_LENGTH_PENALTY:-0.0}"
ASPECT_PENALTY="${ASPECT_PENALTY:-0.0}"
EDGE_ACTION_BONUS="${EDGE_ACTION_BONUS:-0.0}"
EDGE_ACTION_CANDIDATE_TOP_K="${EDGE_ACTION_CANDIDATE_TOP_K:-0}"
EDGE_CHOICE_BONUS="${EDGE_CHOICE_BONUS:-0.0}"
EDGE_CHOICE_CANDIDATE_TOP_K="${EDGE_CHOICE_CANDIDATE_TOP_K:-0}"
SEED_FACE_BONUS="${SEED_FACE_BONUS:-0.0}"
REQUIRE_BOUNDARY_CLOSURE_AFTER="${REQUIRE_BOUNDARY_CLOSURE_AFTER:-1}"
CLOSURE_TARGET_BONUS="${CLOSURE_TARGET_BONUS:-0.0}"
BOUNDARY_BUDGET_CONSTRAINT="${BOUNDARY_BUDGET_CONSTRAINT:-0}"
BEAM_WIDTH="${BEAM_WIDTH:-1}"
BEAM_CANDIDATES="${BEAM_CANDIDATES:-4}"
VERTEX_LINK_CONSTRAINT="${VERTEX_LINK_CONSTRAINT:-0}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

if [ -z "${THUNDER_TOKEN:-}" ]; then
  echo "THUNDER_TOKEN is not set. Export it before running Thunder commands." >&2
  exit 1
fi
if [ -n "$LOCAL_DATASET_TAR" ]; then
  if [ ! -f "$LOCAL_DATASET_TAR" ]; then
    echo "LOCAL_DATASET_TAR not found: $LOCAL_DATASET_TAR" >&2
    exit 1
  fi
  "$TNR_BIN" scp "$LOCAL_DATASET_TAR" "$INSTANCE_ID:$REMOTE_DATASET_TAR"
fi
if [ -n "$LOCAL_CHECKPOINT" ]; then
  if [ ! -f "$LOCAL_CHECKPOINT" ]; then
    echo "LOCAL_CHECKPOINT not found: $LOCAL_CHECKPOINT" >&2
    exit 1
  fi
  "$TNR_BIN" scp "$LOCAL_CHECKPOINT" "$INSTANCE_ID:$REMOTE_CHECKPOINT"
elif [ "$SKIP_TRAIN" = "1" ]; then
  echo "SKIP_TRAIN=1 requires LOCAL_CHECKPOINT." >&2
  exit 1
fi

remote_log="$(mktemp -t clearmesh-face-indexed-remote.XXXXXX.log)"
remote_status=0
set +e
cat <<EOF | "$TNR_BIN" connect "$INSTANCE_ID" | tee "$remote_log"
set -euo pipefail
cd "$REMOTE_DIR"
. "$REMOTE_VENV/bin/activate"
if ! compgen -G '/dev/nvidia[0-9]*' >/dev/null; then
  echo 'missing /dev/nvidia[0-9]*; Thunder GPU device is not mounted' >&2
  exit 20
fi
timeout 45s nvidia-smi
python - <<'PY' || python -m pip install torch==2.8.0 --index-url "$PYTORCH_INDEX_URL"
import torch
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())
PY
timeout 45s python - <<'PY'
import torch
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit('torch CUDA is not available')
print('device', torch.cuda.get_device_name(0), 'bf16', torch.cuda.is_bf16_supported())
PY
rm -rf "$RUN_DIR"
mkdir -p "$RUN_DIR"
if [ -n "$LOCAL_DATASET_TAR" ]; then
  mkdir -p "$RUN_DIR/uploaded"
  tar -xzf "$REMOTE_DATASET_TAR" -C "$RUN_DIR/uploaded"
  DATASET_DIR="\$(find "$RUN_DIR/uploaded" -maxdepth 2 -type f -name 'manifest.jsonl' -print -quit | xargs dirname)"
elif [ -z "$DATASET_DIR" ]; then
  DATASET_DIR="$RUN_DIR/dataset"
  python scripts/research/build_face_token_dataset.py \
    --output-dir "\$DATASET_DIR" \
    --synthetic-count "$SYNTHETIC_COUNT" \
    --synthetic-kind "$SYNTHETIC_KIND" \
    --max-faces "$MAX_FACES" \
    --point-samples "$POINT_SAMPLES" \
    --seed "$SEED"
fi
if [ -z "\$DATASET_DIR" ] || [ ! -d "\$DATASET_DIR" ]; then
  echo "Indexed dataset not found: \$DATASET_DIR" >&2
  exit 2
fi
echo "indexed_dataset_dir=\$DATASET_DIR"
echo "\$DATASET_DIR" > "$RUN_DIR/dataset_dir.txt"
export DATASET_DIR
python - <<'PY'
from pathlib import Path
import json
import os
import numpy as np

dataset = Path(os.environ["DATASET_DIR"])
paths = sorted(path for path in dataset.glob("*.npz") if not path.name.startswith("._"))
print(json.dumps({
    "face_indexed_dataset_probe": str(dataset),
    "npz_files": len(paths),
    "has_manifest": (dataset / "manifest.jsonl").exists(),
}, sort_keys=True))
if paths:
    first = paths[0]
    data = np.load(first)
    print(json.dumps({
        "first_npz": first.name,
        "keys": sorted(data.files),
        "indexed_vertices_shape": list(data["indexed_vertices"].shape) if "indexed_vertices" in data.files else None,
        "indexed_faces_shape": list(data["indexed_faces"].shape) if "indexed_faces" in data.files else None,
        "surface_points_shape": list(data["surface_points"].shape) if "surface_points" in data.files else None,
        "surface_normals_shape": list(data["surface_normals"].shape) if "surface_normals" in data.files else None,
        "num_bins": int(np.asarray(data["num_bins"]).reshape(-1)[0]) if "num_bins" in data.files else None,
    }, sort_keys=True))
PY
if [ "$SKIP_TRAIN" = "1" ]; then
  cp "$REMOTE_CHECKPOINT" "$RUN_DIR/face_indexed_v2_tiny.pt"
else
  python scripts/research/train_face_indexed_conditioned_tiny.py \
    --dataset-dir "\$DATASET_DIR" \
    --output "$RUN_DIR/face_indexed_v2_tiny.pt" \
    --steps "$STEPS" \
    --batch-size "$BATCH_SIZE" \
    --limit "$TRAIN_LIMIT" \
    --point-samples "$TRAIN_POINT_SAMPLES" \
    --hidden-size "$HIDDEN_SIZE" \
    --layers "$LAYERS" \
    --heads "$HEADS" \
    --condition-tokens "$CONDITION_TOKENS" \
    --edge-head-mode "$EDGE_HEAD_MODE" \
    --corner-head "$CORNER_HEAD" \
    --topology-loss-weight "$TOPOLOGY_LOSS_WEIGHT" \
    --edge-action-loss-weight "$EDGE_ACTION_LOSS_WEIGHT" \
    --edge-choice-loss-weight "$EDGE_CHOICE_LOSS_WEIGHT" \
    --edge-choice-candidates "$EDGE_CHOICE_CANDIDATES" \
    --seed-face-loss-weight "$SEED_FACE_LOSS_WEIGHT" \
    --seed-face-loss-stop-step "$SEED_FACE_LOSS_STOP_STEP" \
    --early-face-count "$EARLY_FACE_COUNT" \
    --early-face-loss-weight "$EARLY_FACE_LOSS_WEIGHT" \
    --seed "$SEED"
fi
python scripts/research/eval_face_indexed_conditioned_tiny.py \
  --checkpoint "$RUN_DIR/face_indexed_v2_tiny.pt" \
  --dataset-dir "\$DATASET_DIR" \
  --output "$RUN_DIR/eval_report.json" \
  --export-dir "$RUN_DIR/meshes" \
  --cleanup-export-dir "$RUN_DIR/cleanup" \
  --limit "$EVAL_LIMIT" \
  --offset "$EVAL_OFFSET" \
  --point-samples "$TRAIN_POINT_SAMPLES" \
  --face-count-mode gt \
  --pair-samples "$PAIR_SAMPLES" \
  --token-repair-mode "$TOKEN_REPAIR_MODE" \
  --boundary-fill "$BOUNDARY_FILL" \
  --boundary-fill-max-loop-edges "$BOUNDARY_FILL_MAX_LOOP_EDGES" \
  $([ "$SPLIT_PINCHED_VERTICES" = "1" ] && printf %s "--split-pinched-vertices") \
  $([ "$CLEANUP_SPLIT_NONMANIFOLD_VERTICES" = "1" ] && printf %s "--cleanup-split-nonmanifold-vertices") \
  --decode-strategy "$DECODE_STRATEGY" \
  --decode-mode "$DECODE_MODE" \
  --constraint-top-k "$CONSTRAINT_TOP_K" \
  --local-candidate-neighbors "$LOCAL_CANDIDATE_NEIGHBORS" \
  --closure-bonus "$CLOSURE_BONUS" \
  --new-edge-penalty "$NEW_EDGE_PENALTY" \
  --edge-length-penalty "$EDGE_LENGTH_PENALTY" \
  --aspect-penalty "$ASPECT_PENALTY" \
  --edge-action-bonus "$EDGE_ACTION_BONUS" \
  --edge-action-candidate-top-k "$EDGE_ACTION_CANDIDATE_TOP_K" \
  --edge-choice-bonus "$EDGE_CHOICE_BONUS" \
  --edge-choice-candidate-top-k "$EDGE_CHOICE_CANDIDATE_TOP_K" \
  --seed-face-bonus "$SEED_FACE_BONUS" \
  --require-boundary-closure-after "$REQUIRE_BOUNDARY_CLOSURE_AFTER" \
  --closure-target-bonus "$CLOSURE_TARGET_BONUS" \
  $([ "$BOUNDARY_BUDGET_CONSTRAINT" = "1" ] && printf %s "--boundary-budget-constraint") \
  --beam-width "$BEAM_WIDTH" \
  --beam-candidates "$BEAM_CANDIDATES" \
  $([ "$VERTEX_LINK_CONSTRAINT" = "1" ] && printf %s "--vertex-link-constraint")
if [ "$AUX_TEACHER_FORCED_EVAL" = "1" ]; then
  python scripts/research/eval_face_indexed_conditioned_tiny.py \
    --checkpoint "$RUN_DIR/face_indexed_v2_tiny.pt" \
    --dataset-dir "\$DATASET_DIR" \
    --output "$RUN_DIR/eval_teacher_forced_report.json" \
    --export-dir "$RUN_DIR/teacher_forced_meshes" \
    --limit "$AUX_TEACHER_EVAL_LIMIT" \
    --offset "$EVAL_OFFSET" \
    --point-samples "$TRAIN_POINT_SAMPLES" \
    --face-count-mode gt \
    --pair-samples "$PAIR_SAMPLES" \
    --token-repair-mode "$TOKEN_REPAIR_MODE" \
    --boundary-fill "$BOUNDARY_FILL" \
    --boundary-fill-max-loop-edges "$BOUNDARY_FILL_MAX_LOOP_EDGES" \
    $([ "$SPLIT_PINCHED_VERTICES" = "1" ] && printf %s "--split-pinched-vertices") \
    $([ "$CLEANUP_SPLIT_NONMANIFOLD_VERTICES" = "1" ] && printf %s "--cleanup-split-nonmanifold-vertices") \
    --decode-strategy teacher_forced \
    --decode-mode "$DECODE_MODE" \
    --constraint-top-k "$CONSTRAINT_TOP_K" \
    --local-candidate-neighbors "$LOCAL_CANDIDATE_NEIGHBORS" \
    --closure-bonus "$CLOSURE_BONUS" \
    --new-edge-penalty "$NEW_EDGE_PENALTY" \
    --edge-length-penalty "$EDGE_LENGTH_PENALTY" \
    --aspect-penalty "$ASPECT_PENALTY" \
    --edge-action-bonus "$EDGE_ACTION_BONUS" \
    --edge-action-candidate-top-k "$EDGE_ACTION_CANDIDATE_TOP_K" \
    --edge-choice-bonus "$EDGE_CHOICE_BONUS" \
    --edge-choice-candidate-top-k "$EDGE_CHOICE_CANDIDATE_TOP_K" \
    --seed-face-bonus "$SEED_FACE_BONUS" \
    --require-boundary-closure-after "$REQUIRE_BOUNDARY_CLOSURE_AFTER" \
    --closure-target-bonus "$CLOSURE_TARGET_BONUS" \
    $([ "$BOUNDARY_BUDGET_CONSTRAINT" = "1" ] && printf %s "--boundary-budget-constraint") \
    --beam-width "$BEAM_WIDTH" \
    --beam-candidates "$BEAM_CANDIDATES" \
    $([ "$VERTEX_LINK_CONSTRAINT" = "1" ] && printf %s "--vertex-link-constraint")
fi
if [ "$RUN_SCALE_READINESS" = "1" ] && [ -f "$RUN_DIR/eval_teacher_forced_report.json" ]; then
  readiness_args=(
    --teacher-eval "$RUN_DIR/eval_teacher_forced_report.json"
    --free-run-eval "$RUN_DIR/eval_report.json"
    --output "$RUN_DIR/scale_readiness.json"
	    --min-dataset-samples "$EVAL_LIMIT"
	    --min-eval-samples "$EVAL_LIMIT"
	    --teacher-gate-mode "$TEACHER_GATE_MODE"
	  )
  if [ -f "\$DATASET_DIR/curation_summary.json" ]; then
    readiness_args=(--curation-summary "\$DATASET_DIR/curation_summary.json" "\${readiness_args[@]}")
  fi
  if ! python scripts/research/assess_face_indexed_scale_readiness.py "\${readiness_args[@]}"; then
    if [ "$FAIL_ON_SCALE_NOT_READY" = "1" ]; then
      exit 30
    fi
    echo "scale readiness gate did not promote this run; continuing so artifacts can be fetched" >&2
  fi
fi
python - <<'PY'
import json, torch
from pathlib import Path
run = Path('$RUN_DIR')
report = json.loads((run / 'eval_report.json').read_text())
teacher_report_path = run / 'eval_teacher_forced_report.json'
scale_readiness_path = run / 'scale_readiness.json'
ckpt = torch.load(run / 'face_indexed_v2_tiny.pt', map_location='cpu', weights_only=False)
print(json.dumps({
    'dataset_dir': (run / 'dataset_dir.txt').read_text().strip() if (run / 'dataset_dir.txt').exists() else None,
    'best_loss': ckpt.get('best_loss'),
    'best_step': ckpt.get('best_step'),
    'summary': report['summary'],
    'teacher_forced_summary': json.loads(teacher_report_path.read_text())['summary'] if teacher_report_path.exists() else None,
    'scale_readiness': json.loads(scale_readiness_path.read_text()) if scale_readiness_path.exists() else None,
    'cleanup_summary': report['cleanup_summary'],
}, indent=2, sort_keys=True))
PY
tar -czf /tmp/$(basename "$RUN_DIR").tar.gz -C "$RUN_DIR" .
ls -lh /tmp/$(basename "$RUN_DIR").tar.gz
echo CLEARMESH_FACE_INDEXED_REMOTE_DONE
exit
EOF
remote_status=${PIPESTATUS[1]}
set -e
if ! grep -a -q "CLEARMESH_FACE_INDEXED_REMOTE_DONE" "$remote_log"; then
  echo "FACE indexed remote run did not complete successfully; see $remote_log" >&2
  exit 7
fi
if [ "$remote_status" -ne 0 ]; then
  echo "Thunder connect exited with status $remote_status after completion sentinel; continuing to fetch artifacts" >&2
fi

if [ -n "$DOWNLOAD_DIR" ]; then
  mkdir -p "$DOWNLOAD_DIR"
  archive="/tmp/$(basename "$RUN_DIR").tar.gz"
  rm -f "$DOWNLOAD_DIR/$(basename "$archive")"
  "$TNR_BIN" scp "$INSTANCE_ID:$archive" "$DOWNLOAD_DIR/$(basename "$archive")"
  if [ ! -s "$DOWNLOAD_DIR/$(basename "$archive")" ]; then
    echo "Expected Thunder archive was not downloaded: $archive" >&2
    exit 4
  fi
  tar -xzf "$DOWNLOAD_DIR/$(basename "$archive")" -C "$DOWNLOAD_DIR"
fi
