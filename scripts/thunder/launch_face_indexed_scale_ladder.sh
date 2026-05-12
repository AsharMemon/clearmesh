#!/usr/bin/env bash
# Launch a guarded FACE-indexed scale ladder on Thunder.
#
# This script is intentionally conservative: it refuses to install dependencies
# until the instance exposes /dev/nvidia*, because Thunder has recently returned
# GPU-labeled containers without mounted GPU devices.
set -euo pipefail

TNR_BIN="${TNR_BIN:-/Users/Ashar/.tnr/bin/tnr}"
INSTANCE_ID="${THUNDER_INSTANCE_ID:-}"
CREATE_INSTANCE="${CREATE_INSTANCE:-1}"
DELETE_ON_EXIT="${DELETE_ON_EXIT:-1}"
CREATED_INSTANCE=0
GPU="${GPU:-a100}"
MODE="${MODE:-prototyping}"
VCPUS="${VCPUS:-8}"
PRIMARY_DISK="${PRIMARY_DISK:-200}"
TEMPLATE="${TEMPLATE:-base}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEFAULT_DATASET_TAR="$REPO_ROOT/.codex_outputs/face_indexed_real50_boundary/dataset_4096_decoded_watertight_clean.tar.gz"
if [ ! -f "$DEFAULT_DATASET_TAR" ]; then
  DEFAULT_DATASET_TAR="$REPO_ROOT/.codex_outputs/face_indexed_real50_boundary/dataset_4096_decoded_watertight.tar.gz"
fi
DATASET_TAR="${DATASET_TAR:-$DEFAULT_DATASET_TAR}"
LOCAL_CHECKPOINT="${LOCAL_CHECKPOINT:-}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-$REPO_ROOT/.codex_outputs/face_indexed_scale_ladder_$RUN_STAMP}"
TRAIN_LIMIT="${TRAIN_LIMIT:-8}"
EVAL_LIMIT="${EVAL_LIMIT:-8}"
EVAL_OFFSET="${EVAL_OFFSET:-0}"
TEACHER_GATE_MODE="${TEACHER_GATE_MODE:-}"
if [ -z "$TEACHER_GATE_MODE" ]; then
  if [ "$EVAL_OFFSET" -gt 0 ]; then
    TEACHER_GATE_MODE="generalization"
  else
    TEACHER_GATE_MODE="memorization"
  fi
fi
STEPS="${STEPS:-2500}"
BATCH_SIZE="${BATCH_SIZE:-2}"
HIDDEN_SIZE="${HIDDEN_SIZE:-128}"
LAYERS="${LAYERS:-3}"
HEADS="${HEADS:-4}"
CONDITION_TOKENS="${CONDITION_TOKENS:-8}"
EDGE_HEAD_MODE="${EDGE_HEAD_MODE:-geometry}"
TRAIN_POINT_SAMPLES="${TRAIN_POINT_SAMPLES:-1024}"
PAIR_SAMPLES="${PAIR_SAMPLES:-100}"
DECODE_STRATEGY="${DECODE_STRATEGY:-free_run}"
CORNER_HEAD="${CORNER_HEAD:-causal}"
TOPOLOGY_LOSS_WEIGHT="${TOPOLOGY_LOSS_WEIGHT:-0.2}"
EDGE_ACTION_LOSS_WEIGHT="${EDGE_ACTION_LOSS_WEIGHT:-1.0}"
EDGE_CHOICE_LOSS_WEIGHT="${EDGE_CHOICE_LOSS_WEIGHT:-0.5}"
EDGE_CHOICE_CANDIDATES="${EDGE_CHOICE_CANDIDATES:-32}"
SEED_FACE_LOSS_WEIGHT="${SEED_FACE_LOSS_WEIGHT:-1.0}"
SEED_FACE_LOSS_STOP_STEP="${SEED_FACE_LOSS_STOP_STEP:-0}"
EARLY_FACE_COUNT="${EARLY_FACE_COUNT:-16}"
EARLY_FACE_LOSS_WEIGHT="${EARLY_FACE_LOSS_WEIGHT:-4.0}"
DECODE_MODE="${DECODE_MODE:-boundary_edge}"
CONSTRAINT_TOP_K="${CONSTRAINT_TOP_K:-6}"
LOCAL_CANDIDATE_NEIGHBORS="${LOCAL_CANDIDATE_NEIGHBORS:-2}"
CLOSURE_BONUS="${CLOSURE_BONUS:-2.0}"
NEW_EDGE_PENALTY="${NEW_EDGE_PENALTY:-0.05}"
EDGE_LENGTH_PENALTY="${EDGE_LENGTH_PENALTY:-0.1}"
ASPECT_PENALTY="${ASPECT_PENALTY:-0.05}"
EDGE_ACTION_BONUS="${EDGE_ACTION_BONUS:-1.0}"
EDGE_ACTION_CANDIDATE_TOP_K="${EDGE_ACTION_CANDIDATE_TOP_K:-8}"
EDGE_CHOICE_BONUS="${EDGE_CHOICE_BONUS:-0.5}"
EDGE_CHOICE_CANDIDATE_TOP_K="${EDGE_CHOICE_CANDIDATE_TOP_K:-8}"
SEED_FACE_BONUS="${SEED_FACE_BONUS:-0.0}"
REQUIRE_BOUNDARY_CLOSURE_AFTER="${REQUIRE_BOUNDARY_CLOSURE_AFTER:-1}"
CLOSURE_TARGET_BONUS="${CLOSURE_TARGET_BONUS:-1.0}"
BOUNDARY_BUDGET_CONSTRAINT="${BOUNDARY_BUDGET_CONSTRAINT:-1}"
VERTEX_LINK_CONSTRAINT="${VERTEX_LINK_CONSTRAINT:-1}"
BEAM_WIDTH="${BEAM_WIDTH:-1}"
BEAM_CANDIDATES="${BEAM_CANDIDATES:-4}"
TOKEN_REPAIR_MODE="${TOKEN_REPAIR_MODE:-none}"
BOUNDARY_FILL="${BOUNDARY_FILL:-centroid}"
BOUNDARY_FILL_MAX_LOOP_EDGES="${BOUNDARY_FILL_MAX_LOOP_EDGES:-128}"
SPLIT_PINCHED_VERTICES="${SPLIT_PINCHED_VERTICES:-0}"
CLEANUP_SPLIT_NONMANIFOLD_VERTICES="${CLEANUP_SPLIT_NONMANIFOLD_VERTICES:-0}"
RUN_SCALE_READINESS="${RUN_SCALE_READINESS:-1}"
FAIL_ON_SCALE_NOT_READY="${FAIL_ON_SCALE_NOT_READY:-0}"
WAIT_ATTEMPTS="${WAIT_ATTEMPTS:-60}"
WAIT_INTERVAL_SEC="${WAIT_INTERVAL_SEC:-5}"
WAIT_TIMEOUT_SEC="${WAIT_TIMEOUT_SEC:-$((WAIT_ATTEMPTS * WAIT_INTERVAL_SEC))}"
PREFLIGHT_ATTEMPTS="${PREFLIGHT_ATTEMPTS:-5}"
PREFLIGHT_RETRY_SEC="${PREFLIGHT_RETRY_SEC:-30}"

if [ -z "${THUNDER_TOKEN:-}" ]; then
  echo "THUNDER_TOKEN is not set." >&2
  exit 1
fi
case "$MODE:$GPU" in
  production:a100|production:h100|prototyping:a6000|prototyping:a100|prototyping:h100)
    ;;
  production:*)
    echo "Unsupported Thunder production GPU '$GPU'. Use a100 or h100." >&2
    exit 5
    ;;
  prototyping:*)
    echo "Unsupported Thunder prototyping GPU '$GPU'. Use a6000, a100, or h100." >&2
    exit 5
    ;;
  *)
    echo "Unsupported Thunder mode '$MODE'. Use prototyping or production." >&2
    exit 5
    ;;
esac
if [ ! -f "$DATASET_TAR" ]; then
  echo "Dataset tar not found: $DATASET_TAR" >&2
  exit 2
fi
if [ "$SKIP_TRAIN" = "1" ]; then
  if [ -z "$LOCAL_CHECKPOINT" ] || [ ! -f "$LOCAL_CHECKPOINT" ]; then
    echo "SKIP_TRAIN=1 requires LOCAL_CHECKPOINT." >&2
    exit 2
  fi
fi

if [ "$CREATE_INSTANCE" = "1" ]; then
  create_args=(create --gpu "$GPU" --mode "$MODE" --num-gpus 1 --primary-disk "$PRIMARY_DISK" --template "$TEMPLATE" --yes --json)
  if [ "$MODE" = "prototyping" ]; then
    create_args+=(--vcpus "$VCPUS")
  fi
  create_output="$("$TNR_BIN" "${create_args[@]}")" || {
    echo "Thunder create failed." >&2
    exit 3
  }
  CREATED_INSTANCE=1
  if [ -z "$INSTANCE_ID" ]; then
    INSTANCE_ID="$(CREATE_OUTPUT="$create_output" python3 - <<'PY'
import json
import os
import re

text = os.environ.get("CREATE_OUTPUT", "")
decoder = json.JSONDecoder()
for match in re.finditer(r"[\[{]", text):
    try:
        payload, _ = decoder.raw_decode(text[match.start():])
    except json.JSONDecodeError:
        continue
    items = payload if isinstance(payload, list) else [payload]
    for item in items:
        if isinstance(item, dict):
            for key in ("id", "identifier", "instance_id", "instanceId", "uuid"):
                if item.get(key) is not None:
                    print(item[key])
                    raise SystemExit(0)
raise SystemExit(1)
PY
)" || INSTANCE_ID="0"
  fi
  echo "Created Thunder instance $INSTANCE_ID."
fi
if [ -z "$INSTANCE_ID" ]; then
  echo "Set THUNDER_INSTANCE_ID or CREATE_INSTANCE=1." >&2
  exit 4
fi

cleanup_instance() {
  local exit_code=$?
  if [ "$exit_code" -ne 0 ] && [ "$CREATED_INSTANCE" = "1" ] && [ "$DELETE_ON_EXIT" = "1" ] && [ -n "$INSTANCE_ID" ]; then
    echo "Launcher exiting with status $exit_code; deleting created Thunder instance $INSTANCE_ID." >&2
    "$TNR_BIN" delete "$INSTANCE_ID" --yes || true
  fi
}
trap cleanup_instance EXIT INT TERM HUP

mkdir -p "$DOWNLOAD_ROOT"

echo "Waiting for Thunder instance $INSTANCE_ID to RUNNING..."
running_seen=0
wait_deadline=$(( $(date +%s) + WAIT_TIMEOUT_SEC ))
while [ "$(date +%s)" -lt "$wait_deadline" ]; do
  status_json="$($TNR_BIN status --json || true)"
  printf '%s\n' "$status_json" > "$DOWNLOAD_ROOT/status.latest.json"
  if python3 - "$INSTANCE_ID" "$DOWNLOAD_ROOT/status.latest.json" <<'PY'
import json
import sys
from pathlib import Path
target = sys.argv[1]
text = Path(sys.argv[2]).read_text()
start = text.find("[")
data = json.loads(text[start:]) if start >= 0 else []
for item in data:
    if str(item.get("id")) == str(target) and item.get("status") == "RUNNING":
        raise SystemExit(0)
raise SystemExit(1)
PY
  then
    running_seen=1
    break
  fi
  sleep "$WAIT_INTERVAL_SEC"
done
if [ "$running_seen" != "1" ]; then
  echo "Thunder instance $INSTANCE_ID did not reach RUNNING in time; deleting." >&2
  "$TNR_BIN" delete "$INSTANCE_ID" --yes || true
  CREATED_INSTANCE=0
  exit 21
fi

echo "Preflighting GPU device before install..."
preflight_log="$DOWNLOAD_ROOT/gpu_preflight.log"
preflight_ok=0
for preflight_attempt in $(seq 1 "$PREFLIGHT_ATTEMPTS"); do
  echo "GPU preflight attempt $preflight_attempt/$PREFLIGHT_ATTEMPTS..."
  cat <<'EOF' | "$TNR_BIN" connect "$INSTANCE_ID" | tee "$preflight_log" || true
set -euo pipefail
hostname
ls -la /dev/nvidia* || true
if ! compgen -G '/dev/nvidia[0-9]*' >/dev/null; then
  echo 'missing /dev/nvidia[0-9]*; Thunder GPU device is not mounted' >&2
  exit 20
fi
nvidia-smi
echo CLEARMESH_GPU_PREFLIGHT_OK
exit
EOF
  if python3 - "$preflight_log" <<'PY'
import re
import sys
from pathlib import Path

ansi = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
for raw in Path(sys.argv[1]).read_text(errors="ignore").splitlines():
    clean = ansi.sub("", raw).replace("\r", "").strip()
    if clean == "CLEARMESH_GPU_PREFLIGHT_OK":
        raise SystemExit(0)
raise SystemExit(1)
PY
  then
    preflight_ok=1
    break
  fi
  if [ "$preflight_attempt" -lt "$PREFLIGHT_ATTEMPTS" ]; then
    echo "GPU preflight did not pass; waiting ${PREFLIGHT_RETRY_SEC}s before retry..."
    sleep "$PREFLIGHT_RETRY_SEC"
  fi
done
if [ "$preflight_ok" != "1" ]; then
  echo "GPU preflight failed; deleting instance $INSTANCE_ID." >&2
  "$TNR_BIN" delete "$INSTANCE_ID" --yes || true
  CREATED_INSTANCE=0
  exit 20
fi

echo "Syncing repo and bootstrapping..."
THUNDER_INSTANCE_ID="$INSTANCE_ID" "$REPO_ROOT/scripts/thunder/sync_repo.sh" "$INSTANCE_ID"
THUNDER_INSTANCE_ID="$INSTANCE_ID" INSTALL_MESH_HEAD_REPOS=0 INSTALL_MESH_HEAD_ENVS=0 "$REPO_ROOT/scripts/thunder/bootstrap_remote.sh" "$INSTANCE_ID"

run_id="face_indexed_scale_${RUN_STAMP}_limit${TRAIN_LIMIT}_steps${STEPS}"
if [ "$SKIP_TRAIN" = "1" ]; then
  run_id="face_indexed_eval_${RUN_STAMP}_limit${EVAL_LIMIT}"
fi
run_dir="/tmp/$run_id"
download_dir="$DOWNLOAD_ROOT/$run_id"
mkdir -p "$download_dir"

echo "Launching $run_id"
THUNDER_INSTANCE_ID="$INSTANCE_ID" \
RUN_DIR="$run_dir" \
DOWNLOAD_DIR="$download_dir" \
LOCAL_DATASET_TAR="$DATASET_TAR" \
LOCAL_CHECKPOINT="$LOCAL_CHECKPOINT" \
SKIP_TRAIN="$SKIP_TRAIN" \
TRAIN_LIMIT="$TRAIN_LIMIT" \
	EVAL_LIMIT="$EVAL_LIMIT" \
	EVAL_OFFSET="$EVAL_OFFSET" \
	TEACHER_GATE_MODE="$TEACHER_GATE_MODE" \
	AUX_TEACHER_FORCED_EVAL=1 \
AUX_TEACHER_EVAL_LIMIT="$EVAL_LIMIT" \
STEPS="$STEPS" \
BATCH_SIZE="$BATCH_SIZE" \
HIDDEN_SIZE="$HIDDEN_SIZE" \
LAYERS="$LAYERS" \
HEADS="$HEADS" \
CONDITION_TOKENS="$CONDITION_TOKENS" \
EDGE_HEAD_MODE="$EDGE_HEAD_MODE" \
TRAIN_POINT_SAMPLES="$TRAIN_POINT_SAMPLES" \
POINT_SAMPLES="$TRAIN_POINT_SAMPLES" \
PAIR_SAMPLES="$PAIR_SAMPLES" \
CORNER_HEAD="$CORNER_HEAD" \
TOPOLOGY_LOSS_WEIGHT="$TOPOLOGY_LOSS_WEIGHT" \
EDGE_ACTION_LOSS_WEIGHT="$EDGE_ACTION_LOSS_WEIGHT" \
EDGE_CHOICE_LOSS_WEIGHT="$EDGE_CHOICE_LOSS_WEIGHT" \
EDGE_CHOICE_CANDIDATES="$EDGE_CHOICE_CANDIDATES" \
SEED_FACE_LOSS_WEIGHT="$SEED_FACE_LOSS_WEIGHT" \
SEED_FACE_LOSS_STOP_STEP="$SEED_FACE_LOSS_STOP_STEP" \
EARLY_FACE_COUNT="$EARLY_FACE_COUNT" \
EARLY_FACE_LOSS_WEIGHT="$EARLY_FACE_LOSS_WEIGHT" \
DECODE_STRATEGY="$DECODE_STRATEGY" \
RUN_SCALE_READINESS="$RUN_SCALE_READINESS" \
FAIL_ON_SCALE_NOT_READY="$FAIL_ON_SCALE_NOT_READY" \
DECODE_MODE="$DECODE_MODE" \
CONSTRAINT_TOP_K="$CONSTRAINT_TOP_K" \
LOCAL_CANDIDATE_NEIGHBORS="$LOCAL_CANDIDATE_NEIGHBORS" \
CLOSURE_BONUS="$CLOSURE_BONUS" \
NEW_EDGE_PENALTY="$NEW_EDGE_PENALTY" \
	EDGE_LENGTH_PENALTY="$EDGE_LENGTH_PENALTY" \
	ASPECT_PENALTY="$ASPECT_PENALTY" \
	EDGE_ACTION_BONUS="$EDGE_ACTION_BONUS" \
	EDGE_ACTION_CANDIDATE_TOP_K="$EDGE_ACTION_CANDIDATE_TOP_K" \
	EDGE_CHOICE_BONUS="$EDGE_CHOICE_BONUS" \
	EDGE_CHOICE_CANDIDATE_TOP_K="$EDGE_CHOICE_CANDIDATE_TOP_K" \
SEED_FACE_BONUS="$SEED_FACE_BONUS" \
REQUIRE_BOUNDARY_CLOSURE_AFTER="$REQUIRE_BOUNDARY_CLOSURE_AFTER" \
CLOSURE_TARGET_BONUS="$CLOSURE_TARGET_BONUS" \
BOUNDARY_BUDGET_CONSTRAINT="$BOUNDARY_BUDGET_CONSTRAINT" \
VERTEX_LINK_CONSTRAINT="$VERTEX_LINK_CONSTRAINT" \
BEAM_WIDTH="$BEAM_WIDTH" \
BEAM_CANDIDATES="$BEAM_CANDIDATES" \
TOKEN_REPAIR_MODE="$TOKEN_REPAIR_MODE" \
BOUNDARY_FILL="$BOUNDARY_FILL" \
BOUNDARY_FILL_MAX_LOOP_EDGES="$BOUNDARY_FILL_MAX_LOOP_EDGES" \
SPLIT_PINCHED_VERTICES="$SPLIT_PINCHED_VERTICES" \
CLEANUP_SPLIT_NONMANIFOLD_VERTICES="$CLEANUP_SPLIT_NONMANIFOLD_VERTICES" \
"$REPO_ROOT/scripts/thunder/face_indexed_v2_smoke.sh" "$INSTANCE_ID" | tee "$download_dir/local_launch.log"

echo "Deleting instance $INSTANCE_ID after successful artifact fetch."
"$TNR_BIN" delete "$INSTANCE_ID" --yes || true
CREATED_INSTANCE=0
"$TNR_BIN" status --json | tee "$DOWNLOAD_ROOT/status.final.json"
