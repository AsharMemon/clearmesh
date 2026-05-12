#!/usr/bin/env bash
# Run FACE-indexed scale rungs in order and stop on the first non-promotion.
#
# This wrapper intentionally delegates each rung to the guarded single-rung
# launcher, so the GPU preflight/deletion behavior stays centralized.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_STAMP_ROOT="${RUN_STAMP_ROOT:-$(date -u +%Y%m%d_%H%M%S)}"
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-$REPO_ROOT/.codex_outputs/face_indexed_scale_supervisor_$RUN_STAMP_ROOT}"
LADDER_LIMITS="${LADDER_LIMITS:-16 32 47}"
STEPS_BY_LIMIT_JSON="${STEPS_BY_LIMIT_JSON:-}"
if [ -z "$STEPS_BY_LIMIT_JSON" ]; then
  STEPS_BY_LIMIT_JSON='{"16":3000,"32":5000,"47":7000}'
fi
BATCH_SIZE="${BATCH_SIZE:-2}"
HIDDEN_SIZE="${HIDDEN_SIZE:-128}"
LAYERS="${LAYERS:-3}"
HEADS="${HEADS:-4}"
CONDITION_TOKENS="${CONDITION_TOKENS:-8}"
TRAIN_POINT_SAMPLES="${TRAIN_POINT_SAMPLES:-1024}"
PAIR_SAMPLES="${PAIR_SAMPLES:-10000}"
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
REQUIRE_BOUNDARY_CLOSURE_AFTER="${REQUIRE_BOUNDARY_CLOSURE_AFTER:-1}"
CLOSURE_TARGET_BONUS="${CLOSURE_TARGET_BONUS:-1.0}"
BOUNDARY_BUDGET_CONSTRAINT="${BOUNDARY_BUDGET_CONSTRAINT:-1}"
VERTEX_LINK_CONSTRAINT="${VERTEX_LINK_CONSTRAINT:-1}"
BOUNDARY_FILL="${BOUNDARY_FILL:-centroid}"
BOUNDARY_FILL_MAX_LOOP_EDGES="${BOUNDARY_FILL_MAX_LOOP_EDGES:-128}"
GPU="${GPU:-a100}"
MODE="${MODE:-production}"
CREATE_INSTANCE="${CREATE_INSTANCE:-1}"
FAIL_ON_SCALE_NOT_READY="${FAIL_ON_SCALE_NOT_READY:-0}"
DELETE_ON_EXIT="${DELETE_ON_EXIT:-1}"
WAIT_ATTEMPTS="${WAIT_ATTEMPTS:-60}"
WAIT_INTERVAL_SEC="${WAIT_INTERVAL_SEC:-5}"
WAIT_TIMEOUT_SEC="${WAIT_TIMEOUT_SEC:-$((WAIT_ATTEMPTS * WAIT_INTERVAL_SEC))}"

mkdir -p "$DOWNLOAD_ROOT"

status_path="$DOWNLOAD_ROOT/supervisor_status.jsonl"
touch "$status_path"

step_for_limit() {
  STEPS_BY_LIMIT_JSON_VALUE="$STEPS_BY_LIMIT_JSON" python3 - "$1" <<'PY'
import json
import os
import sys

limit = str(sys.argv[1])
mapping = json.loads(os.environ["STEPS_BY_LIMIT_JSON_VALUE"])
print(int(mapping.get(limit, mapping.get("default", 5000))))
PY
}

readiness_json_for_rung() {
  local rung_dir="$1"
  find "$rung_dir" -path '*/scale_readiness.json' -type f -print -quit
}

scale_ready_from_json() {
  python3 - "$1" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text())
print("1" if payload.get("scale_ready") else "0")
PY
}

append_status() {
  python3 - "$status_path" "$@" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
item = {"time": datetime.now(timezone.utc).isoformat()}
for raw in sys.argv[2:]:
    key, _, value = raw.partition("=")
    item[key] = value
with path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(item, sort_keys=True) + "\n")
PY
}

append_status event=start limits="$LADDER_LIMITS" download_root="$DOWNLOAD_ROOT"

for limit in $LADDER_LIMITS; do
  steps="$(step_for_limit "$limit")"
  rung_stamp="${RUN_STAMP_ROOT}_limit${limit}"
  rung_download="$DOWNLOAD_ROOT/limit_${limit}"
  mkdir -p "$rung_download"
  append_status event=rung_start limit="$limit" steps="$steps" download_dir="$rung_download"

  if ! RUN_STAMP="$rung_stamp" \
    DOWNLOAD_ROOT="$rung_download" \
    TRAIN_LIMIT="$limit" \
    EVAL_LIMIT="$limit" \
    STEPS="$steps" \
    BATCH_SIZE="$BATCH_SIZE" \
    HIDDEN_SIZE="$HIDDEN_SIZE" \
    LAYERS="$LAYERS" \
    HEADS="$HEADS" \
    CONDITION_TOKENS="$CONDITION_TOKENS" \
    TRAIN_POINT_SAMPLES="$TRAIN_POINT_SAMPLES" \
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
    REQUIRE_BOUNDARY_CLOSURE_AFTER="$REQUIRE_BOUNDARY_CLOSURE_AFTER" \
    CLOSURE_TARGET_BONUS="$CLOSURE_TARGET_BONUS" \
    BOUNDARY_BUDGET_CONSTRAINT="$BOUNDARY_BUDGET_CONSTRAINT" \
    VERTEX_LINK_CONSTRAINT="$VERTEX_LINK_CONSTRAINT" \
    BOUNDARY_FILL="$BOUNDARY_FILL" \
    BOUNDARY_FILL_MAX_LOOP_EDGES="$BOUNDARY_FILL_MAX_LOOP_EDGES" \
    GPU="$GPU" \
    MODE="$MODE" \
    CREATE_INSTANCE="$CREATE_INSTANCE" \
    DELETE_ON_EXIT="$DELETE_ON_EXIT" \
    RUN_SCALE_READINESS=1 \
    FAIL_ON_SCALE_NOT_READY="$FAIL_ON_SCALE_NOT_READY" \
    WAIT_ATTEMPTS="$WAIT_ATTEMPTS" \
    WAIT_INTERVAL_SEC="$WAIT_INTERVAL_SEC" \
    WAIT_TIMEOUT_SEC="$WAIT_TIMEOUT_SEC" \
    "$REPO_ROOT/scripts/thunder/launch_face_indexed_scale_ladder.sh"; then
    append_status event=rung_launcher_failed limit="$limit"
    exit 20
  fi

  readiness_path="$(readiness_json_for_rung "$rung_download")"
  if [ -z "$readiness_path" ]; then
    append_status event=rung_missing_readiness limit="$limit"
    exit 21
  fi
  if [ "$(scale_ready_from_json "$readiness_path")" != "1" ]; then
    append_status event=rung_not_promoted limit="$limit" readiness="$readiness_path"
    echo "Rung $limit did not promote; stopping ladder. See $readiness_path" >&2
    exit 30
  fi
  append_status event=rung_promoted limit="$limit" readiness="$readiness_path"
done

append_status event=complete
echo "FACE-indexed scale supervisor completed all rungs. Status: $status_path"
