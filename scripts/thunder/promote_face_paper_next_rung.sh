#!/usr/bin/env bash
# Guarded launcher for the next FACE paper corpus rung.
#
# This script is intentionally conservative. It will not launch anything unless
# the previous gate's inspection says the run is scale-ready, contact sheets
# exist, visual review has been explicitly marked as passed, and CONFIRM_SCALE=1.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

INSPECTION_JSON="${INSPECTION_JSON:-${1:-}}"
LAB_ROOT="${LAB_ROOT:-}"
FAIL_ON_NOT_READY="${FAIL_ON_NOT_READY:-0}"
CONFIRM_SCALE="${CONFIRM_SCALE:-0}"
VISUAL_REVIEW_PASSED="${VISUAL_REVIEW_PASSED:-0}"

NEXT_SELECT_TARGET="${NEXT_SELECT_TARGET:-2048}"
NEXT_STEPS="${NEXT_STEPS:-30000}"
NEXT_GPU="${NEXT_GPU:-a100}"
NEXT_MODE="${NEXT_MODE:-production}"
NEXT_SCAN_LIMIT="${NEXT_SCAN_LIMIT:-100000}"
NEXT_TARGET_FACES="${NEXT_TARGET_FACES:-512}"
NEXT_TOKEN_MAX_FACES="${NEXT_TOKEN_MAX_FACES:-512}"
NEXT_MODEL_MAX_FACES="${NEXT_MODEL_MAX_FACES:-512}"
NEXT_MIN_SCALE_DATASET_SAMPLES="${NEXT_MIN_SCALE_DATASET_SAMPLES:-1024}"
MAX_UNCONFIRMED_TARGET="${MAX_UNCONFIRMED_TARGET:-5000}"
ALLOW_LARGE_RUNG="${ALLOW_LARGE_RUNG:-0}"

usage() {
  cat >&2 <<'EOF'
Usage:
  INSPECTION_JSON=/path/face_gate_inspection.json scripts/thunder/promote_face_paper_next_rung.sh
  LAB_ROOT=/path/extracted_lab_root scripts/thunder/promote_face_paper_next_rung.sh

To actually launch:
  VISUAL_REVIEW_PASSED=1 CONFIRM_SCALE=1 INSPECTION_JSON=... scripts/thunder/promote_face_paper_next_rung.sh

This script never launches from metrics alone.
EOF
}

if [ -z "$INSPECTION_JSON" ]; then
  if [ -z "$LAB_ROOT" ]; then
    usage
    exit 2
  fi
  INSPECTION_JSON="$(mktemp -t clearmesh_face_gate_inspection.XXXXXX.json)"
  python3 "$REPO_ROOT/scripts/research/inspect_face_paper_gate.py" "$LAB_ROOT" --output "$INSPECTION_JSON" >/dev/null
fi
if [ ! -f "$INSPECTION_JSON" ]; then
  echo "INSPECTION_JSON not found: $INSPECTION_JSON" >&2
  exit 2
fi

decision_json="$(python3 - "$INSPECTION_JSON" "$NEXT_SELECT_TARGET" "$NEXT_STEPS" "$NEXT_GPU" "$NEXT_MODE" "$NEXT_MIN_SCALE_DATASET_SAMPLES" "$MAX_UNCONFIRMED_TARGET" "$ALLOW_LARGE_RUNG" <<'PY'
import json
import sys
from pathlib import Path

inspection = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
next_select_target = int(sys.argv[2])
next_steps = int(sys.argv[3])
next_gpu = sys.argv[4]
next_mode = sys.argv[5]
next_min_samples = int(sys.argv[6])
max_unconfirmed = int(sys.argv[7])
allow_large = sys.argv[8] == "1"

scale_ready = bool(inspection.get("scale_ready"))
state = inspection.get("state")
next_action = inspection.get("next_action")
blockers = list(inspection.get("blockers") or [])
galleries = dict(inspection.get("galleries") or {})
warnings = list(inspection.get("warnings") or [])
large_rung_blocked = next_select_target > max_unconfirmed and not allow_large

ready = (
    scale_ready
    and state == "ready_for_visual_review"
    and next_action == "promote_next_corpus_rung_after_visual_review"
    and not blockers
    and "train_ar" in galleries
    and "test_ar" in galleries
    and not large_rung_blocked
)
reasons = []
if not scale_ready:
    reasons.append("scale_ready is false")
if state != "ready_for_visual_review":
    reasons.append(f"inspection state is {state!r}, not 'ready_for_visual_review'")
if next_action != "promote_next_corpus_rung_after_visual_review":
    reasons.append(f"next_action is {next_action!r}")
if blockers:
    reasons.extend(blockers[:8])
if "train_ar" not in galleries or "test_ar" not in galleries:
    reasons.append("train/test AR contact sheets are missing")
if large_rung_blocked:
    reasons.append(f"NEXT_SELECT_TARGET={next_select_target} exceeds MAX_UNCONFIRMED_TARGET={max_unconfirmed}; set ALLOW_LARGE_RUNG=1 only after a deliberate scaling review")

payload = {
    "ready_to_launch_after_human_visual_review": ready,
    "inspection_json": sys.argv[1],
    "inspection_state": state,
    "inspection_next_action": next_action,
    "scale_ready": scale_ready,
    "reasons": reasons,
    "warnings": warnings[:8],
    "next_rung": {
        "select_target": next_select_target,
        "steps": next_steps,
        "gpu": next_gpu,
        "mode": next_mode,
        "min_scale_dataset_samples": next_min_samples,
    },
}
print(json.dumps(payload, indent=2, sort_keys=True))
raise SystemExit(0 if ready else 30)
PY
)" || decision_status=$?
decision_status="${decision_status:-0}"
printf '%s\n' "$decision_json"

if [ "$decision_status" -ne 0 ]; then
  if [ "$FAIL_ON_NOT_READY" = "1" ]; then
    exit "$decision_status"
  fi
  exit 0
fi

if [ "$VISUAL_REVIEW_PASSED" != "1" ]; then
  echo "Not launching: set VISUAL_REVIEW_PASSED=1 after inspecting train/test AR contact sheets." >&2
  exit 0
fi
if [ "$CONFIRM_SCALE" != "1" ]; then
  echo "Not launching: set CONFIRM_SCALE=1 to start the next Thunder rung." >&2
  exit 0
fi

SELECT_TARGET="$NEXT_SELECT_TARGET" \
CURATION_TARGET="$NEXT_SELECT_TARGET" \
SCAN_LIMIT="$NEXT_SCAN_LIMIT" \
TARGET_FACES="$NEXT_TARGET_FACES" \
TOKEN_MAX_FACES="$NEXT_TOKEN_MAX_FACES" \
MODEL_MAX_FACES="$NEXT_MODEL_MAX_FACES" \
STEPS="$NEXT_STEPS" \
GPU="$NEXT_GPU" \
MODE="$NEXT_MODE" \
MIN_SCALE_DATASET_SAMPLES="$NEXT_MIN_SCALE_DATASET_SAMPLES" \
"$REPO_ROOT/scripts/thunder/launch_face_paper_corpus_gate_instance.sh"
