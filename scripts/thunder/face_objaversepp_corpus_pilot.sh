#!/usr/bin/env bash
# Build a small, fully gated Objaverse++ -> FACE-token corpus on a Thunder box.
# This is intended to be run on the remote instance via scripts/thunder/run_remote.sh.
set -euo pipefail

RUN_DIR="${RUN_DIR:-/tmp/clearmesh_face_objpp_corpus_pilot}"
SOURCE_KIND="${SOURCE_KIND:-objaversepp}"
ANNOTATIONS="${ANNOTATIONS:-cindyxl/ObjaversePlusPlus}"
SPLIT="${SPLIT:-train}"
SELECT_TARGET="${SELECT_TARGET:-50}"
SCAN_LIMIT="${SCAN_LIMIT:-10000}"
MIN_QUALITY="${MIN_QUALITY:-2}"
OVERSAMPLE_FACTOR="${OVERSAMPLE_FACTOR:-4}"
SEED="${SEED:-23}"
SHUFFLE="${SHUFFLE:-1}"
DOWNLOAD_PROCESSES="${DOWNLOAD_PROCESSES:-8}"
DOWNLOAD_FALLBACK_PROCESSES="${DOWNLOAD_FALLBACK_PROCESSES:-1}"
DOWNLOAD_BATCH_SIZE="${DOWNLOAD_BATCH_SIZE:-25}"
DOWNLOAD_BATCH_TIMEOUT_SECONDS="${DOWNLOAD_BATCH_TIMEOUT_SECONDS:-0}"
DOWNLOAD_BATCH_RETRIES="${DOWNLOAD_BATCH_RETRIES:-2}"
DOWNLOAD_RETRY_SLEEP_SECONDS="${DOWNLOAD_RETRY_SLEEP_SECONDS:-15}"
DOWNLOAD_RATE_LIMIT_SLEEP_SECONDS="${DOWNLOAD_RATE_LIMIT_SLEEP_SECONDS:-300}"
TEXVERSE_DOWNLOAD_WORKERS="${TEXVERSE_DOWNLOAD_WORKERS:-$DOWNLOAD_PROCESSES}"
CURATION_TARGET="${CURATION_TARGET:-$SELECT_TARGET}"
MESH_TIMEOUT_SECONDS="${MESH_TIMEOUT_SECONDS:-30}"
MESH_MEMORY_LIMIT_GB="${MESH_MEMORY_LIMIT_GB:-12}"
SKIP_MESH_REPORT="${SKIP_MESH_REPORT:-1}"
MAX_FILE_MB="${MAX_FILE_MB:-256}"
MAX_COMPONENTS="${MAX_COMPONENTS:-48}"
MIN_LARGEST_COMPONENT_AREA_RATIO="${MIN_LARGEST_COMPONENT_AREA_RATIO:-0.60}"
SOURCE_MIN_FACES="${SOURCE_MIN_FACES:-64}"
SOURCE_MAX_FACES="${SOURCE_MAX_FACES:-250000}"
STRICT_ENGINE="${STRICT_ENGINE:-voxel_shell}"
TARGET_FACES="${TARGET_FACES:-512}"
MAX_TARGET_FACE_RATIO="${MAX_TARGET_FACE_RATIO:-1.25}"
SAMPLE_POINTS="${SAMPLE_POINTS:-20000}"
VOXEL_RESOLUTION="${VOXEL_RESOLUTION:-64}"
MESH_VOXEL_MAX_FACES="${MESH_VOXEL_MAX_FACES:-5000}"
FALLBACK="${FALLBACK:-convex_hull}"
STRICT_TARGET_PROGRESS_EVERY="${STRICT_TARGET_PROGRESS_EVERY:-25}"
TOKEN_MAX_FACES="${TOKEN_MAX_FACES:-1024}"
POINT_SAMPLES="${POINT_SAMPLES:-8192}"
NUM_BINS="${NUM_BINS:-512}"
PAPER_WITHIN_FACE_ORDER="${PAPER_WITHIN_FACE_ORDER:-rotate_min_zyx}"
GATE_PROFILE="${GATE_PROFILE:-strict}"
TOKEN_FAMILY="${TOKEN_FAMILY:-paper}"
FAIL_ON_GATE="${FAIL_ON_GATE:-0}"
PROMOTE_PASSING="${PROMOTE_PASSING:-1}"
SPLIT_PASSING="${SPLIT_PASSING:-1}"
TEST_RATIO="${TEST_RATIO:-0.2}"
TEST_COUNT="${TEST_COUNT:-0}"
ARCHIVE_PATH="${ARCHIVE_PATH:-}"
LEAN_ARCHIVE_PATH="${LEAN_ARCHIVE_PATH:-}"

cd /home/ubuntu/clearmesh

mkdir -p "$RUN_DIR"

DOWNLOAD_ARGS=()
if [ "$SHUFFLE" = "1" ]; then
  DOWNLOAD_ARGS+=(--shuffle)
fi

case "$SOURCE_KIND" in
  objaversepp)
    python scripts/data/download_objaversepp_face_candidates.py \
      --annotations "$ANNOTATIONS" \
      --split "$SPLIT" \
      --output-dir "$RUN_DIR/raw" \
      --target "$SELECT_TARGET" \
      --scan-limit "$SCAN_LIMIT" \
      --min-quality "$MIN_QUALITY" \
      --oversample-factor "$OVERSAMPLE_FACTOR" \
      --seed "$SEED" \
      --download \
      --processes "$DOWNLOAD_PROCESSES" \
      --fallback-processes "$DOWNLOAD_FALLBACK_PROCESSES" \
      --batch-size "$DOWNLOAD_BATCH_SIZE" \
      --batch-timeout-seconds "$DOWNLOAD_BATCH_TIMEOUT_SECONDS" \
      --batch-retries "$DOWNLOAD_BATCH_RETRIES" \
      --retry-sleep-seconds "$DOWNLOAD_RETRY_SLEEP_SECONDS" \
      --rate-limit-sleep-seconds "$DOWNLOAD_RATE_LIMIT_SLEEP_SECONDS" \
      "${DOWNLOAD_ARGS[@]}"
    ;;
  texverse)
    python scripts/data/download_texverse_face_candidates.py \
      --source-manifest "$ANNOTATIONS" \
      --output-dir "$RUN_DIR/raw" \
      --target "$SELECT_TARGET" \
      --scan-limit "$SCAN_LIMIT" \
      --min-quality "$MIN_QUALITY" \
      --seed "$SEED" \
      --workers "$TEXVERSE_DOWNLOAD_WORKERS" \
      --retries "$DOWNLOAD_BATCH_RETRIES" \
      --retry-sleep-seconds "$DOWNLOAD_RETRY_SLEEP_SECONDS" \
      "${DOWNLOAD_ARGS[@]}"
    ;;
  *)
    echo "Unsupported SOURCE_KIND=$SOURCE_KIND; expected objaversepp or texverse." >&2
    exit 2
    ;;
esac

CURATION_ARGS=()
if [ "$SKIP_MESH_REPORT" = "1" ]; then
  CURATION_ARGS+=(--skip-mesh-report)
fi

python scripts/data/build_face_training_corpus.py \
  --candidates "$RUN_DIR/raw/downloaded_candidates.json" \
  --objaversepp-annotations "$RUN_DIR/raw/selected_annotations.jsonl" \
  --output "$RUN_DIR/curated_candidates.jsonl" \
  --rejects-output "$RUN_DIR/curated_rejects.json" \
  --target "$CURATION_TARGET" \
  --min-quality "$MIN_QUALITY" \
  --min-faces "$SOURCE_MIN_FACES" \
  --max-faces "$SOURCE_MAX_FACES" \
  --max-components "$MAX_COMPONENTS" \
  --min-largest-component-area-ratio "$MIN_LARGEST_COMPONENT_AREA_RATIO" \
  --max-file-mb "$MAX_FILE_MB" \
  --mesh-timeout-seconds "$MESH_TIMEOUT_SECONDS" \
  --mesh-memory-limit-gb "$MESH_MEMORY_LIMIT_GB" \
  --progress-every 10 \
  "${CURATION_ARGS[@]}"

python scripts/research/prepare_face_strict_targets.py \
  --manifest "$RUN_DIR/curated_candidates.jsonl" \
  --output-dir "$RUN_DIR/strict_targets" \
  --engine "$STRICT_ENGINE" \
  --target-faces "$TARGET_FACES" \
  --max-target-face-ratio "$MAX_TARGET_FACE_RATIO" \
  --sample-points "$SAMPLE_POINTS" \
  --voxel-resolution "$VOXEL_RESOLUTION" \
  --mesh-voxel-max-faces "$MESH_VOXEL_MAX_FACES" \
  --fallback "$FALLBACK" \
  --progress-every "$STRICT_TARGET_PROGRESS_EVERY"

python scripts/research/build_face_token_dataset.py \
  --manifest "$RUN_DIR/strict_targets/strict_target_manifest.json" \
  --output-dir "$RUN_DIR/tokens" \
  --max-faces "$TOKEN_MAX_FACES" \
  --point-samples "$POINT_SAMPLES" \
  --num-bins "$NUM_BINS" \
  --paper-within-face-order "$PAPER_WITHIN_FACE_ORDER"

GATE_ARGS=()
if [ "$FAIL_ON_GATE" = "1" ]; then
  GATE_ARGS+=(--fail-on-violations)
fi

python scripts/research/check_face_dataset_targets.py \
  --manifest "$RUN_DIR/tokens/manifest.jsonl" \
  --profile "$GATE_PROFILE" \
  --token-family "$TOKEN_FAMILY" \
  --output "$RUN_DIR/strict_gate.json" \
  "${GATE_ARGS[@]}"

if [ "$PROMOTE_PASSING" = "1" ]; then
  python scripts/research/filter_face_dataset_by_gate.py \
    --dataset-dir "$RUN_DIR/tokens" \
    --gate-report "$RUN_DIR/strict_gate.json" \
    --output-dir "$RUN_DIR/tokens_pass"
fi

if [ "$SPLIT_PASSING" = "1" ] && [ -f "$RUN_DIR/tokens_pass/manifest.jsonl" ]; then
  python scripts/research/split_face_token_dataset.py \
    --dataset-dir "$RUN_DIR/tokens_pass" \
    --output-dir "$RUN_DIR/split_pass" \
    --test-ratio "$TEST_RATIO" \
    --test-count "$TEST_COUNT" \
    --seed "$SEED" \
    --shuffle

  python scripts/research/check_face_dataset_targets.py \
    --manifest "$RUN_DIR/split_pass/train/manifest.jsonl" \
    --profile "$GATE_PROFILE" \
    --token-family "$TOKEN_FAMILY" \
    --output "$RUN_DIR/train_strict_gate.json" \
    --fail-on-violations

  python scripts/research/check_face_dataset_targets.py \
    --manifest "$RUN_DIR/split_pass/test/manifest.jsonl" \
    --profile "$GATE_PROFILE" \
    --token-family "$TOKEN_FAMILY" \
    --output "$RUN_DIR/test_strict_gate.json" \
    --fail-on-violations
fi

python - <<PY
import json
from pathlib import Path

run = Path("$RUN_DIR")

def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None

def count_jsonl(path):
    return len([line for line in path.read_text().splitlines() if line.strip()]) if path.exists() else 0

summary = {
    "run_dir": str(run),
    "raw": read_json(run / "raw" / "summary.json"),
    "curated_count": count_jsonl(run / "curated_candidates.jsonl"),
    "curated_rejects": read_json(run / "curated_rejects.json"),
    "strict_targets": read_json(run / "strict_targets" / "strict_target_manifest.json"),
    "token_count": count_jsonl(run / "tokens" / "manifest.jsonl"),
    "gate": read_json(run / "strict_gate.json"),
    "passing_token_count": count_jsonl(run / "tokens_pass" / "manifest.jsonl"),
    "filter": read_json(run / "tokens_pass" / "filter_summary.json"),
    "split": read_json(run / "split_pass" / "split_summary.json"),
    "train_gate": read_json(run / "train_strict_gate.json"),
    "test_gate": read_json(run / "test_strict_gate.json"),
    "settings": {
        "source_kind": "$SOURCE_KIND",
        "select_target": int("$SELECT_TARGET"),
        "scan_limit": int("$SCAN_LIMIT"),
        "target_faces": int("$TARGET_FACES"),
        "num_bins": int("$NUM_BINS"),
        "point_samples": int("$POINT_SAMPLES"),
        "paper_within_face_order": "$PAPER_WITHIN_FACE_ORDER",
        "promote_passing": "$PROMOTE_PASSING" == "1",
        "split_passing": "$SPLIT_PASSING" == "1",
    },
}
if summary["strict_targets"]:
    summary["strict_targets"].pop("records", None)
if summary["curated_rejects"]:
    summary["curated_rejects"]["rejects"] = len(summary["curated_rejects"].get("rejects", []))
for key in ("gate", "train_gate", "test_gate"):
    if summary[key]:
        summary[key].pop("results", None)
(run / "pilot_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
print(json.dumps(summary, indent=2, sort_keys=True))
PY

if [ -n "$ARCHIVE_PATH" ]; then
  tar -czf "$ARCHIVE_PATH" -C "$(dirname "$RUN_DIR")" "$(basename "$RUN_DIR")"
  echo "Archived $RUN_DIR to $ARCHIVE_PATH"
fi

if [ -n "$LEAN_ARCHIVE_PATH" ]; then
  python scripts/research/package_face_corpus.py \
    --data-run "$RUN_DIR" \
    --output "$LEAN_ARCHIVE_PATH" \
    --manifest-output "$LEAN_ARCHIVE_PATH.json"
  echo "Packaged lean corpus archive to $LEAN_ARCHIVE_PATH"
fi
