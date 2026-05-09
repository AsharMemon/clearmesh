#!/usr/bin/env bash
set -euo pipefail

INSTANCE_ID="${1:-${THUNDER_INSTANCE_ID:-0}}"
REMOTE_DIR="${REMOTE_DIR:-/home/ubuntu/clearmesh}"
REMOTE_VENV="${REMOTE_VENV:-/home/ubuntu/clearmesh-venv}"
TNR_BIN="${TNR_BIN:-/Users/Ashar/.tnr/bin/tnr}"
LOCAL_FIXTURE_DIR="${LOCAL_FIXTURE_DIR:-}"
RUN_DIR="${RUN_DIR:-/tmp/clearmesh_face_level_proxy_fixture_smoke}"
FIXTURE_TEST_COUNT="${FIXTURE_TEST_COUNT:-4}"
FIXTURE_SPLIT_SEED="${FIXTURE_SPLIT_SEED:-17}"
SYNTHETIC_COUNT="${SYNTHETIC_COUNT:-64}"
SYNTHETIC_KIND="${SYNTHETIC_KIND:-mixed}"
SYNTHETIC_SEED="${SYNTHETIC_SEED:-123}"
MAX_FACES="${MAX_FACES:-1024}"
POINT_SAMPLES="${POINT_SAMPLES:-1024}"
TRAIN_POINT_SAMPLES="${TRAIN_POINT_SAMPLES:-512}"
STEPS="${STEPS:-3500}"
BATCH_SIZE="${BATCH_SIZE:-4}"
HIDDEN_SIZE="${HIDDEN_SIZE:-192}"
LAYERS="${LAYERS:-3}"
HEADS="${HEADS:-6}"
CONDITION_TOKENS="${CONDITION_TOKENS:-8}"
LR="${LR:-5e-4}"
REUSE_VERTEX_LOSS_WEIGHT="${REUSE_VERTEX_LOSS_WEIGHT:-0.5}"
EDGE_CLOSURE_LOSS_WEIGHT="${EDGE_CLOSURE_LOSS_WEIGHT:-1.0}"
TOPOLOGY_AUX_LOSS_WEIGHT="${TOPOLOGY_AUX_LOSS_WEIGHT:-0.0}"
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
if [ -z "$LOCAL_FIXTURE_DIR" ]; then
  echo "LOCAL_FIXTURE_DIR is required. Point it at a directory containing prepared fixture meshes." >&2
  exit 1
fi
if [ ! -d "$LOCAL_FIXTURE_DIR" ]; then
  echo "LOCAL_FIXTURE_DIR does not exist: $LOCAL_FIXTURE_DIR" >&2
  exit 1
fi

FIXTURE_ARCHIVE="$(mktemp -t clearmesh-face-fixtures.XXXXXX.tar.gz)"
trap 'rm -f "$FIXTURE_ARCHIVE"' EXIT
COPYFILE_DISABLE=1 tar --no-xattrs --exclude='.DS_Store' --exclude='._*' --exclude='__MACOSX' -czf "$FIXTURE_ARCHIVE" -C "$LOCAL_FIXTURE_DIR" .

"$TNR_BIN" scp "$FIXTURE_ARCHIVE" "$INSTANCE_ID:/tmp/clearmesh_face_proxy_fixtures.tar.gz"

cat <<EOF | "$TNR_BIN" connect "$INSTANCE_ID"
set -euo pipefail
cd "$REMOTE_DIR"
. "$REMOTE_VENV/bin/activate"
python - <<'PY' || python -m pip install torch==2.8.0 --index-url "$PYTORCH_INDEX_URL"
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
PY
rm -rf "$RUN_DIR"
mkdir -p "$RUN_DIR/fixtures_uploaded" "$RUN_DIR/fixture_train_meshes" "$RUN_DIR/fixture_test_meshes"
tar --exclude="._*" --exclude="__MACOSX" -xzf /tmp/clearmesh_face_proxy_fixtures.tar.gz -C "$RUN_DIR/fixtures_uploaded"
python - <<'PY'
import json
import random
import shutil
from pathlib import Path

run = Path("$RUN_DIR")
uploaded = run / "fixtures_uploaded"
mesh_root = uploaded / "meshes" if (uploaded / "meshes").is_dir() else uploaded
paths = sorted(p for p in mesh_root.rglob("*") if p.suffix.lower() in {".glb", ".gltf", ".obj", ".ply", ".stl"})
if len(paths) < 2:
    raise SystemExit(f"Need at least 2 fixture meshes, found {len(paths)} in {mesh_root}")
rng = random.Random(int("$FIXTURE_SPLIT_SEED"))
rng.shuffle(paths)
test_count = max(1, min(int("$FIXTURE_TEST_COUNT"), len(paths) - 1))
test = sorted(paths[:test_count])
train = sorted(paths[test_count:])
for split, split_paths in [("train", train), ("test", test)]:
    out_dir = run / f"fixture_{split}_meshes"
    for idx, src in enumerate(split_paths):
        dst = out_dir / f"{idx:04d}_{src.name}"
        shutil.copy2(src, dst)
summary = {
    "uploaded_meshes": len(paths),
    "train_meshes": [str(p) for p in train],
    "test_meshes": [str(p) for p in test],
}
(run / "fixture_split.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
print(json.dumps(summary, indent=2, sort_keys=True))
PY
python scripts/research/build_face_token_dataset.py \
  --mesh-dir "$RUN_DIR/fixture_train_meshes" \
  --output-dir "$RUN_DIR/train_dataset" \
  --synthetic-count "$SYNTHETIC_COUNT" \
  --synthetic-kind "$SYNTHETIC_KIND" \
  --max-faces "$MAX_FACES" \
  --point-samples "$POINT_SAMPLES" \
  --seed "$SYNTHETIC_SEED"
python scripts/research/build_face_token_dataset.py \
  --mesh-dir "$RUN_DIR/fixture_test_meshes" \
  --output-dir "$RUN_DIR/test_dataset" \
  --synthetic-count 0 \
  --max-faces "$MAX_FACES" \
  --point-samples "$POINT_SAMPLES" \
  --seed "$FIXTURE_SPLIT_SEED"
python scripts/research/check_face_dataset_targets.py \
  --manifest "$RUN_DIR/train_dataset/manifest.jsonl" \
  --output "$RUN_DIR/train_dataset_strict_gate.json" \
  --profile strict || true
python scripts/research/check_face_dataset_targets.py \
  --manifest "$RUN_DIR/test_dataset/manifest.jsonl" \
  --output "$RUN_DIR/test_dataset_strict_gate.json" \
  --profile strict || true
python scripts/research/check_face_dataset_targets.py \
  --manifest "$RUN_DIR/test_dataset/manifest.jsonl" \
  --output "$RUN_DIR/test_dataset_proxy_gate.json" \
  --profile proxy
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
  --output "$RUN_DIR/face_level_proxy_fixture.pt"
CLEANUP_ARGS=()
if [ "$CLEANUP_FILL_HOLES" = "true" ]; then
  CLEANUP_ARGS+=(--cleanup-fill-holes)
fi
python scripts/research/eval_face_level_conditioned_tiny.py \
  --checkpoint "$RUN_DIR/face_level_proxy_fixture.pt" \
  --dataset-dir "$RUN_DIR/test_dataset" \
  --output "$RUN_DIR/eval_fixture_gt_count_report.json" \
  --export-dir "$RUN_DIR/eval_fixture_gt_count_meshes" \
  --point-samples "$TRAIN_POINT_SAMPLES" \
  --pair-samples "$PAIR_SAMPLES" \
  --face-count-mode gt
python scripts/research/eval_face_level_conditioned_tiny.py \
  --checkpoint "$RUN_DIR/face_level_proxy_fixture.pt" \
  --dataset-dir "$RUN_DIR/test_dataset" \
  --output "$RUN_DIR/eval_fixture_predicted_count_report.json" \
  --export-dir "$RUN_DIR/eval_fixture_predicted_count_meshes" \
  --cleanup-export-dir "$RUN_DIR/eval_fixture_predicted_count_cleanup" \
  --point-samples "$TRAIN_POINT_SAMPLES" \
  --pair-samples "$PAIR_SAMPLES" \
  --face-count-mode predicted \
  --cleanup-min-component-faces "$CLEANUP_MIN_COMPONENT_FACES" \
  "\${CLEANUP_ARGS[@]}"
python scripts/research/eval_face_level_conditioned_tiny.py \
  --checkpoint "$RUN_DIR/face_level_proxy_fixture.pt" \
  --dataset-dir "$RUN_DIR/test_dataset" \
  --output "$RUN_DIR/eval_fixture_predicted_count_token_repair_report.json" \
  --export-dir "$RUN_DIR/eval_fixture_predicted_count_token_repair_meshes" \
  --cleanup-export-dir "$RUN_DIR/eval_fixture_predicted_count_token_repair_cleanup" \
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
gt = json.loads((run / "eval_fixture_gt_count_report.json").read_text())
pred = json.loads((run / "eval_fixture_predicted_count_report.json").read_text())
repair = json.loads((run / "eval_fixture_predicted_count_token_repair_report.json").read_text())
summary = {
    "run_dir": str(run),
    "fixture_split": json.loads((run / "fixture_split.json").read_text()),
    "train_dataset_strict_gate": json.loads((run / "train_dataset_strict_gate.json").read_text()),
    "test_dataset_strict_gate": json.loads((run / "test_dataset_strict_gate.json").read_text()),
    "test_dataset_proxy_gate": json.loads((run / "test_dataset_proxy_gate.json").read_text()),
    "train_dataset": [json.loads(line) for line in (run / "train_dataset" / "manifest.jsonl").read_text().splitlines()],
    "test_dataset": [json.loads(line) for line in (run / "test_dataset" / "manifest.jsonl").read_text().splitlines()],
    "gt_count": gt["summary"],
    "predicted_count": pred["summary"],
    "predicted_count_cleanup": pred.get("cleanup_summary"),
    "predicted_count_token_repair": repair["summary"],
    "predicted_count_token_repair_cleanup": repair.get("cleanup_summary"),
    "predicted_count_token_repair_topology": repair.get("token_summary", {}).get("decoded_generated"),
}
(run / "proxy_fixture_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
print(json.dumps(summary, indent=2, sort_keys=True))
PY
ls -lh "$RUN_DIR"/face_level_proxy_fixture.pt "$RUN_DIR"/*_report.json "$RUN_DIR"/proxy_fixture_summary.json
exit
EOF

if [ -n "$DOWNLOAD_DIR" ]; then
  mkdir -p "$DOWNLOAD_DIR"
  REMOTE_ARCHIVE="/tmp/$(basename "$RUN_DIR")_reports.tar.gz"
  cat <<EOF | "$TNR_BIN" connect "$INSTANCE_ID"
set -euo pipefail
tar -czf "$REMOTE_ARCHIVE" -C "$RUN_DIR" \
  face_level_proxy_fixture.pt \
  fixture_split.json \
  proxy_fixture_summary.json \
  train_dataset_strict_gate.json \
  test_dataset_strict_gate.json \
  test_dataset_proxy_gate.json \
  eval_fixture_gt_count_report.json \
  eval_fixture_predicted_count_report.json \
  eval_fixture_predicted_count_token_repair_report.json \
  eval_fixture_gt_count_meshes \
  eval_fixture_predicted_count_meshes \
  eval_fixture_predicted_count_cleanup \
  eval_fixture_predicted_count_token_repair_meshes \
  eval_fixture_predicted_count_token_repair_cleanup
ls -lh "$REMOTE_ARCHIVE"
exit
EOF
  "$TNR_BIN" scp "$INSTANCE_ID:$REMOTE_ARCHIVE" "$DOWNLOAD_DIR/$(basename "$REMOTE_ARCHIVE")"
  tar -xzf "$DOWNLOAD_DIR/$(basename "$REMOTE_ARCHIVE")" -C "$DOWNLOAD_DIR"
fi
