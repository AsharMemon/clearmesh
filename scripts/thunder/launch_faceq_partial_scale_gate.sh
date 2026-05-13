#!/usr/bin/env bash
# Launch a bounded FACE-Q scale gate from already-uploaded B2 shard progress.
#
# This does not touch active shard workers. It creates/uses one A100, downloads
# current strict target meshes/reports from B2, rebuilds a manifest, tokenizes an
# indexed FACE-Q dataset, trains a bounded gate, and runs held-out free-run eval.
set -euo pipefail

TNR_BIN="${TNR_BIN:-/Users/Ashar/.tnr/bin/tnr}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

INSTANCE_ID="${THUNDER_INSTANCE_ID:-}"
CREATE_INSTANCE="${CREATE_INSTANCE:-1}"
GPU="${GPU:-a100}"
MODE="${MODE:-production}"
VCPUS="${VCPUS:-8}"
PRIMARY_DISK="${PRIMARY_DISK:-300}"
TEMPLATE="${TEMPLATE:-base}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%d_%H%M%S)_faceq_partial_gate}"
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-$REPO_ROOT/.codex_outputs/faceq_partial_gate_setup_$RUN_STAMP}"
REMOTE_REPO="${REMOTE_REPO:-/home/ubuntu/clearmesh}"
REMOTE_VENV="${REMOTE_VENV:-/home/ubuntu/clearmesh-venv}"
REMOTE_LAB_ROOT="${REMOTE_LAB_ROOT:-/tmp/clearmesh_faceq_partial_gate_$RUN_STAMP}"
REMOTE_LOG="${REMOTE_LOG:-/tmp/clearmesh_faceq_partial_gate.nohup.log}"
REMOTE_PID="${REMOTE_PID:-/tmp/clearmesh_faceq_partial_gate.pid}"
REMOTE_B2_ENV="${REMOTE_B2_ENV:-$REMOTE_LAB_ROOT/.clearmesh_b2.env}"

B2_BUCKET="${B2_BUCKET:-clearmesh-pairs}"
B2_PREFIX_ROOT="${B2_PREFIX_ROOT:-face-corpora/poolA-shards}"
B2_RUN_PREFIX="${B2_RUN_PREFIX:-face-runs/faceq-partial-gates/$RUN_STAMP}"
SHARD_IDS="${SHARD_IDS:-0010 0011 0012 0013 0014}"
STRICT_LIMIT="${STRICT_LIMIT:-12000}"
MAX_FACES="${MAX_FACES:-512}"
POINT_SAMPLES="${POINT_SAMPLES:-8192}"
NUM_BINS="${NUM_BINS:-128}"
INDEXED_FACE_ORDER="${INDEXED_FACE_ORDER:-boundary_growth}"
PAPER_WITHIN_FACE_ORDER="${PAPER_WITHIN_FACE_ORDER:-rotate_min_zyx}"
TEST_RATIO="${TEST_RATIO:-0.03}"

STEPS="${STEPS:-30000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
TRAIN_LIMIT="${TRAIN_LIMIT:-0}"
TRAIN_POINT_SAMPLES="${TRAIN_POINT_SAMPLES:-2048}"
HIDDEN_SIZE="${HIDDEN_SIZE:-384}"
LAYERS="${LAYERS:-8}"
HEADS="${HEADS:-8}"
CONDITION_TOKENS="${CONDITION_TOKENS:-128}"
LR="${LR:-6e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
OPTIMIZER="${OPTIMIZER:-muon}"
PRECISION="${PRECISION:-bf16}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-5000}"
LOG_EVERY="${LOG_EVERY:-500}"
EVAL_LIMIT="${EVAL_LIMIT:-32}"
PAIR_SAMPLES="${PAIR_SAMPLES:-2048}"

WAIT_INTERVAL_SEC="${WAIT_INTERVAL_SEC:-10}"
WAIT_TIMEOUT_SEC="${WAIT_TIMEOUT_SEC:-1800}"

if [ -z "${THUNDER_TOKEN:-}" ]; then
  echo "THUNDER_TOKEN is not set." >&2
  exit 1
fi
mkdir -p "$DOWNLOAD_ROOT"

parse_create_id() {
  CREATE_OUTPUT="$1" python3 - <<'PY'
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
}

if [ "$CREATE_INSTANCE" = "1" ]; then
  create_args=(create --gpu "$GPU" --mode "$MODE" --num-gpus 1 --primary-disk "$PRIMARY_DISK" --template "$TEMPLATE" --yes --json)
  if [ "$MODE" = "prototyping" ]; then
    create_args+=(--vcpus "$VCPUS")
  fi
  create_output="$("$TNR_BIN" "${create_args[@]}")"
  printf '%s\n' "$create_output" > "$DOWNLOAD_ROOT/create.json"
  INSTANCE_ID="${INSTANCE_ID:-$(parse_create_id "$create_output")}"
  echo "Created Thunder instance $INSTANCE_ID."
fi
if [ -z "$INSTANCE_ID" ]; then
  echo "Set THUNDER_INSTANCE_ID or CREATE_INSTANCE=1." >&2
  exit 2
fi

echo "Waiting for Thunder instance $INSTANCE_ID to RUNNING..."
deadline=$(( $(date +%s) + WAIT_TIMEOUT_SEC ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  status_json="$("$TNR_BIN" status --json || true)"
  printf '%s\n' "$status_json" > "$DOWNLOAD_ROOT/status.latest.json"
  if python3 - "$INSTANCE_ID" "$DOWNLOAD_ROOT/status.latest.json" <<'PY'
import json
import sys
from pathlib import Path

target = str(sys.argv[1])
text = Path(sys.argv[2]).read_text(errors="ignore")
start = text.find("[")
data = json.loads(text[start:]) if start >= 0 else []
for item in data:
    if str(item.get("id")) == target and item.get("status") == "RUNNING":
        raise SystemExit(0)
raise SystemExit(1)
PY
  then
    break
  fi
  sleep "$WAIT_INTERVAL_SEC"
done

echo "Preflighting GPU..."
cat <<'EOF' | "$TNR_BIN" connect "$INSTANCE_ID" | tee "$DOWNLOAD_ROOT/gpu_preflight.log"
set -euo pipefail
hostname
ls -la /dev/nvidia* || true
compgen -G '/dev/nvidia[0-9]*' >/dev/null
nvidia-smi
echo CLEARMESH_GPU_PREFLIGHT_OK
exit
EOF
grep -q CLEARMESH_GPU_PREFLIGHT_OK "$DOWNLOAD_ROOT/gpu_preflight.log"

echo "Syncing repo and bootstrapping..."
THUNDER_INSTANCE_ID="$INSTANCE_ID" "$REPO_ROOT/scripts/thunder/sync_repo.sh" "$INSTANCE_ID"
THUNDER_INSTANCE_ID="$INSTANCE_ID" INSTALL_MESH_HEAD_REPOS=0 INSTALL_MESH_HEAD_ENVS=0 "$REPO_ROOT/scripts/thunder/bootstrap_remote.sh" "$INSTANCE_ID"

mkdir -p "$DOWNLOAD_ROOT"
b2_env_file="$(mktemp "$DOWNLOAD_ROOT/b2_env.XXXXXX")"
{
  printf 'export B2_KEY_ID=%q\n' "${B2_KEY_ID:-${B2_APPLICATION_KEY_ID:-${B2_KEYID:-${BACKBLAZE_B2_KEY_ID:-}}}}"
  printf 'export B2_APP_KEY=%q\n' "${B2_APP_KEY:-${B2_APPLICATION_KEY:-${BACKBLAZE_B2_APPLICATION_KEY:-${BACKBLAZE_B2_APP_KEY:-}}}}"
  printf 'export B2_TOKEN=%q\n' "${B2_TOKEN:-}"
} > "$b2_env_file"
chmod 600 "$b2_env_file"
printf 'mkdir -p %q\nexit\n' "$REMOTE_LAB_ROOT" | "$TNR_BIN" connect "$INSTANCE_ID" >/dev/null
"$TNR_BIN" scp "$b2_env_file" "$INSTANCE_ID:$REMOTE_B2_ENV"
rm -f "$b2_env_file"

remote_script="$DOWNLOAD_ROOT/remote_faceq_partial_gate.sh"
cat > "$remote_script" <<REMOTE
#!/usr/bin/env bash
set -euo pipefail
LOG_TIME() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }
LAB_ROOT=$(printf '%q' "$REMOTE_LAB_ROOT")
REPO=$(printf '%q' "$REMOTE_REPO")
VENV=$(printf '%q' "$REMOTE_VENV")
B2_ENV=$(printf '%q' "$REMOTE_B2_ENV")
B2_BUCKET=$(printf '%q' "$B2_BUCKET")
B2_PREFIX_ROOT=$(printf '%q' "$B2_PREFIX_ROOT")
B2_RUN_PREFIX=$(printf '%q' "$B2_RUN_PREFIX")
SHARD_IDS=$(printf '%q' "$SHARD_IDS")
STRICT_LIMIT=$(printf '%q' "$STRICT_LIMIT")
MAX_FACES=$(printf '%q' "$MAX_FACES")
POINT_SAMPLES=$(printf '%q' "$POINT_SAMPLES")
NUM_BINS=$(printf '%q' "$NUM_BINS")
INDEXED_FACE_ORDER=$(printf '%q' "$INDEXED_FACE_ORDER")
PAPER_WITHIN_FACE_ORDER=$(printf '%q' "$PAPER_WITHIN_FACE_ORDER")
TEST_RATIO=$(printf '%q' "$TEST_RATIO")
STEPS=$(printf '%q' "$STEPS")
BATCH_SIZE=$(printf '%q' "$BATCH_SIZE")
TRAIN_LIMIT=$(printf '%q' "$TRAIN_LIMIT")
TRAIN_POINT_SAMPLES=$(printf '%q' "$TRAIN_POINT_SAMPLES")
HIDDEN_SIZE=$(printf '%q' "$HIDDEN_SIZE")
LAYERS=$(printf '%q' "$LAYERS")
HEADS=$(printf '%q' "$HEADS")
CONDITION_TOKENS=$(printf '%q' "$CONDITION_TOKENS")
LR=$(printf '%q' "$LR")
WEIGHT_DECAY=$(printf '%q' "$WEIGHT_DECAY")
OPTIMIZER=$(printf '%q' "$OPTIMIZER")
PRECISION=$(printf '%q' "$PRECISION")
CHECKPOINT_EVERY=$(printf '%q' "$CHECKPOINT_EVERY")
LOG_EVERY=$(printf '%q' "$LOG_EVERY")
EVAL_LIMIT=$(printf '%q' "$EVAL_LIMIT")
PAIR_SAMPLES=$(printf '%q' "$PAIR_SAMPLES")

mkdir -p "\$LAB_ROOT"
status_jsonl="\$LAB_ROOT/status.jsonl"
log_status() {
  local step="\$1" state="\$2" detail="\${3:-}"
  python - "\$status_jsonl" "\$step" "\$state" "\$detail" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
item = {
    "time": datetime.now(timezone.utc).isoformat(),
    "step": sys.argv[2],
    "state": sys.argv[3],
    "detail": sys.argv[4],
}
with path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(item, sort_keys=True) + "\\n")
PY
}

cd "\$REPO"
source "\$VENV/bin/activate"
python -m pip install -q -r requirements-data.txt pillow scipy
python - <<'PY' || python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
PY
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available(), "bf16", torch.cuda.is_bf16_supported() if torch.cuda.is_available() else None, "muon", hasattr(torch.optim, "Muon"))
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable")
PY
if ! command -v rclone >/dev/null 2>&1; then
  curl -fsSL https://rclone.org/install.sh | sudo bash
fi

source "\$B2_ENV"
if [[ -z "\${B2_KEY_ID:-}" || -z "\${B2_APP_KEY:-}" ]] && [[ -n "\${B2_TOKEN:-}" ]]; then
  parsed_b2="\$(python - <<'PY'
import json, os, sys
token = os.environ.get("B2_TOKEN", "").strip()
key_id = app_key = ""
if token.startswith("{"):
    payload = json.loads(token)
    key_id = (
        payload.get("keyId")
        or payload.get("keyID")
        or payload.get("applicationKeyId")
        or payload.get("applicationKeyID")
        or payload.get("key_id")
        or payload.get("application_key_id")
        or payload.get("accountId")
        or payload.get("accountID")
        or ""
    )
    app_key = (
        payload.get("applicationKey")
        or payload.get("application_key")
        or payload.get("appKey")
        or payload.get("app_key")
        or payload.get("key")
        or ""
    )
elif ":" in token:
    key_id, app_key = token.split(":", 1)
if key_id and app_key:
    sys.stdout.write(key_id + "\\n" + app_key)
PY
)"
  if [[ -n "\$parsed_b2" ]]; then
    export B2_KEY_ID="\${B2_KEY_ID:-\$(printf '%s\\n' "\$parsed_b2" | sed -n '1p')}"
    export B2_APP_KEY="\${B2_APP_KEY:-\$(printf '%s\\n' "\$parsed_b2" | sed -n '2p')}"
  fi
fi
if [[ -n "\${B2_KEY_ID:-}" && -z "\${B2_APP_KEY:-}" && -n "\${B2_TOKEN:-}" ]]; then
  # Some environments expose the application key as B2_TOKEN and the key id
  # separately. Treat opaque, non-JSON, non-pair tokens as the app key.
  case "\$B2_TOKEN" in
    \{*|*:*) ;;
    *) export B2_APP_KEY="\$B2_TOKEN" ;;
  esac
fi
if [[ -z "\${B2_KEY_ID:-}" || -z "\${B2_APP_KEY:-}" ]]; then
  echo "B2 credentials are incomplete: set B2_KEY_ID plus B2_APP_KEY, or B2_TOKEN as JSON/key_id:application_key." >&2
  exit 2
fi
export RCLONE_CONFIG_B2ENV_TYPE=b2
export RCLONE_CONFIG_B2ENV_ACCOUNT="\$B2_KEY_ID"
export RCLONE_CONFIG_B2ENV_KEY="\$B2_APP_KEY"
rclone lsd "b2env:\$B2_BUCKET" >/dev/null

MODE=face_run LOCAL_ROOT="\$LAB_ROOT" B2_BUCKET="\$B2_BUCKET" B2_PREFIX="\$B2_RUN_PREFIX" INTERVAL_SECONDS=300 \\
  B2_KEY_ID="\$B2_KEY_ID" B2_APP_KEY="\$B2_APP_KEY" B2_TOKEN="\${B2_TOKEN:-}" \\
  nohup bash scripts/thunder/b2_continuous_upload.sh > "\$LAB_ROOT/b2_upload.log" 2>&1 &
echo \$! > "\$LAB_ROOT/b2_upload.pid"

log_status b2_download started "shards=\$SHARD_IDS"
for shard in \$SHARD_IDS; do
  mkdir -p "\$LAB_ROOT/b2_shards/shard\$shard"
  rclone copy "b2env:\$B2_BUCKET/\$B2_PREFIX_ROOT/shard\$shard/corpus/strict_targets" \\
    "\$LAB_ROOT/b2_shards/shard\$shard/strict_targets" \\
    --transfers 16 --checkers 32 --stats 30s
done
log_status b2_download complete

strict_args=()
for shard in \$SHARD_IDS; do
  strict_args+=(--strict-root "\$LAB_ROOT/b2_shards/shard\$shard/strict_targets")
done
python scripts/research/build_strict_manifest_from_reports.py \\
  "\${strict_args[@]}" \\
  --output "\$LAB_ROOT/strict_targets/strict_target_manifest.json" \\
  --limit "\$STRICT_LIMIT" | tee "\$LAB_ROOT/build_strict_manifest.log"
log_status manifest complete

python scripts/research/build_face_token_dataset.py \\
  --manifest "\$LAB_ROOT/strict_targets/strict_target_manifest.json" \\
  --output-dir "\$LAB_ROOT/tokens" \\
  --max-faces "\$MAX_FACES" \\
  --point-samples "\$POINT_SAMPLES" \\
  --num-bins "\$NUM_BINS" \\
  --paper-within-face-order "\$PAPER_WITHIN_FACE_ORDER" \\
  --indexed-face-order "\$INDEXED_FACE_ORDER" | tee "\$LAB_ROOT/build_tokens.log"
log_status tokenize complete

python scripts/research/split_face_token_dataset.py \\
  --dataset-dir "\$LAB_ROOT/tokens" \\
  --output-dir "\$LAB_ROOT/split" \\
  --test-ratio "\$TEST_RATIO" \\
  --seed 303 \\
  --shuffle | tee "\$LAB_ROOT/split.log"

python scripts/research/check_face_token_leakage.py \\
  --train-dir "\$LAB_ROOT/split/train" \\
  --test-dir "\$LAB_ROOT/split/test" \\
  --warn-only \\
  --output "\$LAB_ROOT/token_leakage.json" | tee "\$LAB_ROOT/token_leakage.log"

python - <<'PY'
import json
from pathlib import Path

root = Path("${REMOTE_LAB_ROOT}")
rows = [json.loads(line) for line in (root / "tokens" / "manifest.jsonl").read_text().splitlines() if line.strip()]
summary = {
    "written": len(rows),
    "failed": 0,
    "max_faces": max((int(row.get("face_count", row.get("faces", 0)) or 0) for row in rows), default=0),
    "max_vertices": max((int(row.get("indexed_vertices", 0) or 0) for row in rows), default=0),
    "mean_faces": sum(float(row.get("face_count", row.get("faces", 0)) or 0) for row in rows) / max(len(rows), 1),
    "zero_closure_after_first_sum": sum(int(row.get("indexed_zero_closure_after_first", 0) or 0) for row in rows),
}
(root / "tokens" / "curation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
print(json.dumps(summary, indent=2, sort_keys=True))
PY
log_status split complete

mkdir -p "\$LAB_ROOT/runs/faceq_partial_gate"
python scripts/research/train_face_indexed_conditioned_tiny.py \\
  --dataset-dir "\$LAB_ROOT/split/train" \\
  --output "\$LAB_ROOT/runs/faceq_partial_gate/checkpoint.pt" \\
  --steps "\$STEPS" \\
  --batch-size "\$BATCH_SIZE" \\
  --limit "\$TRAIN_LIMIT" \\
  --point-samples "\$TRAIN_POINT_SAMPLES" \\
  --hidden-size "\$HIDDEN_SIZE" \\
  --layers "\$LAYERS" \\
  --heads "\$HEADS" \\
  --condition-tokens "\$CONDITION_TOKENS" \\
  --corner-head causal \\
  --topology-loss-weight 0.2 \\
  --edge-action-loss-weight 1.0 \\
  --edge-choice-loss-weight 0.5 \\
  --edge-choice-candidates 32 \\
  --seed-face-loss-weight 1.0 \\
  --early-face-count 16 \\
  --early-face-loss-weight 4.0 \\
  --optimizer "\$OPTIMIZER" \\
  --precision "\$PRECISION" \\
  --lr "\$LR" \\
  --weight-decay "\$WEIGHT_DECAY" \\
  --checkpoint-every "\$CHECKPOINT_EVERY" \\
  --save-current-checkpoint \\
  --log-every "\$LOG_EVERY" \\
  --device cuda | tee "\$LAB_ROOT/runs/faceq_partial_gate/train.log"
log_status train complete

COMMON_EVAL=(--checkpoint "\$LAB_ROOT/runs/faceq_partial_gate/checkpoint.pt" --point-samples "\$TRAIN_POINT_SAMPLES" --face-count-mode gt --pair-samples "\$PAIR_SAMPLES" --decode-mode boundary_edge --corner-decode causal --constraint-top-k 6 --local-candidate-neighbors 2 --closure-bonus 2.0 --new-edge-penalty 0.05 --edge-length-penalty 0.1 --aspect-penalty 0.05 --edge-action-bonus 1.0 --edge-action-candidate-top-k 8 --edge-choice-bonus 0.5 --edge-choice-candidate-top-k 8 --require-boundary-closure-after 1 --closure-target-bonus 1.0 --boundary-budget-constraint --vertex-link-constraint --boundary-fill centroid --boundary-fill-max-loop-edges 128 --device cuda --limit "\$EVAL_LIMIT")

python scripts/research/eval_face_indexed_conditioned_tiny.py \\
  "\${COMMON_EVAL[@]}" \\
  --dataset-dir "\$LAB_ROOT/split/train" \\
  --output "\$LAB_ROOT/runs/faceq_partial_gate/eval_train_free.json" \\
  --export-dir "\$LAB_ROOT/runs/faceq_partial_gate/meshes_train_free" \\
  --decode-strategy free_run | tee "\$LAB_ROOT/runs/faceq_partial_gate/eval_train_free.log"

python scripts/research/eval_face_indexed_conditioned_tiny.py \\
  "\${COMMON_EVAL[@]}" \\
  --dataset-dir "\$LAB_ROOT/split/test" \\
  --output "\$LAB_ROOT/runs/faceq_partial_gate/eval_test_free.json" \\
  --export-dir "\$LAB_ROOT/runs/faceq_partial_gate/meshes_test_free" \\
  --decode-strategy free_run | tee "\$LAB_ROOT/runs/faceq_partial_gate/eval_test_free.log"

python scripts/research/eval_face_indexed_conditioned_tiny.py \\
  "\${COMMON_EVAL[@]}" \\
  --dataset-dir "\$LAB_ROOT/split/test" \\
  --output "\$LAB_ROOT/runs/faceq_partial_gate/eval_test_teacher.json" \\
  --export-dir "\$LAB_ROOT/runs/faceq_partial_gate/meshes_test_teacher" \\
  --decode-strategy teacher_forced | tee "\$LAB_ROOT/runs/faceq_partial_gate/eval_test_teacher.log"

python scripts/research/assess_face_indexed_scale_readiness.py \\
  --curation-summary "\$LAB_ROOT/tokens/curation_summary.json" \\
  --teacher-eval "\$LAB_ROOT/runs/faceq_partial_gate/eval_test_teacher.json" \\
  --free-run-eval "\$LAB_ROOT/runs/faceq_partial_gate/eval_test_free.json" \\
  --output "\$LAB_ROOT/runs/faceq_partial_gate/scale_readiness.json" \\
  --min-dataset-samples 10000 \\
  --min-eval-samples "\$EVAL_LIMIT" \\
  --teacher-gate-mode generalization || true
log_status eval complete

tar -czf "\$LAB_ROOT/faceq_partial_gate_metadata.tar.gz" -C "\$LAB_ROOT" status.jsonl build_strict_manifest.log build_tokens.log split.log token_leakage.json tokens/curation_summary.json runs/faceq_partial_gate
log_status complete "archive=\$LAB_ROOT/faceq_partial_gate_metadata.tar.gz"
REMOTE

"$TNR_BIN" scp "$remote_script" "$INSTANCE_ID:/tmp/clearmesh_faceq_partial_gate_run.sh"
cat <<REMOTE_LAUNCH | "$TNR_BIN" connect "$INSTANCE_ID" | tee "$DOWNLOAD_ROOT/remote_launch.log"
set -euo pipefail
chmod +x /tmp/clearmesh_faceq_partial_gate_run.sh
nohup /tmp/clearmesh_faceq_partial_gate_run.sh > "$REMOTE_LOG" 2>&1 &
echo \$! > "$REMOTE_PID"
echo CLEARMESH_FACEQ_PARTIAL_GATE_LAUNCHED pid=\$(cat "$REMOTE_PID") root="$REMOTE_LAB_ROOT" log="$REMOTE_LOG"
exit
REMOTE_LAUNCH

cat > "$DOWNLOAD_ROOT/run_info.json" <<JSON
{
  "instance_id": "$INSTANCE_ID",
  "gpu": "$GPU",
  "mode": "$MODE",
  "remote_lab_root": "$REMOTE_LAB_ROOT",
  "remote_log": "$REMOTE_LOG",
  "remote_pid": "$REMOTE_PID",
  "download_root": "$DOWNLOAD_ROOT",
  "b2_bucket": "$B2_BUCKET",
  "b2_prefix_root": "$B2_PREFIX_ROOT",
  "b2_run_prefix": "$B2_RUN_PREFIX",
  "shard_ids": "$SHARD_IDS",
  "strict_limit": $STRICT_LIMIT,
  "steps": $STEPS,
  "batch_size": $BATCH_SIZE,
  "hidden_size": $HIDDEN_SIZE,
  "layers": $LAYERS,
  "heads": $HEADS,
  "condition_tokens": $CONDITION_TOKENS,
  "train_point_samples": $TRAIN_POINT_SAMPLES,
  "eval_limit": $EVAL_LIMIT
}
JSON

echo "FACE-Q partial scale gate launched."
echo "Run info: $DOWNLOAD_ROOT/run_info.json"
