#!/usr/bin/env bash
# Bootstrap a RunCrate instance and launch the FACE-Q 8x FSDP smoke.
#
# Inputs are read from the RunCrate API/launch response by default. Secrets are
# read from local 0600 env files and copied only to the remote node.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="${OUT_DIR:-.codex_outputs/runcrate_20260517}"
LAUNCH_RESPONSE="${LAUNCH_RESPONSE:-$OUT_DIR/create_faceq_smoke_response_latest.json}"
RUNCRATE_ENV_FILE="${RUNCRATE_ENV_FILE:-.codex_secrets/runcrate.env}"
B2_ENV_FILE="${B2_ENV_FILE:-.codex_secrets/b2.env}"
SSH_KEY_FILE="${SSH_KEY_FILE:-$HOME/.ssh/id_ed25519}"
REMOTE_USER="${REMOTE_USER:-root}"
REMOTE_BASE="${REMOTE_BASE:-/workspace}"
REMOTE_REPO="${REMOTE_REPO:-$REMOTE_BASE/clearmesh}"
REMOTE_VENV="${REMOTE_VENV:-$REMOTE_BASE/clearmesh-venv}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%d_faceq_1p4b_fsdp_runcrate_smoke)}"
REMOTE_LAB_ROOT="${REMOTE_LAB_ROOT:-$REMOTE_BASE/clearmesh_faceq_runcrate_scale_train_$RUN_STAMP}"
REMOTE_LOG="${REMOTE_LOG:-$REMOTE_LAB_ROOT/clearmesh_faceq_runcrate_scale_train.nohup.log}"
REMOTE_PID="${REMOTE_PID:-$REMOTE_LAB_ROOT/clearmesh_faceq_runcrate_scale_train.pid}"
REMOTE_B2_ENV="${REMOTE_B2_ENV:-$REMOTE_LAB_ROOT/.clearmesh_b2.env}"
SYNC_REPO="${SYNC_REPO:-1}"

B2_BUCKET="${B2_BUCKET:-clearmesh-pairs}"
B2_CORPUS_PREFIX="${B2_CORPUS_PREFIX:-face-corpora/merged/faceq_rolling_400k_total_380k_train/snapshot_20260517T064949Z_archives658}"
B2_CORPUS_ARCHIVE="${B2_CORPUS_ARCHIVE:-faceq_merged_dedup_split.tar.gz}"
B2_RUN_PREFIX="${B2_RUN_PREFIX:-face-runs/faceq-1p4b-fsdp-smoke/runcrate-$RUN_STAMP}"
ARCHIVE_DATASET_DIR="${ARCHIVE_DATASET_DIR:-split_dedup_tokenhash/train}"
REMOVE_CORPUS_ARCHIVE_AFTER_EXTRACT="${REMOVE_CORPUS_ARCHIVE_AFTER_EXTRACT:-1}"

STEPS="${STEPS:-20}"
BATCH_SIZE="${BATCH_SIZE:-1}"
TRAIN_LIMIT="${TRAIN_LIMIT:-8192}"
TRAIN_POINT_SAMPLES="${TRAIN_POINT_SAMPLES:-65536}"
HIDDEN_SIZE="${HIDDEN_SIZE:-1536}"
LAYERS="${LAYERS:-32}"
HEADS="${HEADS:-16}"
CONDITION_TOKENS="${CONDITION_TOKENS:-2048}"
CONDITION_BACKEND="${CONDITION_BACKEND:-vecset}"
DECODER_BACKEND="${DECODER_BACKEND:-cross_attn}"
ENCODER_LAYERS="${ENCODER_LAYERS:-8}"
LATENT_DIM="${LATENT_DIM:-64}"
FACE_OUTPUT_MODE="${FACE_OUTPUT_MODE:-geometry}"
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
OPTIMIZER="${OPTIMIZER:-muon}"
PRECISION="${PRECISION:-bf16}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-0}"
SKIP_FINAL_CHECKPOINT="${SKIP_FINAL_CHECKPOINT:-1}"
LOG_EVERY="${LOG_EVERY:-1}"
TORCHRUN_NPROC_PER_NODE="${TORCHRUN_NPROC_PER_NODE:-8}"
DISTRIBUTED="${DISTRIBUTED:-auto}"
DISTRIBUTED_BACKEND="${DISTRIBUTED_BACKEND:-nccl}"
DISTRIBUTED_STRATEGY="${DISTRIBUTED_STRATEGY:-fsdp}"
FSDP_MIN_NUM_PARAMS="${FSDP_MIN_NUM_PARAMS:-20000000}"
LAZY_LOAD="${LAZY_LOAD:-1}"
SAMPLE_CACHE_SIZE="${SAMPLE_CACHE_SIZE:-8}"
TRACK_BEST_IN_MEMORY="${TRACK_BEST_IN_MEMORY:-0}"
COUNT_LOSS_WEIGHT="${COUNT_LOSS_WEIGHT:-0.05}"
TOPOLOGY_LOSS_WEIGHT="${TOPOLOGY_LOSS_WEIGHT:-0.2}"
EDGE_ACTION_LOSS_WEIGHT="${EDGE_ACTION_LOSS_WEIGHT:-1.0}"
EDGE_CHOICE_LOSS_WEIGHT="${EDGE_CHOICE_LOSS_WEIGHT:-0.5}"
SEED_FACE_LOSS_WEIGHT="${SEED_FACE_LOSS_WEIGHT:-1.0}"
EARLY_FACE_COUNT="${EARLY_FACE_COUNT:-16}"
EARLY_FACE_LOSS_WEIGHT="${EARLY_FACE_LOSS_WEIGHT:-4.0}"

mkdir -p "$OUT_DIR"
BOOTSTRAP_STARTED="$OUT_DIR/runcrate_bootstrap_started.flag"
BOOTSTRAP_DONE="$OUT_DIR/runcrate_bootstrap_done.flag"
LOCAL_LOG="$OUT_DIR/runcrate_bootstrap.log"

if [[ ! -f "$LAUNCH_RESPONSE" ]]; then
  echo "Missing RunCrate launch response: $LAUNCH_RESPONSE" >&2
  exit 2
fi
if [[ ! -f "$RUNCRATE_ENV_FILE" ]]; then
  echo "Missing RunCrate env file: $RUNCRATE_ENV_FILE" >&2
  exit 2
fi
if [[ ! -f "$B2_ENV_FILE" ]]; then
  echo "Missing B2 env file: $B2_ENV_FILE" >&2
  exit 2
fi
if [[ ! -f "$SSH_KEY_FILE" ]]; then
  echo "Missing SSH key file: $SSH_KEY_FILE" >&2
  exit 2
fi

touch "$BOOTSTRAP_STARTED"
chmod 600 "$BOOTSTRAP_STARTED"
source "$RUNCRATE_ENV_FILE"

API_BASE="${RUNCRATE_API_BASE:-https://www.runcrate.ai/api/v1}"
INSTANCE_ID="${RUNCRATE_INSTANCE_ID:-$(python3 - "$LAUNCH_RESPONSE" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
data = payload.get("data", payload)
print(data.get("id") or data.get("instance_id") or "")
PY
)}"
if [[ -z "$INSTANCE_ID" ]]; then
  echo "Could not parse RunCrate instance id from launch response." >&2
  exit 2
fi

runcrate_get_instances() {
  curl -sS --max-time 45 \
    -H "Authorization: Bearer $RUNCRATE_API_KEY" \
    -H "Accept: application/json" \
    "$API_BASE/instances"
}

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Waiting for RunCrate instance $INSTANCE_ID..." | tee -a "$LOCAL_LOG"
INSTANCE_JSON="$OUT_DIR/runcrate_instance.latest.json"
deadline=$(( $(date +%s) + 1800 ))
REMOTE_HOST=""
REMOTE_PORT="22"
while [[ "$(date +%s)" -lt "$deadline" ]]; do
  if ! runcrate_get_instances > "$INSTANCE_JSON.tmp"; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] RunCrate instance list probe failed; retrying..." | tee -a "$LOCAL_LOG"
    sleep 15
    continue
  fi
  mv "$INSTANCE_JSON.tmp" "$INSTANCE_JSON"
  parse_target="$OUT_DIR/runcrate_instance_target.txt"
  python3 - "$INSTANCE_ID" "$INSTANCE_JSON" > "$parse_target" <<'PY'
import json
import sys

target = sys.argv[1]
payload = json.load(open(sys.argv[2], encoding="utf-8"))
items = payload.get("data", payload if isinstance(payload, list) else [])
for item in items:
    if str(item.get("id")) == target:
        status = (item.get("status") or "").lower()
        ip = item.get("ip") or item.get("public_ip") or item.get("host") or ""
        port = str(item.get("ssh_port") or item.get("port") or 22)
        if status in {"active", "running", "booted", "deployed", "ready"} and ip:
            print(ip, port)
        else:
            print("-", port)
        raise SystemExit
print("-", "22")
PY
  read -r REMOTE_HOST REMOTE_PORT < "$parse_target" || true
  REMOTE_PORT="${REMOTE_PORT:-22}"
  if [[ -n "$REMOTE_HOST" && "$REMOTE_HOST" != "-" ]]; then
    break
  fi
  sleep 15
done
if [[ -z "$REMOTE_HOST" ]]; then
  echo "RunCrate instance did not become SSH-ready before timeout." >&2
  exit 3
fi

SSH_OPTS=(
  -i "$SSH_KEY_FILE"
  -o StrictHostKeyChecking=accept-new
  -o UserKnownHostsFile="$HOME/.ssh/known_hosts"
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=4
  -p "$REMOTE_PORT"
)
SCP_OPTS=(
  -i "$SSH_KEY_FILE"
  -o StrictHostKeyChecking=accept-new
  -o UserKnownHostsFile="$HOME/.ssh/known_hosts"
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=4
  -P "$REMOTE_PORT"
)
SSH_TARGET="$REMOTE_USER@$REMOTE_HOST"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Preflighting RunCrate GPU host..." | tee -a "$LOCAL_LOG"
ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "hostname; nvidia-smi; df -h / /tmp /workspace 2>/dev/null || true; mkdir -p '$REMOTE_REPO' '$REMOTE_LAB_ROOT'" | tee -a "$LOCAL_LOG"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Syncing repository to RunCrate..." | tee -a "$LOCAL_LOG"
if [[ "$SYNC_REPO" = "1" ]]; then
  COPYFILE_DISABLE=1 tar \
    --exclude='.git' \
    --exclude='.codex_outputs' \
    --exclude='.codex_secrets' \
    --exclude='__pycache__' \
    --exclude='.pytest_cache' \
    --exclude='.mypy_cache' \
    --exclude='.ruff_cache' \
    --exclude='.venv' \
    --exclude='node_modules' \
    -czf - . \
    | ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "mkdir -p '$REMOTE_REPO' && tar -xzf - -C '$REMOTE_REPO'"
else
  ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "test -d '$REMOTE_REPO' && test -f '$REMOTE_REPO/scripts/research/train_face_indexed_conditioned_tiny.py'"
fi

tmp_b2_env="$(mktemp "$OUT_DIR/b2_remote_env.XXXXXX")"
{
  source "$B2_ENV_FILE"
  printf 'export B2_KEY_ID=%q\n' "${B2_KEY_ID:-${B2_KEYID:-}}"
  printf 'export B2_APP_KEY=%q\n' "${B2_APP_KEY:-${B2_APPKEY:-}}"
  printf 'export B2_TOKEN=%q\n' "${B2_TOKEN:-}"
} > "$tmp_b2_env"
chmod 600 "$tmp_b2_env"
ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "mkdir -p '$REMOTE_LAB_ROOT'"
scp "${SCP_OPTS[@]}" "$tmp_b2_env" "$SSH_TARGET:$REMOTE_B2_ENV" >/dev/null
rm -f "$tmp_b2_env"

remote_script="$OUT_DIR/remote_runcrate_faceq_smoke.sh"
cat > "$remote_script" <<REMOTE
#!/usr/bin/env bash
set -euo pipefail
LAB_ROOT=$(printf '%q' "$REMOTE_LAB_ROOT")
REPO=$(printf '%q' "$REMOTE_REPO")
VENV=$(printf '%q' "$REMOTE_VENV")
B2_ENV=$(printf '%q' "$REMOTE_B2_ENV")
B2_BUCKET=$(printf '%q' "$B2_BUCKET")
B2_CORPUS_PREFIX=$(printf '%q' "$B2_CORPUS_PREFIX")
B2_CORPUS_ARCHIVE=$(printf '%q' "$B2_CORPUS_ARCHIVE")
B2_RUN_PREFIX=$(printf '%q' "$B2_RUN_PREFIX")
ARCHIVE_DATASET_DIR=$(printf '%q' "$ARCHIVE_DATASET_DIR")
REMOVE_CORPUS_ARCHIVE_AFTER_EXTRACT=$(printf '%q' "$REMOVE_CORPUS_ARCHIVE_AFTER_EXTRACT")
mkdir -p "\$LAB_ROOT/logs" "\$LAB_ROOT/runs/faceq_merged_scale" "\$LAB_ROOT/tmp"
export TMPDIR="\$LAB_ROOT/tmp"
status() {
  python3 - "\$LAB_ROOT/status.jsonl" "\$1" "\${2:-}" <<'PY'
import json, sys, time
path, event, detail = sys.argv[1:4]
with open(path, "a", encoding="utf-8") as handle:
    handle.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "event": event, "detail": detail}) + "\\n")
PY
}
cd "\$REPO"
SUDO=()
if command -v sudo >/dev/null 2>&1; then
  SUDO=(sudo)
fi
if ! command -v rclone >/dev/null 2>&1; then
  "\${SUDO[@]}" apt-get update
  "\${SUDO[@]}" env DEBIAN_FRONTEND=noninteractive apt-get install -y rclone
fi
if [[ ! -x "\$VENV/bin/python" ]]; then
  python3 -m venv "\$VENV" || ("\${SUDO[@]}" apt-get update && "\${SUDO[@]}" env DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv && python3 -m venv "\$VENV")
fi
source "\$VENV/bin/activate"
python -m pip install -U pip setuptools wheel
python -m pip install -q numpy tqdm trimesh scipy networkx
python - <<'PY' || python -m pip install --index-url https://download.pytorch.org/whl/cu128 'torch>=2.4.0'
import torch
print("torch_ready", torch.__version__)
PY
source "\$B2_ENV"
export RCLONE_CONFIG_B2ENV_TYPE=b2
export RCLONE_CONFIG_B2ENV_ACCOUNT="\$B2_KEY_ID"
export RCLONE_CONFIG_B2ENV_KEY="\$B2_APP_KEY"
rclone lsf "b2env:\$B2_BUCKET" >/dev/null
MODE=face_run LOCAL_ROOT="\$LAB_ROOT" B2_BUCKET="\$B2_BUCKET" B2_PREFIX="\$B2_RUN_PREFIX" INTERVAL_SECONDS=300 \\
  B2_KEY_ID="\$B2_KEY_ID" B2_APP_KEY="\$B2_APP_KEY" B2_TOKEN="\${B2_TOKEN:-}" \\
  nohup bash scripts/thunder/b2_continuous_upload.sh > "\$LAB_ROOT/logs/b2_upload.log" 2>&1 &
echo \$! > "\$LAB_ROOT/b2_upload.pid"
status b2_download_started "\$B2_CORPUS_PREFIX/\$B2_CORPUS_ARCHIVE"
rclone copyto "b2env:\$B2_BUCKET/\$B2_CORPUS_PREFIX/\$B2_CORPUS_ARCHIVE" "\$LAB_ROOT/\$B2_CORPUS_ARCHIVE" --stats 30s
tar -xzf "\$LAB_ROOT/\$B2_CORPUS_ARCHIVE" -C "\$LAB_ROOT"
if [[ "\$REMOVE_CORPUS_ARCHIVE_AFTER_EXTRACT" = "1" ]]; then
  rm -f "\$LAB_ROOT/\$B2_CORPUS_ARCHIVE"
fi
DATASET_DIR="\$LAB_ROOT/$(printf '%q' "$ARCHIVE_DATASET_DIR")"
test -d "\$DATASET_DIR"
TRAIN_CMD=(python)
if [[ $(printf '%q' "$TORCHRUN_NPROC_PER_NODE") -gt 1 ]]; then
  TRAIN_CMD=(python -m torch.distributed.run --nproc-per-node $(printf '%q' "$TORCHRUN_NPROC_PER_NODE"))
fi
TRAIN_EXTRA_ARGS=()
[[ $(printf '%q' "$LAZY_LOAD") = "1" ]] && TRAIN_EXTRA_ARGS+=(--lazy-load)
[[ $(printf '%q' "$SAMPLE_CACHE_SIZE") -gt 0 ]] && TRAIN_EXTRA_ARGS+=(--sample-cache-size $(printf '%q' "$SAMPLE_CACHE_SIZE"))
[[ $(printf '%q' "$TRACK_BEST_IN_MEMORY") = "0" ]] && TRAIN_EXTRA_ARGS+=(--no-track-best-in-memory)
[[ $(printf '%q' "$SKIP_FINAL_CHECKPOINT") = "1" ]] && TRAIN_EXTRA_ARGS+=(--skip-final-checkpoint)
status train_started "steps=$(printf '%q' "$STEPS") nproc=$(printf '%q' "$TORCHRUN_NPROC_PER_NODE")"
"\${TRAIN_CMD[@]}" scripts/research/train_face_indexed_conditioned_tiny.py \\
  --dataset-dir "\$DATASET_DIR" \\
  --output "\$LAB_ROOT/runs/faceq_merged_scale/checkpoint.pt" \\
  --steps $(printf '%q' "$STEPS") \\
  --batch-size $(printf '%q' "$BATCH_SIZE") \\
  --limit $(printf '%q' "$TRAIN_LIMIT") \\
  --point-samples $(printf '%q' "$TRAIN_POINT_SAMPLES") \\
  --hidden-size $(printf '%q' "$HIDDEN_SIZE") \\
  --layers $(printf '%q' "$LAYERS") \\
  --heads $(printf '%q' "$HEADS") \\
  --condition-tokens $(printf '%q' "$CONDITION_TOKENS") \\
  --condition-backend $(printf '%q' "$CONDITION_BACKEND") \\
  --decoder-backend $(printf '%q' "$DECODER_BACKEND") \\
  --encoder-layers $(printf '%q' "$ENCODER_LAYERS") \\
  --latent-dim $(printf '%q' "$LATENT_DIM") \\
  --face-output-mode $(printf '%q' "$FACE_OUTPUT_MODE") \\
  --corner-head causal \\
  --count-loss-weight $(printf '%q' "$COUNT_LOSS_WEIGHT") \\
  --topology-loss-weight $(printf '%q' "$TOPOLOGY_LOSS_WEIGHT") \\
  --edge-action-loss-weight $(printf '%q' "$EDGE_ACTION_LOSS_WEIGHT") \\
  --edge-choice-loss-weight $(printf '%q' "$EDGE_CHOICE_LOSS_WEIGHT") \\
  --edge-choice-candidates 32 \\
  --seed-face-loss-weight $(printf '%q' "$SEED_FACE_LOSS_WEIGHT") \\
  --early-face-count $(printf '%q' "$EARLY_FACE_COUNT") \\
  --early-face-loss-weight $(printf '%q' "$EARLY_FACE_LOSS_WEIGHT") \\
  --optimizer $(printf '%q' "$OPTIMIZER") \\
  --precision $(printf '%q' "$PRECISION") \\
  --lr $(printf '%q' "$LR") \\
  --weight-decay $(printf '%q' "$WEIGHT_DECAY") \\
  --checkpoint-every $(printf '%q' "$CHECKPOINT_EVERY") \\
  --save-current-checkpoint \\
  --log-every $(printf '%q' "$LOG_EVERY") \\
  --distributed $(printf '%q' "$DISTRIBUTED") \\
  --distributed-backend $(printf '%q' "$DISTRIBUTED_BACKEND") \\
  --distributed-strategy $(printf '%q' "$DISTRIBUTED_STRATEGY") \\
  --fsdp-min-num-params $(printf '%q' "$FSDP_MIN_NUM_PARAMS") \\
  "\${TRAIN_EXTRA_ARGS[@]}" \\
  --device cuda | tee "\$LAB_ROOT/runs/faceq_merged_scale/train.log"
status train_complete "\$LAB_ROOT/runs/faceq_merged_scale/checkpoint.pt"
REMOTE
chmod 600 "$remote_script"
scp "${SCP_OPTS[@]}" "$remote_script" "$SSH_TARGET:/tmp/clearmesh_runcrate_faceq_smoke.sh" >/dev/null
ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "chmod +x /tmp/clearmesh_runcrate_faceq_smoke.sh; nohup /tmp/clearmesh_runcrate_faceq_smoke.sh > '$REMOTE_LOG' 2>&1 < /dev/null & echo \$! > '$REMOTE_PID'"

cat > "$OUT_DIR/runcrate_run_info.json" <<JSON
{
  "provider": "runcrate",
  "instance_id": "$INSTANCE_ID",
  "host": "$REMOTE_HOST",
  "port": "$REMOTE_PORT",
  "remote_user": "$REMOTE_USER",
  "remote_lab_root": "$REMOTE_LAB_ROOT",
  "remote_log": "$REMOTE_LOG",
  "remote_pid": "$REMOTE_PID",
  "b2_corpus_prefix": "$B2_CORPUS_PREFIX",
  "b2_corpus_archive": "$B2_CORPUS_ARCHIVE",
  "b2_run_prefix": "$B2_RUN_PREFIX",
  "steps": $STEPS,
  "hidden_size": $HIDDEN_SIZE,
  "layers": $LAYERS,
  "heads": $HEADS,
  "condition_tokens": $CONDITION_TOKENS,
  "torchrun_nproc_per_node": $TORCHRUN_NPROC_PER_NODE,
  "distributed_strategy": "$DISTRIBUTED_STRATEGY"
}
JSON
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] RunCrate FACE-Q smoke launched instance=$INSTANCE_ID host=$REMOTE_HOST root=$REMOTE_LAB_ROOT" | tee -a "$LOCAL_LOG"
cp "$OUT_DIR/runcrate_run_info.json" "$BOOTSTRAP_DONE"
