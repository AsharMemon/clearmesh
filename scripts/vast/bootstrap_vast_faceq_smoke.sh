#!/usr/bin/env bash
# Bootstrap a Vast.ai SSH instance and launch the FACE-Q FSDP smoke.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

VAST_BIN="${VAST_BIN:-/Users/Ashar/Library/Python/3.14/bin/vastai}"
INSTANCE_ID="${VAST_INSTANCE_ID:-${1:-}}"
OUT_DIR="${OUT_DIR:-.codex_outputs/vast_scale_20260517}"
B2_ENV_FILE="${B2_ENV_FILE:-.codex_secrets/b2.env}"
SSH_KEY_FILE="${SSH_KEY_FILE:-$HOME/.ssh/id_ed25519}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%d_faceq_1p4b_fsdp_vast_smoke)}"
REMOTE_REPO="${REMOTE_REPO:-/workspace/clearmesh}"
REMOTE_VENV="${REMOTE_VENV:-/workspace/clearmesh-venv}"
REMOTE_LAB_ROOT="${REMOTE_LAB_ROOT:-/workspace/clearmesh_faceq_vast_scale_train_$RUN_STAMP}"
REMOTE_LOG="${REMOTE_LOG:-/workspace/clearmesh_faceq_vast_scale_train.nohup.log}"
REMOTE_PID="${REMOTE_PID:-/workspace/clearmesh_faceq_vast_scale_train.pid}"
REMOTE_B2_ENV="${REMOTE_B2_ENV:-$REMOTE_LAB_ROOT/.clearmesh_b2.env}"

B2_BUCKET="${B2_BUCKET:-clearmesh-pairs}"
B2_CORPUS_PREFIX="${B2_CORPUS_PREFIX:-face-corpora/merged/faceq_rolling_400k_total_380k_train/snapshot_20260517T064949Z_archives658}"
B2_CORPUS_ARCHIVE="${B2_CORPUS_ARCHIVE:-faceq_merged_dedup_split.tar.gz}"
B2_RUN_PREFIX="${B2_RUN_PREFIX:-face-runs/faceq-1p4b-fsdp-smoke/$RUN_STAMP}"
ARCHIVE_DATASET_DIR="${ARCHIVE_DATASET_DIR:-split_dedup_tokenhash/train}"

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
LOCAL_LOG="$OUT_DIR/vast_bootstrap_${INSTANCE_ID:-unknown}.log"

if [[ -z "$INSTANCE_ID" ]]; then
  echo "Set VAST_INSTANCE_ID or pass instance id as argv[1]." >&2
  exit 2
fi
if [[ -z "${VAST_API:-}" ]]; then
  echo "VAST_API is not set." >&2
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

ssh_url_to_args() {
  python3 - "$1" <<'PY'
import re, shlex, sys
text = sys.argv[1].strip()
m = re.search(r'ssh://([^@\\s]+)@([^:/\\s]+):(\\d+)', text)
if m:
    user, host, port = m.groups()
else:
    m = re.search(r'(?:ssh\\s+)?(?:-p\\s+(\\d+)\\s+)?([^@\\s]+)@([^\\s]+)', text)
    if not m:
        raise SystemExit(1)
    port, user, host = m.groups()
    port = port or "22"
print(shlex.quote(user), shlex.quote(host), shlex.quote(port))
PY
}

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Waiting for Vast SSH URL for $INSTANCE_ID..." | tee -a "$LOCAL_LOG"
deadline=$(( $(date +%s) + 1800 ))
USER_HOST_PORT=""
while [[ "$(date +%s)" -lt "$deadline" ]]; do
  raw="$("$VAST_BIN" --api-key "$VAST_API" ssh-url "$INSTANCE_ID" 2>/dev/null || true)"
  if [[ -n "$raw" ]] && USER_HOST_PORT="$(ssh_url_to_args "$raw" 2>/dev/null)"; then
    break
  fi
  sleep 15
done
if [[ -z "$USER_HOST_PORT" ]]; then
  echo "Vast instance did not expose SSH URL before timeout." >&2
  exit 3
fi
read -r REMOTE_USER REMOTE_HOST REMOTE_PORT <<< "$USER_HOST_PORT"
REMOTE_USER="$(printf '%s' "$REMOTE_USER" | tr -d "'")"
REMOTE_HOST="$(printf '%s' "$REMOTE_HOST" | tr -d "'")"
REMOTE_PORT="$(printf '%s' "$REMOTE_PORT" | tr -d "'")"
SSH_OPTS=(
  -i "$SSH_KEY_FILE"
  -p "$REMOTE_PORT"
  -o StrictHostKeyChecking=accept-new
  -o UserKnownHostsFile="$HOME/.ssh/known_hosts"
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=4
)
SSH_TARGET="$REMOTE_USER@$REMOTE_HOST"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Preflighting Vast host..." | tee -a "$LOCAL_LOG"
ssh "${SSH_OPTS[@]}" "$SSH_TARGET" 'hostname; nvidia-smi; mkdir -p /workspace' | tee -a "$LOCAL_LOG"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Syncing repository..." | tee -a "$LOCAL_LOG"
tar \
  --exclude='.git' \
  --exclude='.codex_outputs' \
  --exclude='.codex_secrets' \
  --exclude='__pycache__' \
  --exclude='.pytest_cache' \
  -czf - . \
  | ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "mkdir -p '$REMOTE_REPO' && tar -xzf - -C '$REMOTE_REPO'"

tmp_b2_env="$(mktemp "$OUT_DIR/b2_remote_env.XXXXXX")"
{
  source "$B2_ENV_FILE"
  printf 'export B2_KEY_ID=%q\n' "${B2_KEY_ID:-${B2_KEYID:-}}"
  printf 'export B2_APP_KEY=%q\n' "${B2_APP_KEY:-${B2_APPKEY:-}}"
  printf 'export B2_TOKEN=%q\n' "${B2_TOKEN:-}"
} > "$tmp_b2_env"
chmod 600 "$tmp_b2_env"
ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "mkdir -p '$REMOTE_LAB_ROOT'"
scp "${SSH_OPTS[@]}" "$tmp_b2_env" "$SSH_TARGET:$REMOTE_B2_ENV" >/dev/null
rm -f "$tmp_b2_env"

remote_script="$OUT_DIR/remote_vast_faceq_smoke_${INSTANCE_ID}.sh"
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
mkdir -p "\$LAB_ROOT/logs" "\$LAB_ROOT/runs/faceq_merged_scale"
status() {
  python3 - "\$LAB_ROOT/status.jsonl" "\$1" "\${2:-}" <<'PY'
import json, sys, time
path, event, detail = sys.argv[1:4]
with open(path, "a", encoding="utf-8") as handle:
    handle.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "event": event, "detail": detail}) + "\\n")
PY
}
cd "\$REPO"
if ! command -v rclone >/dev/null 2>&1; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y rclone
fi
if [[ ! -x "\$VENV/bin/python" ]]; then
  python3 -m venv "\$VENV" || (apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv && python3 -m venv "\$VENV")
fi
source "\$VENV/bin/activate"
python -m pip install -U pip setuptools wheel
python -m pip install -q numpy tqdm
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
DATASET_DIR="\$LAB_ROOT/\$ARCHIVE_DATASET_DIR"
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
scp "${SSH_OPTS[@]}" "$remote_script" "$SSH_TARGET:/tmp/clearmesh_vast_faceq_smoke.sh" >/dev/null
ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "chmod +x /tmp/clearmesh_vast_faceq_smoke.sh && nohup /tmp/clearmesh_vast_faceq_smoke.sh > '$REMOTE_LOG' 2>&1 & echo \\$! > '$REMOTE_PID'"

cat > "$OUT_DIR/vast_run_info_${INSTANCE_ID}.json" <<JSON
{
  "provider": "vast",
  "instance_id": "$INSTANCE_ID",
  "ssh_host": "$REMOTE_HOST",
  "ssh_port": "$REMOTE_PORT",
  "ssh_user": "$REMOTE_USER",
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
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Vast FACE-Q smoke launched instance=$INSTANCE_ID root=$REMOTE_LAB_ROOT" | tee -a "$LOCAL_LOG"
