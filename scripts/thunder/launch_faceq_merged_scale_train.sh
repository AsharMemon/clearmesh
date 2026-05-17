#!/usr/bin/env bash
# Launch FACE-Q training from a pre-merged B2 corpus archive.
#
# This is the production-scale path after shard workers have uploaded lean
# corpora and a merge/dedupe job has produced a global split archive. It avoids
# re-tokenizing shard strict targets on the training node.
set -euo pipefail

TNR_BIN="${TNR_BIN:-/Users/Ashar/.tnr/bin/tnr}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

INSTANCE_ID="${THUNDER_INSTANCE_ID:-}"
CREATE_INSTANCE="${CREATE_INSTANCE:-1}"
GPU="${GPU:-a100}"
NUM_GPUS="${NUM_GPUS:-8}"
MODE="${MODE:-production}"
VCPUS="${VCPUS:-32}"
PRIMARY_DISK="${PRIMARY_DISK:-1000}"
TEMPLATE="${TEMPLATE:-base}"

RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%d_%H%M%S)_faceq_merged_scale_train}"
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-$REPO_ROOT/.codex_outputs/faceq_merged_scale_train_$RUN_STAMP}"
REMOTE_REPO="${REMOTE_REPO:-/home/ubuntu/clearmesh}"
REMOTE_VENV="${REMOTE_VENV:-/home/ubuntu/clearmesh-venv}"
REMOTE_LAB_ROOT="${REMOTE_LAB_ROOT:-/tmp/clearmesh_faceq_merged_scale_train_$RUN_STAMP}"
REMOTE_LOG="${REMOTE_LOG:-/tmp/clearmesh_faceq_merged_scale_train.nohup.log}"
REMOTE_PID="${REMOTE_PID:-/tmp/clearmesh_faceq_merged_scale_train.pid}"
REMOTE_B2_ENV="${REMOTE_B2_ENV:-$REMOTE_LAB_ROOT/.clearmesh_b2.env}"

B2_BUCKET="${B2_BUCKET:-clearmesh-pairs}"
B2_CORPUS_PREFIX="${B2_CORPUS_PREFIX:-face-corpora/merged/faceq_poolA_137k_20260516}"
B2_CORPUS_ARCHIVE="${B2_CORPUS_ARCHIVE:-faceq_poolA_merged_dedup_split.tar.gz}"
B2_RUN_PREFIX="${B2_RUN_PREFIX:-face-runs/faceq-merged-scale/$RUN_STAMP}"
ARCHIVE_DATASET_DIR="${ARCHIVE_DATASET_DIR:-split_dedup_tokenhash/train}"

STEPS="${STEPS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-1}"
TRAIN_LIMIT="${TRAIN_LIMIT:-0}"
TRAIN_POINT_SAMPLES="${TRAIN_POINT_SAMPLES:-8192}"
HIDDEN_SIZE="${HIDDEN_SIZE:-1024}"
LAYERS="${LAYERS:-24}"
HEADS="${HEADS:-16}"
CONDITION_TOKENS="${CONDITION_TOKENS:-1024}"
CONDITION_BACKEND="${CONDITION_BACKEND:-vecset}"
DECODER_BACKEND="${DECODER_BACKEND:-cross_attn}"
ENCODER_LAYERS="${ENCODER_LAYERS:-6}"
LATENT_DIM="${LATENT_DIM:-64}"
FACE_OUTPUT_MODE="${FACE_OUTPUT_MODE:-geometry}"
LR="${LR:-6e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
OPTIMIZER="${OPTIMIZER:-muon}"
PRECISION="${PRECISION:-bf16}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-5000}"
SKIP_FINAL_CHECKPOINT="${SKIP_FINAL_CHECKPOINT:-0}"
LOG_EVERY="${LOG_EVERY:-25}"
TORCHRUN_NPROC_PER_NODE="${TORCHRUN_NPROC_PER_NODE:-$NUM_GPUS}"
DISTRIBUTED="${DISTRIBUTED:-auto}"
DISTRIBUTED_BACKEND="${DISTRIBUTED_BACKEND:-auto}"
DISTRIBUTED_STRATEGY="${DISTRIBUTED_STRATEGY:-fsdp}"
FSDP_MIN_NUM_PARAMS="${FSDP_MIN_NUM_PARAMS:-20000000}"
LAZY_LOAD="${LAZY_LOAD:-1}"
SAMPLE_CACHE_SIZE="${SAMPLE_CACHE_SIZE:-256}"
TRACK_BEST_IN_MEMORY="${TRACK_BEST_IN_MEMORY:-0}"
COUNT_LOSS_WEIGHT="${COUNT_LOSS_WEIGHT:-0.05}"
TOPOLOGY_LOSS_WEIGHT="${TOPOLOGY_LOSS_WEIGHT:-0.2}"
EDGE_ACTION_LOSS_WEIGHT="${EDGE_ACTION_LOSS_WEIGHT:-1.0}"
EDGE_CHOICE_LOSS_WEIGHT="${EDGE_CHOICE_LOSS_WEIGHT:-0.5}"
SEED_FACE_LOSS_WEIGHT="${SEED_FACE_LOSS_WEIGHT:-1.0}"
EARLY_FACE_COUNT="${EARLY_FACE_COUNT:-16}"
EARLY_FACE_LOSS_WEIGHT="${EARLY_FACE_LOSS_WEIGHT:-4.0}"

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
  create_args=(create --gpu "$GPU" --mode "$MODE" --num-gpus "$NUM_GPUS" --primary-disk "$PRIMARY_DISK" --template "$TEMPLATE" --yes --json)
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

echo "Preflighting GPUs..."
cat <<'EOF' | "$TNR_BIN" connect "$INSTANCE_ID" | tee "$DOWNLOAD_ROOT/gpu_preflight.log"
set -euo pipefail
hostname
nvidia-smi
python3 - <<'PY'
import glob
gpus = sorted(glob.glob("/dev/nvidia[0-9]*"))
print("gpu_device_count", len(gpus))
if not gpus:
    raise SystemExit(2)
PY
echo CLEARMESH_GPU_PREFLIGHT_OK
exit
EOF
grep -q CLEARMESH_GPU_PREFLIGHT_OK "$DOWNLOAD_ROOT/gpu_preflight.log"

echo "Syncing repo and bootstrapping..."
THUNDER_INSTANCE_ID="$INSTANCE_ID" "$REPO_ROOT/scripts/thunder/sync_repo.sh" "$INSTANCE_ID"
THUNDER_INSTANCE_ID="$INSTANCE_ID" INSTALL_MESH_HEAD_REPOS=0 INSTALL_MESH_HEAD_ENVS=0 "$REPO_ROOT/scripts/thunder/bootstrap_remote.sh" "$INSTANCE_ID"

b2_env_file="$(mktemp "$DOWNLOAD_ROOT/b2_env.XXXXXX")"
{
  printf 'export B2_KEY_ID=%q\n' "${B2_KEY_ID:-${B2_KEYID:-${B2_APPLICATION_KEY_ID:-${BACKBLAZE_B2_KEY_ID:-}}}}"
  printf 'export B2_APP_KEY=%q\n' "${B2_APP_KEY:-${B2_APPKEY:-${B2_APPLICATION_KEY:-${BACKBLAZE_B2_APPLICATION_KEY:-${BACKBLAZE_B2_APP_KEY:-}}}}}"
  printf 'export B2_TOKEN=%q\n' "${B2_TOKEN:-}"
} > "$b2_env_file"
chmod 600 "$b2_env_file"
printf 'mkdir -p %q\nexit\n' "$REMOTE_LAB_ROOT" | "$TNR_BIN" connect "$INSTANCE_ID" >/dev/null
"$TNR_BIN" scp "$b2_env_file" "$INSTANCE_ID:$REMOTE_B2_ENV"
rm -f "$b2_env_file"

remote_script="$DOWNLOAD_ROOT/remote_faceq_merged_scale_train.sh"
cat > "$remote_script" <<REMOTE
#!/usr/bin/env bash
set -euo pipefail
LOG_TIME() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }
LAB_ROOT=$(printf '%q' "$REMOTE_LAB_ROOT")
REPO=$(printf '%q' "$REMOTE_REPO")
VENV=$(printf '%q' "$REMOTE_VENV")
B2_ENV=$(printf '%q' "$REMOTE_B2_ENV")
B2_BUCKET=$(printf '%q' "$B2_BUCKET")
B2_CORPUS_PREFIX=$(printf '%q' "$B2_CORPUS_PREFIX")
B2_CORPUS_ARCHIVE=$(printf '%q' "$B2_CORPUS_ARCHIVE")
B2_RUN_PREFIX=$(printf '%q' "$B2_RUN_PREFIX")
ARCHIVE_DATASET_DIR=$(printf '%q' "$ARCHIVE_DATASET_DIR")
STEPS=$(printf '%q' "$STEPS")
BATCH_SIZE=$(printf '%q' "$BATCH_SIZE")
TRAIN_LIMIT=$(printf '%q' "$TRAIN_LIMIT")
TRAIN_POINT_SAMPLES=$(printf '%q' "$TRAIN_POINT_SAMPLES")
HIDDEN_SIZE=$(printf '%q' "$HIDDEN_SIZE")
LAYERS=$(printf '%q' "$LAYERS")
HEADS=$(printf '%q' "$HEADS")
CONDITION_TOKENS=$(printf '%q' "$CONDITION_TOKENS")
CONDITION_BACKEND=$(printf '%q' "$CONDITION_BACKEND")
DECODER_BACKEND=$(printf '%q' "$DECODER_BACKEND")
ENCODER_LAYERS=$(printf '%q' "$ENCODER_LAYERS")
LATENT_DIM=$(printf '%q' "$LATENT_DIM")
FACE_OUTPUT_MODE=$(printf '%q' "$FACE_OUTPUT_MODE")
LR=$(printf '%q' "$LR")
WEIGHT_DECAY=$(printf '%q' "$WEIGHT_DECAY")
OPTIMIZER=$(printf '%q' "$OPTIMIZER")
PRECISION=$(printf '%q' "$PRECISION")
CHECKPOINT_EVERY=$(printf '%q' "$CHECKPOINT_EVERY")
SKIP_FINAL_CHECKPOINT=$(printf '%q' "$SKIP_FINAL_CHECKPOINT")
LOG_EVERY=$(printf '%q' "$LOG_EVERY")
TORCHRUN_NPROC_PER_NODE=$(printf '%q' "$TORCHRUN_NPROC_PER_NODE")
DISTRIBUTED=$(printf '%q' "$DISTRIBUTED")
DISTRIBUTED_BACKEND=$(printf '%q' "$DISTRIBUTED_BACKEND")
DISTRIBUTED_STRATEGY=$(printf '%q' "$DISTRIBUTED_STRATEGY")
FSDP_MIN_NUM_PARAMS=$(printf '%q' "$FSDP_MIN_NUM_PARAMS")
LAZY_LOAD=$(printf '%q' "$LAZY_LOAD")
SAMPLE_CACHE_SIZE=$(printf '%q' "$SAMPLE_CACHE_SIZE")
TRACK_BEST_IN_MEMORY=$(printf '%q' "$TRACK_BEST_IN_MEMORY")
COUNT_LOSS_WEIGHT=$(printf '%q' "$COUNT_LOSS_WEIGHT")
TOPOLOGY_LOSS_WEIGHT=$(printf '%q' "$TOPOLOGY_LOSS_WEIGHT")
EDGE_ACTION_LOSS_WEIGHT=$(printf '%q' "$EDGE_ACTION_LOSS_WEIGHT")
EDGE_CHOICE_LOSS_WEIGHT=$(printf '%q' "$EDGE_CHOICE_LOSS_WEIGHT")
SEED_FACE_LOSS_WEIGHT=$(printf '%q' "$SEED_FACE_LOSS_WEIGHT")
EARLY_FACE_COUNT=$(printf '%q' "$EARLY_FACE_COUNT")
EARLY_FACE_LOSS_WEIGHT=$(printf '%q' "$EARLY_FACE_LOSS_WEIGHT")

mkdir -p "\$LAB_ROOT/logs" "\$LAB_ROOT/runs/faceq_merged_scale"
status() {
  python3 - "\$LAB_ROOT/status.jsonl" "\$1" "\${2:-}" <<'PY'
import json
import sys
import time
path, event, detail = sys.argv[1:4]
with open(path, "a", encoding="utf-8") as handle:
    handle.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "event": event, "detail": detail}) + "\n")
PY
}

cd "\$REPO"
source "\$VENV/bin/activate"
if ! command -v rclone >/dev/null 2>&1; then
  sudo apt-get update
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y rclone
fi
python - <<'PY' || python -m pip install --index-url https://download.pytorch.org/whl/cu128 'torch>=2.4.0'
import torch
print("torch_ready", torch.__version__)
PY
source "\$B2_ENV"
if [[ -z "\${B2_KEY_ID:-}" || -z "\${B2_APP_KEY:-}" ]] && [[ -n "\${B2_TOKEN:-}" ]]; then
  parsed_b2="\$(python3 - <<'PY'
import json
import os
token = os.environ.get("B2_TOKEN", "").strip()
key_id = app_key = ""
if token:
    if token.startswith("{"):
        payload = json.loads(token)
        key_id = payload.get("keyId") or payload.get("applicationKeyId") or payload.get("key_id") or ""
        app_key = payload.get("applicationKey") or payload.get("application_key") or payload.get("appKey") or ""
    elif ":" in token:
        key_id, app_key = token.split(":", 1)
if key_id and app_key:
    print(key_id)
    print(app_key)
PY
)"
  if [[ -n "\$parsed_b2" ]]; then
    export B2_KEY_ID="\${B2_KEY_ID:-\$(printf '%s\n' "\$parsed_b2" | sed -n '1p')}"
    export B2_APP_KEY="\${B2_APP_KEY:-\$(printf '%s\n' "\$parsed_b2" | sed -n '2p')}"
  fi
fi
if [[ -n "\${B2_KEY_ID:-}" && -z "\${B2_APP_KEY:-}" && -n "\${B2_TOKEN:-}" ]]; then
  case "\$B2_TOKEN" in
    \{*|*:*) ;;
    *) export B2_APP_KEY="\$B2_TOKEN" ;;
  esac
fi
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
status corpus_ready "\$DATASET_DIR"

TRAIN_CMD=(python)
if [ "\$TORCHRUN_NPROC_PER_NODE" -gt 1 ]; then
  TRAIN_CMD=(python -m torch.distributed.run --nproc-per-node "\$TORCHRUN_NPROC_PER_NODE")
fi
TRAIN_EXTRA_ARGS=()
if [ "\$LAZY_LOAD" = "1" ]; then
  TRAIN_EXTRA_ARGS+=(--lazy-load)
fi
if [ "\$SAMPLE_CACHE_SIZE" -gt 0 ]; then
  TRAIN_EXTRA_ARGS+=(--sample-cache-size "\$SAMPLE_CACHE_SIZE")
fi
if [ "\$TRACK_BEST_IN_MEMORY" = "0" ]; then
  TRAIN_EXTRA_ARGS+=(--no-track-best-in-memory)
fi
if [ "\$SKIP_FINAL_CHECKPOINT" = "1" ]; then
  TRAIN_EXTRA_ARGS+=(--skip-final-checkpoint)
fi

status train_started "steps=\$STEPS nproc=\$TORCHRUN_NPROC_PER_NODE"
"\${TRAIN_CMD[@]}" scripts/research/train_face_indexed_conditioned_tiny.py \\
  --dataset-dir "\$DATASET_DIR" \\
  --output "\$LAB_ROOT/runs/faceq_merged_scale/checkpoint.pt" \\
  --steps "\$STEPS" \\
  --batch-size "\$BATCH_SIZE" \\
  --limit "\$TRAIN_LIMIT" \\
  --point-samples "\$TRAIN_POINT_SAMPLES" \\
  --hidden-size "\$HIDDEN_SIZE" \\
  --layers "\$LAYERS" \\
  --heads "\$HEADS" \\
  --condition-tokens "\$CONDITION_TOKENS" \\
  --condition-backend "\$CONDITION_BACKEND" \\
  --decoder-backend "\$DECODER_BACKEND" \\
  --encoder-layers "\$ENCODER_LAYERS" \\
  --latent-dim "\$LATENT_DIM" \\
  --face-output-mode "\$FACE_OUTPUT_MODE" \\
  --corner-head causal \\
  --count-loss-weight "\$COUNT_LOSS_WEIGHT" \\
  --topology-loss-weight "\$TOPOLOGY_LOSS_WEIGHT" \\
  --edge-action-loss-weight "\$EDGE_ACTION_LOSS_WEIGHT" \\
  --edge-choice-loss-weight "\$EDGE_CHOICE_LOSS_WEIGHT" \\
  --edge-choice-candidates 32 \\
  --seed-face-loss-weight "\$SEED_FACE_LOSS_WEIGHT" \\
  --early-face-count "\$EARLY_FACE_COUNT" \\
  --early-face-loss-weight "\$EARLY_FACE_LOSS_WEIGHT" \\
  --optimizer "\$OPTIMIZER" \\
  --precision "\$PRECISION" \\
  --lr "\$LR" \\
  --weight-decay "\$WEIGHT_DECAY" \\
  --checkpoint-every "\$CHECKPOINT_EVERY" \\
  --save-current-checkpoint \\
  --log-every "\$LOG_EVERY" \\
  --distributed "\$DISTRIBUTED" \\
  --distributed-backend "\$DISTRIBUTED_BACKEND" \\
  --distributed-strategy "\$DISTRIBUTED_STRATEGY" \\
  --fsdp-min-num-params "\$FSDP_MIN_NUM_PARAMS" \\
  "\${TRAIN_EXTRA_ARGS[@]}" \\
  --device cuda | tee "\$LAB_ROOT/runs/faceq_merged_scale/train.log"
status train_complete "\$LAB_ROOT/runs/faceq_merged_scale/checkpoint.pt"
REMOTE

"$TNR_BIN" scp "$remote_script" "$INSTANCE_ID:/tmp/clearmesh_faceq_merged_scale_train.sh"
cat <<REMOTE_LAUNCH | "$TNR_BIN" connect "$INSTANCE_ID" | tee "$DOWNLOAD_ROOT/remote_launch.log"
set -euo pipefail
chmod +x /tmp/clearmesh_faceq_merged_scale_train.sh
nohup /tmp/clearmesh_faceq_merged_scale_train.sh > "$REMOTE_LOG" 2>&1 &
echo \$! > "$REMOTE_PID"
echo CLEARMESH_FACEQ_MERGED_SCALE_TRAIN_LAUNCHED pid=\$(cat "$REMOTE_PID") root="$REMOTE_LAB_ROOT" log="$REMOTE_LOG"
exit
REMOTE_LAUNCH

cat > "$DOWNLOAD_ROOT/run_info.json" <<JSON
{
  "instance_id": "$INSTANCE_ID",
  "gpu": "$GPU",
  "num_gpus": $NUM_GPUS,
  "mode": "$MODE",
  "remote_lab_root": "$REMOTE_LAB_ROOT",
  "remote_log": "$REMOTE_LOG",
  "remote_pid": "$REMOTE_PID",
  "b2_bucket": "$B2_BUCKET",
  "b2_corpus_prefix": "$B2_CORPUS_PREFIX",
  "b2_corpus_archive": "$B2_CORPUS_ARCHIVE",
  "b2_run_prefix": "$B2_RUN_PREFIX",
  "archive_dataset_dir": "$ARCHIVE_DATASET_DIR",
  "steps": $STEPS,
  "batch_size": $BATCH_SIZE,
  "skip_final_checkpoint": "$SKIP_FINAL_CHECKPOINT",
  "hidden_size": $HIDDEN_SIZE,
  "layers": $LAYERS,
  "heads": $HEADS,
  "condition_tokens": $CONDITION_TOKENS,
  "condition_backend": "$CONDITION_BACKEND",
  "decoder_backend": "$DECODER_BACKEND",
  "encoder_layers": $ENCODER_LAYERS,
  "latent_dim": $LATENT_DIM,
  "face_output_mode": "$FACE_OUTPUT_MODE",
  "train_point_samples": $TRAIN_POINT_SAMPLES,
  "torchrun_nproc_per_node": $TORCHRUN_NPROC_PER_NODE,
  "distributed": "$DISTRIBUTED",
  "distributed_backend": "$DISTRIBUTED_BACKEND",
  "distributed_strategy": "$DISTRIBUTED_STRATEGY",
  "fsdp_min_num_params": $FSDP_MIN_NUM_PARAMS,
  "lazy_load": "$LAZY_LOAD",
  "sample_cache_size": $SAMPLE_CACHE_SIZE,
  "track_best_in_memory": "$TRACK_BEST_IN_MEMORY"
}
JSON

echo "FACE-Q merged scale train launched."
echo "Run info: $DOWNLOAD_ROOT/run_info.json"
