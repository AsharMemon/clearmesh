#!/usr/bin/env bash
# Launch/bootstrap a small RunCrate GPU for the ClearMesh product inference pipeline.
# The remote bridge runs text-to-image -> TRELLIS.2 -> FACE-Q -> optional Easy3E hook.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_DIR="${OUT_DIR:-.codex_outputs/runcrate_pipeline_inference_$RUN_STAMP}"
RUNCRATE_ENV_FILE="${RUNCRATE_ENV_FILE:-.codex_secrets/runcrate.env}"
B2_ENV_FILE="${B2_ENV_FILE:-.codex_secrets/b2.env}"
SSH_KEY_FILE="${SSH_KEY_FILE:-$HOME/.ssh/id_ed25519}"
KNOWN_HOSTS_FILE="${KNOWN_HOSTS_FILE:-$OUT_DIR/known_hosts}"
SSH_KEY_ID="${RUNCRATE_SSH_KEY_ID:-bb4e76ea-5051-4933-ace6-98b7923509ce}"
INSTANCE_TYPE_ID="${RUNCRATE_INSTANCE_TYPE_ID:-unruffled-perlman-a100_sxm4_80g_dgx-ed84}"
INSTANCE_NAME="${RUNCRATE_INSTANCE_NAME:-clearmesh-pipeline-inference-$RUN_STAMP}"
INSTANCE_STORAGE_GB="${RUNCRATE_INSTANCE_STORAGE_GB:-1000}"
INSTANCE_TEMPLATE="${RUNCRATE_INSTANCE_TEMPLATE:-ubuntu-cuda-devel}"
REMOTE_USER="${REMOTE_USER:-root}"
REMOTE_BASE="${REMOTE_BASE:-/ephemeral}"
REMOTE_REPO="${REMOTE_REPO:-$REMOTE_BASE/clearmesh}"
REMOTE_VENV="${REMOTE_VENV:-$REMOTE_BASE/clearmesh-pipeline-venv}"
REMOTE_TRELLIS2_DIR="${REMOTE_TRELLIS2_DIR:-$REMOTE_BASE/TRELLIS.2}"
REMOTE_HIDREAM_DIR="${REMOTE_HIDREAM_DIR:-$REMOTE_BASE/HiDream-O1-Image}"
REMOTE_MODEL_BUNDLE="${REMOTE_MODEL_BUNDLE:-$REMOTE_BASE/model_bundle_effective100k}"
REMOTE_WORK_ROOT="${REMOTE_WORK_ROOT:-$REMOTE_BASE/clearmesh_pipeline_jobs}"
REMOTE_STATE_ROOT="${REMOTE_STATE_ROOT:-$REMOTE_BASE/clearmesh_pipeline_state}"
REMOTE_LOG="${REMOTE_LOG:-$REMOTE_BASE/clearmesh_pipeline_bootstrap.log}"
REMOTE_SERVER_LOG="${REMOTE_SERVER_LOG:-$REMOTE_BASE/clearmesh_pipeline_server.log}"
REMOTE_SERVER_PID="${REMOTE_SERVER_PID:-$REMOTE_BASE/clearmesh_pipeline_server.pid}"
REMOTE_BOOTSTRAP_PID="${REMOTE_BOOTSTRAP_PID:-$REMOTE_BASE/clearmesh_pipeline_bootstrap.pid}"
REMOTE_PORT="${REMOTE_PORT:-8787}"
LOCAL_PORT="${LOCAL_PORT:-8787}"
START_TUNNEL="${START_TUNNEL:-1}"
INSTALL_TRELLIS2="${INSTALL_TRELLIS2:-1}"
INSTALL_TRELLIS2_EXTENSIONS="${INSTALL_TRELLIS2_EXTENSIONS:-1}"
INSTALL_HIDREAM="${INSTALL_HIDREAM:-1}"
PIPELINE_DRY_RUN="${CLEARMESH_PIPELINE_DRY_RUN:-0}"

B2_BUCKET="${B2_BUCKET:-clearmesh-pairs}"
B2_MODEL_PREFIX="${B2_MODEL_PREFIX:-face-runs/faceq-1p4b-65k1024-257k-full/vast-20260523_lr1e4_clip1_100k/runs/faceq_merged_scale_full_100k_init_from025k_remaining75k/model_bundle_effective100k}"
EXPECTED_CHECKPOINT_SHA="${EXPECTED_CHECKPOINT_SHA:-0c4589a584bfe878e88c55a8abe2c31ba6e19dc97711aa5dd5d81a8bf8cf4f28}"
TEXT_TO_IMAGE_MODEL="${CLEARMESH_TEXT_TO_IMAGE_MODEL:-HiDream-ai/HiDream-O1-Image-Dev-2604}"
TEXT_TO_IMAGE_BACKEND="${CLEARMESH_TEXT_TO_IMAGE_BACKEND:-hidream}"
HIDREAM_REPO_URL="${HIDREAM_REPO_URL:-https://github.com/HiDream-ai/HiDream-O1-Image.git}"
TRELLIS_MODEL="${CLEARMESH_TRELLIS_MODEL:-microsoft/TRELLIS.2-4B}"

mkdir -p "$OUT_DIR"
touch "$KNOWN_HOSTS_FILE"
LOCAL_LOG="$OUT_DIR/bootstrap.log"
CREATE_BODY="$OUT_DIR/create_body.json"
CREATE_RESPONSE="$OUT_DIR/create_response.json"
INSTANCE_JSON="$OUT_DIR/instance.latest.json"
RUN_INFO="$OUT_DIR/run_info.json"
REMOTE_ENV_LOCAL="$OUT_DIR/remote_env.sh"
REMOTE_SCRIPT_LOCAL="$OUT_DIR/remote_bootstrap_pipeline_inference.sh"

for required in "$RUNCRATE_ENV_FILE" "$B2_ENV_FILE" "$SSH_KEY_FILE"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing required file: $required" >&2
    exit 2
  fi
done

source "$RUNCRATE_ENV_FILE"
API_BASE="${RUNCRATE_API_BASE:-https://www.runcrate.ai/api/v1}"

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOCAL_LOG"
}

if [[ -z "${RUNCRATE_INSTANCE_ID:-}" ]]; then
  python3 - "$CREATE_BODY" <<PY
import json, sys
body = {
    "name": "${INSTANCE_NAME}",
    "ssh_key_id": "${SSH_KEY_ID}",
    "instance_type_id": "${INSTANCE_TYPE_ID}",
    "storage": int("${INSTANCE_STORAGE_GB}"),
    "template": "${INSTANCE_TEMPLATE}",
}
json.dump(body, open(sys.argv[1], "w", encoding="utf-8"), indent=2)
PY
  log "Creating RunCrate instance type=$INSTANCE_TYPE_ID storage=${INSTANCE_STORAGE_GB}GB"
  curl -sS --max-time 60 \
    -X POST \
    -H "Authorization: Bearer $RUNCRATE_API_KEY" \
    -H 'Accept: application/json' \
    -H 'Content-Type: application/json' \
    --data-binary "@$CREATE_BODY" \
    "$API_BASE/instances" > "$CREATE_RESPONSE"
  RUNCRATE_INSTANCE_ID="$(python3 - "$CREATE_RESPONSE" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding='utf-8'))
if 'error' in payload:
    raise SystemExit(json.dumps(payload['error']))
data = payload.get('data', payload)
print(data.get('id') or data.get('instance_id') or '')
PY
)"
  if [[ -z "$RUNCRATE_INSTANCE_ID" ]]; then
    echo "RunCrate create response did not include an instance id: $CREATE_RESPONSE" >&2
    exit 3
  fi
else
  log "Using existing RunCrate instance $RUNCRATE_INSTANCE_ID"
fi

log "Waiting for SSH target for RunCrate instance $RUNCRATE_INSTANCE_ID"
deadline=$(( $(date +%s) + 1800 ))
REMOTE_HOST=""
REMOTE_SSH_PORT="22"
while [[ "$(date +%s)" -lt "$deadline" ]]; do
  curl -sS --max-time 45 \
    -H "Authorization: Bearer $RUNCRATE_API_KEY" \
    -H 'Accept: application/json' \
    "$API_BASE/instances" > "$INSTANCE_JSON.tmp" || true
  if [[ -s "$INSTANCE_JSON.tmp" ]]; then
    mv "$INSTANCE_JSON.tmp" "$INSTANCE_JSON"
    parsed="$(python3 - "$RUNCRATE_INSTANCE_ID" "$INSTANCE_JSON" <<'PY'
import json, sys
instance_id, path = sys.argv[1:3]
payload = json.load(open(path, encoding='utf-8'))
items = payload.get('data', payload if isinstance(payload, list) else [])
for item in items:
    if str(item.get('id')) == instance_id:
        status = (item.get('status') or '').lower()
        ip = item.get('ip') or item.get('public_ip') or item.get('host') or item.get('ssh_host') or ''
        port = str(item.get('ssh_port') or item.get('port') or 22)
        print(status, ip or '-', port)
        break
else:
    print('missing - 22')
PY
)"
    status="$(awk '{print $1}' <<< "$parsed")"
    host="$(awk '{print $2}' <<< "$parsed")"
    port="$(awk '{print $3}' <<< "$parsed")"
    if [[ "$host" != "-" && -n "$host" && "$status" != "deploying" && "$status" != "pending" ]]; then
      REMOTE_HOST="$host"
      REMOTE_SSH_PORT="${port:-22}"
      break
    fi
    log "Instance status=$status host=$host; retrying"
  fi
  sleep 15
done
if [[ -z "$REMOTE_HOST" ]]; then
  echo "Instance did not become SSH-ready before timeout." >&2
  exit 4
fi

SSH_OPTS=(
  -i "$SSH_KEY_FILE"
  -o StrictHostKeyChecking=accept-new
  -o UserKnownHostsFile="$KNOWN_HOSTS_FILE"
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=4
  -p "$REMOTE_SSH_PORT"
)
SCP_OPTS=(
  -i "$SSH_KEY_FILE"
  -o StrictHostKeyChecking=accept-new
  -o UserKnownHostsFile="$KNOWN_HOSTS_FILE"
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=4
  -P "$REMOTE_SSH_PORT"
)
SSH_TARGET="$REMOTE_USER@$REMOTE_HOST"

log "Preflighting GPU host $REMOTE_HOST:$REMOTE_SSH_PORT"
ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true; mkdir -p '$REMOTE_BASE' '$REMOTE_REPO'"

log "Syncing repository to remote $REMOTE_REPO"
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
  | ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "rm -rf '$REMOTE_REPO' && mkdir -p '$REMOTE_REPO' && tar -xzf - -C '$REMOTE_REPO'"

{
  source "$B2_ENV_FILE"
  printf 'export B2_KEY_ID=%q\n' "${B2_KEY_ID:-${B2_KEYID:-}}"
  printf 'export B2_APP_KEY=%q\n' "${B2_APP_KEY:-${B2_APPKEY:-}}"
  printf 'export B2_TOKEN=%q\n' "${B2_TOKEN:-}"
  if [[ -n "${HF_TOKEN:-}" ]]; then
    printf 'export HF_TOKEN=%q\n' "$HF_TOKEN"
  elif [[ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
    printf 'export HF_TOKEN=%q\n' "$HUGGING_FACE_HUB_TOKEN"
  fi
} > "$REMOTE_ENV_LOCAL"
chmod 600 "$REMOTE_ENV_LOCAL"
scp "${SCP_OPTS[@]}" "$REMOTE_ENV_LOCAL" "$SSH_TARGET:$REMOTE_BASE/.clearmesh_remote_env" >/dev/null

cat > "$REMOTE_SCRIPT_LOCAL" <<REMOTE
#!/usr/bin/env bash
set -euo pipefail
BASE=$(printf '%q' "$REMOTE_BASE")
REPO=$(printf '%q' "$REMOTE_REPO")
VENV=$(printf '%q' "$REMOTE_VENV")
TRELLIS2_DIR=$(printf '%q' "$REMOTE_TRELLIS2_DIR")
HIDREAM_DIR=$(printf '%q' "$REMOTE_HIDREAM_DIR")
MODEL_BUNDLE=$(printf '%q' "$REMOTE_MODEL_BUNDLE")
WORK_ROOT=$(printf '%q' "$REMOTE_WORK_ROOT")
STATE_ROOT=$(printf '%q' "$REMOTE_STATE_ROOT")
SERVER_LOG=$(printf '%q' "$REMOTE_SERVER_LOG")
SERVER_PID=$(printf '%q' "$REMOTE_SERVER_PID")
REMOTE_PORT=$(printf '%q' "$REMOTE_PORT")
B2_BUCKET=$(printf '%q' "$B2_BUCKET")
B2_MODEL_PREFIX=$(printf '%q' "$B2_MODEL_PREFIX")
EXPECTED_SHA=$(printf '%q' "$EXPECTED_CHECKPOINT_SHA")
TEXT_TO_IMAGE_MODEL=$(printf '%q' "$TEXT_TO_IMAGE_MODEL")
TEXT_TO_IMAGE_BACKEND=$(printf '%q' "$TEXT_TO_IMAGE_BACKEND")
HIDREAM_REPO_URL=$(printf '%q' "$HIDREAM_REPO_URL")
TRELLIS_MODEL=$(printf '%q' "$TRELLIS_MODEL")
INSTALL_TRELLIS2=$(printf '%q' "$INSTALL_TRELLIS2")
INSTALL_TRELLIS2_EXTENSIONS=$(printf '%q' "$INSTALL_TRELLIS2_EXTENSIONS")
INSTALL_HIDREAM=$(printf '%q' "$INSTALL_HIDREAM")
PIPELINE_DRY_RUN=$(printf '%q' "$PIPELINE_DRY_RUN")
ENV_FILE="\$BASE/.clearmesh_remote_env"
source "\$ENV_FILE"
mkdir -p "\$BASE" "\$MODEL_BUNDLE" "\$WORK_ROOT" "\$STATE_ROOT"
cd "\$REPO"
export PYTHONPATH="\$REPO:\$TRELLIS2_DIR:\${PYTHONPATH:-}"
export OPENCV_IO_ENABLE_OPENEXR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="\$BASE/hf-cache"
export TRANSFORMERS_CACHE="\$BASE/hf-cache/transformers"
export HF_HUB_CACHE="\$BASE/hf-cache/hub"
export CLEARMESH_REPO_ROOT="\$REPO"
export CLEARMESH_TRELLIS2_DIR="\$TRELLIS2_DIR"
export CLEARMESH_FACEQ_MODEL_BUNDLE="\$MODEL_BUNDLE"
export CLEARMESH_TEXT_TO_IMAGE_MODEL="\$TEXT_TO_IMAGE_MODEL"
export CLEARMESH_TEXT_TO_IMAGE_BACKEND="\$TEXT_TO_IMAGE_BACKEND"
export CLEARMESH_HIDREAM_DIR="\$HIDREAM_DIR"
export CLEARMESH_TEXT_TO_IMAGE_FALLBACK_MODEL="\${CLEARMESH_TEXT_TO_IMAGE_FALLBACK_MODEL:-stabilityai/stable-diffusion-xl-base-1.0}"
export CLEARMESH_TEXT_TO_IMAGE_FALLBACK_STEPS="\${CLEARMESH_TEXT_TO_IMAGE_FALLBACK_STEPS:-28}"
export CLEARMESH_TEXT_TO_IMAGE_FALLBACK_GUIDANCE_SCALE="\${CLEARMESH_TEXT_TO_IMAGE_FALLBACK_GUIDANCE_SCALE:-7.0}"
export CLEARMESH_TRELLIS_MODEL="\$TRELLIS_MODEL"
export CLEARMESH_TRELLIS_DECIMATION_TARGET="\${CLEARMESH_TRELLIS_DECIMATION_TARGET:-80000}"
export CLEARMESH_FACEQ_PROXY_DECIMATION_TARGET="\${CLEARMESH_FACEQ_PROXY_DECIMATION_TARGET:-4096}"
export CLEARMESH_TRELLIS_TEXTURE_SIZE=1024
export CLEARMESH_TRELLIS_REMESH="\${CLEARMESH_TRELLIS_REMESH:-0}"
export CLEARMESH_TRELLIS_TIMEOUT_SECONDS="\${CLEARMESH_TRELLIS_TIMEOUT_SECONDS:-1200}"
export CLEARMESH_TRELLIS_MAX_ATTEMPTS="\${CLEARMESH_TRELLIS_MAX_ATTEMPTS:-2}"
export CLEARMESH_TRELLIS_ACCEPTANCE_MIN_FACES="\${CLEARMESH_TRELLIS_ACCEPTANCE_MIN_FACES:-1024}"
export CLEARMESH_TRELLIS_ACCEPTANCE_MAX_COMPONENTS="\${CLEARMESH_TRELLIS_ACCEPTANCE_MAX_COMPONENTS:-24}"
export CLEARMESH_TRELLIS_ACCEPTANCE_MIN_LARGEST_COMPONENT_RATIO="\${CLEARMESH_TRELLIS_ACCEPTANCE_MIN_LARGEST_COMPONENT_RATIO:-0.55}"
export CLEARMESH_TRELLIS_ACCEPTANCE_MIN_TWO_COMPONENT_LARGEST_RATIO="\${CLEARMESH_TRELLIS_ACCEPTANCE_MIN_TWO_COMPONENT_LARGEST_RATIO:-0.45}"
export CLEARMESH_ALLOW_TRELLIS_ONLY="\${CLEARMESH_ALLOW_TRELLIS_ONLY:-1}"
export CLEARMESH_PUBLISH_TRELLIS_PREVIEW_BEFORE_FACEQ="\${CLEARMESH_PUBLISH_TRELLIS_PREVIEW_BEFORE_FACEQ:-1}"
export CLEARMESH_MESH_QC_SUBPROCESS="\${CLEARMESH_MESH_QC_SUBPROCESS:-1}"
export CLEARMESH_MESH_QC_MEMORY_MB="\${CLEARMESH_MESH_QC_MEMORY_MB:-4096}"
export CLEARMESH_MESH_QC_TIMEOUT_SECONDS="\${CLEARMESH_MESH_QC_TIMEOUT_SECONDS:-45}"
export CLEARMESH_TEXTURE_UV_ENABLED="\${CLEARMESH_TEXTURE_UV_ENABLED:-1}"
export CLEARMESH_TEXTURE_UV_MODE="\${CLEARMESH_TEXTURE_UV_MODE:-auto}"
export CLEARMESH_TEXTURE_UV_TIMEOUT_SECONDS="\${CLEARMESH_TEXTURE_UV_TIMEOUT_SECONDS:-600}"
export CLEARMESH_TEXTURE_UV_USE_BLENDER="\${CLEARMESH_TEXTURE_UV_USE_BLENDER:-0}"
export CLEARMESH_PIPELINE_DRY_RUN="\$PIPELINE_DRY_RUN"
export CLEARMESH_EASY3E_ENABLED="\${CLEARMESH_EASY3E_ENABLED:-0}"
export CLEARMESH_FACEQ_POINT_SAMPLES="\${CLEARMESH_FACEQ_POINT_SAMPLES:-8192}"
export CLEARMESH_FACEQ_GENERATION_MAX_FACES="\${CLEARMESH_FACEQ_GENERATION_MAX_FACES:-512}"
export CLEARMESH_FACEQ_DECODE_MODE="\${CLEARMESH_FACEQ_DECODE_MODE:-boundary_edge}"
export CLEARMESH_FACEQ_CONSTRAINT_TOP_K="\${CLEARMESH_FACEQ_CONSTRAINT_TOP_K:-12}"
export CLEARMESH_FACEQ_REPAIR_MODE="\${CLEARMESH_FACEQ_REPAIR_MODE:-manifold}"
export CLEARMESH_FACEQ_BOUNDARY_FILL="\${CLEARMESH_FACEQ_BOUNDARY_FILL:-centroid}"
export CLEARMESH_FACEQ_ACCEPTANCE_ENABLE="\${CLEARMESH_FACEQ_ACCEPTANCE_ENABLE:-1}"
export CLEARMESH_FACEQ_RETRY_ON_REJECT="\${CLEARMESH_FACEQ_RETRY_ON_REJECT:-1}"
export CLEARMESH_FACEQ_RETRY_POINT_SAMPLES="\${CLEARMESH_FACEQ_RETRY_POINT_SAMPLES:-16384}"
export CLEARMESH_FACEQ_RETRY_GENERATION_MAX_FACES="\${CLEARMESH_FACEQ_RETRY_GENERATION_MAX_FACES:-768}"
export CLEARMESH_FACEQ_RETRY_CONSTRAINT_TOP_K="\${CLEARMESH_FACEQ_RETRY_CONSTRAINT_TOP_K:-24}"
export CLEARMESH_FACEQ_TIMEOUT_SECONDS="\${CLEARMESH_FACEQ_TIMEOUT_SECONDS:-360}"
export CLEARMESH_FACEQ_ACCEPTANCE_MIN_FACES="\${CLEARMESH_FACEQ_ACCEPTANCE_MIN_FACES:-256}"
export CLEARMESH_FACEQ_ACCEPTANCE_MIN_FACE_RATIO="\${CLEARMESH_FACEQ_ACCEPTANCE_MIN_FACE_RATIO:-0.06}"
export CLEARMESH_FACEQ_ACCEPTANCE_MAX_EXTENT_RATIO="\${CLEARMESH_FACEQ_ACCEPTANCE_MAX_EXTENT_RATIO:-35}"
export CLEARMESH_FACEQ_ACCEPTANCE_MAX_CHAMFER_NORM="\${CLEARMESH_FACEQ_ACCEPTANCE_MAX_CHAMFER_NORM:-0.5}"
unset CLEARMESH_FACEQ_NO_BOUNDARY_BUDGET
unset CLEARMESH_FACEQ_NO_VERTEX_LINK_CONSTRAINT

SUDO=()
if command -v sudo >/dev/null 2>&1; then SUDO=(sudo); fi
"\${SUDO[@]}" apt-get update
"\${SUDO[@]}" env DEBIAN_FRONTEND=noninteractive apt-get install -y git curl rclone python3-venv python3-dev build-essential gcc g++ libgl1 libglib2.0-0
if command -v g++ >/dev/null 2>&1; then
  cc1plus_path="\$(g++ -print-prog-name=cc1plus 2>/dev/null || true)"
  if [[ -n "\$cc1plus_path" && -x "\$cc1plus_path" && ! -x "\$(command -v cc1plus 2>/dev/null || true)" ]]; then
    "\${SUDO[@]}" ln -sf "\$cc1plus_path" /usr/local/bin/cc1plus
  fi
  export CXX="\${CXX:-\$(command -v g++)}"
  export CUDAHOSTCXX="\${CUDAHOSTCXX:-\$CXX}"
fi

if [[ "\$INSTALL_TRELLIS2" = "1" ]]; then
  TRELLIS2_DIR="\$TRELLIS2_DIR" \
  TRELLIS2_VENV="\$VENV" \
  INSTALL_EXTENSIONS="\$INSTALL_TRELLIS2_EXTENSIONS" \
  INSTALL_FLASH_ATTN=1 \
  INSTALL_CUDA_TOOLKIT=1 \
  bash "\$REPO/scripts/setup/install_trellis2_env.sh"
else
  if [[ ! -x "\$VENV/bin/python" ]]; then python3 -m venv "\$VENV"; fi
fi

source "\$VENV/bin/activate"
python -m pip install -U pip setuptools wheel
python -m pip install fastapi 'uvicorn[standard]' diffusers transformers accelerate safetensors pillow trimesh scipy networkx numpy tqdm huggingface_hub einops flask openai sentencepiece protobuf
python - <<'PY'
import torch
print('torch_ready', torch.__version__, 'cuda', torch.cuda.is_available())
PY

if [[ "\$INSTALL_HIDREAM" = "1" ]]; then
  if [[ -d "\$HIDREAM_DIR/.git" ]]; then
    git -C "\$HIDREAM_DIR" fetch --depth 1 origin main
    git -C "\$HIDREAM_DIR" reset --hard FETCH_HEAD
  else
    rm -rf "\$HIDREAM_DIR"
    git clone --depth 1 "\$HIDREAM_REPO_URL" "\$HIDREAM_DIR"
  fi
  if [[ ! -f "\$HIDREAM_DIR/inference.py" ]]; then
    echo "HiDream inference.py missing after clone: \$HIDREAM_DIR" >&2
    exit 21
  fi
fi

export RCLONE_CONFIG_B2ENV_TYPE=b2
export RCLONE_CONFIG_B2ENV_ACCOUNT="\$B2_KEY_ID"
export RCLONE_CONFIG_B2ENV_KEY="\$B2_APP_KEY"
rclone copy "b2env:\$B2_BUCKET/\$B2_MODEL_PREFIX" "\$MODEL_BUNDLE" --stats 30s --transfers 4 --checkers 8
actual_sha="\$(sha256sum "\$MODEL_BUNDLE/checkpoint.final.pt" | awk '{print \$1}')"
if [[ "\$actual_sha" != "\$EXPECTED_SHA" ]]; then
  echo "checkpoint sha mismatch: \$actual_sha" >&2
  exit 20
fi

if [[ -f "\$SERVER_PID" ]] && kill -0 "\$(cat "\$SERVER_PID")" 2>/dev/null; then
  kill "\$(cat "\$SERVER_PID")" || true
  sleep 2
fi
nohup python "\$REPO/scripts/inference/clearmesh_pipeline_server.py" \
  --host 127.0.0.1 \
  --port "\$REMOTE_PORT" \
  --work-root "\$WORK_ROOT" \
  --state-root "\$STATE_ROOT" \
  --model-bundle "\$MODEL_BUNDLE" \
  > "\$SERVER_LOG" 2>&1 < /dev/null &
echo \$! > "\$SERVER_PID"
sleep 5
curl -fsS "http://127.0.0.1:\$REMOTE_PORT/healthz"
echo
REMOTE
chmod 600 "$REMOTE_SCRIPT_LOCAL"
scp "${SCP_OPTS[@]}" "$REMOTE_SCRIPT_LOCAL" "$SSH_TARGET:/tmp/clearmesh_pipeline_bootstrap.sh" >/dev/null

log "Starting remote bootstrap in background; log=$REMOTE_LOG"
ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "chmod +x /tmp/clearmesh_pipeline_bootstrap.sh; nohup /tmp/clearmesh_pipeline_bootstrap.sh > '$REMOTE_LOG' 2>&1 < /dev/null & echo \$! > '$REMOTE_BOOTSTRAP_PID'"

if [[ "$START_TUNNEL" = "1" ]]; then
  TUNNEL_LOG="$OUT_DIR/ssh_tunnel.log"
  TUNNEL_PID="$OUT_DIR/ssh_tunnel.pid"
  if lsof -iTCP:"$LOCAL_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    log "Local port $LOCAL_PORT already has a listener; not starting another tunnel"
  else
    log "Starting local SSH tunnel http://127.0.0.1:$LOCAL_PORT -> remote 127.0.0.1:$REMOTE_PORT"
    nohup ssh "${SSH_OPTS[@]}" -N -L "$LOCAL_PORT:127.0.0.1:$REMOTE_PORT" "$SSH_TARGET" > "$TUNNEL_LOG" 2>&1 < /dev/null &
    echo $! > "$TUNNEL_PID"
  fi
fi

cat > "$RUN_INFO" <<JSON
{
  "provider": "runcrate",
  "instance_id": "$RUNCRATE_INSTANCE_ID",
  "instance_type_id": "$INSTANCE_TYPE_ID",
  "instance_name": "$INSTANCE_NAME",
  "host": "$REMOTE_HOST",
  "ssh_port": "$REMOTE_SSH_PORT",
  "remote_user": "$REMOTE_USER",
  "remote_base": "$REMOTE_BASE",
  "remote_repo": "$REMOTE_REPO",
  "remote_venv": "$REMOTE_VENV",
  "remote_trellis2_dir": "$REMOTE_TRELLIS2_DIR",
  "remote_hidream_dir": "$REMOTE_HIDREAM_DIR",
  "remote_model_bundle": "$REMOTE_MODEL_BUNDLE",
  "remote_log": "$REMOTE_LOG",
  "remote_server_log": "$REMOTE_SERVER_LOG",
  "remote_server_pid": "$REMOTE_SERVER_PID",
  "remote_bootstrap_pid": "$REMOTE_BOOTSTRAP_PID",
  "local_url": "http://127.0.0.1:$LOCAL_PORT",
  "dashboard_bridge_env": "export CLEARMESH_INFERENCE_BASE_URL=http://127.0.0.1:$LOCAL_PORT",
  "b2_model_prefix": "$B2_MODEL_PREFIX",
  "checkpoint_sha256": "$EXPECTED_CHECKPOINT_SHA",
  "text_to_image_model": "$TEXT_TO_IMAGE_MODEL",
  "text_to_image_backend": "$TEXT_TO_IMAGE_BACKEND",
  "trellis_model": "$TRELLIS_MODEL",
  "pipeline_dry_run": "$PIPELINE_DRY_RUN"
}
JSON
printf '%s\n' "$OUT_DIR" > .codex_outputs/latest_runcrate_pipeline_inference_dir.txt
log "Run info: $RUN_INFO"
