#!/usr/bin/env bash
# Create/preflight a Thunder A100 instance and launch the strict FACE paper-knob probe.
#
# Default behavior:
#   - creates one production A100
#   - waits until RUNNING
#   - verifies /dev/nvidia*, nvidia-smi, torch.cuda, bf16, and native Muon
#   - syncs the repo and strict-target seed tar
#   - installs a dedicated venv on the instance
#   - launches scripts/thunder/face_paper_a100_probe.sh under nohup
#   - deletes the created instance on setup/preflight failure only
#
# Useful overrides:
#   CREATE_INSTANCE=0 THUNDER_INSTANCE_ID=<id> scripts/thunder/launch_face_a100_probe_instance.sh
#   STEPS=500 CAUSAL_MLP_VARIANT=legacy_concat scripts/thunder/launch_face_a100_probe_instance.sh
set -euo pipefail

TNR_BIN="${TNR_BIN:-/Users/Ashar/.tnr/bin/tnr}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

INSTANCE_ID="${1:-${THUNDER_INSTANCE_ID:-}}"
CREATE_INSTANCE="${CREATE_INSTANCE:-1}"
DELETE_ON_FAILURE="${DELETE_ON_FAILURE:-1}"
CREATED_INSTANCE=0

GPU="${GPU:-a100}"
MODE="${MODE:-production}"
VCPUS="${VCPUS:-8}"
PRIMARY_DISK="${PRIMARY_DISK:-200}"
TEMPLATE="${TEMPLATE:-base}"
WAIT_INTERVAL_SEC="${WAIT_INTERVAL_SEC:-10}"
WAIT_TIMEOUT_SEC="${WAIT_TIMEOUT_SEC:-1800}"
PREFLIGHT_ATTEMPTS="${PREFLIGHT_ATTEMPTS:-6}"
PREFLIGHT_RETRY_SEC="${PREFLIGHT_RETRY_SEC:-30}"

RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-$REPO_ROOT/.codex_outputs/face_paper_a100_setup_$RUN_STAMP}"
SEED_TAR="${SEED_TAR:-$REPO_ROOT/.codex_outputs/a100_seed_data/clearmesh_face_objpp200_512_strict_targets.tar.gz}"
REMOTE_TAR="${REMOTE_TAR:-/tmp/clearmesh_face_objpp200_512_strict_targets.tar.gz}"
REMOTE_DATA_RUN="${REMOTE_DATA_RUN:-/tmp/clearmesh_face_objpp200_512_20260504_020301}"
REMOTE_LOG="${REMOTE_LOG:-/tmp/clearmesh_face_a100_probe.nohup.log}"
REMOTE_PID="${REMOTE_PID:-/tmp/clearmesh_face_a100_probe.pid}"
REMOTE_VENV="${REMOTE_VENV:-/home/ubuntu/clearmesh-venv}"
REMOTE_REPO="${REMOTE_REPO:-/home/ubuntu/clearmesh}"
REMOTE_LAB_ROOT="${REMOTE_LAB_ROOT:-/tmp/clearmesh_face_a100_probe_$RUN_STAMP}"

# Probe knobs. These default to the paper-faithful A100 gate, not the off-paper indexed lane.
TOKEN_MAX_FACES="${TOKEN_MAX_FACES:-512}"
MODEL_MAX_FACES="${MODEL_MAX_FACES:-512}"
POINT_SAMPLES="${POINT_SAMPLES:-8192}"
VECSET_TOKENS="${VECSET_TOKENS:-2048}"
LATENT_DIM="${LATENT_DIM:-64}"
CAUSAL_MLP_VARIANT="${CAUSAL_MLP_VARIANT:-legacy_concat}"
HIDDEN_SIZE="${HIDDEN_SIZE:-384}"
ENCODER_HIDDEN_SIZE="${ENCODER_HIDDEN_SIZE:-384}"
ENCODER_LAYERS="${ENCODER_LAYERS:-4}"
DECODER_LAYERS="${DECODER_LAYERS:-8}"
HEADS="${HEADS:-8}"
BATCH_SIZE="${BATCH_SIZE:-1}"
STEPS="${STEPS:-5000}"
LOG_EVERY="${LOG_EVERY:-100}"
SELECTION_EVAL_EVERY="${SELECTION_EVAL_EVERY:-1000}"
CAPACITY_STEPS="${CAPACITY_STEPS:-3}"
PRECISION="${PRECISION:-bf16}"
AR_LIMIT="${AR_LIMIT:-5}"
AR_FACE_LIMIT="${AR_FACE_LIMIT:-0}"
PAIR_SAMPLES="${PAIR_SAMPLES:-500}"
REQUIRE_NATIVE_MUON="${REQUIRE_NATIVE_MUON:-1}"

if [ -z "${THUNDER_TOKEN:-}" ]; then
  echo "THUNDER_TOKEN is not set." >&2
  exit 1
fi
if [ ! -x "$TNR_BIN" ]; then
  echo "tnr binary not found or not executable: $TNR_BIN" >&2
  exit 1
fi
if [ ! -f "$SEED_TAR" ]; then
  echo "seed tar not found: $SEED_TAR" >&2
  exit 2
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

mkdir -p "$DOWNLOAD_ROOT"

cleanup_instance() {
  local exit_code=$?
  if [ "$exit_code" -ne 0 ] && [ "$CREATED_INSTANCE" = "1" ] && [ "$DELETE_ON_FAILURE" = "1" ] && [ -n "$INSTANCE_ID" ]; then
    echo "Launcher failed with status $exit_code; deleting created Thunder instance $INSTANCE_ID." >&2
    "$TNR_BIN" delete "$INSTANCE_ID" --yes || true
  fi
}
trap cleanup_instance EXIT INT TERM HUP

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
  create_output="$($TNR_BIN "${create_args[@]}")" || {
    echo "Thunder create failed." >&2
    exit 3
  }
  printf '%s\n' "$create_output" > "$DOWNLOAD_ROOT/create.json"
  CREATED_INSTANCE=1
  if [ -z "$INSTANCE_ID" ]; then
    INSTANCE_ID="$(parse_create_id "$create_output")" || INSTANCE_ID=""
  fi
  echo "Created Thunder instance $INSTANCE_ID."
fi
if [ -z "$INSTANCE_ID" ]; then
  echo "Set THUNDER_INSTANCE_ID or CREATE_INSTANCE=1." >&2
  exit 4
fi

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
    running_seen=1
    break
  fi
  sleep "$WAIT_INTERVAL_SEC"
done
if [ "$running_seen" != "1" ]; then
  echo "Thunder instance $INSTANCE_ID did not reach RUNNING within ${WAIT_TIMEOUT_SEC}s." >&2
  exit 21
fi

echo "Preflighting GPU device before install..."
preflight_log="$DOWNLOAD_ROOT/gpu_preflight.log"
preflight_ok=0
for preflight_attempt in $(seq 1 "$PREFLIGHT_ATTEMPTS"); do
  echo "GPU preflight attempt $preflight_attempt/$PREFLIGHT_ATTEMPTS..."
  cat <<'REMOTE_PREFLIGHT' | "$TNR_BIN" connect "$INSTANCE_ID" 2>&1 | tee "$preflight_log" || true
set -euo pipefail
hostname
ls -la /dev/nvidia* || true
if ! compgen -G '/dev/nvidia[0-9]*' >/dev/null; then
  echo 'missing /dev/nvidia[0-9]*; Thunder GPU device is not mounted' >&2
  exit 20
fi
timeout 60s nvidia-smi
echo CLEARMESH_GPU_PREFLIGHT_OK
exit
REMOTE_PREFLIGHT
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
  echo "GPU preflight failed for instance $INSTANCE_ID." >&2
  exit 20
fi

echo "Syncing repo to $INSTANCE_ID..."
THUNDER_INSTANCE_ID="$INSTANCE_ID" "$REPO_ROOT/scripts/thunder/sync_repo.sh" "$INSTANCE_ID"
echo "Uploading strict seed tar..."
"$TNR_BIN" scp "$SEED_TAR" "$INSTANCE_ID:$REMOTE_TAR"

remote_setup_log="$DOWNLOAD_ROOT/remote_setup_and_launch.log"
setup_status=0
cat <<REMOTE_SETUP | "$TNR_BIN" connect "$INSTANCE_ID" 2>&1 | tee "$remote_setup_log" || setup_status=$?
set -euo pipefail
REMOTE_REPO=$(printf '%q' "$REMOTE_REPO")
REMOTE_VENV=$(printf '%q' "$REMOTE_VENV")
REMOTE_TAR=$(printf '%q' "$REMOTE_TAR")
REMOTE_DATA_RUN=$(printf '%q' "$REMOTE_DATA_RUN")
REMOTE_LOG=$(printf '%q' "$REMOTE_LOG")
REMOTE_PID=$(printf '%q' "$REMOTE_PID")
REMOTE_LAB_ROOT=$(printf '%q' "$REMOTE_LAB_ROOT")
REQUIRE_NATIVE_MUON=$(printf '%q' "$REQUIRE_NATIVE_MUON")

mkdir -p "\$REMOTE_DATA_RUN"
tar -xzf "\$REMOTE_TAR" -C /tmp
echo "\$REMOTE_DATA_RUN" > /tmp/clearmesh_latest_face_objpp_run.txt
cd "\$REMOTE_REPO"

if ! compgen -G '/dev/nvidia[0-9]*' >/dev/null; then
  echo 'missing /dev/nvidia[0-9]* after sync; Thunder GPU device disappeared' >&2
  exit 20
fi
timeout 60s nvidia-smi

if [ ! -x "\$REMOTE_VENV/bin/python" ]; then
  python3 -m venv "\$REMOTE_VENV" || (sudo apt-get update && sudo apt-get install -y python3-venv && python3 -m venv "\$REMOTE_VENV")
fi
# shellcheck disable=SC1091
source "\$REMOTE_VENV/bin/activate"
python -m pip install -U pip setuptools wheel

if ! python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit(1)
if not hasattr(torch.optim, "Muon"):
    raise SystemExit(2)
print("torch_ready", torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), "bf16", torch.cuda.is_bf16_supported(), "native_muon", True)
PY
then
  echo '[clearmesh] installing CUDA PyTorch with native Muon target...'
  python -m pip install --upgrade --pre 'torch>=2.10,<2.11' --index-url https://download.pytorch.org/whl/cu128
fi

python - <<PY
import torch
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device', torch.cuda.get_device_name(0), 'bf16', torch.cuda.is_bf16_supported())
print('native_muon', hasattr(torch.optim, 'Muon'))
if not torch.cuda.is_available():
    raise SystemExit('torch.cuda unavailable')
if not torch.cuda.is_bf16_supported():
    raise SystemExit('bf16 unsupported on this GPU/runtime')
if '$REQUIRE_NATIVE_MUON' == '1' and not hasattr(torch.optim, 'Muon'):
    raise SystemExit('native torch.optim.Muon unavailable')
PY

python -m pip install -q -r requirements-data.txt pytest pillow scipy
python - <<'PY'
import fast_simplification  # noqa: F401
import networkx  # noqa: F401
import pytest  # noqa: F401
import scipy  # noqa: F401
import skimage  # noqa: F401
import trimesh  # noqa: F401
from PIL import Image  # noqa: F401
print('paper_probe_python_deps_ok')
PY

python -m pytest \
  tests/test_face_tokens.py \
  tests/test_face_paper_decode_equivalence.py \
  tests/test_face_paper_optimizer.py \
  tests/test_face_paper_incremental.py \
  tests/test_face_dataset_gate.py

export REMOTE_REPO="\$REMOTE_REPO"
export DATA_RUN="\$REMOTE_DATA_RUN"
export LAB_ROOT="\$REMOTE_LAB_ROOT"
export TOKEN_MAX_FACES=$(printf '%q' "$TOKEN_MAX_FACES")
export MODEL_MAX_FACES=$(printf '%q' "$MODEL_MAX_FACES")
export POINT_SAMPLES=$(printf '%q' "$POINT_SAMPLES")
export VECSET_TOKENS=$(printf '%q' "$VECSET_TOKENS")
export LATENT_DIM=$(printf '%q' "$LATENT_DIM")
export CAUSAL_MLP_VARIANT=$(printf '%q' "$CAUSAL_MLP_VARIANT")
export HIDDEN_SIZE=$(printf '%q' "$HIDDEN_SIZE")
export ENCODER_HIDDEN_SIZE=$(printf '%q' "$ENCODER_HIDDEN_SIZE")
export ENCODER_LAYERS=$(printf '%q' "$ENCODER_LAYERS")
export DECODER_LAYERS=$(printf '%q' "$DECODER_LAYERS")
export HEADS=$(printf '%q' "$HEADS")
export BATCH_SIZE=$(printf '%q' "$BATCH_SIZE")
export STEPS=$(printf '%q' "$STEPS")
export LOG_EVERY=$(printf '%q' "$LOG_EVERY")
export SELECTION_EVAL_EVERY=$(printf '%q' "$SELECTION_EVAL_EVERY")
export CAPACITY_STEPS=$(printf '%q' "$CAPACITY_STEPS")
export PRECISION=$(printf '%q' "$PRECISION")
export AR_LIMIT=$(printf '%q' "$AR_LIMIT")
export AR_FACE_LIMIT=$(printf '%q' "$AR_FACE_LIMIT")
export PAIR_SAMPLES=$(printf '%q' "$PAIR_SAMPLES")

nohup bash scripts/thunder/face_paper_a100_probe.sh > "\$REMOTE_LOG" 2>&1 &
echo \$! > "\$REMOTE_PID"
probe_pid="\$(cat "\$REMOTE_PID")"
echo "a100_probe_pid=\$probe_pid"
echo "a100_probe_log=\$REMOTE_LOG"
echo "a100_probe_lab_root=\$REMOTE_LAB_ROOT"
echo "a100_probe_latest=/tmp/clearmesh_latest_face_a100_probe_run.txt"
echo CLEARMESH_FACE_PAPER_A100_LAUNCHED
exit
REMOTE_SETUP

if [ "$setup_status" -ne 0 ]; then
  if python3 - "$remote_setup_log" <<'PY'
import re
import sys
from pathlib import Path
ansi = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
for raw in Path(sys.argv[1]).read_text(errors="ignore").splitlines():
    if ansi.sub("", raw).replace("\r", "").strip() == "CLEARMESH_FACE_PAPER_A100_LAUNCHED":
        raise SystemExit(0)
raise SystemExit(1)
PY
  then
    echo "Remote launch marker observed despite connect exit status $setup_status; continuing."
  else
    echo "Remote setup failed before launch marker." >&2
    exit "$setup_status"
  fi
fi

if ! python3 - "$remote_setup_log" <<'PY'
import re
import sys
from pathlib import Path
ansi = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
for raw in Path(sys.argv[1]).read_text(errors="ignore").splitlines():
    if ansi.sub("", raw).replace("\r", "").strip() == "CLEARMESH_FACE_PAPER_A100_LAUNCHED":
        raise SystemExit(0)
raise SystemExit(1)
PY
then
  echo "Remote setup completed without launch marker; treating as failure." >&2
  exit 22
fi

cat > "$DOWNLOAD_ROOT/run_info.json" <<JSON
{
  "instance_id": "$INSTANCE_ID",
  "created_instance": $CREATED_INSTANCE,
  "gpu": "$GPU",
  "mode": "$MODE",
  "remote_log": "$REMOTE_LOG",
  "remote_pid": "$REMOTE_PID",
  "remote_lab_root": "$REMOTE_LAB_ROOT",
  "remote_latest_file": "/tmp/clearmesh_latest_face_a100_probe_run.txt",
  "download_root": "$DOWNLOAD_ROOT",
  "seed_tar": "$SEED_TAR",
  "steps": $STEPS,
  "token_max_faces": $TOKEN_MAX_FACES,
  "model_max_faces": $MODEL_MAX_FACES,
  "point_samples": $POINT_SAMPLES,
  "vecset_tokens": $VECSET_TOKENS,
  "latent_dim": $LATENT_DIM,
  "causal_mlp_variant": "$CAUSAL_MLP_VARIANT",
  "precision": "$PRECISION"
}
JSON

echo "FACE paper A100 probe launched on Thunder instance $INSTANCE_ID."
echo "Local setup logs: $DOWNLOAD_ROOT"
echo "Remote nohup log: $REMOTE_LOG"
echo "Remote lab root: $REMOTE_LAB_ROOT"
