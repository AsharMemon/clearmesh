#!/usr/bin/env bash
# Create/preflight a Thunder A100 and launch the bounded FACE paper curated-corpus gate.
# The remote job builds a strict 128-bin corpus, trains with paper knobs, evaluates
# full-face AR, writes scale_readiness.json, and archives everything.
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
PRIMARY_DISK="${PRIMARY_DISK:-300}"
TEMPLATE="${TEMPLATE:-base}"
WAIT_INTERVAL_SEC="${WAIT_INTERVAL_SEC:-10}"
WAIT_TIMEOUT_SEC="${WAIT_TIMEOUT_SEC:-1800}"
PREFLIGHT_ATTEMPTS="${PREFLIGHT_ATTEMPTS:-6}"
PREFLIGHT_RETRY_SEC="${PREFLIGHT_RETRY_SEC:-30}"

RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-$REPO_ROOT/.codex_outputs/face_paper_corpus_gate_setup_$RUN_STAMP}"
REMOTE_REPO="${REMOTE_REPO:-/home/ubuntu/clearmesh}"
REMOTE_VENV="${REMOTE_VENV:-/home/ubuntu/clearmesh-venv}"
REMOTE_LOG="${REMOTE_LOG:-/tmp/clearmesh_face_paper_corpus_gate.nohup.log}"
REMOTE_PID="${REMOTE_PID:-/tmp/clearmesh_face_paper_corpus_gate.pid}"
REMOTE_LAB_ROOT="${REMOTE_LAB_ROOT:-/tmp/clearmesh_face_paper_corpus_gate_$RUN_STAMP}"
REQUIRE_NATIVE_MUON="${REQUIRE_NATIVE_MUON:-1}"
RUN_REMOTE_TESTS="${RUN_REMOTE_TESTS:-1}"
SYNC_HF_TOKEN="${SYNC_HF_TOKEN:-1}"

# Bounded rung defaults. Override these for larger rungs after the gate passes.
SELECT_TARGET="${SELECT_TARGET:-512}"
SCAN_LIMIT="${SCAN_LIMIT:-50000}"
CURATION_TARGET="${CURATION_TARGET:-$SELECT_TARGET}"
MIN_QUALITY="${MIN_QUALITY:-2}"
OVERSAMPLE_FACTOR="${OVERSAMPLE_FACTOR:-5}"
DOWNLOAD_PROCESSES="${DOWNLOAD_PROCESSES:-8}"
DOWNLOAD_BATCH_SIZE="${DOWNLOAD_BATCH_SIZE:-25}"
TARGET_FACES="${TARGET_FACES:-512}"
TOKEN_MAX_FACES="${TOKEN_MAX_FACES:-512}"
MODEL_MAX_FACES="${MODEL_MAX_FACES:-512}"
NUM_BINS="${NUM_BINS:-128}"
PAPER_WITHIN_FACE_ORDER="${PAPER_WITHIN_FACE_ORDER:-rotate_min_zyx}"
POINT_SAMPLES="${POINT_SAMPLES:-8192}"
VOXEL_RESOLUTION="${VOXEL_RESOLUTION:-64}"
MESH_VOXEL_MAX_FACES="${MESH_VOXEL_MAX_FACES:-5000}"
STEPS="${STEPS:-30000}"
BATCH_SIZE="${BATCH_SIZE:-1}"
VECSET_TOKENS="${VECSET_TOKENS:-2048}"
LATENT_DIM="${LATENT_DIM:-64}"
HIDDEN_SIZE="${HIDDEN_SIZE:-384}"
ENCODER_HIDDEN_SIZE="${ENCODER_HIDDEN_SIZE:-384}"
ENCODER_LAYERS="${ENCODER_LAYERS:-4}"
DECODER_LAYERS="${DECODER_LAYERS:-8}"
HEADS="${HEADS:-8}"
PRECISION="${PRECISION:-bf16}"
LOG_EVERY="${LOG_EVERY:-50}"
SELECTION_EVAL_EVERY="${SELECTION_EVAL_EVERY:-1000}"
SELECTION_EVAL_BATCH_SIZE="${SELECTION_EVAL_BATCH_SIZE:-1}"
SKIP_INITIAL_SELECTION_EVAL="${SKIP_INITIAL_SELECTION_EVAL:-0}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1000}"
SAVE_CURRENT_CHECKPOINT="${SAVE_CURRENT_CHECKPOINT:-1}"
PREFETCH_BATCHES="${PREFETCH_BATCHES:-1}"
CACHE_FPS_INDICES="${CACHE_FPS_INDICES:-0}"
DISABLE_AUGMENT="${DISABLE_AUGMENT:-0}"
TEACHER_FORCED_LIMIT="${TEACHER_FORCED_LIMIT:-0}"
AR_LIMIT="${AR_LIMIT:-20}"
AR_FACE_LIMIT="${AR_FACE_LIMIT:-0}"
TEACHER_PREFIX_LIMIT="${TEACHER_PREFIX_LIMIT:-16}"
TEACHER_PREFIX_FACE_COUNTS="${TEACHER_PREFIX_FACE_COUNTS:-1 4 16}"
PREDICTED_LIMIT="${PREDICTED_LIMIT:-5}"
PREDICTED_FACE_LIMIT="${PREDICTED_FACE_LIMIT:-0}"
PAIR_SAMPLES="${PAIR_SAMPLES:-500}"
MIN_SCALE_DATASET_SAMPLES="${MIN_SCALE_DATASET_SAMPLES:-256}"
FAIL_ON_SCALE_NOT_READY="${FAIL_ON_SCALE_NOT_READY:-0}"
CAUSAL_MLP_VARIANT="${CAUSAL_MLP_VARIANT:-legacy_concat}"
FACE_EMBEDDING_VARIANT="${FACE_EMBEDDING_VARIANT:-token_concat_project}"
DECODE_HEAD="${DECODE_HEAD:-causal}"
ALLOW_DEPRECATED_FACE_EMBEDDING="${ALLOW_DEPRECATED_FACE_EMBEDDING:-0}"
STRICT_FACE_PAPER_GATE="${STRICT_FACE_PAPER_GATE:-1}"
SEED="${SEED:-303}"

if [ -z "${THUNDER_TOKEN:-}" ]; then
  echo "THUNDER_TOKEN is not set." >&2
  exit 1
fi
if [ ! -x "$TNR_BIN" ]; then
  echo "tnr binary not found or not executable: $TNR_BIN" >&2
  exit 1
fi
if [ "$STRICT_FACE_PAPER_GATE" = "1" ]; then
  if [ "$FACE_EMBEDDING_VARIANT" != "token_concat_project" ]; then
    echo "Strict FACE paper gates require FACE_EMBEDDING_VARIANT=token_concat_project; got '$FACE_EMBEDDING_VARIANT'." >&2
    echo "Set STRICT_FACE_PAPER_GATE=0 only for an explicitly named compatibility ablation." >&2
    exit 7
  fi
  if [ "$ALLOW_DEPRECATED_FACE_EMBEDDING" = "1" ]; then
    echo "Strict FACE paper gates must not set ALLOW_DEPRECATED_FACE_EMBEDDING=1." >&2
    echo "Set STRICT_FACE_PAPER_GATE=0 only for an explicitly named compatibility ablation." >&2
    exit 7
  fi
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

if [ "$SYNC_HF_TOKEN" = "1" ]; then
  if [ -n "${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}" ]; then
    echo "Syncing Hugging Face token to $INSTANCE_ID..."
    THUNDER_INSTANCE_ID="$INSTANCE_ID" "$REPO_ROOT/scripts/thunder/sync_hf_token.sh" "$INSTANCE_ID"
  else
    echo "HF_TOKEN/HUGGINGFACE_HUB_TOKEN not set locally; remote corpus downloads may be rate limited." >&2
  fi
fi

remote_setup_log="$DOWNLOAD_ROOT/remote_setup_and_launch.log"
setup_status=0
cat <<REMOTE_SETUP | "$TNR_BIN" connect "$INSTANCE_ID" 2>&1 | tee "$remote_setup_log" || setup_status=$?
set -euo pipefail
REMOTE_REPO=$(printf '%q' "$REMOTE_REPO")
REMOTE_VENV=$(printf '%q' "$REMOTE_VENV")
REMOTE_LOG=$(printf '%q' "$REMOTE_LOG")
REMOTE_PID=$(printf '%q' "$REMOTE_PID")
REMOTE_LAB_ROOT=$(printf '%q' "$REMOTE_LAB_ROOT")
REQUIRE_NATIVE_MUON=$(printf '%q' "$REQUIRE_NATIVE_MUON")

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
if '\$REQUIRE_NATIVE_MUON' == '1' and not hasattr(torch.optim, 'Muon'):
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
print('paper_corpus_gate_python_deps_ok')
PY

if [ $(printf '%q' "$RUN_REMOTE_TESTS") = "1" ]; then
  python -m pytest -q \
    tests/test_face_tokens.py \
    tests/test_face_paper_decode_equivalence.py \
    tests/test_face_paper_optimizer.py \
    tests/test_face_paper_incremental.py \
    tests/test_face_dataset_gate.py \
    tests/test_face_paper_scale_readiness.py
else
  echo "[clearmesh] skipping remote pytest because RUN_REMOTE_TESTS=$RUN_REMOTE_TESTS"
fi

export REMOTE_REPO="\$REMOTE_REPO"
export LAB_ROOT="\$REMOTE_LAB_ROOT"
export SELECT_TARGET=$(printf '%q' "$SELECT_TARGET")
export SCAN_LIMIT=$(printf '%q' "$SCAN_LIMIT")
export CURATION_TARGET=$(printf '%q' "$CURATION_TARGET")
export MIN_QUALITY=$(printf '%q' "$MIN_QUALITY")
export OVERSAMPLE_FACTOR=$(printf '%q' "$OVERSAMPLE_FACTOR")
export DOWNLOAD_PROCESSES=$(printf '%q' "$DOWNLOAD_PROCESSES")
export DOWNLOAD_BATCH_SIZE=$(printf '%q' "$DOWNLOAD_BATCH_SIZE")
export TARGET_FACES=$(printf '%q' "$TARGET_FACES")
export TOKEN_MAX_FACES=$(printf '%q' "$TOKEN_MAX_FACES")
export MODEL_MAX_FACES=$(printf '%q' "$MODEL_MAX_FACES")
export NUM_BINS=$(printf '%q' "$NUM_BINS")
export PAPER_WITHIN_FACE_ORDER=$(printf '%q' "$PAPER_WITHIN_FACE_ORDER")
export POINT_SAMPLES=$(printf '%q' "$POINT_SAMPLES")
export VOXEL_RESOLUTION=$(printf '%q' "$VOXEL_RESOLUTION")
export MESH_VOXEL_MAX_FACES=$(printf '%q' "$MESH_VOXEL_MAX_FACES")
export STEPS=$(printf '%q' "$STEPS")
export BATCH_SIZE=$(printf '%q' "$BATCH_SIZE")
export VECSET_TOKENS=$(printf '%q' "$VECSET_TOKENS")
export LATENT_DIM=$(printf '%q' "$LATENT_DIM")
export HIDDEN_SIZE=$(printf '%q' "$HIDDEN_SIZE")
export ENCODER_HIDDEN_SIZE=$(printf '%q' "$ENCODER_HIDDEN_SIZE")
export ENCODER_LAYERS=$(printf '%q' "$ENCODER_LAYERS")
export DECODER_LAYERS=$(printf '%q' "$DECODER_LAYERS")
export HEADS=$(printf '%q' "$HEADS")
export PRECISION=$(printf '%q' "$PRECISION")
export LOG_EVERY=$(printf '%q' "$LOG_EVERY")
export SELECTION_EVAL_EVERY=$(printf '%q' "$SELECTION_EVAL_EVERY")
export SELECTION_EVAL_BATCH_SIZE=$(printf '%q' "$SELECTION_EVAL_BATCH_SIZE")
export SKIP_INITIAL_SELECTION_EVAL=$(printf '%q' "$SKIP_INITIAL_SELECTION_EVAL")
export CHECKPOINT_EVERY=$(printf '%q' "$CHECKPOINT_EVERY")
export SAVE_CURRENT_CHECKPOINT=$(printf '%q' "$SAVE_CURRENT_CHECKPOINT")
export PREFETCH_BATCHES=$(printf '%q' "$PREFETCH_BATCHES")
export CACHE_FPS_INDICES=$(printf '%q' "$CACHE_FPS_INDICES")
export DISABLE_AUGMENT=$(printf '%q' "$DISABLE_AUGMENT")
export TEACHER_FORCED_LIMIT=$(printf '%q' "$TEACHER_FORCED_LIMIT")
export AR_LIMIT=$(printf '%q' "$AR_LIMIT")
export AR_FACE_LIMIT=$(printf '%q' "$AR_FACE_LIMIT")
export TEACHER_PREFIX_LIMIT=$(printf '%q' "$TEACHER_PREFIX_LIMIT")
export TEACHER_PREFIX_FACE_COUNTS=$(printf '%q' "$TEACHER_PREFIX_FACE_COUNTS")
export PREDICTED_LIMIT=$(printf '%q' "$PREDICTED_LIMIT")
export PREDICTED_FACE_LIMIT=$(printf '%q' "$PREDICTED_FACE_LIMIT")
export PAIR_SAMPLES=$(printf '%q' "$PAIR_SAMPLES")
export MIN_SCALE_DATASET_SAMPLES=$(printf '%q' "$MIN_SCALE_DATASET_SAMPLES")
export FAIL_ON_SCALE_NOT_READY=$(printf '%q' "$FAIL_ON_SCALE_NOT_READY")
export CAUSAL_MLP_VARIANT=$(printf '%q' "$CAUSAL_MLP_VARIANT")
export FACE_EMBEDDING_VARIANT=$(printf '%q' "$FACE_EMBEDDING_VARIANT")
export DECODE_HEAD=$(printf '%q' "$DECODE_HEAD")
export ALLOW_DEPRECATED_FACE_EMBEDDING=$(printf '%q' "$ALLOW_DEPRECATED_FACE_EMBEDDING")
export STRICT_FACE_PAPER_GATE=$(printf '%q' "$STRICT_FACE_PAPER_GATE")
export SEED=$(printf '%q' "$SEED")

nohup bash scripts/thunder/face_paper_curated_corpus_gate.sh > "\$REMOTE_LOG" 2>&1 &
echo \$! > "\$REMOTE_PID"
probe_pid="\$(cat "\$REMOTE_PID")"
echo "paper_corpus_gate_pid=\$probe_pid"
echo "paper_corpus_gate_log=\$REMOTE_LOG"
echo "paper_corpus_gate_lab_root=\$REMOTE_LAB_ROOT"
echo "paper_corpus_gate_latest=/tmp/clearmesh_latest_face_paper_corpus_gate_run.txt"
echo CLEARMESH_FACE_PAPER_CORPUS_GATE_LAUNCHED
exit
REMOTE_SETUP

if [ "$setup_status" -ne 0 ]; then
  if python3 - "$remote_setup_log" <<'PY'
import re
import sys
from pathlib import Path
ansi = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
for raw in Path(sys.argv[1]).read_text(errors="ignore").splitlines():
    if ansi.sub("", raw).replace("\r", "").strip() == "CLEARMESH_FACE_PAPER_CORPUS_GATE_LAUNCHED":
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
    if ansi.sub("", raw).replace("\r", "").strip() == "CLEARMESH_FACE_PAPER_CORPUS_GATE_LAUNCHED":
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
  "remote_latest_file": "/tmp/clearmesh_latest_face_paper_corpus_gate_run.txt",
  "download_root": "$DOWNLOAD_ROOT",
  "select_target": $SELECT_TARGET,
  "curation_target": $CURATION_TARGET,
  "steps": $STEPS,
  "num_bins": $NUM_BINS,
  "paper_within_face_order": "$PAPER_WITHIN_FACE_ORDER",
  "token_max_faces": $TOKEN_MAX_FACES,
  "model_max_faces": $MODEL_MAX_FACES,
  "point_samples": $POINT_SAMPLES,
  "vecset_tokens": $VECSET_TOKENS,
  "latent_dim": $LATENT_DIM,
  "precision": "$PRECISION",
  "causal_mlp_variant": "$CAUSAL_MLP_VARIANT",
  "face_embedding_variant": "$FACE_EMBEDDING_VARIANT",
  "allow_deprecated_face_embedding": $([ "$ALLOW_DEPRECATED_FACE_EMBEDDING" = "1" ] && echo true || echo false),
  "strict_face_paper_gate": $([ "$STRICT_FACE_PAPER_GATE" = "1" ] && echo true || echo false),
  "decode_head": "$DECODE_HEAD",
  "selection_eval_every": $SELECTION_EVAL_EVERY,
  "selection_eval_batch_size": $SELECTION_EVAL_BATCH_SIZE,
  "skip_initial_selection_eval": $([ "$SKIP_INITIAL_SELECTION_EVAL" = "1" ] && echo true || echo false),
  "checkpoint_every": $CHECKPOINT_EVERY,
  "save_current_checkpoint": $([ "$SAVE_CURRENT_CHECKPOINT" = "1" ] && echo true || echo false),
  "cache_fps_indices": $([ "$CACHE_FPS_INDICES" = "1" ] && echo true || echo false),
  "disable_augment": $([ "$DISABLE_AUGMENT" = "1" ] && echo true || echo false),
  "predicted_limit": $PREDICTED_LIMIT,
  "predicted_face_limit": $PREDICTED_FACE_LIMIT,
  "teacher_prefix_limit": $TEACHER_PREFIX_LIMIT,
  "teacher_prefix_face_counts": "$TEACHER_PREFIX_FACE_COUNTS",
  "min_scale_dataset_samples": $MIN_SCALE_DATASET_SAMPLES,
  "run_remote_tests": $([ "$RUN_REMOTE_TESTS" = "1" ] && echo true || echo false)
}
JSON

echo "FACE paper corpus gate launched on Thunder instance $INSTANCE_ID."
echo "Local setup logs: $DOWNLOAD_ROOT"
echo "Remote nohup log: $REMOTE_LOG"
echo "Remote lab root: $REMOTE_LAB_ROOT"
