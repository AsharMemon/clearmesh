#!/usr/bin/env bash
# Launch the paper-faithful FACE train/eval gate on an existing Thunder instance
# using a prebuilt strict split. Use this after corpus prep has already produced
# split_pass/train and split_pass/test, so GPU time is used for training/eval only.
set -euo pipefail

TNR_BIN="${TNR_BIN:-/Users/Ashar/.tnr/bin/tnr}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

INSTANCE_ID="${1:-${THUNDER_INSTANCE_ID:-}}"
REMOTE_REPO="${REMOTE_REPO:-/home/ubuntu/clearmesh}"
REMOTE_VENV="${REMOTE_VENV:-/home/ubuntu/clearmesh-venv}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-$REPO_ROOT/.codex_outputs/face_paper_existing_split_setup_$RUN_STAMP}"
REMOTE_LOG="${REMOTE_LOG:-/tmp/clearmesh_face_paper_existing_split_gate.nohup.log}"
REMOTE_PID="${REMOTE_PID:-/tmp/clearmesh_face_paper_existing_split_gate.pid}"
REMOTE_LAB_ROOT="${REMOTE_LAB_ROOT:-/tmp/clearmesh_face_paper_existing_split_gate_$RUN_STAMP}"
REQUIRE_NATIVE_MUON="${REQUIRE_NATIVE_MUON:-1}"
RUN_REMOTE_TESTS="${RUN_REMOTE_TESTS:-1}"
RUN_LABEL="${RUN_LABEL:-paper_existing_split_128_vec2048_muon_aug}"

# Data source. Either point at an existing remote DATA_RUN, or provide a local
# directory that will be archived and uploaded to /tmp on the Thunder instance.
DATA_RUN="${DATA_RUN:-${REMOTE_DATA_RUN:-}}"
SPLIT_DIR="${SPLIT_DIR:-}"
LOCAL_DATA_DIR="${LOCAL_DATA_DIR:-}"
REMOTE_DATA_PARENT="${REMOTE_DATA_PARENT:-/tmp}"
REMOTE_DATA_BASENAME="${REMOTE_DATA_BASENAME:-clearmesh_face_prebuilt_data_$RUN_STAMP}"

# Training knobs. Defaults mirror the strict paper lane.
STEPS="${STEPS:-30000}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_BINS="${NUM_BINS:-128}"
POINT_SAMPLES="${POINT_SAMPLES:-8192}"
MODEL_MAX_FACES="${MODEL_MAX_FACES:-512}"
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
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-2000}"
SAVE_CURRENT_CHECKPOINT="${SAVE_CURRENT_CHECKPOINT:-0}"
PREFETCH_BATCHES="${PREFETCH_BATCHES:-1}"
CACHE_FPS_INDICES="${CACHE_FPS_INDICES:-0}"
TRAIN_LIMIT="${TRAIN_LIMIT:-0}"
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
RELAX_PAPER_KNOBS="${RELAX_PAPER_KNOBS:-0}"
SEED="${SEED:-303}"
OPTIMIZER="${OPTIMIZER:-muon}"
LR="${LR:-0.0006}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
EOS_LOSS_WEIGHT="${EOS_LOSS_WEIGHT:-0.05}"
DISABLE_AUGMENT="${DISABLE_AUGMENT:-0}"
AUGMENT_ROTATION="${AUGMENT_ROTATION:-so3}"
AUGMENT_SCALE_MIN="${AUGMENT_SCALE_MIN:-0.75}"
AUGMENT_SCALE_MAX="${AUGMENT_SCALE_MAX:-1.25}"
AUGMENT_FLIP_PROB="${AUGMENT_FLIP_PROB:-0.5}"
AUGMENT_DIAGNOSTICS="${AUGMENT_DIAGNOSTICS:-0}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"
LOCAL_INIT_CHECKPOINT="${LOCAL_INIT_CHECKPOINT:-}"
CAUSAL_MLP_VARIANT="${CAUSAL_MLP_VARIANT:-legacy_concat}"
FACE_EMBEDDING_VARIANT="${FACE_EMBEDDING_VARIANT:-token_concat_project}"
ALLOW_DEPRECATED_FACE_EMBEDDING="${ALLOW_DEPRECATED_FACE_EMBEDDING:-0}"
DECODE_HEAD="${DECODE_HEAD:-causal}"
STRICT_FACE_PAPER_GATE="${STRICT_FACE_PAPER_GATE:-1}"

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
if [ -z "$INSTANCE_ID" ]; then
  echo "Set THUNDER_INSTANCE_ID or pass instance id as the first argument." >&2
  exit 2
fi
if [ -z "$DATA_RUN" ] && [ -z "$LOCAL_DATA_DIR" ]; then
  echo "Set DATA_RUN/REMOTE_DATA_RUN for an existing remote corpus, or LOCAL_DATA_DIR to upload one." >&2
  exit 2
fi
if [ -n "$LOCAL_DATA_DIR" ] && [ ! -d "$LOCAL_DATA_DIR" ]; then
  echo "LOCAL_DATA_DIR not found: $LOCAL_DATA_DIR" >&2
  exit 2
fi
if [ -n "$LOCAL_INIT_CHECKPOINT" ] && [ ! -f "$LOCAL_INIT_CHECKPOINT" ]; then
  echo "LOCAL_INIT_CHECKPOINT not found: $LOCAL_INIT_CHECKPOINT" >&2
  exit 2
fi

mkdir -p "$DOWNLOAD_ROOT"

echo "Preflighting Thunder instance $INSTANCE_ID..."
cat <<'REMOTE_PREFLIGHT' | "$TNR_BIN" connect "$INSTANCE_ID" 2>&1 | tee "$DOWNLOAD_ROOT/gpu_preflight.log"
set -euo pipefail
hostname
ls -la /dev/nvidia* || true
# Some Thunder runtimes expose the GPU to nvidia-smi/Torch without listing
# /dev/nvidia* in the login shell, so the functional probe is the source of
# truth here.
nvidia-smi
echo CLEARMESH_GPU_PREFLIGHT_OK
exit
REMOTE_PREFLIGHT

echo "Syncing repo..."
THUNDER_INSTANCE_ID="$INSTANCE_ID" "$REPO_ROOT/scripts/thunder/sync_repo.sh" "$INSTANCE_ID"

if [ -n "$LOCAL_DATA_DIR" ]; then
  local_archive="$DOWNLOAD_ROOT/$(basename "$LOCAL_DATA_DIR")_$RUN_STAMP.tar.gz"
  remote_archive="/tmp/$(basename "$local_archive")"
  remote_data_run="$REMOTE_DATA_PARENT/$REMOTE_DATA_BASENAME"
  echo "Packaging LOCAL_DATA_DIR=$LOCAL_DATA_DIR -> $local_archive"
  COPYFILE_DISABLE=1 tar --no-xattrs -czf "$local_archive" -C "$(dirname "$LOCAL_DATA_DIR")" "$(basename "$LOCAL_DATA_DIR")"
  echo "Uploading corpus archive to $INSTANCE_ID:$remote_archive"
  "$TNR_BIN" scp "$local_archive" "$INSTANCE_ID:$remote_archive"
  cat <<REMOTE_EXTRACT | "$TNR_BIN" connect "$INSTANCE_ID" 2>&1 | tee "$DOWNLOAD_ROOT/data_upload.log"
set -euo pipefail
rm -rf $(printf '%q' "$remote_data_run")
mkdir -p $(printf '%q' "$REMOTE_DATA_PARENT")
mkdir -p $(printf '%q' "$remote_data_run")
tar --strip-components=1 -xzf $(printf '%q' "$remote_archive") -C $(printf '%q' "$remote_data_run")
find $(printf '%q' "$remote_data_run") -maxdepth 3 -type f | wc -l
exit
REMOTE_EXTRACT
  DATA_RUN="$remote_data_run"
fi

if [ -n "$LOCAL_INIT_CHECKPOINT" ]; then
  remote_init_checkpoint="/tmp/$(basename "$LOCAL_INIT_CHECKPOINT" .pt)_$RUN_STAMP.pt"
  echo "Uploading init checkpoint to $INSTANCE_ID:$remote_init_checkpoint"
  "$TNR_BIN" scp "$LOCAL_INIT_CHECKPOINT" "$INSTANCE_ID:$remote_init_checkpoint"
  INIT_CHECKPOINT="$remote_init_checkpoint"
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
DATA_RUN=$(printf '%q' "$DATA_RUN")
SPLIT_DIR=$(printf '%q' "$SPLIT_DIR")
REQUIRE_NATIVE_MUON=$(printf '%q' "$REQUIRE_NATIVE_MUON")

cd "\$REMOTE_REPO"
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
if [ $(printf '%q' "$RUN_REMOTE_TESTS") = "1" ]; then
  python -m pytest -q \
    tests/test_face_paper_decode_equivalence.py \
    tests/test_face_paper_optimizer.py \
    tests/test_face_paper_incremental.py \
    tests/test_face_paper_scale_readiness.py
else
  echo "[clearmesh] skipping remote pytest because RUN_REMOTE_TESTS=$RUN_REMOTE_TESTS"
fi

export REMOTE_REPO="\$REMOTE_REPO"
export LAB_ROOT="\$REMOTE_LAB_ROOT"
export RUN_LABEL=$(printf '%q' "$RUN_LABEL")
export DATA_RUN="\$DATA_RUN"
if [ -n "\$SPLIT_DIR" ]; then
  export SPLIT_DIR="\$SPLIT_DIR"
fi
export STEPS=$(printf '%q' "$STEPS")
export BATCH_SIZE=$(printf '%q' "$BATCH_SIZE")
export NUM_BINS=$(printf '%q' "$NUM_BINS")
export POINT_SAMPLES=$(printf '%q' "$POINT_SAMPLES")
export MODEL_MAX_FACES=$(printf '%q' "$MODEL_MAX_FACES")
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
export TRAIN_LIMIT=$(printf '%q' "$TRAIN_LIMIT")
export TEACHER_FORCED_LIMIT=$(printf '%q' "$TEACHER_FORCED_LIMIT")
export AR_LIMIT=$(printf '%q' "$AR_LIMIT")
export AR_FACE_LIMIT=$(printf '%q' "$AR_FACE_LIMIT")
export PREDICTED_LIMIT=$(printf '%q' "$PREDICTED_LIMIT")
export PREDICTED_FACE_LIMIT=$(printf '%q' "$PREDICTED_FACE_LIMIT")
export PAIR_SAMPLES=$(printf '%q' "$PAIR_SAMPLES")
export MIN_SCALE_DATASET_SAMPLES=$(printf '%q' "$MIN_SCALE_DATASET_SAMPLES")
export FAIL_ON_SCALE_NOT_READY=$(printf '%q' "$FAIL_ON_SCALE_NOT_READY")
export RELAX_PAPER_KNOBS=$(printf '%q' "$RELAX_PAPER_KNOBS")
export SEED=$(printf '%q' "$SEED")
export OPTIMIZER=$(printf '%q' "$OPTIMIZER")
export LR=$(printf '%q' "$LR")
export WEIGHT_DECAY=$(printf '%q' "$WEIGHT_DECAY")
export EOS_LOSS_WEIGHT=$(printf '%q' "$EOS_LOSS_WEIGHT")
export DISABLE_AUGMENT=$(printf '%q' "$DISABLE_AUGMENT")
export AUGMENT_ROTATION=$(printf '%q' "$AUGMENT_ROTATION")
export AUGMENT_SCALE_MIN=$(printf '%q' "$AUGMENT_SCALE_MIN")
export AUGMENT_SCALE_MAX=$(printf '%q' "$AUGMENT_SCALE_MAX")
export AUGMENT_FLIP_PROB=$(printf '%q' "$AUGMENT_FLIP_PROB")
export AUGMENT_DIAGNOSTICS=$(printf '%q' "$AUGMENT_DIAGNOSTICS")
export INIT_CHECKPOINT=$(printf '%q' "$INIT_CHECKPOINT")
export CAUSAL_MLP_VARIANT=$(printf '%q' "$CAUSAL_MLP_VARIANT")
export FACE_EMBEDDING_VARIANT=$(printf '%q' "$FACE_EMBEDDING_VARIANT")
export ALLOW_DEPRECATED_FACE_EMBEDDING=$(printf '%q' "$ALLOW_DEPRECATED_FACE_EMBEDDING")
export STRICT_FACE_PAPER_GATE=$(printf '%q' "$STRICT_FACE_PAPER_GATE")
export DECODE_HEAD=$(printf '%q' "$DECODE_HEAD")

nohup bash scripts/thunder/face_paper_existing_split_gate.sh > "\$REMOTE_LOG" 2>&1 &
echo \$! > "\$REMOTE_PID"
echo "paper_existing_split_gate_pid=\$(cat "\$REMOTE_PID")"
echo "paper_existing_split_gate_log=\$REMOTE_LOG"
echo "paper_existing_split_gate_lab_root=\$REMOTE_LAB_ROOT"
echo CLEARMESH_FACE_PAPER_EXISTING_SPLIT_GATE_LAUNCHED
exit
REMOTE_SETUP

if [ "$setup_status" -ne 0 ]; then
  echo "Remote setup failed before launch marker." >&2
  exit "$setup_status"
fi
if ! python3 - "$remote_setup_log" <<'PY'
import re
import sys
from pathlib import Path
ansi = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
for raw in Path(sys.argv[1]).read_text(errors="ignore").splitlines():
    if ansi.sub("", raw).replace("\r", "").strip() == "CLEARMESH_FACE_PAPER_EXISTING_SPLIT_GATE_LAUNCHED":
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
  "remote_log": "$REMOTE_LOG",
  "remote_pid": "$REMOTE_PID",
  "remote_lab_root": "$REMOTE_LAB_ROOT",
  "remote_latest_file": "/tmp/clearmesh_latest_face_paper_existing_split_gate_run.txt",
  "download_root": "$DOWNLOAD_ROOT",
  "data_run": "$DATA_RUN",
  "split_dir": "$SPLIT_DIR",
  "run_label": "$RUN_LABEL",
  "steps": $STEPS,
  "num_bins": $NUM_BINS,
  "model_max_faces": $MODEL_MAX_FACES,
  "point_samples": $POINT_SAMPLES,
  "vecset_tokens": $VECSET_TOKENS,
  "latent_dim": $LATENT_DIM,
  "precision": "$PRECISION",
  "optimizer": "$OPTIMIZER",
  "causal_mlp_variant": "$CAUSAL_MLP_VARIANT",
  "face_embedding_variant": "$FACE_EMBEDDING_VARIANT",
  "allow_deprecated_face_embedding": $([ "$ALLOW_DEPRECATED_FACE_EMBEDDING" = "1" ] && echo true || echo false),
  "strict_face_paper_gate": $([ "$STRICT_FACE_PAPER_GATE" = "1" ] && echo true || echo false),
  "decode_head": "$DECODE_HEAD",
  "selection_eval_every": $SELECTION_EVAL_EVERY,
  "skip_initial_selection_eval": $([ "$SKIP_INITIAL_SELECTION_EVAL" = "1" ] && echo true || echo false),
  "checkpoint_every": $CHECKPOINT_EVERY,
  "disable_augment": $([ "$DISABLE_AUGMENT" = "1" ] && echo true || echo false),
  "cache_fps_indices": $([ "$CACHE_FPS_INDICES" = "1" ] && echo true || echo false),
  "save_current_checkpoint": $([ "$SAVE_CURRENT_CHECKPOINT" = "1" ] && echo true || echo false),
  "train_limit": $TRAIN_LIMIT,
  "predicted_limit": $PREDICTED_LIMIT,
  "predicted_face_limit": $PREDICTED_FACE_LIMIT,
  "relax_paper_knobs": $([ "$RELAX_PAPER_KNOBS" = "1" ] && echo true || echo false),
  "run_remote_tests": $([ "$RUN_REMOTE_TESTS" = "1" ] && echo true || echo false),
  "init_checkpoint": "$INIT_CHECKPOINT",
  "local_init_checkpoint": "$LOCAL_INIT_CHECKPOINT",
  "augment_diagnostics": $([ "$AUGMENT_DIAGNOSTICS" = "1" ] && echo true || echo false),
  "min_scale_dataset_samples": $MIN_SCALE_DATASET_SAMPLES
}
JSON

echo "FACE paper existing-split gate launched on Thunder instance $INSTANCE_ID."
echo "Local setup logs: $DOWNLOAD_ROOT"
echo "Remote nohup log: $REMOTE_LOG"
echo "Remote lab root: $REMOTE_LAB_ROOT"
