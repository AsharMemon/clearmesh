#!/usr/bin/env bash
# Install and start the remote FACE corpus shard queue runner on a Thunder
# worker. This keeps a GPU rolling through annotation shards without a
# Codex-side automation loop.
set -euo pipefail

TNR_BIN="${TNR_BIN:-/Users/Ashar/.tnr/bin/tnr}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

INSTANCE_ID="${1:-${THUNDER_INSTANCE_ID:-}}"
CREATE_INSTANCE="${CREATE_INSTANCE:-0}"
CREATED_INSTANCE=0
GPU="${GPU:-a6000}"
MODE="${MODE:-prototyping}"
VCPUS="${VCPUS:-8}"
PRIMARY_DISK="${PRIMARY_DISK:-200}"
EPHEMERAL_DISK="${EPHEMERAL_DISK:-0}"
TEMPLATE="${TEMPLATE:-base}"
WAIT_INTERVAL_SEC="${WAIT_INTERVAL_SEC:-10}"
WAIT_TIMEOUT_SEC="${WAIT_TIMEOUT_SEC:-1800}"
WAIT_FOR_RUNNING="${WAIT_FOR_RUNNING:-$CREATE_INSTANCE}"
BOOTSTRAP_REMOTE="${BOOTSTRAP_REMOTE:-0}"
SYNC_HF_TOKEN="${SYNC_HF_TOKEN:-1}"
QUEUE_SHARD_IDS="${QUEUE_SHARD_IDS:?QUEUE_SHARD_IDS is required, e.g. '0033 0038 0043'}"
POOL_DIR="${POOL_DIR:-$REPO_ROOT/.codex_outputs/face_source_pool_objpp600k_20260508_061413_managed}"
SHARD_DIR="${SHARD_DIR:-$POOL_DIR/shards}"
REMOTE_REPO="${REMOTE_REPO:-/home/ubuntu/clearmesh}"
REMOTE_VENV="${REMOTE_VENV:-/home/ubuntu/clearmesh-data-venv}"
REMOTE_QUEUE_SOURCE_DIR="${REMOTE_QUEUE_SOURCE_DIR:-/tmp/clearmesh_face_corpus_queue}"
REMOTE_QUEUE_LOG_ROOT="${REMOTE_QUEUE_LOG_ROOT:-/tmp/clearmesh_face_corpus_queue_logs}"
REMOTE_QUEUE_LOG="${REMOTE_QUEUE_LOG:-$REMOTE_QUEUE_LOG_ROOT/queue_worker.nohup.log}"
REMOTE_QUEUE_PID="${REMOTE_QUEUE_PID:-$REMOTE_QUEUE_LOG_ROOT/queue_worker.pid}"
RUN_STAMP_PREFIX="${RUN_STAMP_PREFIX:-$(date -u +%Y%m%d)_poolA_queue}"
LAB_ROOT_PREFIX="${LAB_ROOT_PREFIX:-/tmp/clearmesh_face_corpus_shard}"
SYNC_QUEUE_FILES="${SYNC_QUEUE_FILES:-1}"

START_B2_UPLOAD="${START_B2_UPLOAD:-1}"
B2_BUCKET="${B2_BUCKET:-clearmesh-pairs}"
B2_PREFIX_ROOT="${B2_PREFIX_ROOT:-face-corpora/poolA-shards}"
B2_UPLOAD_INTERVAL_SECONDS="${B2_UPLOAD_INTERVAL_SECONDS:-600}"
REMOTE_B2_ENV="${REMOTE_B2_ENV:-}"
UPLOAD_B2_ENV="${UPLOAD_B2_ENV:-0}"
LOCAL_B2_ENV_FILE="${LOCAL_B2_ENV_FILE:-}"
SOURCE_KIND="${SOURCE_KIND:-objaversepp}"

WAIT_FOR_PID_FILE="${WAIT_FOR_PID_FILE:-}"
WAIT_POLL_SECONDS="${WAIT_POLL_SECONDS:-60}"
PREQUEUE_COMPLETED_ROOT="${PREQUEUE_COMPLETED_ROOT:-}"
PREQUEUE_B2_PREFIX="${PREQUEUE_B2_PREFIX:-}"
PREQUEUE_B2_UPLOAD_PID_FILE="${PREQUEUE_B2_UPLOAD_PID_FILE:-}"
CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-0}"

if [[ -z "${THUNDER_TOKEN:-}" ]]; then
  echo "THUNDER_TOKEN is not set." >&2
  exit 1
fi
if [[ ! -x "$TNR_BIN" ]]; then
  echo "tnr binary not found or not executable: $TNR_BIN" >&2
  exit 1
fi
case "$MODE:$GPU" in
  production:a100|production:h100|prototyping:a6000|prototyping:a100|prototyping:h100)
    ;;
  *)
    echo "Unsupported Thunder mode/GPU '$MODE:$GPU'." >&2
    exit 5
    ;;
esac
if [[ "$START_B2_UPLOAD" = "1" && -z "$REMOTE_B2_ENV" ]]; then
  echo "Set REMOTE_B2_ENV to an existing remote env file, or set UPLOAD_B2_ENV=1 with local B2 env vars." >&2
  exit 2
fi

mkdir -p "$REPO_ROOT/.codex_outputs/face_corpus_queue_setup"
setup_dir="$REPO_ROOT/.codex_outputs/face_corpus_queue_setup/instance_${INSTANCE_ID:-new}_${RUN_STAMP_PREFIX}"
mkdir -p "$setup_dir"

cleanup_instance() {
  local exit_code=$?
  if [[ "$exit_code" -ne 0 && "$CREATED_INSTANCE" = "1" && "${DELETE_ON_FAILURE:-1}" = "1" && -n "$INSTANCE_ID" ]]; then
    echo "Queue launcher failed with status $exit_code; deleting created Thunder instance $INSTANCE_ID." >&2
    "$TNR_BIN" delete "$INSTANCE_ID" --yes || true
  fi
}
# Detached queue launchers are often started under nohup/setsid and may need to
# survive the local Codex shell closing while a Thunder instance is still queued.
# Do not trap SIGHUP by default, or nohup cannot do its job.
if [[ "${TRAP_HUP:-0}" = "1" ]]; then
  trap cleanup_instance EXIT INT TERM HUP
else
  trap cleanup_instance EXIT INT TERM
fi

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

redact_create_output() {
  CREATE_OUTPUT="$1" python3 - <<'PY'
import json
import os
import re
import sys
text = os.environ.get("CREATE_OUTPUT", "")
decoder = json.JSONDecoder()
for match in re.finditer(r"[\[{]", text):
    try:
        payload, _ = decoder.raw_decode(text[match.start():])
    except json.JSONDecodeError:
        continue
    items = payload if isinstance(payload, list) else [payload]
    for item in items:
        if isinstance(item, dict) and "key" in item:
            item["key"] = "[redacted]"
    json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    raise SystemExit(0)
sys.stdout.write(text)
PY
}

run_remote_script_with_marker() {
  local instance_id="$1"
  local script_file="$2"
  local log_file="$3"
  local marker="$4"
  local timeout_sec="${5:-1800}"
  local payload remote_cmd pid deadline
  payload="$(base64 < "$script_file" | tr -d '\n')"
  remote_cmd="printf %s '$payload' | base64 -d >/tmp/clearmesh_queue_payload.sh && bash /tmp/clearmesh_queue_payload.sh"
  (
    printf 'bash -lc %q\nexit\n' "$remote_cmd" | "$TNR_BIN" connect "$instance_id" > "$log_file" 2>&1
  ) &
  pid=$!
  deadline=$(( $(date +%s) + timeout_sec ))
  while [[ "$(date +%s)" -lt "$deadline" ]]; do
    if grep -q "$marker" "$log_file" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      break
    fi
    sleep 2
  done
  kill "$pid" 2>/dev/null || true
  sleep 1
  kill -9 "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  echo "Remote script did not report marker '$marker' within ${timeout_sec}s." >&2
  tail -80 "$log_file" >&2 || true
  return 23
}

if [[ "$CREATE_INSTANCE" = "1" ]]; then
  create_args=(create --gpu "$GPU" --mode "$MODE" --num-gpus 1 --primary-disk "$PRIMARY_DISK" --template "$TEMPLATE")
  if [[ "$EPHEMERAL_DISK" != "0" ]]; then
    create_args+=(--ephemeral-disk "$EPHEMERAL_DISK")
  fi
  if [[ "$MODE" = "prototyping" ]]; then
    create_args+=(--vcpus "$VCPUS")
  fi
  create_args+=(--yes --json)
  create_output="$("$TNR_BIN" "${create_args[@]}")" || {
    echo "Thunder create failed." >&2
    exit 3
  }
  redact_create_output "$create_output" > "$setup_dir/create.json"
  CREATED_INSTANCE=1
  if [[ -z "$INSTANCE_ID" ]]; then
    INSTANCE_ID="$(parse_create_id "$create_output")" || INSTANCE_ID=""
  fi
  echo "Created Thunder queue instance $INSTANCE_ID."
fi
if [[ -z "$INSTANCE_ID" ]]; then
  echo "Set THUNDER_INSTANCE_ID/pass an instance id, or set CREATE_INSTANCE=1." >&2
  exit 2
fi

setup_dir="$REPO_ROOT/.codex_outputs/face_corpus_queue_setup/instance_${INSTANCE_ID}_${RUN_STAMP_PREFIX}"
mkdir -p "$setup_dir"

if [[ "$WAIT_FOR_RUNNING" = "1" ]]; then
  echo "Waiting for Thunder instance $INSTANCE_ID to RUNNING..."
  running_seen=0
  wait_deadline=$(( $(date +%s) + WAIT_TIMEOUT_SEC ))
  while [[ "$(date +%s)" -lt "$wait_deadline" ]]; do
    status_json="$("$TNR_BIN" status --json || true)"
    printf '%s\n' "$status_json" > "$setup_dir/status.latest.json"
    if python3 - "$INSTANCE_ID" "$setup_dir/status.latest.json" <<'PY'
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
  if [[ "$running_seen" != "1" ]]; then
    echo "Thunder instance $INSTANCE_ID did not reach RUNNING within ${WAIT_TIMEOUT_SEC}s." >&2
    exit 21
  fi
fi

if [[ "$BOOTSTRAP_REMOTE" = "1" ]]; then
  echo "Syncing repo to queue worker $INSTANCE_ID..."
  THUNDER_INSTANCE_ID="$INSTANCE_ID" "$REPO_ROOT/scripts/thunder/sync_repo.sh" "$INSTANCE_ID"
  if [[ "$SYNC_HF_TOKEN" = "1" ]]; then
    if [[ -n "${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}" ]]; then
      echo "Syncing Hugging Face token to queue worker $INSTANCE_ID..."
      THUNDER_INSTANCE_ID="$INSTANCE_ID" "$REPO_ROOT/scripts/thunder/sync_hf_token.sh" "$INSTANCE_ID"
    else
      echo "HF_TOKEN/HUGGINGFACE_HUB_TOKEN not set locally; remote downloads may be rate limited." >&2
    fi
  fi
  bootstrap_log="$setup_dir/bootstrap_remote.log"
  bootstrap_script="$setup_dir/bootstrap_remote.sh"
  cat > "$bootstrap_script" <<REMOTE_BOOTSTRAP
set -euo pipefail
cd $(printf '%q' "$REMOTE_REPO")
if [ ! -x $(printf '%q' "$REMOTE_VENV/bin/python") ]; then
  python3 -m venv $(printf '%q' "$REMOTE_VENV") || (sudo apt-get update && sudo apt-get install -y python3-venv && python3 -m venv $(printf '%q' "$REMOTE_VENV"))
fi
source $(printf '%q' "$REMOTE_VENV/bin/activate")
python -m pip install -U pip setuptools wheel
python -m pip install -q -r requirements-data.txt pillow scipy
if ! command -v rclone >/dev/null 2>&1; then
  sudo apt-get update
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y rclone
fi
python - <<'PY'
import fast_simplification  # noqa: F401
import networkx  # noqa: F401
import objaverse  # noqa: F401
import scipy  # noqa: F401
import skimage  # noqa: F401
import trimesh  # noqa: F401
print('face_corpus_queue_python_deps_ok')
PY
rclone version | head -n 1
echo __FACE_CORPUS_QUEUE_BOOTSTRAP_DONE__
REMOTE_BOOTSTRAP
  run_remote_script_with_marker "$INSTANCE_ID" "$bootstrap_script" "$bootstrap_log" "__FACE_CORPUS_QUEUE_BOOTSTRAP_DONE__" "${BOOTSTRAP_TIMEOUT_SEC:-1800}"
fi

if [[ "$SYNC_QUEUE_FILES" = "1" ]]; then
  "$TNR_BIN" scp "$REPO_ROOT/scripts/thunder/face_corpus_shard_queue_worker.sh" "$INSTANCE_ID:$REMOTE_REPO/scripts/thunder/face_corpus_shard_queue_worker.sh"
  "$TNR_BIN" scp "$REPO_ROOT/scripts/thunder/face_objaversepp_corpus_pilot.sh" "$INSTANCE_ID:$REMOTE_REPO/scripts/thunder/face_objaversepp_corpus_pilot.sh"
  "$TNR_BIN" scp "$REPO_ROOT/scripts/thunder/b2_continuous_upload.sh" "$INSTANCE_ID:$REMOTE_REPO/scripts/thunder/b2_continuous_upload.sh"
  "$TNR_BIN" scp "$REPO_ROOT/scripts/data/download_texverse_face_candidates.py" "$INSTANCE_ID:$REMOTE_REPO/scripts/data/download_texverse_face_candidates.py"
  "$TNR_BIN" scp "$REPO_ROOT/scripts/data/download_objaversexl_face_candidates.py" "$INSTANCE_ID:$REMOTE_REPO/scripts/data/download_objaversexl_face_candidates.py"
  "$TNR_BIN" scp "$REPO_ROOT/scripts/research/prepare_face_strict_targets.py" "$INSTANCE_ID:$REMOTE_REPO/scripts/research/prepare_face_strict_targets.py"
  "$TNR_BIN" scp "$REPO_ROOT/scripts/research/check_face_dataset_targets.py" "$INSTANCE_ID:$REMOTE_REPO/scripts/research/check_face_dataset_targets.py"
fi

printf 'mkdir -p %q %q\nexit\n' "$REMOTE_QUEUE_SOURCE_DIR" "$REMOTE_QUEUE_LOG_ROOT" | "$TNR_BIN" connect "$INSTANCE_ID"

for shard_id in $QUEUE_SHARD_IDS; do
  shard_file="$SHARD_DIR/shard_${shard_id}.jsonl"
  if [[ ! -f "$shard_file" ]]; then
    echo "Missing local shard file: $shard_file" >&2
    exit 3
  fi
  "$TNR_BIN" scp "$shard_file" "$INSTANCE_ID:$REMOTE_QUEUE_SOURCE_DIR/shard_${shard_id}.jsonl"
done

if [[ "$START_B2_UPLOAD" = "1" && -n "$LOCAL_B2_ENV_FILE" ]]; then
  if [[ ! -f "$LOCAL_B2_ENV_FILE" ]]; then
    echo "LOCAL_B2_ENV_FILE does not exist: $LOCAL_B2_ENV_FILE" >&2
    exit 4
  fi
  "$TNR_BIN" scp "$LOCAL_B2_ENV_FILE" "$INSTANCE_ID:$REMOTE_B2_ENV"
elif [[ "$START_B2_UPLOAD" = "1" && "$UPLOAD_B2_ENV" = "1" ]]; then
  b2_env_file="$(mktemp "$setup_dir/b2_env.XXXXXX")"
  {
    printf 'export B2_KEY_ID=%q\n' "${B2_KEY_ID:-${B2_KEYID:-${B2_APPLICATION_KEY_ID:-${BACKBLAZE_B2_KEY_ID:-}}}}"
    printf 'export B2_APP_KEY=%q\n' "${B2_APP_KEY:-${B2_APPKEY:-${B2_APPLICATION_KEY:-${BACKBLAZE_B2_APPLICATION_KEY:-${BACKBLAZE_B2_APP_KEY:-}}}}}"
    printf 'export B2_TOKEN=%q\n' "${B2_TOKEN:-}"
  } > "$b2_env_file"
  chmod 600 "$b2_env_file"
  "$TNR_BIN" scp "$b2_env_file" "$INSTANCE_ID:$REMOTE_B2_ENV"
  rm -f "$b2_env_file"
fi

remote_launch_log="$setup_dir/remote_launch.log"
cat <<REMOTE | "$TNR_BIN" connect "$INSTANCE_ID" 2>&1 | tee "$remote_launch_log"
set -euo pipefail
cd $(printf '%q' "$REMOTE_REPO")
chmod +x scripts/thunder/face_corpus_shard_queue_worker.sh scripts/thunder/b2_continuous_upload.sh
mkdir -p $(printf '%q' "$REMOTE_QUEUE_LOG_ROOT")
nohup env \\
  QUEUE_SHARD_IDS=$(printf '%q' "$QUEUE_SHARD_IDS") \\
  QUEUE_SOURCE_DIR=$(printf '%q' "$REMOTE_QUEUE_SOURCE_DIR") \\
  QUEUE_LOG_ROOT=$(printf '%q' "$REMOTE_QUEUE_LOG_ROOT") \\
  REMOTE_VENV=$(printf '%q' "$REMOTE_VENV") \\
  HF_ENV_FILE=$(printf '%q' "${HF_ENV_FILE:-/home/ubuntu/.clearmesh_hf.env}") \\
  RUN_STAMP_PREFIX=$(printf '%q' "$RUN_STAMP_PREFIX") \\
  SOURCE_KIND=$(printf '%q' "$SOURCE_KIND") \\
  LAB_ROOT_PREFIX=$(printf '%q' "$LAB_ROOT_PREFIX") \\
  START_B2_UPLOAD=$(printf '%q' "$START_B2_UPLOAD") \\
  B2_BUCKET=$(printf '%q' "$B2_BUCKET") \\
  B2_PREFIX_ROOT=$(printf '%q' "$B2_PREFIX_ROOT") \\
  B2_UPLOAD_INTERVAL_SECONDS=$(printf '%q' "$B2_UPLOAD_INTERVAL_SECONDS") \\
  B2_ENV_FILE=$(printf '%q' "$REMOTE_B2_ENV") \\
  WAIT_FOR_PID_FILE=$(printf '%q' "$WAIT_FOR_PID_FILE") \\
  WAIT_POLL_SECONDS=$(printf '%q' "$WAIT_POLL_SECONDS") \\
  PREQUEUE_COMPLETED_ROOT=$(printf '%q' "$PREQUEUE_COMPLETED_ROOT") \\
  PREQUEUE_B2_PREFIX=$(printf '%q' "$PREQUEUE_B2_PREFIX") \\
  PREQUEUE_B2_UPLOAD_PID_FILE=$(printf '%q' "$PREQUEUE_B2_UPLOAD_PID_FILE") \\
  CONTINUE_ON_FAILURE=$(printf '%q' "$CONTINUE_ON_FAILURE") \\
  CLEANUP_FAILED_ROOT_ON_FAILURE=$(printf '%q' "${CLEANUP_FAILED_ROOT_ON_FAILURE:-0}") \\
  DATA_LANE=$(printf '%q' "${DATA_LANE:-unspecified}") \\
  SOURCE_POOL_NAME=$(printf '%q' "${SOURCE_POOL_NAME:-}") \\
  SELECT_TARGET=$(printf '%q' "${SELECT_TARGET:-auto}") \\
  CURATION_TARGET=$(printf '%q' "${CURATION_TARGET:-auto}") \\
  SCAN_LIMIT=$(printf '%q' "${SCAN_LIMIT:-0}") \\
  MIN_QUALITY=$(printf '%q' "${MIN_QUALITY:-2}") \\
  DOWNLOAD_PROCESSES=$(printf '%q' "${DOWNLOAD_PROCESSES:-16}") \\
  DOWNLOAD_FALLBACK_PROCESSES=$(printf '%q' "${DOWNLOAD_FALLBACK_PROCESSES:-1}") \\
  DOWNLOAD_BATCH_SIZE=$(printf '%q' "${DOWNLOAD_BATCH_SIZE:-50}") \\
  DOWNLOAD_BATCH_TIMEOUT_SECONDS=$(printf '%q' "${DOWNLOAD_BATCH_TIMEOUT_SECONDS:-600}") \\
  DOWNLOAD_BATCH_RETRIES=$(printf '%q' "${DOWNLOAD_BATCH_RETRIES:-2}") \\
  DOWNLOAD_RETRY_SLEEP_SECONDS=$(printf '%q' "${DOWNLOAD_RETRY_SLEEP_SECONDS:-15}") \\
  DOWNLOAD_RATE_LIMIT_SLEEP_SECONDS=$(printf '%q' "${DOWNLOAD_RATE_LIMIT_SLEEP_SECONDS:-600}") \\
  TEXVERSE_DOWNLOAD_WORKERS=$(printf '%q' "${TEXVERSE_DOWNLOAD_WORKERS:-${DOWNLOAD_PROCESSES:-16}}") \\
  TEXVERSE_MAX_SIZE_MB=$(printf '%q' "${TEXVERSE_MAX_SIZE_MB:-0}") \\
  TEXVERSE_CLEANUP_CACHE_EACH=$(printf '%q' "${TEXVERSE_CLEANUP_CACHE_EACH:-0}") \\
  SOURCE_MIN_FACES=$(printf '%q' "${SOURCE_MIN_FACES:-64}") \\
  SOURCE_MAX_FACES=$(printf '%q' "${SOURCE_MAX_FACES:-250000}") \\
  MAX_FILE_MB=$(printf '%q' "${MAX_FILE_MB:-256}") \\
  MAX_COMPONENTS=$(printf '%q' "${MAX_COMPONENTS:-48}") \\
  MIN_LARGEST_COMPONENT_AREA_RATIO=$(printf '%q' "${MIN_LARGEST_COMPONENT_AREA_RATIO:-0.60}") \\
  TARGET_FACES=$(printf '%q' "${TARGET_FACES:-512}") \\
  TOKEN_MAX_FACES=$(printf '%q' "${TOKEN_MAX_FACES:-512}") \\
  POINT_SAMPLES=$(printf '%q' "${POINT_SAMPLES:-8192}") \\
  NUM_BINS=$(printf '%q' "${NUM_BINS:-128}") \\
  PAPER_WITHIN_FACE_ORDER=$(printf '%q' "${PAPER_WITHIN_FACE_ORDER:-rotate_min_zyx}") \\
  STRICT_ENGINE=$(printf '%q' "${STRICT_ENGINE:-voxel_shell}") \\
  FALLBACK=$(printf '%q' "${FALLBACK:-convex_hull}") \\
  VOXEL_RESOLUTION=$(printf '%q' "${VOXEL_RESOLUTION:-64}") \\
  MESH_VOXEL_MAX_FACES=$(printf '%q' "${MESH_VOXEL_MAX_FACES:-5000}") \\
  STRICT_TARGET_PROGRESS_EVERY=$(printf '%q' "${STRICT_TARGET_PROGRESS_EVERY:-100}") \\
  TARGET_MAX_OUTPUT_COMPONENTS=$(printf '%q' "${TARGET_MAX_OUTPUT_COMPONENTS:-1}") \\
  TARGET_MAX_BOUNDARY_LOOPS=$(printf '%q' "${TARGET_MAX_BOUNDARY_LOOPS:-0}") \\
  TARGET_MAX_NONMANIFOLD_EDGES=$(printf '%q' "${TARGET_MAX_NONMANIFOLD_EDGES:-0}") \\
  TARGET_REQUIRE_WATERTIGHT=$(printf '%q' "${TARGET_REQUIRE_WATERTIGHT:-1}") \\
  GATE_PROFILE=$(printf '%q' "${GATE_PROFILE:-strict}") \\
  GATE_MAX_BOUNDARY_EDGES=$(printf '%q' "${GATE_MAX_BOUNDARY_EDGES:-}") \\
  GATE_MAX_NONMANIFOLD_EDGES=$(printf '%q' "${GATE_MAX_NONMANIFOLD_EDGES:-}") \\
  GATE_MIN_EDGE_PAIRING_RATIO=$(printf '%q' "${GATE_MIN_EDGE_PAIRING_RATIO:-}") \\
  TEST_RATIO=$(printf '%q' "${TEST_RATIO:-0.02}") \\
  bash scripts/thunder/face_corpus_shard_queue_worker.sh > $(printf '%q' "$REMOTE_QUEUE_LOG") 2>&1 &
echo \$! > $(printf '%q' "$REMOTE_QUEUE_PID")
echo "face_corpus_queue_pid=\$(cat $(printf '%q' "$REMOTE_QUEUE_PID"))"
echo "face_corpus_queue_log=$(printf '%q' "$REMOTE_QUEUE_LOG")"
echo CLEARMESH_FACE_CORPUS_QUEUE_LAUNCHED
exit
REMOTE

cat > "$setup_dir/queue_info.json" <<JSON
{
  "instance_id": "$INSTANCE_ID",
  "queue_shard_ids": "$QUEUE_SHARD_IDS",
  "remote_queue_source_dir": "$REMOTE_QUEUE_SOURCE_DIR",
  "remote_queue_log": "$REMOTE_QUEUE_LOG",
  "remote_queue_pid": "$REMOTE_QUEUE_PID",
  "remote_venv": "$REMOTE_VENV",
  "run_stamp_prefix": "$RUN_STAMP_PREFIX",
  "data_lane": "${DATA_LANE:-unspecified}",
  "source_pool_name": "${SOURCE_POOL_NAME:-}",
  "source_kind": "$SOURCE_KIND",
  "lab_root_prefix": "$LAB_ROOT_PREFIX",
  "start_b2_upload": "$START_B2_UPLOAD",
  "b2_bucket": "$B2_BUCKET",
  "b2_prefix_root": "$B2_PREFIX_ROOT",
  "remote_b2_env": "$REMOTE_B2_ENV",
  "wait_for_pid_file": "$WAIT_FOR_PID_FILE",
  "prequeue_completed_root": "$PREQUEUE_COMPLETED_ROOT",
  "prequeue_b2_prefix": "$PREQUEUE_B2_PREFIX",
  "texverse_download_workers": ${TEXVERSE_DOWNLOAD_WORKERS:-${DOWNLOAD_PROCESSES:-16}},
  "texverse_max_size_mb": ${TEXVERSE_MAX_SIZE_MB:-0},
  "texverse_cleanup_cache_each": "${TEXVERSE_CLEANUP_CACHE_EACH:-0}",
  "source_min_faces": ${SOURCE_MIN_FACES:-64},
  "source_max_faces": ${SOURCE_MAX_FACES:-250000},
  "max_file_mb": ${MAX_FILE_MB:-256},
  "max_components": ${MAX_COMPONENTS:-48},
  "target_faces": ${TARGET_FACES:-512},
  "token_max_faces": ${TOKEN_MAX_FACES:-512},
  "target_max_output_components": ${TARGET_MAX_OUTPUT_COMPONENTS:-1},
  "target_max_boundary_loops": ${TARGET_MAX_BOUNDARY_LOOPS:-0},
  "target_max_nonmanifold_edges": ${TARGET_MAX_NONMANIFOLD_EDGES:-0},
  "target_require_watertight": "${TARGET_REQUIRE_WATERTIGHT:-1}",
  "gate_profile": "${GATE_PROFILE:-strict}",
  "gate_max_boundary_edges": "${GATE_MAX_BOUNDARY_EDGES:-}",
  "gate_max_nonmanifold_edges": "${GATE_MAX_NONMANIFOLD_EDGES:-}",
  "gate_min_edge_pairing_ratio": "${GATE_MIN_EDGE_PAIRING_RATIO:-}"
}
JSON

echo "FACE corpus queue launched on Thunder instance $INSTANCE_ID."
echo "Local setup: $setup_dir"
echo "Remote queue log: $REMOTE_QUEUE_LOG"
