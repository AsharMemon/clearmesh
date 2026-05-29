#!/usr/bin/env bash
# Launch a B2-backed Localizable FACE smoke on an existing RunCrate SSH host.
#
# This deliberately uses a small shard by default so we can validate the
# voxel-anchored patch contract on real corpus samples without downloading the
# full merged training archive or interrupting the inference app.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="${OUT_DIR:-.codex_outputs/runcrate_localizable_face_smoke_$(date -u +%Y%m%dT%H%M%SZ)}"
SSH_PROBE_FILE="${SSH_PROBE_FILE:-.codex_outputs/runcrate_pipeline_inference_20260526T200820Z/ssh_probe.sh}"
B2_ENV_FILE="${B2_ENV_FILE:-.codex_secrets/b2.env}"
SSH_KEY_FILE="${SSH_KEY_FILE:-$HOME/.ssh/id_ed25519}"
REMOTE_BASE="${REMOTE_BASE:-/ephemeral}"
REMOTE_REPO="${REMOTE_REPO:-$REMOTE_BASE/clearmesh-localizable-smoke-repo}"
REMOTE_VENV="${REMOTE_VENV:-$REMOTE_BASE/clearmesh-localizable-smoke-venv}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
REMOTE_LAB_ROOT="${REMOTE_LAB_ROOT:-$REMOTE_BASE/clearmesh_localizable_face_smoke_$RUN_STAMP}"
REMOTE_B2_ENV="${REMOTE_B2_ENV:-$REMOTE_LAB_ROOT/.clearmesh_b2.env}"

B2_BUCKET="${B2_BUCKET:-clearmesh-pairs}"
B2_SOURCE_PREFIX="${B2_SOURCE_PREFIX:-face-corpora/poolA-shards/shard0010}"
B2_SOURCE_PREFIXES="${B2_SOURCE_PREFIXES:-}"
B2_SOURCE_ARCHIVE="${B2_SOURCE_ARCHIVE:-lean_face_corpus.tar.gz}"
B2_RUN_PREFIX="${B2_RUN_PREFIX:-face-runs/localizable-face-smoke/$RUN_STAMP}"
ARCHIVE_DATASET_DIR="${ARCHIVE_DATASET_DIR:-corpus/split_pass/train}"
RCLONE_TRANSFERS="${RCLONE_TRANSFERS:-4}"
RCLONE_CHECKERS="${RCLONE_CHECKERS:-8}"
RCLONE_MULTI_THREAD_STREAMS="${RCLONE_MULTI_THREAD_STREAMS:-8}"
RCLONE_MULTI_THREAD_CUTOFF="${RCLONE_MULTI_THREAD_CUTOFF:-64M}"

DATASET_LIMIT="${DATASET_LIMIT:-512}"
VOXEL_RESOLUTION="${VOXEL_RESOLUTION:-32}"
MAX_FACES_PER_PATCH="${MAX_FACES_PER_PATCH:-64}"
POINT_SAMPLES_PER_PATCH="${POINT_SAMPLES_PER_PATCH:-32}"
PATCH_WORKERS="${PATCH_WORKERS:-4}"
VERIFY_LIMIT_SOURCES="${VERIFY_LIMIT_SOURCES:-0}"
DEBUG_MESH_LIMIT="${DEBUG_MESH_LIMIT:-4}"
TRAIN_STEPS="${TRAIN_STEPS:-25}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
TRAIN_LIMIT="${TRAIN_LIMIT:-4096}"
TRAIN_POINT_SAMPLES="${TRAIN_POINT_SAMPLES:-32}"
HIDDEN_SIZE="${HIDDEN_SIZE:-128}"
LAYERS="${LAYERS:-4}"
HEADS="${HEADS:-4}"
CONDITION_TOKENS="${CONDITION_TOKENS:-8}"
LOG_EVERY="${LOG_EVERY:-5}"
EVAL_LIMIT_PATCHES="${EVAL_LIMIT_PATCHES:-4096}"
EVAL_GREEDY_PATCHES="${EVAL_GREEDY_PATCHES:-256}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
EVAL_GREEDY_BATCH_SIZE="${EVAL_GREEDY_BATCH_SIZE:-8}"

mkdir -p "$OUT_DIR"
LOCAL_LOG="$OUT_DIR/bootstrap.log"

if [[ ! -f "$SSH_PROBE_FILE" ]]; then
  echo "Missing SSH probe file: $SSH_PROBE_FILE" >&2
  exit 2
fi
if [[ ! -f "$B2_ENV_FILE" ]]; then
  echo "Missing B2 env file: $B2_ENV_FILE" >&2
  exit 2
fi
if [[ ! -f "$SSH_KEY_FILE" ]]; then
  echo "Missing SSH key: $SSH_KEY_FILE" >&2
  exit 2
fi

source "$SSH_PROBE_FILE"
REMOTE_USER="${REMOTE_USER:-root}"
SSH_PORT="${SSH_PORT:-22}"
if [[ -z "${HOST:-}" ]]; then
  echo "SSH probe did not define HOST." >&2
  exit 2
fi
SSH_TARGET="$REMOTE_USER@$HOST"
SSH_OPTS=(
  -i "$SSH_KEY_FILE"
  -p "$SSH_PORT"
  -o StrictHostKeyChecking=accept-new
  -o UserKnownHostsFile="$HOME/.ssh/known_hosts"
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=4
)
SCP_OPTS=(
  -i "$SSH_KEY_FILE"
  -P "$SSH_PORT"
  -o StrictHostKeyChecking=accept-new
  -o UserKnownHostsFile="$HOME/.ssh/known_hosts"
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=4
)

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Preflighting RunCrate host..." | tee -a "$LOCAL_LOG"
ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "hostname; nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader; df -h '$REMOTE_BASE' || true; mkdir -p '$REMOTE_REPO' '$REMOTE_LAB_ROOT'" | tee -a "$LOCAL_LOG"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Syncing isolated smoke repo..." | tee -a "$LOCAL_LOG"
COPYFILE_DISABLE=1 tar \
  --no-xattrs \
  --exclude='.git' \
  --exclude='.codex_outputs' \
  --exclude='.codex_secrets' \
  --exclude='__pycache__' \
  --exclude='.pytest_cache' \
  --exclude='.mypy_cache' \
  --exclude='.ruff_cache' \
  --exclude='.venv' \
  --exclude='node_modules' \
  --exclude='.DS_Store' \
  -czf - . \
  | ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "rm -rf '$REMOTE_REPO' && mkdir -p '$REMOTE_REPO' && tar -xzf - -C '$REMOTE_REPO'"

tmp_b2_env="$(mktemp "$OUT_DIR/b2_remote_env.XXXXXX")"
{
  source "$B2_ENV_FILE"
  printf 'export B2_KEY_ID=%q\n' "${B2_KEY_ID:-${B2_KEYID:-${B2_APPLICATION_KEY_ID:-${BACKBLAZE_B2_KEY_ID:-}}}}"
  printf 'export B2_APP_KEY=%q\n' "${B2_APP_KEY:-${B2_APPKEY:-${B2_APPLICATION_KEY:-${BACKBLAZE_B2_APPLICATION_KEY:-${BACKBLAZE_B2_APP_KEY:-}}}}}"
  printf 'export B2_TOKEN=%q\n' "${B2_TOKEN:-}"
} > "$tmp_b2_env"
chmod 600 "$tmp_b2_env"
scp "${SCP_OPTS[@]}" "$tmp_b2_env" "$SSH_TARGET:$REMOTE_B2_ENV" >/dev/null
rm -f "$tmp_b2_env"

remote_script="$OUT_DIR/remote_localizable_face_smoke.sh"
cat > "$remote_script" <<REMOTE
#!/usr/bin/env bash
set -euo pipefail
LAB_ROOT=$(printf '%q' "$REMOTE_LAB_ROOT")
REPO=$(printf '%q' "$REMOTE_REPO")
VENV=$(printf '%q' "$REMOTE_VENV")
B2_ENV=$(printf '%q' "$REMOTE_B2_ENV")
B2_BUCKET=$(printf '%q' "$B2_BUCKET")
B2_SOURCE_PREFIX=$(printf '%q' "$B2_SOURCE_PREFIX")
B2_SOURCE_PREFIXES=$(printf '%q' "$B2_SOURCE_PREFIXES")
B2_SOURCE_ARCHIVE=$(printf '%q' "$B2_SOURCE_ARCHIVE")
B2_RUN_PREFIX=$(printf '%q' "$B2_RUN_PREFIX")
ARCHIVE_DATASET_DIR=$(printf '%q' "$ARCHIVE_DATASET_DIR")
RCLONE_TRANSFERS=$(printf '%q' "$RCLONE_TRANSFERS")
RCLONE_CHECKERS=$(printf '%q' "$RCLONE_CHECKERS")
RCLONE_MULTI_THREAD_STREAMS=$(printf '%q' "$RCLONE_MULTI_THREAD_STREAMS")
RCLONE_MULTI_THREAD_CUTOFF=$(printf '%q' "$RCLONE_MULTI_THREAD_CUTOFF")
mkdir -p "\$LAB_ROOT/logs" "\$LAB_ROOT/data" "\$LAB_ROOT/runs"
status() {
  python3 - "\$LAB_ROOT/status.jsonl" "\$1" "\${2:-}" <<'PY'
import json, sys, time
path, event, detail = sys.argv[1:4]
with open(path, "a", encoding="utf-8") as handle:
    handle.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "event": event, "detail": detail}) + "\\n")
PY
}
cd "\$REPO"
export PYTHONPATH="\$REPO:\${PYTHONPATH:-}"
if ! command -v rclone >/dev/null 2>&1; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y rclone
fi
if [[ ! -x "\$VENV/bin/python" ]]; then
  python3 -m venv --system-site-packages "\$VENV" || (apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv && python3 -m venv --system-site-packages "\$VENV")
fi
source "\$VENV/bin/activate"
python -m pip install -U pip setuptools wheel
python -m pip install -q numpy trimesh scipy networkx tqdm
python - <<'PY' || python -m pip install --index-url https://download.pytorch.org/whl/cu128 'torch>=2.4.0'
import torch
print("torch_ready", torch.__version__)
PY
source "\$B2_ENV"
export RCLONE_CONFIG_B2ENV_TYPE=b2
export RCLONE_CONFIG_B2ENV_ACCOUNT="\$B2_KEY_ID"
export RCLONE_CONFIG_B2ENV_KEY="\$B2_APP_KEY"
COMBINED_DATASET_DIR="\$LAB_ROOT/data/combined_train_npz"
mkdir -p "\$COMBINED_DATASET_DIR"
python - "\$B2_SOURCE_PREFIX" "\$B2_SOURCE_PREFIXES" "\$LAB_ROOT/source_prefixes.txt" <<'PY'
import re, sys
single, many, out_path = sys.argv[1:4]
text = many.strip() or single.strip()
prefixes = [part for part in re.split(r"[\s,]+", text) if part]
open(out_path, "w", encoding="utf-8").write("\n".join(prefixes) + "\n")
print(len(prefixes))
PY
source_count=0
while IFS= read -r source_prefix; do
  [[ -n "\$source_prefix" ]] || continue
  shard_dir="\$LAB_ROOT/data/source_\$(printf '%04d' "\$source_count")"
  mkdir -p "\$shard_dir"
  archive_path="\$shard_dir/\$B2_SOURCE_ARCHIVE"
  status b2_download_started "\$source_prefix/\$B2_SOURCE_ARCHIVE"
  rclone copyto "b2env:\$B2_BUCKET/\$source_prefix/\$B2_SOURCE_ARCHIVE" "\$archive_path" \\
    --stats 20s \\
    --transfers "\$RCLONE_TRANSFERS" \\
    --checkers "\$RCLONE_CHECKERS" \\
    --multi-thread-streams "\$RCLONE_MULTI_THREAD_STREAMS" \\
    --multi-thread-cutoff "\$RCLONE_MULTI_THREAD_CUTOFF"
  status extract_started "\$source_prefix/\$B2_SOURCE_ARCHIVE"
  extract_dir="\$shard_dir/extracted"
  mkdir -p "\$extract_dir"
  tar -xzf "\$archive_path" -C "\$extract_dir"
  dataset_part="\$extract_dir/\$ARCHIVE_DATASET_DIR"
  test -d "\$dataset_part"
  python - "\$dataset_part" "\$COMBINED_DATASET_DIR" "\$source_count" <<'PY'
import os, sys
from pathlib import Path
src, dst, index = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
dst.mkdir(parents=True, exist_ok=True)
for n, path in enumerate(sorted(src.rglob("*.npz"))):
    target = dst / f"{index:04d}_{n:08d}_{path.name}"
    if not target.exists():
        os.symlink(path, target)
PY
  source_count=\$((source_count + 1))
done < "\$LAB_ROOT/source_prefixes.txt"
DATASET_DIR="\$COMBINED_DATASET_DIR"
test -d "\$DATASET_DIR"
status patch_build_started "\$DATASET_DIR"
python scripts/research/build_localizable_face_patch_dataset.py \\
  --dataset-dir "\$DATASET_DIR" \\
  --output-dir "\$LAB_ROOT/localizable_patches" \\
  --limit $(printf '%q' "$DATASET_LIMIT") \\
  --voxel-resolution $(printf '%q' "$VOXEL_RESOLUTION") \\
  --max-faces-per-patch $(printf '%q' "$MAX_FACES_PER_PATCH") \\
  --point-samples-per-patch $(printf '%q' "$POINT_SAMPLES_PER_PATCH") \\
  --workers $(printf '%q' "$PATCH_WORKERS") \\
  > "\$LAB_ROOT/logs/patch_build.log" 2>&1
status patch_build_completed "\$(cat "\$LAB_ROOT/localizable_patches/summary.json")"
status verify_started "\$LAB_ROOT/localizable_patches/manifest.jsonl"
python scripts/research/verify_localizable_face_patch_dataset.py \\
  --manifest "\$LAB_ROOT/localizable_patches/manifest.jsonl" \\
  --report "\$LAB_ROOT/localizable_patches/verify_report.json" \\
  --limit-sources $(printf '%q' "$VERIFY_LIMIT_SOURCES") \\
  --export-debug-mesh-dir "\$LAB_ROOT/localizable_patches/debug_meshes" \\
  --debug-mesh-limit $(printf '%q' "$DEBUG_MESH_LIMIT") \\
  > "\$LAB_ROOT/logs/verify.log" 2>&1
status verify_completed "\$(cat "\$LAB_ROOT/localizable_patches/verify_report.json")"
status train_started "steps=$(printf '%q' "$TRAIN_STEPS")"
python scripts/research/train_localizable_face_patch_tiny.py \\
  --manifest "\$LAB_ROOT/localizable_patches/manifest.jsonl" \\
  --output "\$LAB_ROOT/runs/localizable_face_patch_tiny.pt" \\
  --steps $(printf '%q' "$TRAIN_STEPS") \\
  --batch-size $(printf '%q' "$TRAIN_BATCH_SIZE") \\
  --limit-patches $(printf '%q' "$TRAIN_LIMIT") \\
  --point-samples $(printf '%q' "$TRAIN_POINT_SAMPLES") \\
  --hidden-size $(printf '%q' "$HIDDEN_SIZE") \\
  --layers $(printf '%q' "$LAYERS") \\
  --heads $(printf '%q' "$HEADS") \\
  --condition-tokens $(printf '%q' "$CONDITION_TOKENS") \\
  --log-every $(printf '%q' "$LOG_EVERY") \\
  --device auto \\
  > "\$LAB_ROOT/logs/train.log" 2>&1
status train_completed "\$(tail -n 20 "\$LAB_ROOT/logs/train.log")"
status eval_started "\$LAB_ROOT/runs/localizable_face_patch_tiny.pt"
python scripts/research/eval_localizable_face_patch_tiny.py \\
  --manifest "\$LAB_ROOT/localizable_patches/manifest.jsonl" \\
  --checkpoint "\$LAB_ROOT/runs/localizable_face_patch_tiny.pt" \\
  --report "\$LAB_ROOT/localizable_eval/eval_report.json" \\
  --limit-patches $(printf '%q' "$EVAL_LIMIT_PATCHES") \\
  --greedy-patches $(printf '%q' "$EVAL_GREEDY_PATCHES") \\
  --batch-size $(printf '%q' "$EVAL_BATCH_SIZE") \\
  --greedy-batch-size $(printf '%q' "$EVAL_GREEDY_BATCH_SIZE") \\
  --point-samples $(printf '%q' "$TRAIN_POINT_SAMPLES") \\
  --device auto \\
  > "\$LAB_ROOT/logs/eval.log" 2>&1
status eval_completed "\$(cat "\$LAB_ROOT/localizable_eval/eval_report.json")"
rclone copy "\$LAB_ROOT/status.jsonl" "b2env:\$B2_BUCKET/\$B2_RUN_PREFIX/" --stats 0
rclone copy "\$LAB_ROOT/logs" "b2env:\$B2_BUCKET/\$B2_RUN_PREFIX/logs" --stats 0
rclone copy "\$LAB_ROOT/localizable_patches/summary.json" "b2env:\$B2_BUCKET/\$B2_RUN_PREFIX/localizable_patches/" --stats 0
rclone copy "\$LAB_ROOT/localizable_patches/verify_report.json" "b2env:\$B2_BUCKET/\$B2_RUN_PREFIX/localizable_patches/" --stats 0
rclone copy "\$LAB_ROOT/localizable_patches/debug_meshes" "b2env:\$B2_BUCKET/\$B2_RUN_PREFIX/localizable_patches/debug_meshes" --stats 0 || true
rclone copy "\$LAB_ROOT/localizable_eval" "b2env:\$B2_BUCKET/\$B2_RUN_PREFIX/localizable_eval" --stats 0
rclone copy "\$LAB_ROOT/runs" "b2env:\$B2_BUCKET/\$B2_RUN_PREFIX/runs" --stats 0
status b2_upload_completed "\$B2_RUN_PREFIX"
REMOTE
chmod +x "$remote_script"
scp "${SCP_OPTS[@]}" "$remote_script" "$SSH_TARGET:$REMOTE_LAB_ROOT/remote_localizable_face_smoke.sh" >/dev/null

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Launching remote smoke..." | tee -a "$LOCAL_LOG"
ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "cd '$REMOTE_LAB_ROOT' && setsid -f bash remote_localizable_face_smoke.sh > localizable_face_smoke.nohup.log 2>&1 < /dev/null && sleep 1 && pgrep -af 'remote_localizable_face_smoke.sh' | awk 'NR==1 {print \$1}' > localizable_face_smoke.pid && cat localizable_face_smoke.pid" | tee "$OUT_DIR/remote_pid.txt"
cat > "$OUT_DIR/run_info.json" <<JSON
{
  "ssh_probe_file": "$SSH_PROBE_FILE",
  "host": "$HOST",
  "remote_user": "$REMOTE_USER",
  "ssh_port": "$SSH_PORT",
  "remote_lab_root": "$REMOTE_LAB_ROOT",
  "remote_repo": "$REMOTE_REPO",
  "b2_bucket": "$B2_BUCKET",
  "b2_source_prefix": "$B2_SOURCE_PREFIX",
  "b2_source_prefixes": "$B2_SOURCE_PREFIXES",
  "b2_source_archive": "$B2_SOURCE_ARCHIVE",
  "b2_run_prefix": "$B2_RUN_PREFIX",
  "dataset_limit": $DATASET_LIMIT,
  "voxel_resolution": $VOXEL_RESOLUTION,
  "max_faces_per_patch": $MAX_FACES_PER_PATCH,
  "patch_workers": $PATCH_WORKERS,
  "train_steps": $TRAIN_STEPS
}
JSON
echo "$OUT_DIR"
