#!/usr/bin/env bash
# Rent a cheap single-GPU Vast instance and launch the bounded Localizable FACE M2 run.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

VAST_BIN="${VAST_BIN:-/Users/Ashar/Library/Python/3.14/bin/vastai}"
OUT_DIR="${OUT_DIR:-.codex_outputs/vast_localizable_face_m2_$(date -u +%Y%m%dT%H%M%SZ)}"
IMAGE="${VAST_IMAGE:-pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel}"
LABEL="${VAST_LABEL:-clearmesh-localizable-face-m2}"
DISK_GB="${DISK_GB:-220}"
SSH_KEY_FILE="${SSH_KEY_FILE:-$HOME/.ssh/id_ed25519}"

VAST_QUERY="${VAST_QUERY:-num_gpus=1 rented=False gpu_ram>23 reliability>0.95 inet_down>100 disk_space>180}"
VAST_ORDER="${VAST_ORDER:-dph}"
VAST_LIMIT="${VAST_LIMIT:-80}"
MAX_DPH="${MAX_DPH:-0.45}"

RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
B2_SOURCE_PREFIX="${B2_SOURCE_PREFIX:-face-corpora/paper-large-65k1024-repack8k128-v2/objpp-minquality1/shard0010}"
B2_SOURCE_PREFIXES="${B2_SOURCE_PREFIXES:-}"
B2_SOURCE_ARCHIVE="${B2_SOURCE_ARCHIVE:-lean_face_corpus.tar.gz}"
B2_RUN_PREFIX="${B2_RUN_PREFIX:-face-runs/localizable-face-m2/vast-$RUN_STAMP}"
ARCHIVE_DATASET_DIR="${ARCHIVE_DATASET_DIR:-corpus/split_pass/train}"
DATASET_LIMIT="${DATASET_LIMIT:-4096}"
VOXEL_RESOLUTION="${VOXEL_RESOLUTION:-32}"
MAX_FACES_PER_PATCH="${MAX_FACES_PER_PATCH:-96}"
POINT_SAMPLES_PER_PATCH="${POINT_SAMPLES_PER_PATCH:-32}"
PATCH_WORKERS="${PATCH_WORKERS:-6}"
VERIFY_LIMIT_SOURCES="${VERIFY_LIMIT_SOURCES:-0}"
DEBUG_MESH_LIMIT="${DEBUG_MESH_LIMIT:-8}"
TRAIN_STEPS="${TRAIN_STEPS:-500}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
TRAIN_LIMIT="${TRAIN_LIMIT:-131072}"
TRAIN_POINT_SAMPLES="${TRAIN_POINT_SAMPLES:-32}"
HIDDEN_SIZE="${HIDDEN_SIZE:-256}"
LAYERS="${LAYERS:-6}"
HEADS="${HEADS:-8}"
CONDITION_TOKENS="${CONDITION_TOKENS:-16}"
LOG_EVERY="${LOG_EVERY:-25}"

mkdir -p "$OUT_DIR"
LOCAL_LOG="$OUT_DIR/launch.log"

if [[ -z "${VAST_API:-}" ]]; then
  echo "VAST_API is not set." >&2
  exit 2
fi
if [[ ! -x "$VAST_BIN" ]]; then
  echo "vastai CLI not found at $VAST_BIN" >&2
  exit 2
fi
if [[ ! -f "$SSH_KEY_FILE" ]]; then
  echo "Missing SSH key file: $SSH_KEY_FILE" >&2
  exit 2
fi

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOCAL_LOG"
}

ssh_url_to_probe() {
  python3 - "$1" "$2" <<'PY'
import re, sys
from pathlib import Path
from urllib.parse import urlparse

text, out_path = sys.argv[1:3]
text = text.strip()
if text.startswith("ssh://"):
    parsed = urlparse(text)
    user = parsed.username or "root"
    host = parsed.hostname or ""
    port = str(parsed.port or 22)
else:
    match = re.search(r'([^@\s]+)@([^:\s]+):(\d+)$', text)
    if match:
        user, host, port = match.groups()
    else:
        match = re.search(r'(?:ssh\s+)?(?:-p\s+(\d+)\s+)?([^@\s]+)@([^\s]+)', text)
        if not match:
            raise SystemExit(1)
        port, user, host = match.groups()
        port = port or "22"
if not user or not host or not port:
    raise SystemExit(1)
Path(out_path).write_text(f"HOST={host}\nSSH_PORT={port}\nREMOTE_USER={user}\n", encoding="utf-8")
print(f"{user}@{host}:{port}")
PY
}

log "Searching Vast offers query=$VAST_QUERY max_dph=$MAX_DPH"
"$VAST_BIN" --api-key "$VAST_API" search offers "$VAST_QUERY" --raw --limit "$VAST_LIMIT" -o "$VAST_ORDER" \
  > "$OUT_DIR/offers.json" 2> "$OUT_DIR/offers.err" || true

offer_id="$(
  python3 - "$OUT_DIR/offers.json" "$OUT_DIR/selected_offer.json" "$MAX_DPH" <<'PY'
import json, sys
from pathlib import Path

offers_path, selected_path, max_dph_text = sys.argv[1:4]
max_dph = float(max_dph_text)
try:
    offers = json.loads(Path(offers_path).read_text() or "[]")
except Exception as exc:
    raise SystemExit(f"failed to parse offers: {exc}")

def gpu_class(name: str, ram: int) -> int:
    upper = name.upper()
    if "A100" in upper and ram >= 80000:
        return 0
    if "H100" in upper:
        return 1
    if "A100" in upper:
        return 2
    if "L40" in upper or "RTX 6000 ADA" in upper:
        return 3
    if "A6000" in upper or "A5000" in upper or "A10" in upper:
        return 4
    if "3090" in upper or "4090" in upper or "V100" in upper or "TITAN RTX" in upper or "Q RTX" in upper:
        return 5
    return 9

candidates = []
for offer in offers:
    dph = float(offer.get("dph_total") or offer.get("dph") or 999.0)
    if dph > max_dph:
        continue
    ram = int(offer.get("gpu_ram") or 0)
    name = str(offer.get("gpu_name") or "")
    if ram < 24000:
        continue
    candidates.append((gpu_class(name, ram), dph, -float(offer.get("inet_down") or 0), offer))

if not candidates:
    raise SystemExit("no suitable Vast offers under MAX_DPH")

# For M2, price matters more than GPU class after the card has >=24GB VRAM.
selected = sorted(candidates, key=lambda item: (item[1], item[0], item[2]))[0][3]
Path(selected_path).write_text(json.dumps(selected, indent=2, sort_keys=True) + "\n")
print(selected["id"])
PY
)"
log "Selected Vast offer $offer_id"

"$VAST_BIN" --api-key "$VAST_API" create instance "$offer_id" \
  --image "$IMAGE" \
  --disk "$DISK_GB" \
  --ssh \
  --direct \
  --cancel-unavail \
  --label "$LABEL" \
  --onstart-cmd 'mkdir -p /workspace; sleep infinity' \
  --raw > "$OUT_DIR/create_response.json" 2>&1

instance_id="$(
  python3 - "$OUT_DIR/create_response.json" <<'PY'
import json, re, sys
text = open(sys.argv[1], encoding="utf-8", errors="ignore").read()
try:
    data = json.loads(text)
except Exception:
    start = text.find("{")
    data = json.loads(text[start:]) if start >= 0 else {}
for key in ("new_contract", "id", "instance_id"):
    if data.get(key) is not None:
        print(data[key])
        raise SystemExit
match = re.search(r"new_contract['\"]?\s*[:=]\s*([0-9]+)", text)
if match:
    print(match.group(1))
PY
)"
if [[ -z "$instance_id" ]]; then
  echo "Vast create did not return instance id. See $OUT_DIR/create_response.json" >&2
  exit 3
fi
log "Created Vast instance $instance_id"

probe="$OUT_DIR/ssh_probe.sh"
deadline=$(( $(date +%s) + 1800 ))
while [[ "$(date +%s)" -lt "$deadline" ]]; do
  raw="$("$VAST_BIN" --api-key "$VAST_API" ssh-url "$instance_id" 2>/dev/null || true)"
  if [[ -n "$raw" ]] && ssh_url_to_probe "$raw" "$probe" >> "$LOCAL_LOG" 2>&1; then
    break
  fi
  sleep 15
done
if [[ ! -s "$probe" ]]; then
  echo "Vast instance did not expose SSH URL before timeout." >&2
  exit 4
fi
log "SSH probe ready"

SSH_PROBE_FILE="$probe" \
SSH_KEY_FILE="$SSH_KEY_FILE" \
REMOTE_BASE=/workspace \
REMOTE_REPO=/workspace/clearmesh-localizable-m2-repo \
REMOTE_VENV=/workspace/clearmesh-localizable-m2-venv \
REMOTE_LAB_ROOT="/workspace/clearmesh_localizable_face_m2_$RUN_STAMP" \
B2_SOURCE_PREFIX="$B2_SOURCE_PREFIX" \
B2_SOURCE_PREFIXES="$B2_SOURCE_PREFIXES" \
B2_SOURCE_ARCHIVE="$B2_SOURCE_ARCHIVE" \
B2_RUN_PREFIX="$B2_RUN_PREFIX" \
ARCHIVE_DATASET_DIR="$ARCHIVE_DATASET_DIR" \
DATASET_LIMIT="$DATASET_LIMIT" \
VOXEL_RESOLUTION="$VOXEL_RESOLUTION" \
MAX_FACES_PER_PATCH="$MAX_FACES_PER_PATCH" \
POINT_SAMPLES_PER_PATCH="$POINT_SAMPLES_PER_PATCH" \
PATCH_WORKERS="$PATCH_WORKERS" \
VERIFY_LIMIT_SOURCES="$VERIFY_LIMIT_SOURCES" \
DEBUG_MESH_LIMIT="$DEBUG_MESH_LIMIT" \
TRAIN_STEPS="$TRAIN_STEPS" \
TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE" \
TRAIN_LIMIT="$TRAIN_LIMIT" \
TRAIN_POINT_SAMPLES="$TRAIN_POINT_SAMPLES" \
HIDDEN_SIZE="$HIDDEN_SIZE" \
LAYERS="$LAYERS" \
HEADS="$HEADS" \
CONDITION_TOKENS="$CONDITION_TOKENS" \
LOG_EVERY="$LOG_EVERY" \
OUT_DIR="$OUT_DIR/remote_launch" \
  bash scripts/runcrate/bootstrap_runcrate_localizable_face_smoke.sh | tee -a "$LOCAL_LOG"

cat > "$OUT_DIR/run_info.json" <<JSON
{
  "provider": "vast",
  "instance_id": "$instance_id",
  "offer_id": "$offer_id",
  "ssh_probe_file": "$probe",
  "b2_source_prefix": "$B2_SOURCE_PREFIX",
  "b2_source_prefixes": "$B2_SOURCE_PREFIXES",
  "b2_source_archive": "$B2_SOURCE_ARCHIVE",
  "b2_run_prefix": "$B2_RUN_PREFIX",
  "dataset_limit": $DATASET_LIMIT,
  "train_steps": $TRAIN_STEPS,
  "selected_offer_file": "$OUT_DIR/selected_offer.json",
  "remote_launch_dir": "$OUT_DIR/remote_launch",
  "remote_lab_root": "/workspace/clearmesh_localizable_face_m2_$RUN_STAMP",
  "destroy_command": "$VAST_BIN --api-key [REDACTED] destroy instance $instance_id"
}
JSON
printf '%s\n' "$OUT_DIR" > .codex_outputs/latest_vast_localizable_face_m2_dir.txt
log "Launched Localizable FACE M2 on Vast instance $instance_id"
