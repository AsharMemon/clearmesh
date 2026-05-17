#!/usr/bin/env bash
# Poll Vast.ai for an 8x A100 80GB offer, rent it, and bootstrap FACE-Q smoke.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

VAST_BIN="${VAST_BIN:-/Users/Ashar/Library/Python/3.14/bin/vastai}"
OUT_DIR="${OUT_DIR:-.codex_outputs/vast_scale_20260517}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MAX_DPH_TOTAL="${MAX_DPH_TOTAL:-14.0}"
MIN_RELIABILITY="${MIN_RELIABILITY:-0.90}"
MIN_DISK_GB="${MIN_DISK_GB:-800}"
REQUIRE_VERIFIED="${REQUIRE_VERIFIED:-0}"
DISK_GB="${DISK_GB:-1000}"
IMAGE="${VAST_IMAGE:-pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel}"
LABEL="${VAST_LABEL:-clearmesh-faceq-8xa10080-smoke}"
BOOTSTRAP="${BOOTSTRAP:-1}"
LOG="$OUT_DIR/vast_a100_80gb_poller.log"

mkdir -p "$OUT_DIR"
chmod 700 "$OUT_DIR"

if [[ -z "${VAST_API:-}" ]]; then
  echo "VAST_API is not set." >&2
  exit 2
fi

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG"
}

select_offer() {
  local offers_json="$1" selected_json="$2"
  python3 - "$offers_json" "$selected_json" "$MAX_DPH_TOTAL" "$MIN_RELIABILITY" "$MIN_DISK_GB" "$REQUIRE_VERIFIED" <<'PY'
import json
import sys

offers_path, selected_path, max_dph, min_rel, min_disk, require_verified = sys.argv[1:7]
max_dph = float(max_dph)
min_rel = float(min_rel)
min_disk = float(min_disk)
require_verified = require_verified == "1"
offers = json.load(open(offers_path, encoding="utf-8"))
matches = []
for offer in offers:
    gpu_name = str(offer.get("gpu_name", ""))
    if "A100" not in gpu_name:
        continue
    if int(offer.get("num_gpus") or 0) != 8:
        continue
    # Vast reports MB here: 81920 ~= 80GB.
    if float(offer.get("gpu_ram") or 0) < 75000:
        continue
    if float(offer.get("reliability") or 0) < min_rel:
        continue
    if float(offer.get("disk_space") or 0) < min_disk:
        continue
    if int(offer.get("direct_port_count") or 0) < 1:
        continue
    if require_verified and offer.get("verification") != "verified":
        continue
    dph = offer.get("dph_total")
    if dph is None:
        dph = offer.get("search", {}).get("totalHour")
    if dph is None or float(dph) > max_dph:
        continue
    matches.append(offer)
matches.sort(key=lambda x: (float(x.get("dph_total") or x.get("search", {}).get("totalHour") or 1e9), -float(x.get("reliability") or 0)))
if not matches:
    raise SystemExit(1)
json.dump(matches[0], open(selected_path, "w", encoding="utf-8"), indent=2, sort_keys=True)
print(matches[0]["id"])
PY
}

launch_offer() {
  local offer_id="$1" create_raw="$2"
  "$VAST_BIN" --api-key "$VAST_API" create instance "$offer_id" \
    --image "$IMAGE" \
    --disk "$DISK_GB" \
    --ssh \
    --direct \
    --cancel-unavail \
    --label "$LABEL" \
    --onstart-cmd 'mkdir -p /workspace; sleep infinity' \
    --raw > "$create_raw" 2>&1
}

parse_instance_id() {
  python3 - "$1" <<'PY'
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
m = re.search(r"new_contract['\"]?\\s*[:=]\\s*([0-9]+)", text)
if m:
    print(m.group(1))
    raise SystemExit
raise SystemExit(1)
PY
}

log "poller_started poll_seconds=$POLL_SECONDS max_dph_total=$MAX_DPH_TOTAL min_reliability=$MIN_RELIABILITY min_disk_gb=$MIN_DISK_GB disk_gb=$DISK_GB"
while true; do
  if [[ -f "$OUT_DIR/vast_launched.flag" ]]; then
    log "already_launched $(cat "$OUT_DIR/vast_launched.flag")"
    exit 0
  fi
  ts="$(date -u +%Y%m%dT%H%M%SZ)"
  offers="$OUT_DIR/a100_80gb_8x_offers_$ts.json"
  selected="$OUT_DIR/a100_80gb_8x_selected_$ts.json"
  log "searching Vast 8x A100 80GB offers"
  "$VAST_BIN" --api-key "$VAST_API" search offers \
    "external=any rentable=true rented=false num_gpus=8 gpu_ram>=75 direct_port_count>=1 disk_space>=$MIN_DISK_GB reliability>=$MIN_RELIABILITY" \
    --storage "$DISK_GB" \
    --raw \
    -n \
    -o 'dph_total,reliability-' \
    --limit 100 > "$offers"
  if offer_id="$(select_offer "$offers" "$selected" 2>/dev/null)"; then
    dph="$(python3 - "$selected" <<'PY'
import json, sys
o=json.load(open(sys.argv[1]))
print(o.get("dph_total") or o.get("search", {}).get("totalHour") or "")
PY
)"
    gpu_name="$(python3 - "$selected" <<'PY'
import json, sys
o=json.load(open(sys.argv[1]))
print(o.get("gpu_name", ""))
PY
)"
    log "selected offer_id=$offer_id gpu=$gpu_name dph_total=$dph selected=$selected"
    create_raw="$OUT_DIR/vast_create_${offer_id}_$ts.json"
    if launch_offer "$offer_id" "$create_raw"; then
      chmod 600 "$create_raw"
      instance_id="$(parse_instance_id "$create_raw")"
      printf '%s offer_id=%s instance_id=%s dph_total=%s selected=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$offer_id" "$instance_id" "$dph" "$selected" > "$OUT_DIR/vast_launched.flag"
      log "launched Vast instance_id=$instance_id offer_id=$offer_id"
      if [[ "$BOOTSTRAP" = "1" ]]; then
        VAST_INSTANCE_ID="$instance_id" OUT_DIR="$OUT_DIR" bash scripts/vast/bootstrap_vast_faceq_smoke.sh "$instance_id" >> "$LOG" 2>&1
      fi
      exit 0
    fi
    log "launch failed for offer_id=$offer_id; see $create_raw"
  else
    count="$(python3 - "$offers" <<'PY'
import json, sys
print(len(json.load(open(sys.argv[1]))))
PY
)"
    log "no qualifying 8x A100 80GB offer this pass raw_offer_count=$count"
  fi
  sleep "$POLL_SECONDS"
done
