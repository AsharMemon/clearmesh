#!/usr/bin/env bash
set -euo pipefail

BASE_RUN_DIR="${RUN_DIR:-/tmp/clearmesh_face_paper_ab}"
BASE_ARCHIVE_PATH="${ARCHIVE_PATH:-/tmp/$(basename "$BASE_RUN_DIR").tar.gz}"
HEADS_CSV="${HEADS_CSV:-causal,parallel}"

IFS=',' read -r -a HEADS_TO_RUN <<< "$HEADS_CSV"
SUMMARY_DIR="$BASE_RUN_DIR"
rm -rf "$SUMMARY_DIR"
mkdir -p "$SUMMARY_DIR"

for HEAD in "${HEADS_TO_RUN[@]}"; do
  HEAD="$(echo "$HEAD" | xargs)"
  [ -n "$HEAD" ] || continue
  SUB_RUN_DIR="${BASE_RUN_DIR}_${HEAD}"
  SUB_ARCHIVE="/tmp/$(basename "$SUB_RUN_DIR").tar.gz"
  echo "=== FACE paper A/B: DECODE_HEAD=$HEAD RUN_DIR=$SUB_RUN_DIR ==="
  RUN_DIR="$SUB_RUN_DIR" ARCHIVE_PATH="$SUB_ARCHIVE" DECODE_HEAD="$HEAD" scripts/thunder/face_paper_holdout_remote_job.sh
  mkdir -p "$SUMMARY_DIR/$HEAD"
  tar -xzf "$SUB_ARCHIVE" -C "$SUMMARY_DIR/$HEAD"
done

python - <<PY
import json
from pathlib import Path
base = Path("$SUMMARY_DIR")
comparison = {}
for child in sorted(p for p in base.iterdir() if p.is_dir()):
    summary_path = child / "holdout_summary.json"
    if summary_path.exists():
        comparison[child.name] = json.loads(summary_path.read_text())
(base / "ab_comparison.json").write_text(json.dumps(comparison, indent=2, sort_keys=True))
print(json.dumps(comparison, indent=2, sort_keys=True))
PY

tar -czf "$BASE_ARCHIVE_PATH" -C "$SUMMARY_DIR" .
ls -lh "$BASE_ARCHIVE_PATH"
