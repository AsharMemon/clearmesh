#!/usr/bin/env bash
# Periodically run the FACE corpus idle-worker reaper. This is intentionally
# local-side: it uses Thunder CLI plus B2 checks before deleting anything.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-600}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/.codex_outputs/thunder_idle_reaper_watch_$(date -u +%Y%m%dT%H%M%SZ)}"

usage() {
  cat <<'USAGE'
Usage:
  INTERVAL_SECONDS=600 scripts/thunder/watch_reap_idle_face_corpus_workers.sh --all-a6000 --delete
  INTERVAL_SECONDS=600 scripts/thunder/watch_reap_idle_face_corpus_workers.sh --ids "1 2 5" --delete

This wrapper repeats reap_idle_face_corpus_workers.sh forever. The underlying
reaper still performs the safety checks: live process detection, B2 durable
archive + split_summary validation, and production-instance skip by default.
USAGE
}

if [[ "${1:-}" = "-h" || "${1:-}" = "--help" ]]; then
  usage
  exit 0
fi
if [[ $# -eq 0 ]]; then
  echo "Pass reaper arguments, for example: --all-a6000 --delete" >&2
  usage >&2
  exit 2
fi

mkdir -p "$LOG_DIR"
echo "Idle reaper watch log dir: $LOG_DIR"
while true; do
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  echo "[$(date -u +%FT%TZ)] reaper pass starting"
  if "$REPO_ROOT/scripts/thunder/reap_idle_face_corpus_workers.sh" "$@" \
      > "$LOG_DIR/reap_$stamp.log" 2>&1; then
    tail -40 "$LOG_DIR/reap_$stamp.log"
  else
    rc=$?
    echo "[$(date -u +%FT%TZ)] reaper pass failed rc=$rc; tail follows" >&2
    tail -80 "$LOG_DIR/reap_$stamp.log" >&2 || true
  fi
  echo "[$(date -u +%FT%TZ)] sleeping ${INTERVAL_SECONDS}s"
  sleep "$INTERVAL_SECONDS"
done
