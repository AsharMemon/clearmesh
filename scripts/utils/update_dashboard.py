#!/usr/bin/env python3
"""Generate dashboard/status.json and optionally push to git.

Usage:
    # Generate + push (run from cron on a pod):
    python scripts/utils/update_dashboard.py --push

    # Generate only (for local testing):
    python scripts/utils/update_dashboard.py
"""

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PAIR_DIR = os.environ.get("OUTPUT_DIR", "/workspace/data/training_pairs")
INPUT_JSON = os.environ.get("INPUT_JSON", "/workspace/data/filtered/valid_models.json")
NUM_SHARDS = int(os.environ.get("NUM_SHARDS", "3"))
REPO_DIR = os.environ.get("REPO_DIR", "/workspace/clearmesh")
OUTPUT_FILE = os.path.join(REPO_DIR, "dashboard", "status.json")
PUSH = "--push" in sys.argv


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def parse_log_tail(log_path, max_bytes=60_000):
    try:
        with open(log_path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            tail = f.read().decode("utf-8", errors="replace").splitlines()
    except Exception:
        return {}

    successes = empties = quality_rejects = attempts = 0
    models_in_run = 0
    rss_vals = []
    first_ts = last_ts = backend = attn = None

    for line in tail:
        if "[trellis]" in line:
            attempts += 1
            if "SUCCESS" in line:
                successes += 1
            elif "empty" in line:
                empties += 1
            elif "quality_reject" in line:
                quality_rejects += 1

        m = re.search(r"RSS=(\d+\.\d+)GiB", line)
        if m:
            rss_vals.append(float(m.group(1)))

        m = re.search(r"Shard \d+:\s+\d+%\|.*\|\s+(\d+)/(\d+)", line)
        if m:
            models_in_run = int(m.group(1))

        m = re.search(r"\[(\d{4}-\d{2}-\d{2}T[\d:]+)", line)
        if m:
            ts = m.group(1)
            if first_ts is None:
                first_ts = ts
            last_ts = ts

        if "attention backends:" in line:
            m2 = re.search(r"dense=(\S+)", line)
            if m2:
                attn = m2.group(1)

        if "render backend:" in line:
            m2 = re.search(r"render backend:\s*(\S+)", line)
            if m2:
                backend = m2.group(1)

    rate = 0
    if first_ts and last_ts and models_in_run > 1:
        try:
            t0 = datetime.fromisoformat(first_ts)
            t1 = datetime.fromisoformat(last_ts)
            elapsed = (t1 - t0).total_seconds()
            if elapsed > 0:
                rate = models_in_run / elapsed * 3600
        except Exception:
            pass

    return {
        "trellis_successes": successes,
        "trellis_empties": empties,
        "trellis_quality_rejects": quality_rejects,
        "trellis_attempts": attempts,
        "models_in_run": models_in_run,
        "rss_last": rss_vals[-1] if rss_vals else None,
        "rss_max": max(rss_vals) if rss_vals else None,
        "backend": backend,
        "attn": attn,
        "rate": rate,
    }


def main():
    models = read_json(INPUT_JSON)
    total_models = len(models) if isinstance(models, list) else 0

    total_completed = total_failed = total_remaining = 0
    combined_rate = 0
    shards = []

    for s in range(NUM_SHARDS):
        progress = read_json(os.path.join(PAIR_DIR, f"shard_{s}", "progress.json")) or []
        failures = read_json(os.path.join(PAIR_DIR, f"shard_{s}", "failures.json")) or {}
        completed = len(progress) if isinstance(progress, list) else 0
        failed = len(failures) if isinstance(failures, dict) else 0

        shard_total = total_models // NUM_SHARDS + (1 if s < total_models % NUM_SHARDS else 0)
        remaining = max(0, shard_total - completed - failed)

        log_info = parse_log_tail(f"/workspace/logs/shard{s}.log")
        rate = log_info.get("rate", 0)

        total_completed += completed
        total_failed += failed
        total_remaining += remaining
        combined_rate += rate

        shards.append({
            "id": s,
            "completed": completed,
            "failed": failed,
            "remaining": remaining,
            "total": shard_total,
            "rate": round(rate, 2) if rate else 0,
            "rss_last": log_info.get("rss_last"),
            "rss_max": log_info.get("rss_max"),
            "backend": log_info.get("backend"),
            "attn": log_info.get("attn"),
            "models_in_run": log_info.get("models_in_run", 0),
            "trellis_attempts": log_info.get("trellis_attempts", 0),
            "trellis_successes": log_info.get("trellis_successes", 0),
            "trellis_empties": log_info.get("trellis_empties", 0),
        })

    eta_seconds = (total_remaining / combined_rate * 3600) if combined_rate > 0 else None
    eta_date = None
    if eta_seconds:
        eta_dt = datetime.now(timezone.utc) + __import__("datetime").timedelta(seconds=eta_seconds)
        eta_date = eta_dt.isoformat()

    status = {
        "total_models": total_models,
        "total_completed": total_completed,
        "total_failed": total_failed,
        "total_remaining": total_remaining,
        "combined_rate": round(combined_rate, 2) if combined_rate else 0,
        "eta_seconds": round(eta_seconds) if eta_seconds else None,
        "eta_date": eta_date,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "shards": shards,
    }

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(status, f, indent=2)
    print(f"Wrote {OUTPUT_FILE}")

    if PUSH:
        try:
            subprocess.run(
                ["git", "add", "dashboard/status.json"],
                cwd=REPO_DIR, check=True, capture_output=True,
            )
            result = subprocess.run(
                ["git", "diff", "--cached", "--quiet"],
                cwd=REPO_DIR, capture_output=True,
            )
            if result.returncode != 0:
                subprocess.run(
                    ["git", "commit", "-m", "Update dashboard status"],
                    cwd=REPO_DIR, check=True, capture_output=True,
                )
                subprocess.run(
                    ["git", "push"],
                    cwd=REPO_DIR, check=True, capture_output=True,
                )
                print("Pushed to git")
            else:
                print("No changes to push")
        except subprocess.CalledProcessError as e:
            print(f"Git error: {e.stderr.decode() if e.stderr else e}", file=sys.stderr)


if __name__ == "__main__":
    main()
