#!/usr/bin/env python3
"""Live progress dashboard for pair generation across shards.

Usage:
    python scripts/utils/progress_dashboard.py [--interval 30]
    # Or via SSH:
    ssh pod 'cd /workspace/clearmesh && python scripts/utils/progress_dashboard.py'
"""

import json
import os
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

PAIR_DIR = os.environ.get("OUTPUT_DIR", "/workspace/data/training_pairs")
INPUT_JSON = os.environ.get("INPUT_JSON", "/workspace/data/filtered/valid_models.json")
NUM_SHARDS = int(os.environ.get("NUM_SHARDS", "3"))
REFRESH = int(sys.argv[sys.argv.index("--interval") + 1]) if "--interval" in sys.argv else 30


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def parse_log_tail(log_path, lines=200):
    """Extract recent timing + status from log file."""
    try:
        with open(log_path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - lines * 300))
            tail = f.read().decode("utf-8", errors="replace").splitlines()
    except Exception:
        return {}

    successes = 0
    empties = 0
    quality_rejects = 0
    attempts = 0
    last_ts = None
    first_ts = None
    backend = None
    attn = None
    models_processed = 0
    rss_vals = []

    for line in tail:
        if "[trellis]" in line:
            attempts += 1
            if "SUCCESS" in line:
                successes += 1
            elif "empty" in line:
                empties += 1
            elif "quality_reject" in line:
                quality_rejects += 1

        # Extract RSS
        m = re.search(r"RSS=(\d+\.\d+)GiB", line)
        if m:
            rss_vals.append(float(m.group(1)))

        # Shard progress line
        m = re.search(r"Shard \d+:\s+\d+%\|.*\|\s+(\d+)/(\d+)", line)
        if m:
            models_processed = int(m.group(1))

        # Timestamps
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

    return {
        "successes": successes,
        "empties": empties,
        "quality_rejects": quality_rejects,
        "attempts": attempts,
        "models_in_run": models_processed,
        "rss_last": rss_vals[-1] if rss_vals else None,
        "rss_max": max(rss_vals) if rss_vals else None,
        "backend": backend,
        "attn": attn,
        "first_ts": first_ts,
        "last_ts": last_ts,
    }


def fmt_duration(secs):
    if secs is None or secs <= 0:
        return "--"
    h, rem = divmod(int(secs), 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m:02d}m"
    return f"{m}m {s:02d}s"


def fmt_rate(count, secs):
    if not secs or secs <= 0 or count <= 0:
        return "--"
    per_hour = count / secs * 3600
    return f"{per_hour:.1f}/hr"


def dashboard():
    # Load total model count
    models = read_json(INPUT_JSON)
    total_models = len(models) if isinstance(models, list) else 0

    now = datetime.utcnow()
    print(f"\033[2J\033[H")  # Clear screen
    print(f"{'=' * 72}")
    print(f"  PAIR GENERATION DASHBOARD    {now.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"{'=' * 72}")

    total_completed = 0
    total_failed = 0
    total_remaining = 0
    shard_rows = []

    for s in range(NUM_SHARDS):
        progress_file = os.path.join(PAIR_DIR, f"shard_{s}", "progress.json")
        failures_file = os.path.join(PAIR_DIR, f"shard_{s}", "failures.json")
        log_file = f"/workspace/logs/shard{s}.log"

        completed_list = read_json(progress_file) or []
        failed_dict = read_json(failures_file) or {}
        completed = len(completed_list) if isinstance(completed_list, list) else 0
        failed = len(failed_dict) if isinstance(failed_dict, dict) else 0

        shard_total = total_models // NUM_SHARDS + (1 if s < total_models % NUM_SHARDS else 0)
        remaining = max(0, shard_total - completed - failed)

        log_info = parse_log_tail(log_file)

        total_completed += completed
        total_failed += failed
        total_remaining += remaining

        shard_rows.append((s, completed, failed, remaining, shard_total, log_info))

    # Overall summary
    grand_total = total_completed + total_failed + total_remaining
    pct = (total_completed / grand_total * 100) if grand_total > 0 else 0
    print(f"\n  Total models: {grand_total:,}")
    print(f"  Completed:    {total_completed:,}  ({pct:.1f}%)")
    print(f"  Failed:       {total_failed:,}")
    print(f"  Remaining:    {total_remaining:,}")

    # Per-shard table
    print(f"\n  {'Shard':<7} {'Done':>7} {'Fail':>7} {'Left':>7} {'Pct':>7} {'Rate':>9} {'RSS':>8} {'Attn':>10} {'Backend':>9}")
    print(f"  {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 9} {'-' * 8} {'-' * 10} {'-' * 9}")

    total_rate_per_hour = 0
    for s, completed, failed, remaining, shard_total, info in shard_rows:
        pct_s = (completed / shard_total * 100) if shard_total > 0 else 0

        # Calculate rate from log timestamps
        rate_str = "--"
        rate_per_hour = 0
        if info.get("first_ts") and info.get("last_ts") and info["models_in_run"] > 1:
            try:
                t0 = datetime.fromisoformat(info["first_ts"])
                t1 = datetime.fromisoformat(info["last_ts"])
                elapsed = (t1 - t0).total_seconds()
                if elapsed > 0:
                    rate_per_hour = info["models_in_run"] / elapsed * 3600
                    rate_str = f"{rate_per_hour:.1f}/hr"
                    total_rate_per_hour += rate_per_hour
            except Exception:
                pass

        rss_str = f"{info['rss_last']:.1f}G" if info.get("rss_last") else "--"
        attn_str = info.get("attn") or "--"
        backend_str = info.get("backend") or "--"

        print(f"  {s:<7} {completed:>7,} {failed:>7,} {remaining:>7,} {pct_s:>6.1f}% {rate_str:>9} {rss_str:>8} {attn_str:>10} {backend_str:>9}")

    # ETA
    print()
    if total_rate_per_hour > 0:
        eta_hours = total_remaining / total_rate_per_hour
        eta_dt = now + timedelta(hours=eta_hours)
        print(f"  Combined rate: {total_rate_per_hour:.1f} models/hr")
        print(f"  ETA:           {fmt_duration(eta_hours * 3600)}  ({eta_dt.strftime('%b %d %H:%M')} UTC)")
    else:
        print(f"  Combined rate: calculating...")
        print(f"  ETA:           calculating...")

    # Recent TRELLIS stats from logs
    print(f"\n  {'Shard':<7} {'TRELLIS Attempts':>17} {'Successes':>10} {'Empties':>10} {'Hit Rate':>9}")
    print(f"  {'-' * 7} {'-' * 17} {'-' * 10} {'-' * 10} {'-' * 9}")
    for s, _, _, _, _, info in shard_rows:
        att = info.get("attempts", 0)
        suc = info.get("successes", 0)
        emp = info.get("empties", 0)
        hr = f"{suc / att * 100:.0f}%" if att > 0 else "--"
        print(f"  {s:<7} {att:>17} {suc:>10} {emp:>10} {hr:>9}")

    print(f"\n{'=' * 72}")
    print(f"  Refreshing every {REFRESH}s  |  Ctrl+C to exit")
    print(f"{'=' * 72}")


def main():
    try:
        while True:
            dashboard()
            time.sleep(REFRESH)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
