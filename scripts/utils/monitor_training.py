#!/usr/bin/env python3
"""Monitor training progress: loss curves, checkpoint status, ETA.

Usage:
    python monitor_training.py --checkpoint_dir /mnt/data/checkpoints/clearmesh_stage2

    # Continuous monitoring (refresh every 30s)
    python monitor_training.py --checkpoint_dir /mnt/data/checkpoints/clearmesh_stage2 --watch
"""

import argparse
import os
import time
from datetime import timedelta
from pathlib import Path

import torch


def get_checkpoint_info(checkpoint_dir: str) -> list[dict]:
    """List all checkpoints with metadata."""
    ckpt_dir = Path(checkpoint_dir)
    checkpoints = []

    for ckpt_file in sorted(ckpt_dir.glob("checkpoint_*.pt")):
        try:
            # Load just the metadata (not the full model weights)
            ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
            info = {
                "file": ckpt_file.name,
                "step": ckpt.get("global_step", 0),
                "epoch": ckpt.get("epoch", 0),
                "size_mb": ckpt_file.stat().st_size / (1024 * 1024),
                "modified": ckpt_file.stat().st_mtime,
            }
            checkpoints.append(info)
        except Exception as e:
            checkpoints.append({"file": ckpt_file.name, "error": str(e)})

    return checkpoints


def estimate_eta(checkpoints: list[dict], total_steps: int) -> str | None:
    """Estimate time remaining based on checkpoint timestamps."""
    valid = [c for c in checkpoints if "step" in c and "modified" in c and c["step"] > 0]

    if len(valid) < 2:
        return None

    # Use first and last checkpoint to estimate speed
    first = min(valid, key=lambda c: c["step"])
    last = max(valid, key=lambda c: c["step"])

    steps_done = last["step"] - first["step"]
    time_elapsed = last["modified"] - first["modified"]

    if steps_done <= 0 or time_elapsed <= 0:
        return None

    steps_per_second = steps_done / time_elapsed
    steps_remaining = total_steps - last["step"]
    seconds_remaining = steps_remaining / steps_per_second

    return str(timedelta(seconds=int(seconds_remaining)))


def print_status(checkpoint_dir: str, total_steps: int = 100_000):
    """Print current training status."""
    checkpoints = get_checkpoint_info(checkpoint_dir)

    if not checkpoints:
        print("No checkpoints found.")
        return

    print(f"\n{'='*60}")
    print(f"  Training Monitor: {checkpoint_dir}")
    print(f"{'='*60}")

    latest = max(
        [c for c in checkpoints if "step" in c], key=lambda c: c["step"], default=None
    )

    if latest:
        progress = latest["step"] / total_steps * 100
        print(f"  Current step: {latest['step']:,} / {total_steps:,} ({progress:.1f}%)")
        print(f"  Current epoch: {latest['epoch']}")

        eta = estimate_eta(checkpoints, total_steps)
        if eta:
            print(f"  Estimated time remaining: {eta}")

    print(f"\n  Checkpoints ({len(checkpoints)} total):")
    for ckpt in checkpoints[-5:]:  # Show last 5
        if "error" in ckpt:
            print(f"    {ckpt['file']}: ERROR - {ckpt['error']}")
        else:
            timestamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(ckpt["modified"]))
            print(f"    {ckpt['file']}: step {ckpt['step']:,}, {ckpt['size_mb']:.1f}MB ({timestamp})")

    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Monitor training progress")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--total_steps", type=int, default=100_000)
    parser.add_argument("--watch", action="store_true", help="Continuous monitoring")
    parser.add_argument("--interval", type=int, default=30, help="Refresh interval (seconds)")
    args = parser.parse_args()

    if args.watch:
        try:
            while True:
                os.system("clear")
                print_status(args.checkpoint_dir, args.total_steps)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nMonitor stopped.")
    else:
        print_status(args.checkpoint_dir, args.total_steps)


if __name__ == "__main__":
    main()
