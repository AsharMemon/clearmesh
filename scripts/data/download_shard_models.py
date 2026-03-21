#!/usr/bin/env python3
"""Download 3D model files for a specific shard from ObjaverseXL.

Used on Vast.ai pods to download only the models assigned to this pod's shard,
rather than downloading the entire TRELLIS-500K dataset.

Reads the shard JSON (from prepare_vastai_shards.py) and downloads matching
models from ObjaverseXL using the SHA256 as lookup key.

Output:
  - Downloads model files to --download_dir
  - Updates shard JSON entries with local file paths
  - Creates download_progress.json for resume support

Usage:
    # On a Vast.ai pod:
    python download_shard_models.py \\
        --shard_json /workspace/data/shards/shard_0.json \\
        --download_dir /workspace/data/models \\
        --processes 16

    # Resume interrupted download:
    python download_shard_models.py \\
        --shard_json /workspace/data/shards/shard_0.json \\
        --download_dir /workspace/data/models
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def load_shard(path: str) -> list:
    """Load shard JSON."""
    with open(path) as f:
        entries = json.load(f)
    print(f"Shard: {len(entries):,} models from {path}")
    return entries


def load_progress(progress_path: str) -> dict:
    """Load download progress: sha256 → local_path."""
    if os.path.exists(progress_path):
        with open(progress_path) as f:
            progress = json.load(f)
        valid = {k: v for k, v in progress.items() if os.path.exists(v)}
        if len(valid) < len(progress):
            print(f"  Removed {len(progress) - len(valid)} stale entries")
        return valid
    return {}


def save_progress(progress: dict, progress_path: str):
    """Save progress atomically."""
    tmp = progress_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(progress, f)
    os.replace(tmp, progress_path)


def download_models(
    entries: list,
    download_dir: str,
    progress: dict,
    progress_path: str,
    processes: int = 16,
    batch_size: int = 200,
) -> dict:
    """Download models from ObjaverseXL using SHA256 matching."""
    import objaverse.xl as oxl

    # Get SHA256s we need
    target_sha = set(e["uid"] for e in entries)  # uid = sha256 in our format
    remaining = target_sha - set(progress.keys())

    print(f"\nTotal in shard: {len(target_sha):,}")
    print(f"Already downloaded: {len(target_sha) - len(remaining):,}")
    print(f"Remaining: {len(remaining):,}")

    if not remaining:
        print("All models already downloaded!")
        return progress

    # Get ObjaverseXL annotations
    print("\nLoading ObjaverseXL annotations...")
    annotations = oxl.get_annotations(
        download_dir=os.path.expanduser("~/.objaverse")
    )

    # Match by SHA256
    matched = annotations[annotations["sha256"].isin(remaining)].copy()
    print(f"Matched {len(matched):,} models in ObjaverseXL catalog")

    if len(matched) == 0:
        print("WARNING: No matches found. Check SHA256 format.")
        return progress

    # Split by source for efficient downloading
    sf_matched = matched[matched["source"] == "sketchfab"]
    gh_matched = matched[matched["source"] == "github"]
    other_matched = matched[~matched["source"].isin(["sketchfab", "github"])]

    print(f"  Sketchfab: {len(sf_matched):,}")
    print(f"  GitHub: {len(gh_matched):,}")
    if len(other_matched) > 0:
        print(f"  Other: {len(other_matched):,}")

    # Download Sketchfab first (faster, individual GLBs)
    total_new = 0
    for source_name, source_df in [("Sketchfab", sf_matched), ("GitHub", gh_matched), ("Other", other_matched)]:
        if len(source_df) == 0:
            continue

        print(f"\nDownloading {source_name} models...")
        bs = batch_size if source_name == "Sketchfab" else min(batch_size, 100)

        for i in tqdm(range(0, len(source_df), bs), desc=source_name):
            batch = source_df.iloc[i:i + bs]

            try:
                paths = oxl.download_objects(
                    batch,
                    download_dir=download_dir,
                    processes=processes,
                )
                sha_map = dict(zip(batch["fileIdentifier"], batch["sha256"]))
                for fi, local_path in paths.items():
                    if os.path.exists(local_path):
                        sha = sha_map.get(fi, "")
                        if sha:
                            progress[sha] = str(local_path)
                            total_new += 1
            except Exception as e:
                print(f"\n  Batch {i // bs} failed: {e}")
                continue

            # Save progress every 10 batches
            if (i // bs + 1) % 10 == 0:
                save_progress(progress, progress_path)
                print(f"\n  Progress: {total_new} new, {len(progress)} total")

    save_progress(progress, progress_path)
    print(f"\nDownload complete: {total_new} new models")
    return progress


def update_shard_json(shard_path: str, entries: list, progress: dict):
    """Update shard JSON with resolved local paths."""
    updated = []
    resolved = 0
    for entry in entries:
        sha = entry["uid"]
        if sha in progress:
            entry["path"] = progress[sha]
            resolved += 1
        updated.append(entry)

    with open(shard_path, "w") as f:
        json.dump(updated, f, indent=2)

    print(f"\nUpdated shard JSON: {resolved:,} / {len(entries):,} have local paths")


def main():
    parser = argparse.ArgumentParser(
        description="Download model files for a shard from ObjaverseXL"
    )
    parser.add_argument("--shard_json", type=str, required=True,
                        help="Path to shard JSON (from prepare_vastai_shards.py)")
    parser.add_argument("--download_dir", type=str,
                        default="/workspace/data/models",
                        help="Where to download 3D model files")
    parser.add_argument("--processes", type=int, default=16,
                        help="Parallel download workers")
    parser.add_argument("--batch_size", type=int, default=200,
                        help="Download batch size")

    args = parser.parse_args()

    print("=" * 70)
    print("ClearMesh Shard Model Download")
    print("=" * 70)

    # Setup
    os.makedirs(args.download_dir, exist_ok=True)
    os.environ.setdefault("HF_HOME", "/workspace/.hf_cache")

    # Symlink ~/.objaverse cache
    objaverse_cache = Path("/workspace/.objaverse_cache")
    objaverse_cache.mkdir(parents=True, exist_ok=True)
    home_cache = Path.home() / ".objaverse"
    if not home_cache.is_symlink():
        if home_cache.exists():
            import shutil
            for item in home_cache.iterdir():
                dest = objaverse_cache / item.name
                if not dest.exists():
                    shutil.move(str(item), str(dest))
            shutil.rmtree(str(home_cache))
        home_cache.symlink_to(objaverse_cache)

    # Load shard
    entries = load_shard(args.shard_json)

    # Progress file next to shard JSON
    progress_dir = os.path.dirname(os.path.abspath(args.shard_json))
    shard_name = os.path.splitext(os.path.basename(args.shard_json))[0]
    progress_path = os.path.join(progress_dir, f"{shard_name}_download_progress.json")

    # Load progress
    progress = load_progress(progress_path)
    print(f"Existing progress: {len(progress):,}")

    # Download
    t0 = time.time()
    progress = download_models(
        entries, args.download_dir, progress, progress_path,
        processes=args.processes,
        batch_size=args.batch_size,
    )
    dt = time.time() - t0

    # Update shard JSON with paths
    update_shard_json(args.shard_json, entries, progress)

    # Stats
    total_size = sum(
        os.path.getsize(v) for v in progress.values() if os.path.exists(v)
    )
    print(f"\n{'=' * 70}")
    print(f"Download complete in {dt / 60:.1f} minutes")
    print(f"Total models: {len(progress):,}")
    print(f"Total size: {total_size / 1e9:.1f} GB")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
