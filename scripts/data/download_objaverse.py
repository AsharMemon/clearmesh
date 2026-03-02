#!/usr/bin/env python3
"""Download Objaverse dataset for ClearMesh training.

Objaverse contains ~800K 3D models. We download them in batches,
supporting resume and subset selection for testing.

Usage:
    # Download first 1000 for testing
    python download_objaverse.py --limit 1000 --output_dir /mnt/data/objaverse

    # Download full dataset (~1TB, takes hours)
    python download_objaverse.py --output_dir /mnt/data/objaverse
"""

import argparse
import json
import os
from pathlib import Path

import objaverse
from tqdm import tqdm


def download_objaverse(output_dir: str, limit: int | None = None, processes: int = 8):
    """Download Objaverse models to disk."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get all available UIDs
    uids = objaverse.load_uids()
    print(f"Total available models: {len(uids)}")

    if limit:
        uids = uids[:limit]
        print(f"Downloading subset: {limit} models")

    # Check which are already downloaded (resume support)
    manifest_path = output_path / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            downloaded = set(json.load(f).keys())
        remaining = [uid for uid in uids if uid not in downloaded]
        print(f"Already downloaded: {len(downloaded)}, remaining: {len(remaining)}")
        uids = remaining

    if not uids:
        print("All models already downloaded.")
        return

    # Download in batches
    batch_size = 500
    all_paths = {}

    for i in tqdm(range(0, len(uids), batch_size), desc="Batches"):
        batch = uids[i : i + batch_size]
        paths = objaverse.load_objects(batch, download_processes=processes)
        all_paths.update(paths)

        # Update manifest after each batch (resume support)
        if manifest_path.exists():
            with open(manifest_path) as f:
                existing = json.load(f)
            existing.update({uid: str(p) for uid, p in paths.items()})
            all_paths_save = existing
        else:
            all_paths_save = {uid: str(p) for uid, p in all_paths.items()}

        with open(manifest_path, "w") as f:
            json.dump(all_paths_save, f)

    print(f"\nDownloaded {len(all_paths)} models")
    print(f"Manifest: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="Download Objaverse dataset")
    parser.add_argument("--output_dir", type=str, default="/mnt/data/objaverse")
    parser.add_argument("--limit", type=int, default=None, help="Max models to download (None = all)")
    parser.add_argument("--processes", type=int, default=8, help="Parallel download processes")
    args = parser.parse_args()

    download_objaverse(args.output_dir, args.limit, args.processes)


if __name__ == "__main__":
    main()
