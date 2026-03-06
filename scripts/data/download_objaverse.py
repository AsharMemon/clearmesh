#!/usr/bin/env python3
"""Download Objaverse dataset for ClearMesh training.

Objaverse contains ~800K 3D models. We download them in batches,
supporting resume and subset selection for testing.

IMPORTANT: On RunPod, the root filesystem (/) is ephemeral and gets
wiped on pod restart. This script redirects the objaverse cache to
the persistent /workspace volume via HF_HOME and symlinks.

Usage:
    # Download first 1000 for testing
    python download_objaverse.py --limit 1000 --output_dir /workspace/data/objaverse

    # Download 100K for training
    python download_objaverse.py --limit 100000 --output_dir /workspace/data/objaverse

    # Resume interrupted download (automatic via manifest)
    python download_objaverse.py --limit 100000 --output_dir /workspace/data/objaverse
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import objaverse
from tqdm import tqdm


def setup_persistent_cache(output_dir: str) -> None:
    """Redirect objaverse cache to persistent storage.

    The objaverse pip package downloads to ~/.objaverse/ by default,
    which is on the ephemeral root filesystem. On RunPod (spot instances),
    this gets wiped on pod restart/eviction.

    This function:
      1. Creates a persistent cache dir on /workspace
      2. Symlinks ~/.objaverse → /workspace/.objaverse_cache
      3. Sets HF_HOME to persistent storage (objaverse uses HF hub)

    Args:
        output_dir: The output directory (should be on persistent storage).
    """
    # Determine persistent base (parent of output_dir, assumed persistent)
    persistent_base = Path(output_dir).parent

    # Persistent cache for objaverse downloads
    persistent_cache = persistent_base / ".objaverse_cache"
    persistent_cache.mkdir(parents=True, exist_ok=True)

    # Symlink ~/.objaverse → persistent cache
    home_cache = Path.home() / ".objaverse"
    if home_cache.is_symlink():
        # Already symlinked — check it points to right place
        if home_cache.resolve() != persistent_cache.resolve():
            home_cache.unlink()
            home_cache.symlink_to(persistent_cache)
    elif home_cache.exists():
        # Existing dir (from previous ephemeral download) — move contents
        print(f"Moving existing cache {home_cache} → {persistent_cache}")
        for item in home_cache.iterdir():
            dest = persistent_cache / item.name
            if not dest.exists():
                shutil.move(str(item), str(dest))
        shutil.rmtree(str(home_cache))
        home_cache.symlink_to(persistent_cache)
    else:
        home_cache.symlink_to(persistent_cache)

    print(f"Objaverse cache: {home_cache} → {persistent_cache}")

    # Also redirect HuggingFace cache
    hf_cache = persistent_base / ".hf_cache"
    hf_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_cache)


def download_objaverse(output_dir: str, limit: int | None = None, processes: int = 8):
    """Download Objaverse models to persistent disk.

    Args:
        output_dir: Output directory (should be on persistent /workspace).
        limit: Maximum number of models to download.
        processes: Number of parallel download workers.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Setup persistent cache to survive pod restarts
    setup_persistent_cache(output_dir)

    # Get all available UIDs
    uids = objaverse.load_uids()
    print(f"Total available models: {len(uids)}")

    if limit:
        uids = uids[:limit]
        print(f"Downloading subset: {limit} models")

    # Check which are already downloaded (resume support)
    manifest_path = output_path / "manifest.json"
    existing_manifest = {}
    if manifest_path.exists():
        with open(manifest_path) as f:
            existing_manifest = json.load(f)

        # Only count entries whose files actually exist on disk
        valid = {uid: p for uid, p in existing_manifest.items() if os.path.exists(p)}
        stale = len(existing_manifest) - len(valid)
        if stale > 0:
            print(f"Removing {stale} stale manifest entries (files missing)")
            existing_manifest = valid

        remaining = [uid for uid in uids if uid not in existing_manifest]
        print(f"Already downloaded: {len(existing_manifest)}, remaining: {len(remaining)}")
        uids = remaining

    if not uids:
        print("All models already downloaded.")
        return

    # Download in batches
    batch_size = 500
    total_new = 0

    for i in tqdm(range(0, len(uids), batch_size), desc="Batches"):
        batch = uids[i : i + batch_size]

        try:
            paths = objaverse.load_objects(batch, download_processes=processes)
        except Exception as e:
            print(f"\nBatch {i // batch_size} failed: {e}")
            print("Saving progress and continuing...")
            continue

        # Verify downloaded files exist and count them
        valid_paths = {uid: str(p) for uid, p in paths.items() if os.path.exists(p)}
        total_new += len(valid_paths)

        # Update manifest (merge with existing)
        existing_manifest.update(valid_paths)

        with open(manifest_path, "w") as f:
            json.dump(existing_manifest, f)

        if (i // batch_size + 1) % 10 == 0:
            print(f"\n  Progress: {len(existing_manifest)} total, {total_new} new this run")

    print(f"\nDownload complete!")
    print(f"  New this run: {total_new}")
    print(f"  Total in manifest: {len(existing_manifest)}")
    print(f"  Manifest: {manifest_path}")

    # Show a sample path to verify persistent storage
    if existing_manifest:
        sample_path = next(iter(existing_manifest.values()))
        print(f"  Sample path: {sample_path}")
        print(f"  File exists: {os.path.exists(sample_path)}")


def main():
    parser = argparse.ArgumentParser(description="Download Objaverse dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/workspace/data/objaverse",
        help="Output directory (use /workspace for persistent storage on RunPod)",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Max models to download (None = all)"
    )
    parser.add_argument(
        "--processes", type=int, default=8, help="Parallel download processes"
    )
    args = parser.parse_args()

    download_objaverse(args.output_dir, args.limit, args.processes)


if __name__ == "__main__":
    main()
