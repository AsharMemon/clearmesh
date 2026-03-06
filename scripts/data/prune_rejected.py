#!/usr/bin/env python3
"""Prune rejected Objaverse models to free disk space.

After filtering, this script deletes models that didn't pass quality checks
from the objaverse cache. Critical for staying within volume storage quotas.

Usage:
    python prune_rejected.py \
        --manifest /workspace/data/objaverse/manifest.json \
        --valid_models /workspace/data/filtered/valid_models.json \
        --dry_run  # Preview what would be deleted

    # Actually delete:
    python prune_rejected.py \
        --manifest /workspace/data/objaverse/manifest.json \
        --valid_models /workspace/data/filtered/valid_models.json
"""

import argparse
import json
import os
from pathlib import Path


def prune_rejected(
    manifest_path: str,
    valid_models_path: str,
    dry_run: bool = True,
):
    """Delete rejected models from the objaverse cache.

    Args:
        manifest_path: Path to the full download manifest.
        valid_models_path: Path to valid_models.json from filter step.
        dry_run: If True, only print what would be deleted.
    """
    # Load full manifest (uid → file path)
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Load valid model UIDs
    with open(valid_models_path) as f:
        valid_models = json.load(f)

    valid_uids = {m["uid"] for m in valid_models}

    # Find rejected UIDs
    rejected_uids = set(manifest.keys()) - valid_uids
    print(f"Total in manifest: {len(manifest)}")
    print(f"Valid (keeping):   {len(valid_uids)}")
    print(f"Rejected (delete): {len(rejected_uids)}")

    # Calculate space to be freed
    total_size = 0
    files_to_delete = []

    for uid in rejected_uids:
        path = manifest[uid]
        if os.path.exists(path):
            size = os.path.getsize(path)
            total_size += size
            files_to_delete.append((path, size))

    print(f"\nFiles to delete: {len(files_to_delete)}")
    print(f"Space to free:   {total_size / 1024**3:.1f} GB")

    if dry_run:
        print("\n[DRY RUN] No files deleted. Remove --dry_run to delete.")
        # Show first 5 examples
        for path, size in files_to_delete[:5]:
            print(f"  Would delete: {path} ({size/1024:.0f} KB)")
        return

    # Actually delete files
    deleted = 0
    freed = 0
    for path, size in files_to_delete:
        try:
            os.remove(path)
            deleted += 1
            freed += size
        except OSError as e:
            print(f"  Failed to delete {path}: {e}")

    print(f"\nDeleted {deleted} files, freed {freed / 1024**3:.1f} GB")

    # Update manifest to only contain valid models
    new_manifest = {uid: path for uid, path in manifest.items() if uid in valid_uids}
    with open(manifest_path, "w") as f:
        json.dump(new_manifest, f)
    print(f"Updated manifest: {len(new_manifest)} entries")


def main():
    parser = argparse.ArgumentParser(description="Prune rejected Objaverse models")
    parser.add_argument(
        "--manifest", type=str, required=True, help="Path to download manifest.json"
    )
    parser.add_argument(
        "--valid_models", type=str, required=True, help="Path to valid_models.json from filter"
    )
    parser.add_argument(
        "--dry_run", action="store_true", help="Preview without deleting"
    )
    args = parser.parse_args()

    prune_rejected(args.manifest, args.valid_models, args.dry_run)


if __name__ == "__main__":
    main()
