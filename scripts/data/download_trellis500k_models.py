#!/usr/bin/env python3
"""Download all TRELLIS-500K 3D models from ObjaverseXL.

This is the streamlined download script — assumes metadata CSVs are already
downloaded to /workspace/data/trellis500k/metadata/ (by download_trellis500k.py
or manually).

Strategy:
  1. Download Sketchfab models first (168K, fast individual downloads)
  2. Download GitHub models second (312K, slower repo-based downloads)
  3. Create valid_models.json for pair generation

Designed for long-running execution on RunPod pods with resume support.

Usage:
    # Download everything
    python download_trellis500k_models.py

    # Download only Sketchfab (faster, 168K models)
    python download_trellis500k_models.py --sketchfab_only

    # Resume interrupted download
    python download_trellis500k_models.py  # automatic

    # Create valid_models.json from already-downloaded files
    python download_trellis500k_models.py --build_json_only
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def load_trellis_metadata(meta_dir: str) -> pd.DataFrame:
    """Load TRELLIS-500K metadata CSVs."""
    meta_path = Path(meta_dir)
    dfs = []

    for name, csv in [("sketchfab", "ObjaverseXL_sketchfab.csv"),
                       ("github", "ObjaverseXL_github.csv")]:
        path = meta_path / csv
        if not path.exists():
            print(f"WARNING: {path} not found — skipping {name}")
            continue
        df = pd.read_csv(path)
        df["trellis_source"] = name
        dfs.append(df)
        print(f"  {name}: {len(df)} entries")

    if not dfs:
        print("ERROR: No metadata CSVs found. Run download_trellis500k.py first.")
        sys.exit(1)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"  Total: {len(combined)} entries")
    return combined


def load_progress(progress_path: str) -> dict:
    """Load download progress (sha256 → local_path mapping)."""
    if os.path.exists(progress_path):
        with open(progress_path) as f:
            progress = json.load(f)
        # Validate existing entries
        valid = {k: v for k, v in progress.items() if os.path.exists(v)}
        stale = len(progress) - len(valid)
        if stale > 0:
            print(f"  Removed {stale} stale entries")
        return valid
    return {}


def save_progress(progress: dict, progress_path: str):
    """Save download progress atomically."""
    tmp = progress_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(progress, f)
    os.replace(tmp, progress_path)


def download_sketchfab(
    trellis_metadata: pd.DataFrame,
    download_dir: str,
    progress: dict,
    progress_path: str,
    processes: int = 16,
    batch_size: int = 200,
) -> dict:
    """Download Sketchfab models (individual GLB downloads, fast)."""
    import objaverse.xl as oxl

    sf_meta = trellis_metadata[trellis_metadata["trellis_source"] == "sketchfab"]
    target_sha = set(sf_meta["sha256"].values)
    remaining = target_sha - set(progress.keys())

    print(f"\nSketchfab: {len(sf_meta)} total, {len(remaining)} remaining")
    if not remaining:
        print("  All Sketchfab models already downloaded!")
        return progress

    # Get ObjaverseXL annotations for matching
    print("  Loading ObjaverseXL annotations...")
    annotations = oxl.get_annotations(download_dir=os.path.expanduser("~/.objaverse"))
    sf_annotations = annotations[
        (annotations["source"] == "sketchfab") &
        (annotations["sha256"].isin(remaining))
    ].copy()
    print(f"  Matched {len(sf_annotations)} Sketchfab models in ObjaverseXL catalog")

    if len(sf_annotations) == 0:
        return progress

    # Download in batches
    total_new = 0
    for i in tqdm(range(0, len(sf_annotations), batch_size),
                  desc="Sketchfab download"):
        batch = sf_annotations.iloc[i:i + batch_size]

        try:
            paths = oxl.download_objects(
                batch,
                download_dir=download_dir,
                processes=processes,
            )
            # Map file_identifier paths back to sha256
            sha_map = dict(zip(batch["fileIdentifier"], batch["sha256"]))
            for fi, local_path in paths.items():
                if os.path.exists(local_path):
                    sha = sha_map.get(fi, "")
                    if sha:
                        progress[sha] = str(local_path)
                        total_new += 1
        except Exception as e:
            print(f"\n  Batch {i // batch_size} failed: {e}")
            continue

        # Save progress every 10 batches
        if (i // batch_size + 1) % 10 == 0:
            save_progress(progress, progress_path)
            print(f"\n  Progress: {total_new} new, {len(progress)} total")

    save_progress(progress, progress_path)
    print(f"\n  Sketchfab download: {total_new} new models")
    return progress


def download_github(
    trellis_metadata: pd.DataFrame,
    download_dir: str,
    progress: dict,
    progress_path: str,
    processes: int = 8,
    batch_size: int = 100,
) -> dict:
    """Download GitHub models (repo-based downloads, slower)."""
    import objaverse.xl as oxl

    gh_meta = trellis_metadata[trellis_metadata["trellis_source"] == "github"]
    target_sha = set(gh_meta["sha256"].values)
    remaining = target_sha - set(progress.keys())

    print(f"\nGitHub: {len(gh_meta)} total, {len(remaining)} remaining")
    if not remaining:
        print("  All GitHub models already downloaded!")
        return progress

    # Get ObjaverseXL annotations for matching
    print("  Loading ObjaverseXL annotations...")
    annotations = oxl.get_annotations(download_dir=os.path.expanduser("~/.objaverse"))
    gh_annotations = annotations[
        (annotations["source"] == "github") &
        (annotations["sha256"].isin(remaining))
    ].copy()
    print(f"  Matched {len(gh_annotations)} GitHub models in ObjaverseXL catalog")

    if len(gh_annotations) == 0:
        return progress

    # GitHub downloads are per-repo, so larger batches are more efficient
    # but also more memory-intensive. Use moderate batch size.
    total_new = 0
    for i in tqdm(range(0, len(gh_annotations), batch_size),
                  desc="GitHub download"):
        batch = gh_annotations.iloc[i:i + batch_size]

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
            print(f"\n  Batch {i // batch_size} failed: {e}")
            continue

        # Save progress every 5 batches (GitHub batches are slower)
        if (i // batch_size + 1) % 5 == 0:
            save_progress(progress, progress_path)
            print(f"\n  Progress: {total_new} new, {len(progress)} total")

    save_progress(progress, progress_path)
    print(f"\n  GitHub download: {total_new} new models")
    return progress


def build_valid_models_json(
    progress: dict,
    output_path: str,
    trellis_metadata: pd.DataFrame,
):
    """Build valid_models.json compatible with generate_pairs.py."""
    # Merge with metadata (aesthetic scores, captions)
    meta_by_sha = trellis_metadata.set_index("sha256")

    valid = []
    for sha, path in progress.items():
        if not os.path.exists(path):
            continue

        entry = {"uid": sha, "path": path}

        # Add metadata if available
        if sha in meta_by_sha.index:
            row = meta_by_sha.loc[sha]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            if pd.notna(row.get("aesthetic_score")):
                entry["aesthetic_score"] = float(row["aesthetic_score"])

        valid.append(entry)

    # Sort by aesthetic score (highest first) for prioritized pair generation
    valid.sort(key=lambda x: x.get("aesthetic_score", 0), reverse=True)

    with open(output_path, "w") as f:
        json.dump(valid, f, indent=2)

    print(f"\nvalid_models.json: {len(valid)} models")
    print(f"  Path: {output_path}")

    # Stats
    scores = [m["aesthetic_score"] for m in valid if "aesthetic_score" in m]
    if scores:
        import statistics
        print(f"  Aesthetic scores: min={min(scores):.2f}, "
              f"max={max(scores):.2f}, "
              f"mean={statistics.mean(scores):.2f}, "
              f"median={statistics.median(scores):.2f}")

    return valid


def main():
    parser = argparse.ArgumentParser(description="Download TRELLIS-500K 3D models")
    parser.add_argument("--meta_dir", type=str,
                        default="/workspace/data/trellis500k/metadata",
                        help="Directory with TRELLIS-500K CSV files")
    parser.add_argument("--download_dir", type=str,
                        default="/workspace/data/trellis500k/models",
                        help="Where to download 3D files")
    parser.add_argument("--output_json", type=str,
                        default="/workspace/data/trellis500k/valid_models.json",
                        help="Output valid_models.json path")
    parser.add_argument("--sketchfab_only", action="store_true",
                        help="Only download Sketchfab models (faster)")
    parser.add_argument("--github_only", action="store_true",
                        help="Only download GitHub models")
    parser.add_argument("--build_json_only", action="store_true",
                        help="Only build valid_models.json from existing downloads")
    parser.add_argument("--processes", type=int, default=16,
                        help="Parallel download workers")
    args = parser.parse_args()

    # Setup
    os.makedirs(args.download_dir, exist_ok=True)
    os.environ["HF_HOME"] = "/workspace/.hf_cache"

    # Symlink ~/.objaverse to persistent storage
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

    progress_path = os.path.join(args.download_dir, "..", "download_progress.json")
    progress_path = os.path.abspath(progress_path)

    print("=" * 80)
    print("TRELLIS-500K Model Download")
    print("=" * 80)

    # Load metadata
    print("\nLoading TRELLIS-500K metadata...")
    metadata = load_trellis_metadata(args.meta_dir)

    # Load progress
    print("\nLoading download progress...")
    progress = load_progress(progress_path)
    print(f"  Already downloaded: {len(progress)}")

    if args.build_json_only:
        build_valid_models_json(progress, args.output_json, metadata)
        return

    # Download
    t0 = time.time()

    if not args.github_only:
        progress = download_sketchfab(
            metadata, args.download_dir, progress, progress_path,
            processes=args.processes,
        )

    if not args.sketchfab_only:
        progress = download_github(
            metadata, args.download_dir, progress, progress_path,
            processes=args.processes,
        )

    dt = time.time() - t0
    print(f"\n{'=' * 80}")
    print(f"Download complete in {dt / 3600:.1f} hours")
    print(f"Total models: {len(progress)}")

    # Build valid_models.json
    build_valid_models_json(progress, args.output_json, metadata)


if __name__ == "__main__":
    main()
