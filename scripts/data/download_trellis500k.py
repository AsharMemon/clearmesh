#!/usr/bin/env python3
"""Download TRELLIS-500K dataset — the exact ~500K models TRELLIS was trained on.

This script:
  1. Downloads the TRELLIS-500K metadata CSVs from HuggingFace
  2. Downloads the actual 3D model files from ObjaverseXL (GitHub + Sketchfab sources)
  3. Runs basic geometry filtering (face count, aspect ratio)
  4. Outputs valid_models.json compatible with generate_pairs.py

The TRELLIS-500K dataset (https://huggingface.co/datasets/JeffreyXiang/TRELLIS-500K)
contains ~500K curated 3D models from ObjaverseXL, including ~120K high-quality
UltraShape-identified models. The SHA256 hash is the primary key.

Sources:
  - ObjaverseXL GitHub:    ~312K models (code repos with 3D files)
  - ObjaverseXL Sketchfab: ~168K models (overlaps with Objaverse v1)
  - ABO:                   ~4.5K models (Amazon Berkeley Objects)
  - 3D-FUTURE:             ~9.5K models (furniture)
  - HSSD:                  ~6.7K models (indoor scenes)

IMPORTANT: On RunPod, use persistent /workspace storage. Downloads to ephemeral
root filesystem will be lost on pod restart.

Usage:
    # Download everything (full 500K)
    python download_trellis500k.py --output_dir /workspace/data/trellis500k

    # Download only Sketchfab subset (faster, ~168K)
    python download_trellis500k.py --output_dir /workspace/data/trellis500k --sources sketchfab

    # Download + filter in one step
    python download_trellis500k.py --output_dir /workspace/data/trellis500k --filter

    # Resume interrupted download
    python download_trellis500k.py --output_dir /workspace/data/trellis500k  # automatic

    # Just generate valid_models.json from already-downloaded files
    python download_trellis500k.py --output_dir /workspace/data/trellis500k --skip_download --filter
"""

import argparse
import json
import os
import sys
import time
import hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TRELLIS_500K_REPO = "JeffreyXiang/TRELLIS-500K"
CSV_FILES = {
    "github": "ObjaverseXL_github.csv",
    "sketchfab": "ObjaverseXL_sketchfab.csv",
    "abo": "ABO.csv",
    "3d_future": "3D-FUTURE.csv",
    "hssd": "HSSD.csv",
}
# Toys4k is evaluation-only, not training
EVAL_CSV_FILES = {
    "toys4k": "Toys4k.csv",
}


def setup_persistent_cache(output_dir: str) -> None:
    """Redirect objaverse/HF cache to persistent storage (RunPod)."""
    persistent_base = Path(output_dir).parent
    hf_cache = persistent_base / ".hf_cache"
    hf_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_cache)

    objaverse_cache = persistent_base / ".objaverse_cache"
    objaverse_cache.mkdir(parents=True, exist_ok=True)

    home_cache = Path.home() / ".objaverse"
    if home_cache.is_symlink():
        if home_cache.resolve() != objaverse_cache.resolve():
            home_cache.unlink()
            home_cache.symlink_to(objaverse_cache)
    elif home_cache.exists():
        import shutil
        for item in home_cache.iterdir():
            dest = objaverse_cache / item.name
            if not dest.exists():
                shutil.move(str(item), str(dest))
        shutil.rmtree(str(home_cache))
        home_cache.symlink_to(objaverse_cache)
    else:
        home_cache.symlink_to(objaverse_cache)

    print(f"Cache: HF_HOME={hf_cache}")
    print(f"Cache: ~/.objaverse → {objaverse_cache}")


def download_metadata(output_dir: str, sources: list[str]) -> pd.DataFrame:
    """Download TRELLIS-500K metadata CSVs from HuggingFace."""
    meta_dir = Path(output_dir) / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    all_dfs = []
    for source in sources:
        csv_name = CSV_FILES[source]
        local_path = meta_dir / csv_name

        if local_path.exists():
            print(f"  {source}: Loading cached {local_path}")
            df = pd.read_csv(local_path)
        else:
            hf_url = f"hf://datasets/{TRELLIS_500K_REPO}/{csv_name}"
            print(f"  {source}: Downloading from {hf_url}...")
            df = pd.read_csv(hf_url)
            df.to_csv(local_path, index=False)
            print(f"  {source}: Saved {len(df)} entries to {local_path}")

        df["source"] = source
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal TRELLIS-500K entries: {len(combined)}")
    for src in sources:
        count = len(combined[combined["source"] == src])
        print(f"  {src}: {count}")

    return combined


def download_objaversexl_models(
    metadata: pd.DataFrame,
    output_dir: str,
    processes: int = 16,
    batch_size: int = 500,
) -> dict:
    """Download 3D model files from ObjaverseXL using sha256 matching.

    Returns dict mapping sha256 → local file path.
    """
    import objaverse.xl as oxl

    output_path = Path(output_dir)
    models_dir = output_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Load progress
    progress_path = output_path / "download_progress.json"
    downloaded = {}
    if progress_path.exists():
        with open(progress_path) as f:
            downloaded = json.load(f)
        # Validate: remove entries whose files don't exist
        valid = {k: v for k, v in downloaded.items() if os.path.exists(v)}
        stale = len(downloaded) - len(valid)
        if stale > 0:
            print(f"Removed {stale} stale progress entries")
            downloaded = valid

    print(f"\nAlready downloaded: {len(downloaded)}")

    # Split by source for ObjaverseXL download
    target_sha256s = set(metadata["sha256"].values)
    remaining_sha256s = target_sha256s - set(downloaded.keys())
    print(f"Remaining to download: {len(remaining_sha256s)}")

    if not remaining_sha256s:
        print("All models already downloaded!")
        return downloaded

    # Get ObjaverseXL annotations — this is the catalog of all available models
    print("\nFetching ObjaverseXL annotations (this may take a few minutes)...")

    # Process each source separately since they use different download mechanisms
    for source in metadata["source"].unique():
        source_df = metadata[metadata["source"] == source]
        source_sha256s = set(source_df["sha256"].values) - set(downloaded.keys())

        if not source_sha256s:
            print(f"\n  {source}: All {len(source_df)} models already downloaded")
            continue

        print(f"\n  {source}: Need to download {len(source_sha256s)} models")

        try:
            if source in ("github", "sketchfab"):
                # Use ObjaverseXL API
                source_key = f"objaverse-xl-{source}"
                print(f"  Getting {source} annotations...")
                annotations = oxl.get_annotations(download_dir=str(models_dir))

                # Filter to our target sha256s
                if isinstance(annotations, pd.DataFrame):
                    matched = annotations[annotations["sha256"].isin(source_sha256s)]
                elif isinstance(annotations, dict):
                    # Some versions return dict of DataFrames keyed by source
                    if source in annotations:
                        matched = annotations[source]
                        matched = matched[matched["sha256"].isin(source_sha256s)]
                    else:
                        print(f"  WARNING: Source '{source}' not found in annotations")
                        continue
                else:
                    print(f"  WARNING: Unexpected annotations type: {type(annotations)}")
                    continue

                print(f"  Matched {len(matched)} models in ObjaverseXL catalog")

                if len(matched) == 0:
                    continue

                # Download in batches
                matched_list = matched.to_dict("records") if hasattr(matched, "to_dict") else list(matched)
                for i in tqdm(range(0, len(matched_list), batch_size),
                              desc=f"  {source} batches"):
                    batch = matched_list[i:i + batch_size]
                    batch_df = pd.DataFrame(batch)

                    try:
                        paths = oxl.download_objects(
                            batch_df,
                            download_dir=str(models_dir),
                            save_repo_format="zip",
                            processes=processes,
                        )
                        # paths is dict: file_identifier → local_path
                        # Map back to sha256
                        for _, row in batch_df.iterrows():
                            fi = row.get("file_identifier", row.get("fileIdentifier", ""))
                            sha = row["sha256"]
                            if fi in paths and os.path.exists(paths[fi]):
                                downloaded[sha] = str(paths[fi])
                    except Exception as e:
                        print(f"  Batch {i // batch_size} failed: {e}")
                        continue

                    # Save progress periodically
                    if (i // batch_size + 1) % 10 == 0:
                        with open(progress_path, "w") as f:
                            json.dump(downloaded, f)

            else:
                # For ABO, 3D-FUTURE, HSSD — these are separate datasets
                # The TRELLIS dataset toolkit has custom loaders for these
                print(f"  {source}: Custom dataset — skipping ObjaverseXL download")
                print(f"  (These {len(source_sha256s)} models need manual download)")
                continue

        except Exception as e:
            print(f"  ERROR downloading {source}: {e}")
            import traceback
            traceback.print_exc()

        # Save progress after each source
        with open(progress_path, "w") as f:
            json.dump(downloaded, f)
        print(f"  Progress saved: {len(downloaded)} total downloaded")

    # Final save
    with open(progress_path, "w") as f:
        json.dump(downloaded, f)

    print(f"\nDownload complete: {len(downloaded)} / {len(target_sha256s)} models")
    return downloaded


def download_objaverse_v1_models(
    metadata: pd.DataFrame,
    output_dir: str,
    processes: int = 16,
    batch_size: int = 500,
) -> dict:
    """Alternative: Download from Objaverse v1 for Sketchfab models.

    This is simpler and more reliable than ObjaverseXL for the Sketchfab subset,
    since Objaverse v1 has the same models with simpler UID-based access.
    """
    import objaverse

    output_path = Path(output_dir)
    models_dir = output_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Load progress
    progress_path = output_path / "download_progress.json"
    downloaded = {}
    if progress_path.exists():
        with open(progress_path) as f:
            downloaded = json.load(f)
        valid = {k: v for k, v in downloaded.items() if os.path.exists(v)}
        stale = len(downloaded) - len(valid)
        if stale > 0:
            print(f"Removed {stale} stale progress entries")
            downloaded = valid

    print(f"\nAlready downloaded: {len(downloaded)}")

    # Get all v1 UIDs
    all_v1_uids = objaverse.load_uids()
    print(f"Objaverse v1 total UIDs: {len(all_v1_uids)}")

    # For Sketchfab models, the file_identifier in TRELLIS-500K corresponds
    # to the Objaverse v1 UID. Try matching by file_identifier.
    sketchfab_df = metadata[metadata["source"] == "sketchfab"]
    if len(sketchfab_df) > 0:
        # The file_identifier for Sketchfab models IS the Objaverse UID
        target_uids = list(sketchfab_df["file_identifier"].dropna().unique())
        # Filter to UIDs that exist in v1
        valid_uids = [u for u in target_uids if u in set(all_v1_uids)]
        # Filter out already downloaded
        remaining_uids = [u for u in valid_uids if u not in downloaded]

        print(f"Sketchfab: {len(target_uids)} target → {len(valid_uids)} in v1 → {len(remaining_uids)} remaining")

        if remaining_uids:
            for i in tqdm(range(0, len(remaining_uids), batch_size), desc="v1 download"):
                batch = remaining_uids[i:i + batch_size]
                try:
                    paths = objaverse.load_objects(batch, download_processes=processes)
                    for uid, path in paths.items():
                        if os.path.exists(path):
                            downloaded[uid] = str(path)
                except Exception as e:
                    print(f"Batch {i // batch_size} failed: {e}")
                    continue

                if (i // batch_size + 1) % 20 == 0:
                    with open(progress_path, "w") as f:
                        json.dump(downloaded, f)

    with open(progress_path, "w") as f:
        json.dump(downloaded, f)

    print(f"\nv1 download complete: {len(downloaded)} models")
    return downloaded


def _geometry_filter_one(args: tuple) -> dict | None:
    """Filter a single model (runs in subprocess for parallelism)."""
    sha256, path = args
    try:
        import trimesh
        mesh = trimesh.load(path, force="mesh")
    except Exception:
        return None

    n_faces = mesh.faces.shape[0]
    if n_faces < 500 or n_faces > 500_000:
        return None

    extents = mesh.extents
    if extents.min() < 1e-6:
        return None
    if extents.max() / extents.min() > 50:
        return None

    return {
        "uid": sha256,
        "path": str(path),
        "faces": int(n_faces),
        "vertices": int(mesh.vertices.shape[0]),
        "watertight": bool(mesh.is_watertight),
        "volume": bool(mesh.is_volume),
        "extents": extents.tolist(),
    }


def geometry_filter(
    downloaded: dict,
    output_dir: str,
    num_workers: int = 8,
) -> tuple[list, list]:
    """Run basic geometry filtering on downloaded models."""
    output_path = Path(output_dir)

    # Check for cached results
    valid_path = output_path / "valid_models.json"
    hq_path = output_path / "high_quality_models.json"

    if valid_path.exists():
        with open(valid_path) as f:
            existing_valid = json.load(f)
        existing_uids = {m["uid"] for m in existing_valid}
        new_models = {k: v for k, v in downloaded.items() if k not in existing_uids}
        if not new_models:
            print(f"All {len(existing_valid)} models already filtered")
            with open(hq_path) as f:
                existing_hq = json.load(f)
            return existing_valid, existing_hq
        print(f"Filtering {len(new_models)} new models (keeping {len(existing_valid)} cached)")
    else:
        existing_valid = []
        new_models = downloaded

    print(f"\nRunning geometry filter on {len(new_models)} models ({num_workers} workers)...")
    items = list(new_models.items())
    valid = list(existing_valid)
    high_quality = [m for m in valid if m.get("watertight") and m.get("faces", 0) >= 2000]

    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(_geometry_filter_one, item): item for item in items}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Geometry filter"):
            result = future.result()
            if result is not None:
                valid.append(result)
                if result["watertight"] and result["faces"] >= 2000:
                    high_quality.append(result)

    print(f"\nValid: {len(valid)} / {len(downloaded)}")
    print(f"High quality: {len(high_quality)} / {len(downloaded)}")

    with open(valid_path, "w") as f:
        json.dump(valid, f, indent=2)
    with open(hq_path, "w") as f:
        json.dump(high_quality, f, indent=2)

    return valid, high_quality


def main():
    parser = argparse.ArgumentParser(
        description="Download TRELLIS-500K dataset from ObjaverseXL"
    )
    parser.add_argument(
        "--output_dir", type=str, default="/workspace/data/trellis500k",
        help="Output directory (use /workspace for RunPod persistence)"
    )
    parser.add_argument(
        "--sources", type=str, nargs="+",
        default=["github", "sketchfab", "abo", "3d_future", "hssd"],
        choices=list(CSV_FILES.keys()),
        help="Which TRELLIS-500K sources to download"
    )
    parser.add_argument(
        "--use_v1", action="store_true",
        help="Use Objaverse v1 API for Sketchfab models (simpler, more reliable)"
    )
    parser.add_argument(
        "--processes", type=int, default=16,
        help="Parallel download workers"
    )
    parser.add_argument(
        "--batch_size", type=int, default=500,
        help="Download batch size"
    )
    parser.add_argument(
        "--filter", action="store_true",
        help="Run geometry filtering after download"
    )
    parser.add_argument(
        "--filter_workers", type=int, default=8,
        help="Workers for geometry filtering"
    )
    parser.add_argument(
        "--skip_download", action="store_true",
        help="Skip download, only run filtering on existing files"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of models per source (for testing)"
    )
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Setup persistent cache
    setup_persistent_cache(args.output_dir)

    # Step 1: Download metadata
    print("=" * 80)
    print("STEP 1: Download TRELLIS-500K Metadata")
    print("=" * 80)
    metadata = download_metadata(args.output_dir, args.sources)

    if args.limit:
        print(f"\nLimiting to {args.limit} models per source")
        limited = []
        for src in args.sources:
            src_df = metadata[metadata["source"] == src].head(args.limit)
            limited.append(src_df)
        metadata = pd.concat(limited, ignore_index=True)
        print(f"Total after limit: {len(metadata)}")

    # Save combined metadata
    metadata.to_csv(output_path / "trellis500k_combined.csv", index=False)

    if args.skip_download:
        print("\nSkipping download (--skip_download)")
        progress_path = output_path / "download_progress.json"
        if progress_path.exists():
            with open(progress_path) as f:
                downloaded = json.load(f)
            # Validate
            downloaded = {k: v for k, v in downloaded.items() if os.path.exists(v)}
        else:
            print("ERROR: No download_progress.json found")
            sys.exit(1)
    else:
        # Step 2: Download 3D models
        print("\n" + "=" * 80)
        print("STEP 2: Download 3D Models")
        print("=" * 80)

        if args.use_v1:
            downloaded = download_objaverse_v1_models(
                metadata, args.output_dir, args.processes, args.batch_size
            )
        else:
            downloaded = download_objaversexl_models(
                metadata, args.output_dir, args.processes, args.batch_size
            )

    # Step 3: Optional geometry filtering
    if args.filter:
        print("\n" + "=" * 80)
        print("STEP 3: Geometry Filtering")
        print("=" * 80)
        valid, high_quality = geometry_filter(
            downloaded, args.output_dir, args.filter_workers
        )
        print(f"\nFinal counts:")
        print(f"  Valid models:        {len(valid)}")
        print(f"  High-quality models: {len(high_quality)}")
        print(f"  valid_models.json:   {output_path / 'valid_models.json'}")
    else:
        # Create a simple valid_models.json from all downloaded files
        valid = [
            {"uid": sha256, "path": path}
            for sha256, path in downloaded.items()
            if os.path.exists(path)
        ]
        with open(output_path / "valid_models.json", "w") as f:
            json.dump(valid, f, indent=2)
        print(f"\nCreated valid_models.json with {len(valid)} models (unfiltered)")

    print(f"\n{'=' * 80}")
    print(f"DONE")
    print(f"{'=' * 80}")
    print(f"Output directory:  {args.output_dir}")
    print(f"Models downloaded: {len(downloaded)}")
    print(f"Valid models:      {len(valid)}")
    print(f"Next step: Run pair generation with:")
    print(f"  ./scripts/data/run_pairs_watchdog.sh 0 <num_shards>")
    print(f"  with INPUT_JSON={output_path / 'valid_models.json'}")


if __name__ == "__main__":
    main()
