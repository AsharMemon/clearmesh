#!/usr/bin/env python3
"""Build a quality-filtered candidate pool from TRELLIS-500K for pair generation.

Implements Layers 0-2 of the ClearMesh data quality filter pipeline:
  Layer 0: Source aggregation + geometry fingerprint dedup (simplified for pilot)
  Layer 1: Metadata pre-filter (vertex/face counts, aspect ratio, degenerates)
  Layer 2: Objaverse++ quality scores (High + Superior only)

For models without Objaverse++ scores:
  - Include if in Step1X-3D curated list
  - Include if TRELLIS aesthetic_score > 6.0

Inputs:
  - TRELLIS-500K metadata CSVs (ObjaverseXL_sketchfab.csv, ObjaverseXL_github.csv)
  - Objaverse++ annotations (annotated_800k.json from HuggingFace)
  - Step1X-3D UIDs (objaverse_320k.json from HuggingFace)
  - Download progress (download_progress.json from download_trellis500k_models.py)

Output:
  - candidates_{N}k.json: List of {uid, sha256, path, quality_score, aesthetic_score, source, ...}

Usage:
    # Build 50K candidate pool (for pilot)
    python build_candidate_pool.py --target 50000

    # Build full candidate pool (no limit)
    python build_candidate_pool.py --target 0

    # Use custom paths
    python build_candidate_pool.py \\
        --meta_dir /workspace/data/trellis500k/metadata \\
        --annotations /tmp/annotated_800k.json \\
        --step1x_uids /tmp/objaverse_320k.json \\
        --progress /workspace/data/trellis500k/download_progress.json \\
        --output /workspace/data/trellis500k/candidates_50k.json \\
        --target 50000
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# UID extraction
# ---------------------------------------------------------------------------

def extract_objaverse_uid(file_identifier: str) -> str:
    """Extract Objaverse v1 UID from a Sketchfab file_identifier URL.

    Sketchfab URLs look like:
      https://sketchfab.com/3d-models/18e8e405446849afb22b3760d3c73a31
      https://sketchfab.com/3d-models/some-title-18e8e405446849afb22b3760d3c73a31

    The UID is the last 32 hex chars in the URL path.
    """
    if not isinstance(file_identifier, str):
        return ""
    # Match 32 hex chars at the end of the URL path
    match = re.search(r'([0-9a-f]{32})(?:\?|$|#)', file_identifier)
    if match:
        return match.group(1)
    # Fallback: try the last path component
    parts = file_identifier.rstrip("/").split("/")
    if parts:
        last = parts[-1]
        # UID might be at the end of a slug like "title-uid"
        if len(last) >= 32:
            candidate = last[-32:]
            if re.match(r'^[0-9a-f]{32}$', candidate):
                return candidate
    return ""


def extract_step1x_uid(path: str) -> str:
    """Extract UID from Step1X-3D path like '000-130/f7654b3f336249d8bb6ae12503e6b543.glb'."""
    if not isinstance(path, str):
        return ""
    basename = os.path.splitext(os.path.basename(path))[0]
    if re.match(r'^[0-9a-f]{32}$', basename):
        return basename
    return ""


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_trellis_metadata(meta_dir: str) -> pd.DataFrame:
    """Load TRELLIS-500K metadata CSVs."""
    meta_path = Path(meta_dir)
    dfs = []

    for name, csv_name in [("sketchfab", "ObjaverseXL_sketchfab.csv"),
                            ("github", "ObjaverseXL_github.csv")]:
        path = meta_path / csv_name
        if not path.exists():
            print(f"  WARNING: {path} not found — skipping {name}")
            continue
        df = pd.read_csv(path)
        df["trellis_source"] = name
        dfs.append(df)
        print(f"  {name}: {len(df):,} entries")

    if not dfs:
        print("ERROR: No metadata CSVs found.")
        sys.exit(1)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"  Total: {len(combined):,} entries")
    return combined


def load_objaversepp_annotations(path: str) -> dict:
    """Load Objaverse++ annotations: UID → {score, style, ...}.

    Supports:
      - JSON file (annotated_800k.json): list of dicts
      - Parquet file: DataFrame with UID, score, style, ... columns
      - HuggingFace dataset (path='hf://cindyxl/ObjaversePlusPlus')
    """
    print(f"  Loading Objaverse++ annotations from {path}...")

    if path.startswith("hf://"):
        # Load from HuggingFace datasets
        from datasets import load_dataset
        ds = load_dataset(path.replace("hf://", ""), split="train")
        ann = {}
        for row in ds:
            uid = row.get("UID", "")
            if uid:
                ann[uid] = dict(row)
        print(f"  Loaded {len(ann):,} annotations from HuggingFace")
        return ann

    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
        ann = {}
        for _, row in df.iterrows():
            uid = str(row.get("UID", ""))
            if uid:
                ann[uid] = row.to_dict()
        print(f"  Loaded {len(ann):,} annotations from parquet")
        return ann

    # JSON format
    with open(path) as f:
        data = json.load(f)

    # Convert list of dicts to UID-keyed dict
    ann = {}
    if isinstance(data, list):
        for entry in data:
            uid = entry.get("UID", "")
            if uid:
                ann[uid] = entry
    elif isinstance(data, dict):
        # Could be {uid: {score, ...}} or {uid: score}
        for uid, val in data.items():
            if isinstance(val, dict):
                ann[uid] = val
            else:
                ann[uid] = {"score": val}
    print(f"  Loaded {len(ann):,} annotations")
    return ann


def load_step1x_uids(path: str) -> set:
    """Load Step1X-3D curated UIDs."""
    print(f"  Loading Step1X-3D UIDs from {path}...")
    with open(path) as f:
        data = json.load(f)

    uids = set()
    for item in data:
        uid = extract_step1x_uid(item)
        if uid:
            uids.add(uid)
    print(f"  Loaded {len(uids):,} UIDs")
    return uids


def load_download_progress(path: str) -> dict:
    """Load download progress: sha256 → local_path."""
    if not os.path.exists(path):
        print(f"  WARNING: No download progress at {path}")
        return {}
    with open(path) as f:
        progress = json.load(f)
    print(f"  Download progress: {len(progress):,} models")
    return progress


# ---------------------------------------------------------------------------
# Layer 1: Metadata pre-filter
# ---------------------------------------------------------------------------

def layer1_metadata_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Layer 1: Filter by basic metadata.

    For the pilot, this is lightweight since TRELLIS-500K metadata
    doesn't include vertex/face counts directly. We filter on:
    - Must have valid sha256
    - Must have valid file_identifier
    - Aesthetic score not NaN (if available)

    Full Layer 1 (for Phase B) will add:
    - Vertex count > 100, < 5M
    - Face count > 200, < 10M
    - Aspect ratio < 100
    - Degenerate face ratio < 5%
    """
    before = len(df)

    # Must have sha256
    df = df[df["sha256"].notna() & (df["sha256"] != "")].copy()

    # Must have file_identifier
    df = df[df["file_identifier"].notna() & (df["file_identifier"] != "")].copy()

    # Remove exact sha256 duplicates (keep first occurrence)
    df = df.drop_duplicates(subset="sha256", keep="first")

    after = len(df)
    print(f"  Layer 1: {before:,} → {after:,} ({before - after:,} removed)")
    return df


# ---------------------------------------------------------------------------
# Layer 2: Quality scoring
# ---------------------------------------------------------------------------

def layer2_quality_filter(
    df: pd.DataFrame,
    objaversepp: dict,
    step1x_uids: set,
    min_quality_score: int = 2,
    fallback_aesthetic_threshold: float = 6.0,
) -> pd.DataFrame:
    """Layer 2: Filter by Objaverse++ quality scores.

    Score mapping: 0=Low, 1=Medium, 2=High, 3=Superior
    Keep: score >= min_quality_score (default: High + Superior)

    For models without Objaverse++ scores:
    - Include if UID is in Step1X-3D curated list
    - Include if TRELLIS aesthetic_score > fallback_aesthetic_threshold
    """
    before = len(df)

    # Extract Objaverse UIDs for Sketchfab models
    sf_mask = df["trellis_source"] == "sketchfab"
    df.loc[sf_mask, "objaverse_uid"] = df.loc[sf_mask, "file_identifier"].apply(
        extract_objaverse_uid
    )
    df.loc[~sf_mask, "objaverse_uid"] = ""

    # Look up Objaverse++ scores
    def get_quality_score(uid):
        if not uid or uid not in objaversepp:
            return -1  # No score available
        return objaversepp[uid].get("score", -1)

    df["quality_score"] = df["objaverse_uid"].apply(get_quality_score)

    # Look up style info
    def get_style(uid):
        if not uid or uid not in objaversepp:
            return "unknown"
        return objaversepp[uid].get("style", "unknown")

    df["style"] = df["objaverse_uid"].apply(get_style)

    # Check Step1X-3D membership
    df["in_step1x"] = df["objaverse_uid"].apply(lambda uid: uid in step1x_uids if uid else False)

    # Determine inclusion
    # 1. Has Objaverse++ score >= threshold
    has_score = df["quality_score"] >= 0
    passes_quality = df["quality_score"] >= min_quality_score

    # 2. No score but in Step1X-3D
    no_score = ~has_score
    in_step1x = df["in_step1x"]

    # 3. No score, not in Step1X, but high aesthetic score
    has_aesthetic = df["aesthetic_score"].notna()
    high_aesthetic = df["aesthetic_score"] >= fallback_aesthetic_threshold

    keep_mask = (
        passes_quality |
        (no_score & in_step1x) |
        (no_score & ~in_step1x & has_aesthetic & high_aesthetic)
    )

    df_filtered = df[keep_mask].copy()

    # Stats
    kept_by_quality = passes_quality.sum()
    kept_by_step1x = (no_score & in_step1x & ~passes_quality).sum()
    kept_by_aesthetic = (no_score & ~in_step1x & has_aesthetic & high_aesthetic & ~passes_quality).sum()
    rejected = before - len(df_filtered)

    print(f"  Layer 2: {before:,} → {len(df_filtered):,} ({rejected:,} removed)")
    print(f"    Kept by Objaverse++ quality ≥ {min_quality_score}: {kept_by_quality:,}")
    print(f"    Kept by Step1X-3D membership: {kept_by_step1x:,}")
    print(f"    Kept by aesthetic score > {fallback_aesthetic_threshold}: {kept_by_aesthetic:,}")

    # Quality score distribution
    score_dist = Counter(df_filtered["quality_score"].values)
    score_labels = {-1: "No score", 0: "Low", 1: "Medium", 2: "High", 3: "Superior"}
    print(f"    Quality distribution:")
    for score in sorted(score_dist.keys()):
        label = score_labels.get(score, f"Score {score}")
        print(f"      {label}: {score_dist[score]:,}")

    return df_filtered


# ---------------------------------------------------------------------------
# Ranking and selection
# ---------------------------------------------------------------------------

def rank_and_select(
    df: pd.DataFrame,
    download_progress: dict,
    target: int = 50000,
    require_downloaded: bool = False,
) -> list:
    """Rank models by quality and select top N.

    Ranking: quality_score (primary) × aesthetic_score (secondary)
    If require_downloaded=True, only include models we've already downloaded.
    """
    # Compute composite score for ranking
    # quality_score: -1 to 3 → normalize to 0-1 range
    # aesthetic_score: typically 3-9 → normalize to 0-1 range
    df = df.copy()

    # Normalize quality score: -1 → 0.5 (unknown), 0→0.25, 1→0.5, 2→0.75, 3→1.0
    quality_map = {-1: 0.5, 0: 0.25, 1: 0.5, 2: 0.75, 3: 1.0}
    df["norm_quality"] = df["quality_score"].map(quality_map).fillna(0.5)

    # Normalize aesthetic score to 0-1
    aes = df["aesthetic_score"]
    if aes.notna().any():
        aes_min, aes_max = aes.min(), aes.max()
        if aes_max > aes_min:
            df["norm_aesthetic"] = (aes - aes_min) / (aes_max - aes_min)
        else:
            df["norm_aesthetic"] = 0.5
    else:
        df["norm_aesthetic"] = 0.5

    df["norm_aesthetic"] = df["norm_aesthetic"].fillna(0.5)

    # Composite: 60% quality, 40% aesthetic
    df["composite_score"] = 0.6 * df["norm_quality"] + 0.4 * df["norm_aesthetic"]

    # Check download status
    df["is_downloaded"] = df["sha256"].apply(lambda sha: sha in download_progress)
    df["local_path"] = df["sha256"].apply(lambda sha: download_progress.get(sha, ""))

    downloaded_count = df["is_downloaded"].sum()
    print(f"\n  Downloaded models in filtered set: {downloaded_count:,} / {len(df):,}")

    if require_downloaded:
        df = df[df["is_downloaded"]].copy()
        print(f"  After requiring downloaded: {len(df):,}")

    # Sort by composite score (descending)
    df = df.sort_values("composite_score", ascending=False)

    # Select top N
    if target > 0:
        df = df.head(target)
        print(f"  Selected top {target:,}: {len(df):,}")
    else:
        print(f"  No target limit: keeping all {len(df):,}")

    # Build output list
    candidates = []
    for _, row in df.iterrows():
        entry = {
            "sha256": str(row["sha256"]),
            "file_identifier": str(row["file_identifier"]),
            "trellis_source": str(row["trellis_source"]),
            "quality_score": int(row["quality_score"]),
            "aesthetic_score": float(row["aesthetic_score"]) if pd.notna(row.get("aesthetic_score")) else None,
            "composite_score": float(row["composite_score"]),
            "style": str(row.get("style", "unknown")),
            "in_step1x": bool(row.get("in_step1x", False)),
        }

        # Add objaverse UID if available
        uid = row.get("objaverse_uid", "")
        if uid:
            entry["objaverse_uid"] = str(uid)

        # Add download path if available
        if row.get("is_downloaded"):
            entry["path"] = str(row["local_path"])

        candidates.append(entry)

    return candidates


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build quality-filtered candidate pool for pair generation"
    )
    parser.add_argument("--meta_dir", type=str,
                        default="/workspace/data/trellis500k/metadata",
                        help="Directory with TRELLIS-500K CSV files")
    parser.add_argument("--annotations", type=str,
                        default="/tmp/annotated_800k.json",
                        help="Path to Objaverse++ annotated_800k.json")
    parser.add_argument("--step1x_uids", type=str,
                        default="/tmp/objaverse_320k.json",
                        help="Path to Step1X-3D objaverse_320k.json")
    parser.add_argument("--progress", type=str,
                        default="/workspace/data/trellis500k/download_progress.json",
                        help="Path to download_progress.json")
    parser.add_argument("--output", type=str,
                        default="/workspace/data/trellis500k/candidates_50k.json",
                        help="Output candidates JSON path")
    parser.add_argument("--target", type=int, default=50000,
                        help="Target number of candidates (0 = no limit)")
    parser.add_argument("--min_quality", type=int, default=2,
                        help="Minimum Objaverse++ quality score (0-3, default=2=High)")
    parser.add_argument("--aesthetic_threshold", type=float, default=6.0,
                        help="Fallback aesthetic score threshold for unscored models")
    parser.add_argument("--require_downloaded", action="store_true",
                        help="Only include models that are already downloaded")
    parser.add_argument("--sources", nargs="*",
                        choices=["sketchfab", "github", "all"],
                        default=["sketchfab"],
                        help="Which TRELLIS-500K sources to include (default: sketchfab only for pilot)")
    parser.add_argument("--stats_only", action="store_true",
                        help="Only show statistics, don't write output")

    args = parser.parse_args()

    print("=" * 70)
    print("ClearMesh Candidate Pool Builder")
    print("=" * 70)

    # ---- Load data ----
    print("\n[1/5] Loading TRELLIS-500K metadata...")
    metadata = load_trellis_metadata(args.meta_dir)

    # Filter by source
    if "all" not in args.sources:
        metadata = metadata[metadata["trellis_source"].isin(args.sources)].copy()
        print(f"  Filtered to sources {args.sources}: {len(metadata):,}")

    print("\n[2/5] Loading quality annotations...")
    objaversepp = load_objaversepp_annotations(args.annotations)
    step1x_uids = load_step1x_uids(args.step1x_uids)

    print("\n[3/5] Loading download progress...")
    download_progress = load_download_progress(args.progress)

    # ---- Apply filters ----
    print("\n[4/5] Applying filters...")
    print("\n  --- Layer 1: Metadata pre-filter ---")
    filtered = layer1_metadata_filter(metadata)

    print("\n  --- Layer 2: Quality filter ---")
    filtered = layer2_quality_filter(
        filtered,
        objaversepp,
        step1x_uids,
        min_quality_score=args.min_quality,
        fallback_aesthetic_threshold=args.aesthetic_threshold,
    )

    if args.stats_only:
        print("\n  Stats-only mode — not writing output.")
        return

    # ---- Rank and select ----
    print("\n[5/5] Ranking and selecting candidates...")
    candidates = rank_and_select(
        filtered,
        download_progress,
        target=args.target,
        require_downloaded=args.require_downloaded,
    )

    # ---- Write output ----
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(candidates, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"Candidate pool: {len(candidates):,} models")
    print(f"Output: {args.output}")

    # Summary stats
    downloaded = sum(1 for c in candidates if "path" in c)
    with_quality = sum(1 for c in candidates if c["quality_score"] >= 0)
    in_step1x = sum(1 for c in candidates if c["in_step1x"])
    superior = sum(1 for c in candidates if c["quality_score"] == 3)
    high = sum(1 for c in candidates if c["quality_score"] == 2)

    scores = [c["aesthetic_score"] for c in candidates if c["aesthetic_score"] is not None]

    print(f"\n  Summary:")
    print(f"    Downloaded:       {downloaded:,} / {len(candidates):,}")
    print(f"    With quality:     {with_quality:,}")
    print(f"    Superior (3):     {superior:,}")
    print(f"    High (2):         {high:,}")
    print(f"    In Step1X-3D:     {in_step1x:,}")
    if scores:
        import statistics
        print(f"    Aesthetic scores: min={min(scores):.2f}, max={max(scores):.2f}, "
              f"mean={statistics.mean(scores):.2f}, median={statistics.median(scores):.2f}")

    # Source breakdown
    source_counts = Counter(c["trellis_source"] for c in candidates)
    print(f"\n  Source breakdown:")
    for source, count in source_counts.most_common():
        print(f"    {source}: {count:,}")

    # Style breakdown
    style_counts = Counter(c["style"] for c in candidates)
    print(f"\n  Style breakdown:")
    for style, count in style_counts.most_common(10):
        print(f"    {style}: {count:,}")

    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    main()
