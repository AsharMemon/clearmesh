#!/usr/bin/env python3
"""Prepare sharded candidate pools for Vast.ai pair generation.

Takes a candidates JSON (from build_candidate_pool.py) and splits it into
N shards for parallel pair generation across Vast.ai pods.

Sharding strategy:
  1. Shuffle with fixed seed (deterministic)
  2. Modulo interleave: model i goes to shard (i % num_shards)
     This ensures even distribution of quality scores across shards

Output:
  shards/shard_0.json ... shards/shard_{N-1}.json
  Each shard JSON is compatible with generate_pairs.py --input_json

Optionally creates tar archives of model files per shard for upload to B2.

Usage:
    # Split 50K candidates into 8 shards
    python prepare_vastai_shards.py \\
        --input /workspace/data/trellis500k/candidates_50k.json \\
        --output_dir /workspace/data/trellis500k/shards \\
        --num_shards 8

    # Also create tar archives for B2 upload
    python prepare_vastai_shards.py \\
        --input /workspace/data/trellis500k/candidates_50k.json \\
        --output_dir /workspace/data/trellis500k/shards \\
        --num_shards 8 \\
        --create_tars
"""

import argparse
import json
import os
import random
import subprocess
import sys
import tarfile
from pathlib import Path

from tqdm import tqdm


def load_candidates(path: str) -> list:
    """Load candidates JSON."""
    with open(path) as f:
        candidates = json.load(f)
    print(f"Loaded {len(candidates):,} candidates from {path}")
    return candidates


def validate_paths(candidates: list) -> tuple:
    """Check which candidates have valid local paths."""
    with_path = [c for c in candidates if c.get("path") and os.path.exists(c["path"])]
    without_path = [c for c in candidates if not c.get("path") or not os.path.exists(c.get("path", ""))]

    print(f"  With valid local path: {len(with_path):,}")
    print(f"  Without valid path:    {len(without_path):,}")
    return with_path, without_path


def shard_candidates(candidates: list, num_shards: int, seed: int = 42) -> list:
    """Split candidates into N shards with modulo interleaving.

    Returns list of N lists, where shard i contains candidates[j]
    for all j where j % num_shards == i (after shuffling).
    """
    # Shuffle deterministically
    shuffled = list(candidates)
    random.seed(seed)
    random.shuffle(shuffled)

    # Modulo interleave
    shards = [[] for _ in range(num_shards)]
    for i, candidate in enumerate(shuffled):
        shard_id = i % num_shards
        shards[shard_id].append(candidate)

    for i, shard in enumerate(shards):
        print(f"  Shard {i}: {len(shard):,} models")

    return shards


def convert_to_pairgen_format(candidates: list) -> list:
    """Convert candidate entries to generate_pairs.py input format.

    generate_pairs.py expects: [{uid, path, ...}]
    where uid is used as the output directory name.
    """
    entries = []
    for c in candidates:
        entry = {
            "uid": c["sha256"],  # Use SHA256 as UID for pair generation
            "path": c.get("path", ""),
        }

        # Pass through metadata for potential future use
        if c.get("aesthetic_score") is not None:
            entry["aesthetic_score"] = c["aesthetic_score"]
        if c.get("quality_score") is not None and c["quality_score"] >= 0:
            entry["quality_score"] = c["quality_score"]

        entries.append(entry)
    return entries


def create_shard_tar(
    shard: list,
    shard_id: int,
    output_dir: str,
    tar_name: str = None,
) -> str:
    """Create a tar archive of model files for a shard.

    The tar preserves the relative path structure so models can be
    extracted to the same relative locations on Vast.ai pods.
    """
    if tar_name is None:
        tar_name = f"shard_{shard_id}_models.tar"

    tar_path = os.path.join(output_dir, tar_name)

    # Find common prefix for relative paths
    paths = [c["path"] for c in shard if c.get("path") and os.path.exists(c.get("path", ""))]
    if not paths:
        print(f"  Shard {shard_id}: No valid paths to archive")
        return ""

    # Use /workspace as the base for relative paths
    base = "/workspace"

    print(f"  Creating {tar_name} ({len(paths):,} files)...")
    with tarfile.open(tar_path, "w") as tar:
        for path in tqdm(paths, desc=f"  Shard {shard_id}", leave=False):
            if os.path.exists(path):
                # Store with relative path from /workspace
                arcname = os.path.relpath(path, base)
                tar.add(path, arcname=arcname)

    size_gb = os.path.getsize(tar_path) / (1024**3)
    print(f"  {tar_name}: {size_gb:.1f} GB ({len(paths):,} files)")
    return tar_path


def main():
    parser = argparse.ArgumentParser(
        description="Split candidates into shards for Vast.ai pair generation"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Input candidates JSON (from build_candidate_pool.py)")
    parser.add_argument("--output_dir", type=str,
                        default="/workspace/data/trellis500k/shards",
                        help="Output directory for shard JSONs")
    parser.add_argument("--num_shards", type=int, default=8,
                        help="Number of shards (one per Vast.ai pod)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling")
    parser.add_argument("--create_tars", action="store_true",
                        help="Create tar archives of model files per shard")
    parser.add_argument("--require_paths", action="store_true",
                        help="Only include models with valid local paths")

    args = parser.parse_args()

    print("=" * 70)
    print("ClearMesh Shard Preparation")
    print("=" * 70)

    # Load candidates
    candidates = load_candidates(args.input)

    # Validate paths
    with_paths, without_paths = validate_paths(candidates)

    if args.require_paths:
        if not with_paths:
            print("ERROR: No candidates have valid local paths!")
            sys.exit(1)
        candidates = with_paths
        print(f"  Using {len(candidates):,} candidates with valid paths")

    # Shard
    print(f"\nSharding into {args.num_shards} shards (seed={args.seed})...")
    shards = shard_candidates(candidates, args.num_shards, args.seed)

    # Write shard JSONs
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nWriting shard JSONs to {args.output_dir}/...")

    for i, shard in enumerate(shards):
        # Convert to pairgen format
        pairgen_entries = convert_to_pairgen_format(shard)

        shard_path = os.path.join(args.output_dir, f"shard_{i}.json")
        with open(shard_path, "w") as f:
            json.dump(pairgen_entries, f, indent=2)
        print(f"  shard_{i}.json: {len(pairgen_entries):,} entries")

    # Create tar archives if requested
    if args.create_tars:
        print(f"\nCreating tar archives...")
        for i, shard in enumerate(shards):
            create_shard_tar(shard, i, args.output_dir)

    # Write metadata
    meta = {
        "total_candidates": len(candidates),
        "num_shards": args.num_shards,
        "seed": args.seed,
        "shard_sizes": [len(s) for s in shards],
        "with_paths": len(with_paths),
        "without_paths": len(without_paths),
        "source_file": os.path.abspath(args.input),
    }
    meta_path = os.path.join(args.output_dir, "sharding_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"Sharding complete!")
    print(f"  Shards: {args.num_shards}")
    print(f"  Output: {args.output_dir}")
    print(f"  Meta:   {meta_path}")

    if not args.require_paths and without_paths:
        print(f"\n  NOTE: {len(without_paths):,} candidates don't have local paths yet.")
        print(f"  These models need to be downloaded before pair generation.")
        print(f"  Run with --require_paths to exclude them.")

    print(f"\n  Next steps:")
    print(f"  1. Upload shards to B2: rclone copy {args.output_dir}/ b2:clearmesh-pairs/shards/")
    print(f"  2. Upload weights:      rclone copy /workspace/models/ b2:clearmesh-pairs/models/")
    print(f"  3. Launch Vast.ai pods with shard IDs 0..{args.num_shards - 1}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
