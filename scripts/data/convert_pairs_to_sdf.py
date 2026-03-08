#!/usr/bin/env python3
"""Compute ground-truth SDF from fine meshes at saved voxel positions.

This script bridges pair generation (Phase 2) and Stage 2 training (Phase 3).
It takes the output of generate_pairs.py (coarse/fine mesh pairs with saved
SLAT positions) and computes signed distance values from the fine mesh at
each voxel position.

Output per pair:
    {uid}/fine_sdf.npy   (N, 1) float32 — signed distance at each voxel position

Prerequisites:
    - generate_pairs.py must have been run with return_intermediates=True
    - Each pair directory must contain positions.npy and a fine mesh

Usage:
    # Process all pairs across shards:
    python scripts/data/convert_pairs_to_sdf.py \
        --pairs_dir /workspace/data/training_pairs

    # With parallelism:
    python scripts/data/convert_pairs_to_sdf.py \
        --pairs_dir /workspace/data/training_pairs \
        --num_workers 8

    # Force recompute:
    python scripts/data/convert_pairs_to_sdf.py \
        --pairs_dir /workspace/data/training_pairs --force
"""

import argparse
import json
import os
import sys
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import trimesh
from tqdm import tqdm


def find_fine_mesh(pair_dir: Path) -> Path | None:
    """Find the fine mesh file in a pair directory."""
    for ext in [".glb", ".obj", ".ply", ".stl", ".off", ".gltf"]:
        candidate = pair_dir / f"fine{ext}"
        if candidate.exists():
            # Resolve symlinks
            return candidate.resolve()
    return None


def load_and_normalize_mesh(mesh_path: Path) -> trimesh.Trimesh | None:
    """Load a mesh and normalize to [-0.5, 0.5] centered at origin."""
    try:
        mesh = trimesh.load(str(mesh_path), force="mesh")
        if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
            return None

        # Center at origin
        mesh.vertices -= mesh.centroid

        # Scale to [-0.5, 0.5]
        scale = mesh.extents.max()
        if scale > 0:
            mesh.vertices /= scale

        return mesh
    except Exception:
        return None


def compute_sdf_for_pair(args: tuple) -> tuple[str, bool, str]:
    """Compute ground-truth SDF for a single pair.

    Args:
        args: Tuple of (pair_dir_str, resolution, force).

    Returns:
        Tuple of (uid, success, error_message).
    """
    pair_dir_str, resolution, force = args
    pair_dir = Path(pair_dir_str)
    uid = pair_dir.name
    output_file = pair_dir / "fine_sdf.npy"

    # Idempotency: skip if already done
    if output_file.exists() and not force:
        return (uid, True, "skipped (exists)")

    # Load positions (from generate_pairs.py intermediates)
    positions_file = pair_dir / "positions.npy"
    if not positions_file.exists():
        return (uid, False, "positions.npy missing")

    try:
        positions = np.load(str(positions_file))  # (N, 3) int32
    except Exception as e:
        return (uid, False, f"positions.npy load error: {e}")

    if positions.ndim != 2 or positions.shape[1] != 3:
        return (uid, False, f"positions.npy bad shape: {positions.shape}")

    # Find fine mesh
    fine_path = find_fine_mesh(pair_dir)
    if fine_path is None:
        return (uid, False, "fine mesh not found")

    # Load and normalize
    mesh = load_and_normalize_mesh(fine_path)
    if mesh is None:
        return (uid, False, "fine mesh load/normalize failed")

    # Read resolution from meta.json if available
    meta_file = pair_dir / "meta.json"
    actual_res = resolution
    if meta_file.exists():
        try:
            with open(meta_file) as f:
                meta = json.load(f)
            if meta.get("ss_res") is not None:
                actual_res = int(meta["ss_res"])
        except Exception:
            pass

    try:
        # Convert integer voxel coords to world coords in [-0.5, 0.5]
        # TRELLIS.2 sparse structure coords are in [0, ss_res-1]
        world_positions = positions.astype(np.float64) / max(actual_res - 1, 1) - 0.5

        # Compute signed distance
        # trimesh uses winding number method — positive inside, negative outside
        sdf_values = trimesh.proximity.signed_distance(mesh, world_positions)

        # Save as (N, 1) float32
        sdf_out = sdf_values.astype(np.float32).reshape(-1, 1)
        np.save(str(output_file), sdf_out)

        return (uid, True, f"ok N={len(sdf_values)}")

    except Exception as e:
        return (uid, False, f"SDF computation error: {e}")


def discover_pair_dirs(pairs_dir: Path) -> list[Path]:
    """Find all pair directories that have positions.npy."""
    pair_dirs = []

    # Check for shard subdirectories
    shard_dirs = sorted(pairs_dir.glob("shard_*"))
    search_dirs = shard_dirs if shard_dirs else [pairs_dir]

    for parent in search_dirs:
        if not parent.is_dir():
            continue
        for d in sorted(parent.iterdir()):
            if not d.is_dir():
                continue
            if (d / "positions.npy").exists():
                pair_dirs.append(d)

    return pair_dirs


def main():
    parser = argparse.ArgumentParser(
        description="Compute ground-truth SDF from fine meshes at saved voxel positions"
    )
    parser.add_argument(
        "--pairs_dir",
        type=str,
        required=True,
        help="Root directory containing pair data (with shard_* subdirs)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=32,
        help="Sparse structure resolution (default: 32, matching '512' pipeline). "
             "Overridden per-pair by meta.json ss_res if available.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even if fine_sdf.npy already exists",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N pairs (for testing)",
    )
    args = parser.parse_args()

    pairs_dir = Path(args.pairs_dir)
    if not pairs_dir.exists():
        print(f"Error: {pairs_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    # Discover pairs
    pair_dirs = discover_pair_dirs(pairs_dir)
    if args.limit:
        pair_dirs = pair_dirs[: args.limit]

    if not pair_dirs:
        print("No pair directories with positions.npy found.")
        print("Make sure generate_pairs.py was run with return_intermediates=True.")
        sys.exit(0)

    # Count already done
    already_done = sum(1 for d in pair_dirs if (d / "fine_sdf.npy").exists())
    need_compute = len(pair_dirs) - already_done if not args.force else len(pair_dirs)

    print(f"SDF Conversion")
    print(f"  Pairs found:    {len(pair_dirs)}")
    print(f"  Already done:   {already_done}")
    print(f"  To compute:     {need_compute}")
    print(f"  Resolution:     {args.resolution} (overridden per-pair by meta.json)")
    print(f"  Workers:        {args.num_workers}")
    print(f"  Force:          {args.force}")
    print()

    if need_compute == 0 and not args.force:
        print("Nothing to do.")
        return

    # Build work items
    work_items = [
        (str(d), args.resolution, args.force)
        for d in pair_dirs
    ]

    success = 0
    skipped = 0
    failed = 0
    errors = []

    if args.num_workers <= 1:
        # Sequential
        for item in tqdm(work_items, desc="Computing SDF"):
            uid, ok, msg = compute_sdf_for_pair(item)
            if ok:
                if "skipped" in msg:
                    skipped += 1
                else:
                    success += 1
            else:
                failed += 1
                errors.append((uid, msg))
    else:
        # Parallel
        with Pool(args.num_workers, maxtasksperchild=50) as pool:
            results = pool.imap_unordered(compute_sdf_for_pair, work_items, chunksize=10)
            for uid, ok, msg in tqdm(results, total=len(work_items), desc="Computing SDF"):
                if ok:
                    if "skipped" in msg:
                        skipped += 1
                    else:
                        success += 1
                else:
                    failed += 1
                    errors.append((uid, msg))

    print(f"\nSDF Conversion Complete")
    print(f"  Computed:  {success}")
    print(f"  Skipped:   {skipped}")
    print(f"  Failed:    {failed}")

    if errors:
        print(f"\nFirst 10 errors:")
        for uid, msg in errors[:10]:
            print(f"  {uid}: {msg}")

    # Save error log
    if errors:
        error_log = pairs_dir / "sdf_conversion_errors.json"
        with open(error_log, "w") as f:
            json.dump({uid: msg for uid, msg in errors}, f, indent=2)
        print(f"\nFull error log: {error_log}")


if __name__ == "__main__":
    main()
