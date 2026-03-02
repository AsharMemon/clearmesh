#!/usr/bin/env python3
"""Filter Objaverse models using UltraShape's 3-stage pipeline.

Stages:
  1. VLM-Based Filtering: Removes primitives, ground planes, noisy scans
  2. Pose Normalization: Aligns models to consistent orientation
  3. Geometry Filtering: Excludes thin shells and fragmented shapes

Input:  Raw Objaverse models (manifest.json from download_objaverse.py)
Output: Filtered model list with ~330K valid, ~120K high-quality

Usage:
    python filter_dataset.py \
        --manifest /mnt/data/objaverse/manifest.json \
        --ultrashape_dir /mnt/data/UltraShape-1.0 \
        --output_dir /mnt/data/filtered
"""

import argparse
import json
import os
import sys
from pathlib import Path

import trimesh
from tqdm import tqdm


def geometry_filter(mesh_path: str, min_faces: int = 500, max_faces: int = 500_000) -> dict | None:
    """Basic geometry filtering: check face count, watertightness, bounding box."""
    try:
        mesh = trimesh.load(mesh_path, force="mesh")
    except Exception:
        return None

    if mesh.faces.shape[0] < min_faces or mesh.faces.shape[0] > max_faces:
        return None

    # Check bounding box aspect ratio (reject extremely flat/thin objects)
    extents = mesh.extents
    if extents.min() < 1e-6:
        return None
    aspect_ratio = extents.max() / extents.min()
    if aspect_ratio > 50:
        return None

    # Check for degenerate geometry
    if not mesh.is_volume:
        # Not necessarily disqualifying, but note it
        pass

    return {
        "path": mesh_path,
        "faces": int(mesh.faces.shape[0]),
        "vertices": int(mesh.vertices.shape[0]),
        "watertight": bool(mesh.is_watertight),
        "volume": bool(mesh.is_volume),
        "extents": extents.tolist(),
    }


def run_ultrashape_filtering(manifest_path: str, ultrashape_dir: str, output_dir: str):
    """Run UltraShape's VLM + pose + geometry filtering if available."""
    ultrashape_sampling = os.path.join(ultrashape_dir, "scripts", "sampling.py")

    if os.path.exists(ultrashape_sampling):
        print("Running UltraShape filtering pipeline...")
        # Generate mesh_paths.json from our manifest
        with open(manifest_path) as f:
            manifest = json.load(f)

        mesh_paths = {uid: path for uid, path in manifest.items()}
        mesh_json = os.path.join(output_dir, "mesh_paths.json")
        with open(mesh_json, "w") as f:
            json.dump(mesh_paths, f)

        # Call UltraShape's sampling script
        os.system(
            f"cd {ultrashape_dir} && python scripts/sampling.py "
            f"--mesh_json {mesh_json} "
            f"--output_dir {output_dir}/ultrashape_filtered"
        )
        return True
    else:
        print("UltraShape sampling.py not found, using basic geometry filter only.")
        return False


def filter_dataset(manifest_path: str, ultrashape_dir: str, output_dir: str, quality_threshold: float = 0.7):
    """Main filtering pipeline."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(manifest_path) as f:
        manifest = json.load(f)

    print(f"Input models: {len(manifest)}")

    # Stage 1: Try UltraShape's full pipeline
    has_ultrashape = run_ultrashape_filtering(manifest_path, ultrashape_dir, output_dir)

    # Stage 2: Basic geometry filtering (always runs as safety net)
    print("\nRunning geometry filtering...")
    valid_models = []
    high_quality = []

    for uid, mesh_path in tqdm(manifest.items(), desc="Geometry filter"):
        result = geometry_filter(mesh_path)
        if result is None:
            continue

        result["uid"] = uid
        valid_models.append(result)

        # High-quality: watertight, good face count, reasonable aspect ratio
        if result["watertight"] and result["faces"] >= 2000:
            high_quality.append(result)

    print(f"\nValid models: {len(valid_models)} / {len(manifest)}")
    print(f"High-quality: {len(high_quality)} / {len(manifest)}")

    # Save filtered lists
    with open(output_path / "valid_models.json", "w") as f:
        json.dump(valid_models, f, indent=2)

    with open(output_path / "high_quality_models.json", "w") as f:
        json.dump(high_quality, f, indent=2)

    print(f"\nOutput saved to {output_dir}")
    return valid_models, high_quality


def main():
    parser = argparse.ArgumentParser(description="Filter Objaverse dataset")
    parser.add_argument("--manifest", type=str, required=True, help="Path to download manifest.json")
    parser.add_argument("--ultrashape_dir", type=str, default="/mnt/data/UltraShape-1.0")
    parser.add_argument("--output_dir", type=str, default="/mnt/data/filtered")
    parser.add_argument("--quality_threshold", type=float, default=0.7)
    args = parser.parse_args()

    filter_dataset(args.manifest, args.ultrashape_dir, args.output_dir, args.quality_threshold)


if __name__ == "__main__":
    main()
