#!/usr/bin/env python3
"""Convert training meshes to O-Voxel format using TRELLIS.2's dataset toolkit.

TRELLIS.2's training code requires data in O-Voxel format. This script wraps
their dataset_toolkits/voxelize.py to convert our filtered meshes.

Usage:
    python convert_ovoxel.py \
        --trellis2_dir /mnt/data/TRELLIS.2 \
        --input_dir /mnt/data/training_pairs \
        --output_dir /mnt/data/datasets/ClearMesh
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def convert_to_ovoxel(trellis2_dir: str, input_dir: str, output_dir: str):
    """Convert meshes to O-Voxel format using TRELLIS.2 toolkits."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    trellis2 = Path(trellis2_dir)

    # Step 1: Voxelize meshes
    print("=== Step 1/3: Voxelizing meshes to O-Voxel format ===")
    voxelize_script = trellis2 / "dataset_toolkits" / "voxelize.py"

    if voxelize_script.exists():
        subprocess.run(
            [
                sys.executable,
                str(voxelize_script),
                "CustomDataset",
                "--output_dir",
                output_dir,
            ],
            cwd=str(trellis2),
            check=True,
        )
    else:
        print(f"WARNING: {voxelize_script} not found.")
        print("Ensure TRELLIS.2 is installed at the specified path.")
        print("Falling back to manual voxelization...")
        manual_voxelize(input_dir, output_dir)

    # Step 2: Extract DINO features for conditioning
    print("\n=== Step 2/3: Extracting DINO features ===")
    features_script = trellis2 / "dataset_toolkits" / "extract_features.py"

    if features_script.exists():
        subprocess.run(
            [
                sys.executable,
                str(features_script),
                "--output_dir",
                output_dir,
            ],
            cwd=str(trellis2),
            check=True,
        )
    else:
        print(f"WARNING: {features_script} not found. Skipping DINO feature extraction.")

    # Step 3: Encode structured latents (for VAE)
    print("\n=== Step 3/3: Encoding structured latents ===")
    encode_script = trellis2 / "dataset_toolkits" / "encode_ss_latent.py"

    if encode_script.exists():
        subprocess.run(
            [
                sys.executable,
                str(encode_script),
                "--output_dir",
                output_dir,
            ],
            cwd=str(trellis2),
            check=True,
        )
    else:
        print(f"WARNING: {encode_script} not found. Skipping latent encoding.")

    print(f"\n=== O-Voxel conversion complete ===")
    print(f"Output: {output_dir}")


def manual_voxelize(input_dir: str, output_dir: str):
    """Fallback voxelization using trimesh if TRELLIS.2 toolkit not available."""
    import numpy as np
    import trimesh

    input_path = Path(input_dir)
    output_path = Path(output_dir) / "voxels"
    output_path.mkdir(parents=True, exist_ok=True)

    mesh_files = list(input_path.rglob("*.obj")) + list(input_path.rglob("*.glb"))
    print(f"Found {len(mesh_files)} meshes to voxelize")

    for mesh_file in mesh_files:
        try:
            mesh = trimesh.load(str(mesh_file), force="mesh")

            # Center and normalize
            mesh.vertices -= mesh.centroid
            scale = mesh.extents.max()
            if scale > 0:
                mesh.vertices /= scale

            # Voxelize at 128^3 resolution
            voxel_grid = mesh.voxelized(pitch=1.0 / 128)
            matrix = voxel_grid.matrix.astype(np.float32)

            # Save as numpy array
            uid = mesh_file.stem
            np.save(str(output_path / f"{uid}.npy"), matrix)

        except Exception as e:
            print(f"Failed to voxelize {mesh_file}: {e}")
            continue


def main():
    parser = argparse.ArgumentParser(description="Convert meshes to O-Voxel format")
    parser.add_argument("--trellis2_dir", type=str, default="/mnt/data/TRELLIS.2")
    parser.add_argument("--input_dir", type=str, required=True, help="Training pairs directory")
    parser.add_argument("--output_dir", type=str, default="/mnt/data/datasets/ClearMesh")
    args = parser.parse_args()

    convert_to_ovoxel(args.trellis2_dir, args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
