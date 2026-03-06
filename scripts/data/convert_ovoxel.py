#!/usr/bin/env python3
"""Convert training meshes to O-Voxel format using TRELLIS.2's data toolkit.

TRELLIS.2 uses O-Voxel (dual grid) format for its sparse voxel representation.
This script converts filtered meshes to .vxz files using the o_voxel library.

The TRELLIS.2 data_toolkit scripts (dual_grid.py, encode_ss_latent.py, etc.)
expect a specific dataset format with metadata.csv and sha256-indexed files.
This script provides a simpler interface that works with our manifest format.

Usage:
    python convert_ovoxel.py \
        --trellis2_dir /workspace/TRELLIS.2 \
        --input_json /workspace/data/filtered/valid_models.json \
        --output_dir /workspace/data/ovoxel \
        --resolution 256

    # Preview (process only first 10):
    python convert_ovoxel.py \
        --input_json /workspace/data/filtered/valid_models.json \
        --output_dir /workspace/data/ovoxel \
        --limit 10
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
from tqdm import tqdm


def mesh_to_ovoxel_direct(
    mesh_path: str,
    output_path: str,
    resolution: int = 256,
) -> bool:
    """Convert a mesh to O-Voxel format using the o_voxel library.

    Args:
        mesh_path: Path to input mesh file (GLB/OBJ/PLY).
        output_path: Path for output .vxz file.
        resolution: Voxel grid resolution.

    Returns:
        True if conversion succeeded.
    """
    try:
        import o_voxel

        # Load mesh
        mesh = trimesh.load(mesh_path, force="mesh")

        # Center and normalize to [-0.5, 0.5]
        mesh.vertices -= mesh.centroid
        scale = mesh.extents.max()
        if scale > 0:
            mesh.vertices /= scale

        # Convert to torch tensors
        vertices = torch.from_numpy(np.array(mesh.vertices)).float()
        faces = torch.from_numpy(np.array(mesh.faces)).long()

        # Convert to flexible dual grid (O-Voxel format)
        dual_grid = o_voxel.convert.mesh_to_flexible_dual_grid(
            vertices, faces, resolution=resolution
        )

        # Write to .vxz file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        o_voxel.io.write_vxz(output_path, dual_grid)

        return True

    except Exception as e:
        print(f"  Failed: {mesh_path}: {e}")
        return False


def manual_voxelize(
    mesh_path: str,
    output_path: str,
    resolution: int = 128,
) -> bool:
    """Fallback voxelization using trimesh (no o_voxel needed).

    Produces numpy voxel grids instead of .vxz files.
    Less efficient but works without CUDA extensions.
    """
    try:
        mesh = trimesh.load(mesh_path, force="mesh")

        # Center and normalize
        mesh.vertices -= mesh.centroid
        scale = mesh.extents.max()
        if scale > 0:
            mesh.vertices /= scale

        # Voxelize
        pitch = 1.0 / resolution
        voxel_grid = mesh.voxelized(pitch=pitch)
        matrix = voxel_grid.matrix.astype(np.float32)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, matrix)

        return True

    except Exception as e:
        print(f"  Failed manual voxelize {mesh_path}: {e}")
        return False


def convert_to_ovoxel(
    input_json: str,
    output_dir: str,
    resolution: int = 256,
    limit: int | None = None,
    use_ovoxel: bool = True,
):
    """Convert filtered meshes to O-Voxel format.

    Args:
        input_json: Path to valid_models.json from filter step.
        output_dir: Output directory for voxel files.
        resolution: Voxel grid resolution (256 for standard, 1024 for high-res).
        limit: Max number of models to process.
        use_ovoxel: Use o_voxel library (True) or numpy fallback (False).
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load filtered model list
    with open(input_json) as f:
        models = json.load(f)

    if limit:
        models = models[:limit]

    print(f"Converting {len(models)} meshes to O-Voxel at resolution {resolution}")

    # Check if o_voxel is available
    if use_ovoxel:
        try:
            import o_voxel
            print(f"Using o_voxel library (version: {getattr(o_voxel, '__version__', 'unknown')})")
        except ImportError:
            print("WARNING: o_voxel not available. Falling back to numpy voxelization.")
            use_ovoxel = False

    # Track progress for resume
    progress_path = output_path / "progress.json"
    if progress_path.exists():
        with open(progress_path) as f:
            completed = set(json.load(f))
    else:
        completed = set()

    success = 0
    skipped = 0
    failed = 0

    for model in tqdm(models, desc="O-Voxel conversion"):
        uid = model["uid"]
        mesh_path = model["path"]

        if uid in completed:
            skipped += 1
            continue

        if not os.path.exists(mesh_path):
            failed += 1
            continue

        if use_ovoxel:
            out_file = os.path.join(output_dir, f"{uid}.vxz")
            ok = mesh_to_ovoxel_direct(mesh_path, out_file, resolution)
        else:
            out_file = os.path.join(output_dir, f"{uid}.npy")
            ok = manual_voxelize(mesh_path, out_file, min(resolution, 128))

        if ok:
            success += 1
            completed.add(uid)
        else:
            failed += 1

        # Save progress periodically
        if success % 100 == 0 and success > 0:
            with open(progress_path, "w") as f:
                json.dump(list(completed), f)

    # Final save
    with open(progress_path, "w") as f:
        json.dump(list(completed), f)

    ext = ".vxz" if use_ovoxel else ".npy"
    print(f"\nO-Voxel conversion complete!")
    print(f"  Success: {success}")
    print(f"  Skipped (already done): {skipped}")
    print(f"  Failed: {failed}")
    print(f"  Output: {output_dir}/*{ext}")


def main():
    parser = argparse.ArgumentParser(description="Convert meshes to O-Voxel format")
    parser.add_argument(
        "--trellis2_dir",
        type=str,
        default="/workspace/TRELLIS.2",
        help="Path to TRELLIS.2 installation",
    )
    parser.add_argument(
        "--input_json",
        type=str,
        required=True,
        help="Path to valid_models.json from filter step",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/workspace/data/ovoxel",
        help="Output directory for voxel files",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="Voxel grid resolution (256=standard, 1024=high-res)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of models to process",
    )
    args = parser.parse_args()

    # Add TRELLIS.2 to path for o_voxel access
    if args.trellis2_dir:
        sys.path.insert(0, args.trellis2_dir)

    convert_to_ovoxel(
        args.input_json,
        args.output_dir,
        args.resolution,
        args.limit,
    )


if __name__ == "__main__":
    main()
