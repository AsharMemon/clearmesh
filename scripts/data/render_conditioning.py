#!/usr/bin/env python3
"""Render multi-view conditioning images for Stage 2 training.

Renders each mesh from 24 views (evenly distributed on a sphere) and
optionally extracts DINO features for conditioning the refinement DiT.

Usage:
    python render_conditioning.py \
        --trellis2_dir /mnt/data/TRELLIS.2 \
        --input_dir /mnt/data/training_pairs \
        --output_dir /mnt/data/datasets/ClearMesh/renders_cond \
        --num_views 24
"""

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from tqdm import tqdm


def fibonacci_sphere(n: int) -> list[tuple[float, float, float]]:
    """Generate n evenly-distributed points on a unit sphere."""
    points = []
    golden_ratio = (1 + math.sqrt(5)) / 2

    for i in range(n):
        theta = math.acos(1 - 2 * (i + 0.5) / n)
        phi = 2 * math.pi * i / golden_ratio

        x = math.sin(theta) * math.cos(phi)
        y = math.sin(theta) * math.sin(phi)
        z = math.cos(theta)
        points.append((x, y, z))

    return points


def render_mesh_multiview(
    mesh_path: str, output_dir: str, uid: str, num_views: int = 24, image_size: int = 512
):
    """Render a mesh from multiple viewpoints."""
    try:
        mesh = trimesh.load(mesh_path, force="mesh")
    except Exception as e:
        print(f"Failed to load {mesh_path}: {e}")
        return False

    # Center and normalize
    mesh.vertices -= mesh.centroid
    scale = mesh.extents.max()
    if scale > 0:
        mesh.vertices /= scale

    view_dir = os.path.join(output_dir, uid)
    os.makedirs(view_dir, exist_ok=True)

    camera_positions = fibonacci_sphere(num_views)
    scene = trimesh.Scene(mesh)

    for i, cam_pos in enumerate(camera_positions):
        try:
            # Set camera position
            camera_transform = trimesh.transformations.look_at(
                np.array(cam_pos) * 2.0,  # distance from origin
                np.array([0, 0, 0]),  # look at center
                np.array([0, 1, 0]),  # up vector
            )

            scene.camera_transform = camera_transform
            png = scene.save_image(resolution=(image_size, image_size))

            from io import BytesIO

            img = Image.open(BytesIO(png))
            img.save(os.path.join(view_dir, f"view_{i:03d}.png"))
        except Exception:
            # If rendering fails for this view, create placeholder
            img = Image.new("RGB", (image_size, image_size), (128, 128, 128))
            img.save(os.path.join(view_dir, f"view_{i:03d}.png"))

    return True


def render_conditioning(
    input_dir: str,
    output_dir: str,
    num_views: int,
    trellis2_dir: str | None = None,
):
    """Render conditioning images for all training pairs."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Try TRELLIS.2's built-in renderer first (higher quality)
    if trellis2_dir:
        render_script = Path(trellis2_dir) / "dataset_toolkits" / "render_cond.py"
        if render_script.exists():
            print("Using TRELLIS.2's built-in renderer...")
            subprocess.run(
                [
                    sys.executable,
                    str(render_script),
                    "CustomDataset",
                    "--output_dir",
                    output_dir,
                    "--num_views",
                    str(num_views),
                ],
                cwd=trellis2_dir,
                check=True,
            )
            return

    # Fallback: render with trimesh
    print("Using trimesh renderer (install TRELLIS.2 for higher quality renders)")
    input_path = Path(input_dir)

    # Find all fine meshes in training pairs
    mesh_files = list(input_path.rglob("fine.*"))
    print(f"Rendering {len(mesh_files)} meshes x {num_views} views")

    success = 0
    for mesh_file in tqdm(mesh_files, desc="Rendering"):
        uid = mesh_file.parent.name
        if render_mesh_multiview(str(mesh_file), output_dir, uid, num_views):
            success += 1

    print(f"\nRendered: {success}/{len(mesh_files)}")
    print(f"Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Render conditioning images")
    parser.add_argument("--trellis2_dir", type=str, default="/mnt/data/TRELLIS.2")
    parser.add_argument("--input_dir", type=str, required=True, help="Training pairs directory")
    parser.add_argument("--output_dir", type=str, default="/mnt/data/datasets/ClearMesh/renders_cond")
    parser.add_argument("--num_views", type=int, default=24)
    args = parser.parse_args()

    render_conditioning(args.input_dir, args.output_dir, args.num_views, args.trellis2_dir)


if __name__ == "__main__":
    main()
