#!/usr/bin/env python3
"""Render multi-view conditioning images for Stage 2 training.

Renders each mesh from 24 views (evenly distributed on a sphere) and
optionally renders paired normal maps for Ctrl-Adapter training.

Usage:
    # RGB only (original behavior)
    python render_conditioning.py \
        --input_dir /mnt/data/training_pairs \
        --output_dir /mnt/data/datasets/ClearMesh/renders_cond \
        --num_views 24

    # RGB + normal maps (for Easy3E Ctrl-Adapter)
    python render_conditioning.py \
        --input_dir /mnt/data/training_pairs \
        --output_dir /mnt/data/datasets/ClearMesh/renders_cond \
        --num_views 24 \
        --render_normals
"""

import argparse
import math
import os
import subprocess
import sys
from io import BytesIO
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


def render_normal_map(
    mesh: trimesh.Trimesh,
    camera_transform: np.ndarray,
    image_size: int = 512,
) -> Image.Image:
    """Render a view-space normal map from a mesh at a given camera pose.

    Normal maps encode surface orientation as RGB:
      R = nx mapped from [-1,1] to [0,255]
      G = ny mapped from [-1,1] to [0,255]
      B = nz mapped from [-1,1] to [0,255]

    Args:
        mesh: Input mesh (must have valid vertex normals).
        camera_transform: 4x4 camera-to-world transform matrix.
        image_size: Output image resolution (square).

    Returns:
        PIL Image with normal map encoded as RGB colors.
    """
    mesh_copy = mesh.copy()

    # Ensure vertex normals are computed
    if mesh_copy.vertex_normals is None or len(mesh_copy.vertex_normals) == 0:
        mesh_copy.fix_normals()

    # Transform vertex normals into camera/view space
    # camera_transform is world-to-camera, so we use its rotation part
    cam_rotation = np.array(camera_transform[:3, :3])
    cam_rotation_inv = np.linalg.inv(cam_rotation)
    view_normals = (cam_rotation_inv @ mesh_copy.vertex_normals.T).T

    # Normalize (some transforms may slightly denormalize)
    norms = np.linalg.norm(view_normals, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    view_normals = view_normals / norms

    # Map [-1, 1] -> [0, 255] as uint8 RGB
    normal_colors = ((view_normals + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)
    # Add alpha channel (fully opaque)
    alpha = np.full((len(normal_colors), 1), 255, dtype=np.uint8)
    rgba = np.hstack([normal_colors, alpha])

    # Assign as vertex colors and render
    mesh_copy.visual = trimesh.visual.ColorVisuals(
        mesh=mesh_copy,
        vertex_colors=rgba,
    )

    scene = trimesh.Scene(mesh_copy)
    scene.camera_transform = camera_transform

    try:
        png = scene.save_image(resolution=(image_size, image_size))
        return Image.open(BytesIO(png))
    except Exception:
        # Fallback: neutral normal map (pointing up)
        return Image.new("RGB", (image_size, image_size), (128, 128, 255))


def render_mesh_multiview(
    mesh_path: str,
    output_dir: str,
    uid: str,
    num_views: int = 24,
    image_size: int = 512,
    render_normals: bool = False,
):
    """Render a mesh from multiple viewpoints, optionally with normal maps.

    Args:
        mesh_path: Path to the mesh file.
        output_dir: Root output directory.
        uid: Unique identifier for this mesh.
        num_views: Number of viewpoints to render.
        image_size: Output image resolution (square).
        render_normals: If True, also render normal maps as view_XXX_normal.png.

    Returns:
        True if rendering succeeded, False otherwise.
    """
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

    # Create scene for RGB rendering
    scene = trimesh.Scene(mesh)

    for i, cam_pos in enumerate(camera_positions):
        try:
            # Set camera position looking at origin
            camera_transform = trimesh.transformations.look_at(
                np.array(cam_pos) * 2.0,  # distance from origin
                np.array([0, 0, 0]),  # look at center
                np.array([0, 1, 0]),  # up vector
            )

            # Render RGB view
            scene.camera_transform = camera_transform
            png = scene.save_image(resolution=(image_size, image_size))
            img = Image.open(BytesIO(png))
            img.save(os.path.join(view_dir, f"view_{i:03d}.png"))

            # Render normal map if requested
            if render_normals:
                normal_img = render_normal_map(mesh, camera_transform, image_size)
                normal_img.save(os.path.join(view_dir, f"view_{i:03d}_normal.png"))

        except Exception:
            # If rendering fails for this view, create placeholders
            img = Image.new("RGB", (image_size, image_size), (128, 128, 128))
            img.save(os.path.join(view_dir, f"view_{i:03d}.png"))
            if render_normals:
                normal_img = Image.new("RGB", (image_size, image_size), (128, 128, 255))
                normal_img.save(os.path.join(view_dir, f"view_{i:03d}_normal.png"))

    return True


def render_conditioning(
    input_dir: str,
    output_dir: str,
    num_views: int,
    trellis2_dir: str | None = None,
    render_normals: bool = False,
):
    """Render conditioning images for all training pairs.

    Args:
        input_dir: Directory containing training pairs (each pair in a subdirectory).
        output_dir: Output directory for rendered images.
        num_views: Number of viewpoints per mesh.
        trellis2_dir: Path to TRELLIS.2 installation (for higher-quality rendering).
        render_normals: If True, also render normal maps alongside RGB views.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Try TRELLIS.2's built-in renderer first (higher quality, but no normal maps)
    if trellis2_dir and not render_normals:
        render_script = Path(trellis2_dir) / "data_toolkit" / "render_cond.py"
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

    # Fallback: render with trimesh (supports normal maps)
    mode = "RGB + normal maps" if render_normals else "RGB only"
    print(f"Using trimesh renderer ({mode})")
    input_path = Path(input_dir)

    # Find all fine meshes in training pairs
    mesh_files = list(input_path.rglob("fine.*"))
    if not mesh_files:
        # Also try finding any GLB/OBJ files directly
        mesh_files = sorted(
            list(input_path.rglob("*.glb"))
            + list(input_path.rglob("*.obj"))
            + list(input_path.rglob("*.ply"))
        )
    print(f"Rendering {len(mesh_files)} meshes x {num_views} views")

    success = 0
    for mesh_file in tqdm(mesh_files, desc="Rendering"):
        uid = mesh_file.stem if mesh_file.parent == input_path else mesh_file.parent.name
        if render_mesh_multiview(
            str(mesh_file), output_dir, uid, num_views, render_normals=render_normals
        ):
            success += 1

    print(f"\nRendered: {success}/{len(mesh_files)}")
    if render_normals:
        print(f"  RGB views: {output_dir}/{{uid}}/view_XXX.png")
        print(f"  Normal maps: {output_dir}/{{uid}}/view_XXX_normal.png")
    print(f"Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Render conditioning images")
    parser.add_argument("--trellis2_dir", type=str, default=None)
    parser.add_argument("--input_dir", type=str, required=True, help="Training pairs directory")
    parser.add_argument(
        "--output_dir", type=str, default="/mnt/data/datasets/ClearMesh/renders_cond"
    )
    parser.add_argument("--num_views", type=int, default=24)
    parser.add_argument(
        "--render_normals",
        action="store_true",
        help="Also render normal maps (for Easy3E Ctrl-Adapter training)",
    )
    args = parser.parse_args()

    render_conditioning(
        args.input_dir, args.output_dir, args.num_views, args.trellis2_dir, args.render_normals
    )


if __name__ == "__main__":
    main()
