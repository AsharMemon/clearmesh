#!/usr/bin/env python3
"""Render 6-view RGB + normal map pairs for Ctrl-Adapter training.

Unlike the 24-view rendering for Stage 2 conditioning, this uses
6 canonical orthographic-style views matching ERA3D's output format:
  View 0: Front  (camera at +Z)
  View 1: Back   (camera at -Z)
  View 2: Left   (camera at -X)
  View 3: Right  (camera at +X)
  View 4: Top    (camera at +Y)
  View 5: Bottom (camera at -Y)

Each view produces both an RGB image and a paired normal map.
Only a 5K-10K subset is needed for Ctrl-Adapter training (~12GB disk).

Usage:
    python render_ctrl_adapter_data.py \
        --input_dir /workspace/data/filtered \
        --output_dir /workspace/data/ctrl_adapter \
        --limit 5000

Output structure:
    {output_dir}/{uid}/rgb_000.png ... rgb_005.png
    {output_dir}/{uid}/normal_000.png ... normal_005.png
    {output_dir}/manifest.json
"""

import argparse
import json
import os
from io import BytesIO
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from tqdm import tqdm

# 6 canonical camera positions (looking at origin from each cardinal direction)
CANONICAL_VIEWS = [
    {"name": "front", "eye": (0, 0, 2), "up": (0, 1, 0)},
    {"name": "back", "eye": (0, 0, -2), "up": (0, 1, 0)},
    {"name": "left", "eye": (-2, 0, 0), "up": (0, 1, 0)},
    {"name": "right", "eye": (2, 0, 0), "up": (0, 1, 0)},
    {"name": "top", "eye": (0, 2, 0), "up": (0, 0, -1)},
    {"name": "bottom", "eye": (0, -2, 0), "up": (0, 0, 1)},
]


def render_normal_map(
    mesh: trimesh.Trimesh,
    camera_transform: np.ndarray,
    image_size: int = 512,
) -> Image.Image:
    """Render a view-space normal map from a mesh.

    Normal maps encode surface orientation as RGB:
      R = nx mapped from [-1,1] to [0,255]
      G = ny mapped from [-1,1] to [0,255]
      B = nz mapped from [-1,1] to [0,255]
    """
    mesh_copy = mesh.copy()

    if mesh_copy.vertex_normals is None or len(mesh_copy.vertex_normals) == 0:
        mesh_copy.fix_normals()

    # Transform vertex normals into camera/view space
    cam_rotation = np.array(camera_transform[:3, :3])
    cam_rotation_inv = np.linalg.inv(cam_rotation)
    view_normals = (cam_rotation_inv @ mesh_copy.vertex_normals.T).T

    # Normalize
    norms = np.linalg.norm(view_normals, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    view_normals = view_normals / norms

    # Map [-1, 1] -> [0, 255]
    normal_colors = ((view_normals + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)
    alpha = np.full((len(normal_colors), 1), 255, dtype=np.uint8)
    rgba = np.hstack([normal_colors, alpha])

    mesh_copy.visual = trimesh.visual.ColorVisuals(mesh=mesh_copy, vertex_colors=rgba)
    scene = trimesh.Scene(mesh_copy)
    scene.camera_transform = camera_transform

    try:
        png = scene.save_image(resolution=(image_size, image_size))
        return Image.open(BytesIO(png))
    except Exception:
        return Image.new("RGB", (image_size, image_size), (128, 128, 255))


def render_6view_with_normals(
    mesh_path: str,
    output_dir: str,
    uid: str,
    image_size: int = 512,
) -> bool:
    """Render 6 canonical views with paired normal maps.

    Args:
        mesh_path: Path to mesh file (GLB/OBJ/PLY).
        output_dir: Root output directory.
        uid: Unique identifier for this mesh.
        image_size: Output resolution (square).

    Returns:
        True on success, False on failure.
    """
    try:
        mesh = trimesh.load(mesh_path, force="mesh")
    except Exception as e:
        print(f"Failed to load {mesh_path}: {e}")
        return False

    # Center and normalize to [-0.5, 0.5]
    mesh.vertices -= mesh.centroid
    scale = mesh.extents.max()
    if scale > 0:
        mesh.vertices /= scale

    view_dir = os.path.join(output_dir, uid)
    os.makedirs(view_dir, exist_ok=True)

    # Create scene for RGB rendering
    scene = trimesh.Scene(mesh)

    for i, view in enumerate(CANONICAL_VIEWS):
        try:
            camera_transform = trimesh.transformations.look_at(
                np.array(view["eye"]),
                np.array([0, 0, 0]),
                np.array(view["up"]),
            )

            # Render RGB
            scene.camera_transform = camera_transform
            png = scene.save_image(resolution=(image_size, image_size))
            rgb_img = Image.open(BytesIO(png))
            rgb_img.save(os.path.join(view_dir, f"rgb_{i:03d}.png"))

            # Render normal map
            normal_img = render_normal_map(mesh, camera_transform, image_size)
            normal_img.save(os.path.join(view_dir, f"normal_{i:03d}.png"))

        except Exception as e:
            print(f"  View {i} ({view['name']}) failed for {uid}: {e}")
            # Create placeholders
            Image.new("RGB", (image_size, image_size), (128, 128, 128)).save(
                os.path.join(view_dir, f"rgb_{i:03d}.png")
            )
            Image.new("RGB", (image_size, image_size), (128, 128, 255)).save(
                os.path.join(view_dir, f"normal_{i:03d}.png")
            )

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Render 6-view RGB + normal map pairs for Ctrl-Adapter training"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing mesh files (or subdirectories with meshes)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/workspace/data/ctrl_adapter",
        help="Output directory for rendered views",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="JSON manifest mapping uid -> mesh_path (alternative to scanning input_dir)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5000,
        help="Maximum number of meshes to render (default: 5000)",
    )
    parser.add_argument("--image_size", type=int, default=512, help="Image resolution (square)")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers (not yet implemented)")
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Collect mesh files
    if args.manifest:
        with open(args.manifest) as f:
            manifest = json.load(f)
        mesh_items = list(manifest.items())[:args.limit]
    else:
        input_path = Path(args.input_dir)
        mesh_files = sorted(
            list(input_path.rglob("*.glb"))
            + list(input_path.rglob("*.obj"))
            + list(input_path.rglob("*.ply"))
        )[:args.limit]
        mesh_items = [(f.stem, str(f)) for f in mesh_files]

    print(f"Rendering {len(mesh_items)} meshes x 6 views x 2 (RGB + normal)")
    print(f"Estimated disk: ~{len(mesh_items) * 12 * 200 / 1024 / 1024:.1f} GB")

    # Render all meshes
    success = 0
    rendered_uids = []
    for uid, mesh_path in tqdm(mesh_items, desc="Rendering 6-view pairs"):
        if render_6view_with_normals(mesh_path, args.output_dir, uid, args.image_size):
            success += 1
            rendered_uids.append(uid)

    # Save manifest
    manifest_out = {
        "total": success,
        "views_per_mesh": 6,
        "image_size": args.image_size,
        "view_names": [v["name"] for v in CANONICAL_VIEWS],
        "uids": rendered_uids,
    }
    manifest_path = output_path / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest_out, f, indent=2)

    print(f"\nRendered: {success}/{len(mesh_items)} meshes")
    print(f"Manifest: {manifest_path}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
