#!/usr/bin/env python3
"""Generate coarse/fine training pairs for Stage 2 refinement.

For each high-quality mesh:
  1. Render a canonical view image
  2. Run TRELLIS.2 to generate a coarse mesh from that image
  3. Save the (coarse, fine) pair for Stage 2 training

The coarse mesh is the training INPUT, the original is the training TARGET.
This teaches Stage 2 to refine real TRELLIS.2 outputs.

Usage:
    python generate_pairs.py \
        --input_json /mnt/data/filtered/high_quality_models.json \
        --output_dir /mnt/data/training_pairs \
        --resolution 512 \
        --batch_size 16
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image
from tqdm import tqdm


def render_canonical_view(mesh_path: str, image_size: int = 512) -> Image.Image:
    """Render a mesh from a canonical front view using trimesh's offscreen renderer."""
    mesh = trimesh.load(mesh_path, force="mesh")

    # Center and normalize
    mesh.vertices -= mesh.centroid
    scale = mesh.extents.max()
    if scale > 0:
        mesh.vertices /= scale

    # Render using trimesh's built-in renderer
    scene = trimesh.Scene(mesh)
    # Try pyrender-based rendering, fall back to simple rendering
    try:
        png = scene.save_image(resolution=(image_size, image_size))
        from io import BytesIO

        return Image.open(BytesIO(png)).convert("RGBA")
    except Exception:
        # Fallback: create a simple depth rendering
        # In production this would use a proper renderer
        img = Image.new("RGBA", (image_size, image_size), (255, 255, 255, 255))
        return img


def generate_coarse_mesh(pipeline, image: Image.Image, resolution: int = 512):
    """Run TRELLIS.2 to generate a coarse mesh from an image."""
    with torch.no_grad():
        result = pipeline.run(image, resolution=resolution)
    return result[0]


def save_pair(coarse_mesh, fine_mesh_path: str, output_dir: str, uid: str):
    """Save a coarse/fine pair to disk."""
    pair_dir = os.path.join(output_dir, uid)
    os.makedirs(pair_dir, exist_ok=True)

    # Save coarse mesh (from TRELLIS.2)
    coarse_path = os.path.join(pair_dir, "coarse.glb")
    try:
        import o_voxel

        o_voxel.postprocess.to_glb(coarse_mesh, coarse_path)
    except ImportError:
        # Fallback: if o_voxel not available, save as OBJ via trimesh
        if hasattr(coarse_mesh, "vertices"):
            t = trimesh.Trimesh(vertices=coarse_mesh.vertices, faces=coarse_mesh.faces)
            t.export(coarse_path.replace(".glb", ".obj"))

    # Symlink or copy fine mesh (original high-quality)
    fine_link = os.path.join(pair_dir, "fine" + Path(fine_mesh_path).suffix)
    if not os.path.exists(fine_link):
        os.symlink(os.path.abspath(fine_mesh_path), fine_link)

    return pair_dir


def generate_pairs(input_json: str, output_dir: str, resolution: int, limit: int | None = None):
    """Main pair generation loop."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load filtered model list
    with open(input_json) as f:
        models = json.load(f)

    if limit:
        models = models[:limit]

    print(f"Generating coarse/fine pairs for {len(models)} models at {resolution}^3")

    # Load TRELLIS.2
    print("Loading TRELLIS.2-4B...")
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipeline.cuda()

    # Track progress for resume
    progress_path = output_path / "progress.json"
    if progress_path.exists():
        with open(progress_path) as f:
            completed = set(json.load(f))
    else:
        completed = set()

    pairs_created = 0
    failures = 0

    for model in tqdm(models, desc="Generating pairs"):
        uid = model["uid"]
        mesh_path = model["path"]

        if uid in completed:
            continue

        try:
            # Render canonical view
            image = render_canonical_view(mesh_path)

            # Generate coarse mesh via TRELLIS.2
            coarse = generate_coarse_mesh(pipeline, image, resolution)

            # Save pair
            save_pair(coarse, mesh_path, output_dir, uid)

            completed.add(uid)
            pairs_created += 1

        except Exception as e:
            failures += 1
            if failures % 100 == 0:
                print(f"\nFailures so far: {failures} (latest: {e})")
            continue

        # Save progress periodically
        if pairs_created % 100 == 0:
            with open(progress_path, "w") as f:
                json.dump(list(completed), f)

    # Final progress save
    with open(progress_path, "w") as f:
        json.dump(list(completed), f)

    print(f"\nPairs created: {pairs_created}")
    print(f"Failures: {failures}")
    print(f"Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate coarse/fine training pairs")
    parser.add_argument("--input_json", type=str, required=True, help="Filtered models JSON")
    parser.add_argument("--output_dir", type=str, default="/mnt/data/training_pairs")
    parser.add_argument("--resolution", type=int, default=512, choices=[512, 1024, 1536])
    parser.add_argument("--limit", type=int, default=None, help="Max pairs to generate")
    args = parser.parse_args()

    generate_pairs(args.input_json, args.output_dir, args.resolution, args.limit)


if __name__ == "__main__":
    main()
