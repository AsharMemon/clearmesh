#!/usr/bin/env python3
"""
Batch experiment: TRELLIS.2 coarse → UltraShape refine.

Generates coarse meshes from TRELLIS.2, refines them with UltraShape,
and compares the results side-by-side.

Usage:
    # With pre-generated coarse meshes + images:
    python experiments/ultrashape/run_experiment.py \
        --image_dir experiments/ultrashape/inputs/images \
        --mesh_dir experiments/ultrashape/inputs/coarse_meshes

    # End-to-end: generate coarse with TRELLIS.2 first, then refine:
    python experiments/ultrashape/run_experiment.py \
        --image_dir experiments/ultrashape/inputs/images \
        --generate_coarse

    # Quick test with lower resolution:
    python experiments/ultrashape/run_experiment.py \
        --image_dir experiments/ultrashape/inputs/images \
        --mesh_dir experiments/ultrashape/inputs/coarse_meshes \
        --octree_res 384 --low_vram
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import numpy as np


def find_pairs(image_dir, mesh_dir):
    """Find matching image-mesh pairs by filename stem."""
    image_exts = {".png", ".jpg", ".jpeg", ".webp"}
    mesh_exts = {".glb", ".obj", ".ply", ".stl"}

    images = {}
    for f in Path(image_dir).iterdir():
        if f.suffix.lower() in image_exts:
            images[f.stem] = f

    pairs = []
    for f in Path(mesh_dir).iterdir():
        if f.suffix.lower() in mesh_exts:
            stem = f.stem
            # Try exact match first, then strip common suffixes
            for key in [stem, stem.replace("_coarse", ""), stem.replace("_512", "")]:
                if key in images:
                    pairs.append((str(images[key]), str(f)))
                    break

    pairs.sort(key=lambda x: x[0])
    return pairs


def generate_coarse_meshes(image_dir, output_dir, device="cuda"):
    """Generate coarse meshes from images using TRELLIS.2 (512 steps)."""
    from PIL import Image

    try:
        from trellis2.pipelines import Trellis2ImageTo3DPipeline
    except ImportError:
        print("ERROR: TRELLIS.2 not installed. Cannot generate coarse meshes.")
        print("Either install TRELLIS.2 or provide pre-generated meshes with --mesh_dir")
        sys.exit(1)

    print("Loading TRELLIS.2 pipeline (512 steps)...")
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipeline = pipeline.to(device)

    image_exts = {".png", ".jpg", ".jpeg", ".webp"}
    os.makedirs(output_dir, exist_ok=True)

    pairs = []
    for f in sorted(Path(image_dir).iterdir()):
        if f.suffix.lower() not in image_exts:
            continue

        output_path = os.path.join(output_dir, f"{f.stem}_coarse.glb")
        if os.path.exists(output_path):
            print(f"  Skipping {f.name} (already generated)")
            pairs.append((str(f), output_path))
            continue

        print(f"  Generating coarse mesh for {f.name}...")
        t0 = time.time()

        img = Image.open(f).convert("RGBA")
        # Use 512-step model for coarse output
        result = pipeline(img, num_inference_steps=512)

        if hasattr(result, "export"):
            result.export(output_path)
        elif isinstance(result, dict) and "mesh" in result:
            result["mesh"].export(output_path)

        elapsed = time.time() - t0
        print(f"    Done in {elapsed:.1f}s -> {output_path}")
        pairs.append((str(f), output_path))

    # Free TRELLIS.2 VRAM
    del pipeline
    torch.cuda.empty_cache()

    return pairs


def run_single_refinement(refine_script, image, mesh, output, args):
    """Run a single refinement as a subprocess (to isolate VRAM)."""
    import subprocess

    cmd = [
        sys.executable, refine_script,
        "--image", image,
        "--mesh", mesh,
        "--output", output,
        "--steps", str(args.steps),
        "--guidance", str(args.guidance),
        "--octree_res", str(args.octree_res),
        "--seed", str(args.seed),
    ]
    if args.low_vram:
        cmd.append("--low_vram")
    if args.checkpoint:
        cmd.extend(["--checkpoint", args.checkpoint])

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    return result.returncode, result.stdout, result.stderr


def compute_mesh_stats(mesh_path):
    """Compute basic mesh statistics."""
    import trimesh

    try:
        mesh = trimesh.load(mesh_path, force="mesh")
        return {
            "vertices": len(mesh.vertices),
            "faces": len(mesh.faces),
            "watertight": mesh.is_watertight,
            "volume": float(mesh.volume) if mesh.is_watertight else None,
            "surface_area": float(mesh.area),
            "bounding_box": mesh.bounds.tolist(),
        }
    except Exception as e:
        return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Batch UltraShape refinement experiment")
    parser.add_argument("--image_dir", required=True, help="Directory of reference images")
    parser.add_argument("--mesh_dir", default=None, help="Directory of coarse meshes (GLB)")
    parser.add_argument("--generate_coarse", action="store_true",
                        help="Generate coarse meshes from images using TRELLIS.2")
    parser.add_argument("--output_dir", default=None, help="Output directory")
    parser.add_argument("--checkpoint", default=None, help="Path to ultrashape_v1.pt")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=5.0)
    parser.add_argument("--octree_res", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--low_vram", action="store_true")
    parser.add_argument("--subprocess", action="store_true",
                        help="Run each refinement in a subprocess (better VRAM isolation)")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    refine_script = os.path.join(script_dir, "refine.py")

    if args.output_dir is None:
        args.output_dir = os.path.join(script_dir, "outputs")
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("UltraShape Batch Refinement Experiment")
    print("=" * 60)

    # Get image-mesh pairs
    if args.generate_coarse:
        coarse_dir = os.path.join(script_dir, "inputs", "coarse_meshes")
        pairs = generate_coarse_meshes(args.image_dir, coarse_dir)
    elif args.mesh_dir:
        pairs = find_pairs(args.image_dir, args.mesh_dir)
    else:
        print("ERROR: Must provide --mesh_dir or --generate_coarse")
        sys.exit(1)

    if not pairs:
        print("ERROR: No matching image-mesh pairs found.")
        print(f"  Image dir: {args.image_dir}")
        print(f"  Mesh dir:  {args.mesh_dir}")
        sys.exit(1)

    print(f"\nFound {len(pairs)} image-mesh pairs:")
    for img, mesh in pairs:
        print(f"  {os.path.basename(img)} <-> {os.path.basename(mesh)}")
    print()

    # Run refinements
    results = []
    for i, (image_path, mesh_path) in enumerate(pairs):
        name = Path(mesh_path).stem
        output_path = os.path.join(args.output_dir, f"{name}_refined.glb")

        print(f"\n[{i+1}/{len(pairs)}] Refining {name}...")

        t0 = time.time()

        if args.subprocess:
            retcode, stdout, stderr = run_single_refinement(
                refine_script, image_path, mesh_path, output_path, args
            )
            elapsed = time.time() - t0
            success = retcode == 0 and os.path.exists(output_path)
            if not success:
                print(f"  FAILED (returncode={retcode})")
                if stderr:
                    print(f"  stderr: {stderr[-500:]}")
        else:
            # Run in-process (shares model loading across runs)
            if i == 0:
                # First run: set up everything
                sys.path.insert(0, os.path.join(script_dir, "UltraShape-1.0"))
                from refine import setup_ultrashape_path, load_config, load_models, run_refinement

                ultrashape_dir = setup_ultrashape_path()
                config = load_config(ultrashape_dir, args)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                ckpt = args.checkpoint or os.path.join(script_dir, "checkpoints", "ultrashape_v1.pt")
                vae, dit, conditioner, scheduler, image_processor = load_models(
                    config, ckpt, device, low_vram=args.low_vram
                )

            try:
                _, elapsed = run_refinement(
                    vae, dit, conditioner, scheduler, image_processor,
                    image_path, mesh_path, output_path,
                    num_steps=args.steps, guidance_scale=args.guidance,
                    num_latents=32768, octree_res=args.octree_res,
                    chunk_size=8000, scale=0.99,
                    seed=args.seed, device=device, low_vram=args.low_vram,
                )
                success = os.path.exists(output_path)
            except Exception as e:
                print(f"  FAILED: {e}")
                import traceback
                traceback.print_exc()
                elapsed = time.time() - t0
                success = False

        # Compute stats
        entry = {
            "name": name,
            "image": image_path,
            "coarse_mesh": mesh_path,
            "refined_mesh": output_path if success else None,
            "success": success,
            "elapsed_seconds": round(elapsed, 1),
        }
        if success:
            entry["coarse_stats"] = compute_mesh_stats(mesh_path)
            entry["refined_stats"] = compute_mesh_stats(output_path)

            cs = entry["coarse_stats"]
            rs = entry["refined_stats"]
            if "vertices" in cs and "vertices" in rs:
                print(f"  Coarse:  {cs['vertices']} verts, {cs['faces']} faces")
                print(f"  Refined: {rs['vertices']} verts, {rs['faces']} faces")
                print(f"  Time:    {elapsed:.1f}s")

        results.append(entry)

    # Save summary
    summary_path = os.path.join(args.output_dir, "experiment_summary.json")
    summary = {
        "experiment": "ultrashape_refinement",
        "config": {
            "steps": args.steps,
            "guidance": args.guidance,
            "octree_res": args.octree_res,
            "seed": args.seed,
            "low_vram": args.low_vram,
        },
        "results": results,
        "total_pairs": len(pairs),
        "successful": sum(1 for r in results if r["success"]),
        "failed": sum(1 for r in results if not r["success"]),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("Experiment Summary")
    print("=" * 60)
    print(f"  Total:      {summary['total_pairs']}")
    print(f"  Successful: {summary['successful']}")
    print(f"  Failed:     {summary['failed']}")
    print(f"  Results:    {summary_path}")
    print()

    if summary["successful"] > 0:
        times = [r["elapsed_seconds"] for r in results if r["success"]]
        print(f"  Avg time:   {np.mean(times):.1f}s")
        print(f"  Min time:   {np.min(times):.1f}s")
        print(f"  Max time:   {np.max(times):.1f}s")

    print(f"\nRefined meshes saved to: {args.output_dir}/")
    print("Compare coarse vs refined meshes visually to evaluate quality.")


if __name__ == "__main__":
    main()
