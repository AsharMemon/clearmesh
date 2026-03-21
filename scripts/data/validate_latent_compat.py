#!/usr/bin/env python3
"""Gate 0A-2: Encoder vs Diffusion Latent Compatibility.

Tests whether encoder-produced SLAT lives in the same distribution as
diffusion-produced SLAT. This matters because at inference, the refiner
takes diffusion-generated coarse SLAT as input but trains on encoder-
generated target SLAT.

For each test mesh:
  1. Generate SLAT via diffusion (TRELLIS.2 at 1024) → slat_diffusion
  2. Encode the mesh via shape encoder → slat_encoded
  3. Compare per-channel statistics (mean, std, cosine similarity)
  4. Decode both through shape decoder → two meshes
  5. Compare decoded mesh quality against original

Pass criteria:
  - Per-channel mean/std within 2× of each other
  - Cosine similarity between matched tokens: median > 0.5
  - Decoded mesh quality from encoder within 50% Chamfer of diffusion

Usage:
    python scripts/data/validate_latent_compat.py \\
        --mesh_dir /workspace/data/test_meshes \\
        --image_dir /workspace/data/test_images \\
        --output_dir /workspace/data/gate_0a2_results \\
        --trellis2_dir /workspace/TRELLIS.2 \\
        --model_dir /workspace/models/trellis2-4b
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


def compute_latent_statistics(slat: np.ndarray) -> dict:
    """Compute per-channel statistics for a SLAT tensor."""
    return {
        "mean_per_channel": slat.mean(axis=0).tolist(),
        "std_per_channel": slat.std(axis=0).tolist(),
        "global_mean": float(slat.mean()),
        "global_std": float(slat.std()),
        "l2_norm_mean": float(np.linalg.norm(slat, axis=1).mean()),
        "l2_norm_std": float(np.linalg.norm(slat, axis=1).std()),
        "shape": list(slat.shape),
    }


def compare_distributions(
    slat_a: np.ndarray,
    slat_b: np.ndarray,
    name_a: str = "A",
    name_b: str = "B",
) -> dict:
    """Compare two SLAT distributions channel by channel."""
    stats_a = compute_latent_statistics(slat_a)
    stats_b = compute_latent_statistics(slat_b)

    mean_a = np.array(stats_a["mean_per_channel"])
    mean_b = np.array(stats_b["mean_per_channel"])
    std_a = np.array(stats_a["std_per_channel"])
    std_b = np.array(stats_b["std_per_channel"])

    # Per-channel mean/std ratio
    mean_ratio = np.abs(mean_a - mean_b) / (np.abs(mean_a) + np.abs(mean_b) + 1e-8)
    std_ratio = std_a / (std_b + 1e-8)

    # Cosine similarity between matched tokens (nearest-neighbor by position)
    # This requires position information — compute overall feature cosine instead
    n = min(slat_a.shape[0], slat_b.shape[0])
    cos_sims = []
    for i in range(n):
        cos = np.dot(slat_a[i], slat_b[i]) / (
            np.linalg.norm(slat_a[i]) * np.linalg.norm(slat_b[i]) + 1e-8
        )
        cos_sims.append(cos)
    cos_sims = np.array(cos_sims)

    return {
        f"stats_{name_a}": stats_a,
        f"stats_{name_b}": stats_b,
        "mean_ratio_max": float(np.max(mean_ratio)),
        "std_ratio_range": [float(np.min(std_ratio)), float(np.max(std_ratio))],
        "cosine_similarity_median": float(np.median(cos_sims)),
        "cosine_similarity_mean": float(np.mean(cos_sims)),
        "cosine_similarity_std": float(np.std(cos_sims)),
        "channels_with_mean_ratio_gt_1": int(np.sum(mean_ratio > 1.0)),
        "channels_with_std_ratio_gt_2": int(np.sum(std_ratio > 2.0)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Gate 0A-2: Encoder vs Diffusion Latent Compatibility"
    )
    parser.add_argument("--mesh_dir", required=True,
                        help="Directory of test meshes")
    parser.add_argument("--image_dir", required=True,
                        help="Directory of corresponding images (for TRELLIS.2 diffusion)")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--trellis2_dir", default="/workspace/TRELLIS.2")
    parser.add_argument("--model_dir", default="/workspace/models/trellis2-4b")
    parser.add_argument("--max_meshes", type=int, default=20)
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Gate 0A-2: Encoder vs Diffusion Latent Compatibility")
    print(f"{'='*60}")
    print(f"  Mesh dir: {args.mesh_dir}")
    print(f"  Image dir: {args.image_dir}")
    print(f"  Output: {args.output_dir}")

    # Load TRELLIS.2
    print(f"\nLoading TRELLIS.2...")
    sys.path.insert(0, args.trellis2_dir)

    try:
        from trellis2.pipelines import Trellis2ImageTo3DPipeline
        pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.model_dir)
        pipeline.cuda()
        print("  Pipeline loaded")
    except Exception as e:
        print(f"  ERROR: {e}")
        sys.exit(1)

    # Find matching mesh/image pairs
    mesh_exts = {".glb", ".obj", ".ply", ".stl"}
    image_exts = {".png", ".jpg", ".jpeg", ".webp"}

    mesh_dir = Path(args.mesh_dir)
    image_dir = Path(args.image_dir)

    pairs = []
    for mesh_path in sorted(mesh_dir.rglob("*")):
        if mesh_path.suffix.lower() not in mesh_exts:
            continue
        stem = mesh_path.stem
        for ext in image_exts:
            img_path = image_dir / f"{stem}{ext}"
            if img_path.exists():
                pairs.append((mesh_path, img_path))
                break

    pairs = pairs[:args.max_meshes]
    print(f"  Found {len(pairs)} mesh/image pairs")

    if not pairs:
        print("ERROR: No matching mesh/image pairs found")
        sys.exit(1)

    results = []
    for i, (mesh_path, image_path) in enumerate(pairs):
        name = mesh_path.stem
        print(f"\n[{i+1}/{len(pairs)}] {name}")

        try:
            # === Step 1: Generate SLAT via diffusion ===
            # Run TRELLIS.2 normally → diffusion SLAT
            from PIL import Image
            image = Image.open(str(image_path)).convert("RGBA")
            processed = pipeline.preprocess_image(image)
            torch.manual_seed(42)

            cond_1024 = pipeline.get_cond([processed], 1024)
            coords = pipeline.sample_sparse_structure(
                cond_1024, 32, 1, {"steps": 12, "guidance_strength": 9.0},
            )
            shape_slat_diff = pipeline.sample_shape_slat(
                cond_1024,
                pipeline.models["shape_slat_flow_model_1024"],
                coords,
                {"steps": 12, "guidance_strength": 4.5},
            )

            # Extract features
            if hasattr(shape_slat_diff, "feats"):
                slat_diffusion = shape_slat_diff.feats.cpu().numpy()
            elif hasattr(shape_slat_diff, "F"):
                slat_diffusion = shape_slat_diff.F.cpu().numpy()
            else:
                slat_diffusion = shape_slat_diff.squeeze(0).cpu().numpy()

            print(f"  Diffusion SLAT: {slat_diffusion.shape}")

            # === Step 2: Encode mesh via shape encoder ===
            # NOTE: Requires Gate 0C to determine encoder API
            print(f"  TODO: Encode mesh via shape encoder")
            print(f"  Need Gate 0C to determine encoder API")

            # Placeholder comparison
            result = {
                "name": name,
                "diffusion_slat_shape": list(slat_diffusion.shape),
                "diffusion_stats": compute_latent_statistics(slat_diffusion.astype(np.float32)),
                "status": "diffusion_only",
                "note": "Encoder not yet available — run Gate 0C first",
            }
            results.append(result)

            del shape_slat_diff, coords, cond_1024
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({"name": name, "status": "error", "error": str(e)})

    # Summary
    summary = {
        "gate": "0A-2",
        "description": "Encoder vs Diffusion Latent Compatibility",
        "total_pairs": len(pairs),
        "results": results,
    }

    summary_path = output_path / "gate_0a2_results.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Gate 0A-2 Results")
    print(f"{'='*60}")
    print(f"  Total pairs: {len(pairs)}")
    print(f"  Results: {summary_path}")
    print(f"\nNOTE: Encoder comparison requires Gate 0C findings.")
    print(f"Run inspect_trellis_dit.py first to determine encoder API.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
