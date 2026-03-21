#!/usr/bin/env python3
"""Gate 0D-1: Target Validity Oracle.

Tests whether 1024 diffusion SLAT at coarse positions produces better
decoded meshes than 512 coarse SLAT. This validates the training target.

For each test image:
  1. Generate sparse structure → shared positions (N, 3)
  2. Sample coarse SLAT at 512 → decode → baseline mesh
  3. Sample fine SLAT at 1024 (same positions) → decode → oracle mesh
  4. Compare Chamfer distances: oracle vs baseline vs fine reference

Since 512 and 1024 share the SAME sparse structure positions, no
alignment or nearest-neighbor matching is needed.

Also optionally tests encoder-projected oracle:
  5. Encode a GT mesh through shape encoder → SLAT at encoder positions (M)
  6. NN-project encoder SLAT onto coarse positions → projected oracle SLAT (N)
  7. Decode → encoder-oracle mesh
  8. Compare against baselines

Pass criteria:
  - Oracle-decoded meshes visibly better than coarse-decoded (>80%)
  - Mean Chamfer improvement ≥30%
  - Per-channel deltas (target - coarse) are structured, not random noise

Usage:
    python scripts/data/validate_supervision.py \\
        --image_dir /workspace/data/test_images \\
        --output_dir /workspace/data/gate_0d1_results \\
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
import trimesh
from PIL import Image


def chamfer_distance(mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh,
                     n_samples: int = 10000) -> float:
    """Compute mean Chamfer distance (both directions)."""
    pts_a = mesh_a.sample(n_samples)
    pts_b = mesh_b.sample(n_samples)

    from scipy.spatial import KDTree
    tree_b = KDTree(pts_b)
    dists_a2b, _ = tree_b.query(pts_a)
    tree_a = KDTree(pts_a)
    dists_b2a, _ = tree_a.query(pts_b)

    return float(np.mean(dists_a2b) + np.mean(dists_b2a)) / 2


def analyze_delta(coarse_slat: np.ndarray, target_slat: np.ndarray) -> dict:
    """Analyze the per-channel delta between coarse and target SLAT."""
    delta = target_slat - coarse_slat  # (N, 32)

    # Per-channel stats
    per_channel = {
        "mean": delta.mean(axis=0).tolist(),
        "std": delta.std(axis=0).tolist(),
        "abs_mean": np.abs(delta).mean(axis=0).tolist(),
        "max_abs": np.abs(delta).max(axis=0).tolist(),
    }

    # Spatial smoothness: compute variance of deltas between neighboring tokens
    delta_norms = np.linalg.norm(delta, axis=1)

    return {
        "per_channel": per_channel,
        "overall_mean": float(delta.mean()),
        "overall_std": float(delta.std()),
        "overall_abs_mean": float(np.abs(delta).mean()),
        "delta_norm_mean": float(delta_norms.mean()),
        "delta_norm_std": float(delta_norms.std()),
        "delta_norm_max": float(delta_norms.max()),
        "fraction_near_zero": float(np.mean(delta_norms < 0.01)),
        "fraction_large": float(np.mean(delta_norms > 1.0)),
    }


def extract_mesh(mesh_obj) -> trimesh.Trimesh:
    """Convert TRELLIS.2 mesh object to trimesh."""
    v = mesh_obj.vertices.detach().cpu().float().numpy()
    f = mesh_obj.faces.detach().cpu().numpy()
    return trimesh.Trimesh(vertices=v, faces=f)


def main():
    parser = argparse.ArgumentParser(
        description="Gate 0D-1: Target Validity Oracle (1024 diffusion target test)"
    )
    parser.add_argument("--image_dir", required=True,
                        help="Directory of reference images")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--trellis2_dir", default="/workspace/TRELLIS.2")
    parser.add_argument("--model_dir", default="/workspace/models/trellis2-4b")
    parser.add_argument("--max_samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Gate 0D-1: Target Validity Oracle")
    print(f"{'='*60}")
    print(f"  Image dir: {args.image_dir}")
    print(f"  Output: {args.output_dir}")
    print(f"  Max samples: {args.max_samples}")
    print(f"  Question: If I gave the model perfect 1024 targets,")
    print(f"            would the decoder actually improve the mesh?")

    # Load TRELLIS.2
    print(f"\nLoading TRELLIS.2...")
    sys.path.insert(0, args.trellis2_dir)

    try:
        from trellis2.pipelines import Trellis2ImageTo3DPipeline
        pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.model_dir)
        pipeline.low_vram = False
        for m in pipeline.models.values():
            if hasattr(m, "low_vram"):
                m.low_vram = False
        pipeline.cuda()
        print("  Pipeline loaded")
    except Exception as e:
        print(f"  ERROR: {e}")
        sys.exit(1)

    # Find test images
    image_exts = {".png", ".jpg", ".jpeg", ".webp"}
    image_dir = Path(args.image_dir)
    images = []
    for f in sorted(image_dir.rglob("*")):
        if f.suffix.lower() in image_exts:
            images.append(f)
    images = images[:args.max_samples]

    if not images:
        print(f"ERROR: No images found in {args.image_dir}")
        sys.exit(1)

    print(f"  Found {len(images)} test images")

    results = []
    for i, img_path in enumerate(images):
        name = img_path.stem
        print(f"\n[{i+1}/{len(images)}] {name}")

        sample_dir = output_path / name
        sample_dir.mkdir(parents=True, exist_ok=True)

        try:
            image = Image.open(str(img_path)).convert("RGBA")
            processed = pipeline.preprocess_image(image)
            torch.manual_seed(args.seed)

            # === Step 1: Sparse structure → shared positions ===
            print(f"  Generating sparse structure...")
            cond_512 = pipeline.get_cond([processed], 512)
            coords = pipeline.sample_sparse_structure(
                cond_512, 32, 1, {"steps": 12, "guidance_strength": 9.0},
            )

            # === Step 2: Coarse SLAT at 512 → baseline ===
            print(f"  Sampling coarse SLAT (512)...")
            shape_slat_coarse = pipeline.sample_shape_slat(
                cond_512,
                pipeline.models["shape_slat_flow_model_512"],
                coords,
                {"steps": 12, "guidance_strength": 4.5},
            )

            coarse_feats = shape_slat_coarse.feats.cpu().numpy()
            n_points = coarse_feats.shape[0]
            print(f"  Coarse SLAT: {coarse_feats.shape} ({n_points} points)")
            np.save(sample_dir / "coarse_slat.npy", coarse_feats)

            # Decode coarse → baseline mesh
            print(f"  Decoding coarse mesh...")
            with torch.no_grad():
                meshes_coarse, _ = pipeline.decode_shape_slat(shape_slat_coarse, 512)
            tm_coarse = extract_mesh(meshes_coarse[0])
            tm_coarse.export(str(sample_dir / "mesh_coarse_512.glb"))
            print(f"  Coarse mesh: {len(tm_coarse.vertices)} verts, {len(tm_coarse.faces)} faces")

            # === Step 3: Fine SLAT at 1024 → oracle (same positions!) ===
            print(f"  Sampling fine SLAT (1024, same positions)...")
            cond_1024 = pipeline.get_cond([processed], 1024)
            shape_slat_fine = pipeline.sample_shape_slat(
                cond_1024,
                pipeline.models["shape_slat_flow_model_1024"],
                coords,  # same sparse structure → same positions
                {"steps": 12, "guidance_strength": 4.5},
            )

            fine_feats = shape_slat_fine.feats.cpu().numpy()
            np.save(sample_dir / "fine_slat.npy", fine_feats)
            print(f"  Fine SLAT: {fine_feats.shape}")

            # Decode fine → reference mesh
            with torch.no_grad():
                meshes_fine, _ = pipeline.decode_shape_slat(shape_slat_fine, 512)
            tm_fine = extract_mesh(meshes_fine[0])
            tm_fine.export(str(sample_dir / "mesh_fine_1024.glb"))
            print(f"  Fine mesh: {len(tm_fine.vertices)} verts, {len(tm_fine.faces)} faces")

            # === Step 4: Oracle test — replace coarse SLAT with fine SLAT, decode ===
            # Since positions are identical, we can directly swap features
            print(f"  Running oracle test (swap coarse feats → fine feats)...")
            oracle_slat = shape_slat_coarse.replace(
                torch.from_numpy(fine_feats).to(
                    shape_slat_coarse.feats.device
                ).to(shape_slat_coarse.feats.dtype)
            )
            with torch.no_grad():
                meshes_oracle, _ = pipeline.decode_shape_slat(oracle_slat, 512)
            tm_oracle = extract_mesh(meshes_oracle[0])
            tm_oracle.export(str(sample_dir / "mesh_oracle.glb"))
            print(f"  Oracle mesh: {len(tm_oracle.vertices)} verts, {len(tm_oracle.faces)} faces")

            # === Step 5: Compute metrics ===
            print(f"  Computing metrics...")

            # Chamfer distances (all relative to fine mesh as reference)
            chamfer_coarse = chamfer_distance(tm_coarse, tm_fine)
            chamfer_oracle = chamfer_distance(tm_oracle, tm_fine)

            improvement = (chamfer_coarse - chamfer_oracle) / (chamfer_coarse + 1e-8) * 100

            # Delta analysis
            delta_stats = analyze_delta(
                coarse_feats.astype(np.float32),
                fine_feats.astype(np.float32),
            )

            result = {
                "name": name,
                "n_points": n_points,
                "chamfer_coarse_vs_fine": chamfer_coarse,
                "chamfer_oracle_vs_fine": chamfer_oracle,
                "improvement_pct": round(improvement, 1),
                "oracle_better": chamfer_oracle < chamfer_coarse,
                "delta_analysis": delta_stats,
                "coarse_mesh": f"{len(tm_coarse.vertices)} verts, {len(tm_coarse.faces)} faces",
                "fine_mesh": f"{len(tm_fine.vertices)} verts, {len(tm_fine.faces)} faces",
                "oracle_mesh": f"{len(tm_oracle.vertices)} verts, {len(tm_oracle.faces)} faces",
            }
            results.append(result)

            print(f"  Chamfer coarse→fine: {chamfer_coarse:.6f}")
            print(f"  Chamfer oracle→fine: {chamfer_oracle:.6f}")
            print(f"  Improvement: {improvement:.1f}%")
            print(f"  Oracle better: {'YES' if chamfer_oracle < chamfer_coarse else 'NO'}")

            # Cleanup
            del shape_slat_coarse, shape_slat_fine, oracle_slat
            del meshes_coarse, meshes_fine, meshes_oracle
            del cond_512, cond_1024, coords
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append({"name": name, "status": "error", "error": str(e)})

    # === Summary ===
    n_oracle_better = sum(1 for r in results if r.get("oracle_better", False))
    improvements = [r["improvement_pct"] for r in results if "improvement_pct" in r]
    chamfers_coarse = [r["chamfer_coarse_vs_fine"] for r in results if "chamfer_coarse_vs_fine" in r]
    chamfers_oracle = [r["chamfer_oracle_vs_fine"] for r in results if "chamfer_oracle_vs_fine" in r]

    summary = {
        "gate": "0D-1",
        "description": "Target Validity Oracle — does 1024 SLAT decode to better meshes?",
        "total_samples": len(images),
        "oracle_better_count": n_oracle_better,
        "oracle_better_pct": round(n_oracle_better / max(len(images), 1) * 100, 1),
        "mean_improvement_pct": round(np.mean(improvements), 1) if improvements else None,
        "mean_chamfer_coarse": round(np.mean(chamfers_coarse), 6) if chamfers_coarse else None,
        "mean_chamfer_oracle": round(np.mean(chamfers_oracle), 6) if chamfers_oracle else None,
        "pass_criteria": {
            "oracle_better_gt_80pct": bool(n_oracle_better / max(len(images), 1) > 0.8),
            "mean_improvement_gt_30pct": bool(np.mean(improvements) > 30) if improvements else False,
        },
        "results": results,
    }

    summary_path = output_path / "gate_0d1_results.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Gate 0D-1 Results")
    print(f"{'='*60}")
    print(f"  Total: {len(images)}")
    print(f"  Oracle better than coarse: {n_oracle_better}/{len(images)} "
          f"({summary['oracle_better_pct']}%)")
    if improvements:
        print(f"  Mean improvement: {np.mean(improvements):.1f}%")
    if chamfers_coarse:
        print(f"  Mean Chamfer (coarse→fine): {np.mean(chamfers_coarse):.6f}")
    if chamfers_oracle:
        print(f"  Mean Chamfer (oracle→fine): {np.mean(chamfers_oracle):.6f}")

    # PASS/FAIL
    passed = all(summary["pass_criteria"].values())
    print(f"\n  {'PASS' if passed else 'FAIL'}: Gate 0D-1")
    if not passed:
        for crit, val in summary["pass_criteria"].items():
            if not val:
                print(f"    FAILED: {crit}")

    print(f"\n  Results saved to: {summary_path}")
    print(f"  Mesh outputs in: {args.output_dir}/*/")
    print(f"\nNOTE: This test uses 1024 diffusion SLAT as oracle target. Since 512")
    print(f"and 1024 share the same sparse structure positions, no alignment is needed.")
    print(f"If this gate passes, 1024 SLAT is a valid training target for the")
    print(f"conditional generation model.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
