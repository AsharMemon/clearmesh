#!/usr/bin/env python3
"""Gate 0A-1: Encoder/Decoder Round-Trip Quality.

Tests whether TRELLIS.2's shape encoder + decoder produces acceptable
reconstructions. If the round-trip is lossy, the quality ceiling for
the refinement model is that lossy reconstruction.

For each test mesh:
  1. Convert to O-Voxel (sparse voxel representation)
  2. Encode to SLAT via shape encoder
  3. Decode back via shape decoder
  4. Extract mesh
  5. Compute: Chamfer distance, normal consistency, F-score

Pass criteria:
  - Mean Chamfer < 2× the distance between 512→1024 SLAT pairs
  - No catastrophic failures (holes, missing limbs) on >5% of meshes
  - Visual: reconstructed meshes recognizably the same object

Usage:
    python scripts/data/validate_roundtrip.py \\
        --mesh_dir /workspace/data/test_meshes \\
        --output_dir /workspace/data/gate_0a1_results \\
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


def compute_chamfer_distance(mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh,
                              n_samples: int = 10000) -> dict:
    """Compute bidirectional Chamfer distance between two meshes."""
    pts_a = mesh_a.sample(n_samples)
    pts_b = mesh_b.sample(n_samples)

    # A → B
    from scipy.spatial import KDTree
    tree_b = KDTree(pts_b)
    dists_a2b, _ = tree_b.query(pts_a)

    # B → A
    tree_a = KDTree(pts_a)
    dists_b2a, _ = tree_a.query(pts_b)

    return {
        "chamfer_mean": float(np.mean(dists_a2b) + np.mean(dists_b2a)) / 2,
        "chamfer_a2b": float(np.mean(dists_a2b)),
        "chamfer_b2a": float(np.mean(dists_b2a)),
        "chamfer_max_a2b": float(np.max(dists_a2b)),
        "chamfer_max_b2a": float(np.max(dists_b2a)),
    }


def compute_normal_consistency(mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh,
                                n_samples: int = 10000) -> float:
    """Compute normal consistency (mean dot product of matched normals)."""
    pts_a, face_idx_a = trimesh.sample.sample_surface(mesh_a, n_samples)
    normals_a = mesh_a.face_normals[face_idx_a]

    from scipy.spatial import KDTree
    tree_b = KDTree(mesh_b.sample(n_samples))

    pts_b, face_idx_b = trimesh.sample.sample_surface(mesh_b, n_samples)
    normals_b = mesh_b.face_normals[face_idx_b]

    # Match A points to nearest B points
    tree_b_full = KDTree(pts_b)
    _, idx = tree_b_full.query(pts_a)

    matched_normals_b = normals_b[idx]
    dots = np.sum(normals_a * matched_normals_b, axis=1)
    return float(np.mean(np.abs(dots)))


def compute_f_score(mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh,
                    tau: float = 0.01, n_samples: int = 10000) -> float:
    """Compute F-score at threshold tau."""
    pts_a = mesh_a.sample(n_samples)
    pts_b = mesh_b.sample(n_samples)

    from scipy.spatial import KDTree
    tree_b = KDTree(pts_b)
    dists_a2b, _ = tree_b.query(pts_a)
    precision = np.mean(dists_a2b < tau)

    tree_a = KDTree(pts_a)
    dists_b2a, _ = tree_a.query(pts_b)
    recall = np.mean(dists_b2a < tau)

    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def main():
    parser = argparse.ArgumentParser(
        description="Gate 0A-1: Encoder/Decoder Round-Trip Quality"
    )
    parser.add_argument("--mesh_dir", required=True,
                        help="Directory of test meshes (GLB/OBJ/PLY)")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for results")
    parser.add_argument("--trellis2_dir", default="/workspace/TRELLIS.2")
    parser.add_argument("--model_dir", default="/workspace/models/trellis2-4b")
    parser.add_argument("--max_meshes", type=int, default=50)
    parser.add_argument("--n_samples", type=int, default=10000,
                        help="Points to sample for Chamfer/F-score")
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find test meshes
    mesh_exts = {".glb", ".obj", ".ply", ".stl", ".off"}
    mesh_files = []
    for f in sorted(Path(args.mesh_dir).rglob("*")):
        if f.suffix.lower() in mesh_exts:
            mesh_files.append(f)
    mesh_files = mesh_files[:args.max_meshes]

    if not mesh_files:
        print(f"ERROR: No meshes found in {args.mesh_dir}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Gate 0A-1: Encoder/Decoder Round-Trip Quality")
    print(f"{'='*60}")
    print(f"  Test meshes: {len(mesh_files)}")
    print(f"  Output: {args.output_dir}")

    # Load TRELLIS.2
    print(f"\nLoading TRELLIS.2...")
    sys.path.insert(0, args.trellis2_dir)

    # Import TRELLIS.2 components
    # NOTE: The exact encoder/decoder API depends on TRELLIS.2's codebase.
    # This script provides the structure — adjust imports based on Gate 0C findings.
    try:
        from trellis2.pipelines import Trellis2ImageTo3DPipeline
        pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.model_dir)
        pipeline.cuda()
        print("  Pipeline loaded")
    except Exception as e:
        print(f"  ERROR loading TRELLIS.2: {e}")
        print("  This script requires TRELLIS.2 with encoder/decoder access.")
        print("  Adjust imports based on Gate 0C architecture inspection.")
        sys.exit(1)

    # Check if shape encoder is available
    has_encoder = hasattr(pipeline, "models") and any(
        "enc" in k.lower() for k in pipeline.models.keys()
    )
    if not has_encoder:
        print("\n  WARNING: No shape encoder found in pipeline.models")
        print("  Available models:", list(pipeline.models.keys()))
        print("  The encoder may be loaded separately — check TRELLIS.2 docs")
        print("  Listing model keys for reference:")
        for k, v in pipeline.models.items():
            print(f"    {k}: {type(v).__name__}")

    results = []
    failures = 0

    for i, mesh_path in enumerate(mesh_files):
        name = mesh_path.stem
        print(f"\n[{i+1}/{len(mesh_files)}] {name}")

        try:
            # Load original mesh
            original = trimesh.load(str(mesh_path), force="mesh")
            if len(original.vertices) == 0:
                print(f"  Empty mesh, skipping")
                continue

            # Normalize to unit cube
            bounds = original.bounds
            center = (bounds[0] + bounds[1]) / 2
            scale = np.max(bounds[1] - bounds[0])
            original.vertices = (original.vertices - center) / scale

            t0 = time.time()

            # === ENCODE → DECODE ROUND-TRIP ===
            # NOTE: The exact API calls depend on TRELLIS.2's encoder.
            # This is a placeholder structure — fill in after Gate 0C.
            #
            # Expected flow:
            #   1. mesh → o_voxel → sparse voxel representation
            #   2. sparse voxels → shape_encoder → SLAT (N, 32)
            #   3. SLAT → shape_decoder → O-Voxel → mesh
            #
            # If TRELLIS.2's encoder isn't directly accessible via the pipeline,
            # you may need to load it separately:
            #   from trellis2.models import ShapeEncoder
            #   encoder = ShapeEncoder.from_pretrained(...)

            # Placeholder — replace with actual encoder/decoder calls:
            print(f"  TODO: Implement encode/decode round-trip")
            print(f"  Need Gate 0C to determine encoder API")

            elapsed = time.time() - t0

            # For now, record as skipped
            results.append({
                "name": name,
                "status": "needs_implementation",
                "elapsed_sec": round(elapsed, 1),
            })

        except Exception as e:
            print(f"  FAILED: {e}")
            failures += 1
            results.append({
                "name": name,
                "status": "error",
                "error": str(e),
            })

    # Summary
    summary = {
        "gate": "0A-1",
        "description": "Encoder/Decoder Round-Trip Quality",
        "total_meshes": len(mesh_files),
        "failures": failures,
        "results": results,
    }

    summary_path = output_path / "gate_0a1_results.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Gate 0A-1 Results")
    print(f"{'='*60}")
    print(f"  Total: {len(mesh_files)}")
    print(f"  Failures: {failures}")
    print(f"  Results: {summary_path}")
    print(f"\nNOTE: This script needs Gate 0C results to determine the")
    print(f"exact encoder/decoder API. Run inspect_trellis_dit.py first.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
