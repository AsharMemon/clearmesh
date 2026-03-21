#!/usr/bin/env python3
"""Gate 0D-2: Alignment Noise Severity.

Tests whether nearest-neighbor projection from encoder positions to coarse
positions introduces acceptable or fatal noise.

For each test mesh (generated from a test image via TRELLIS.2):
  1. Generate coarse SLAT at 512 → positions (N, 3), features (N, 32)
  2. Generate fine SLAT at 1024 (same positions, same features) → reference
  3. Encode the decoded fine mesh through shape encoder
     → encoder SLAT at encoder's own positions (M, 32)
  4. NN-project encoder SLAT onto coarse positions → projected SLAT (N, 32)
  5. Decode projected SLAT → projected mesh
  6. Measure:
     - Distance from each coarse position to its NN-matched encoder position
     - Feature mismatch between projected SLAT and 1024 SLAT (at same positions)
     - Geometry error of projected mesh vs fine reference mesh
     - Correlation between NN distance and local feature/geometry error

Questions answered:
  - Is NN projection mildly noisy or fundamentally bad?
  - Does projection noise correlate with spatial NN distance?
  - Should we use 1024 diffusion targets (no alignment) or encoder targets (with NN)?

Usage:
    python scripts/data/validate_alignment_noise.py \\
        --image_dir /workspace/data/test_images \\
        --output_dir /workspace/data/gate_0d2_results \\
        --trellis2_dir /workspace/TRELLIS.2 \\
        --model_dir /workspace/models/trellis2-4b
"""

import argparse
import json
import sys
import traceback
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


def extract_mesh(mesh_obj) -> trimesh.Trimesh:
    """Convert TRELLIS.2 mesh object to trimesh."""
    v = mesh_obj.vertices.detach().cpu().float().numpy()
    f = mesh_obj.faces.detach().cpu().numpy()
    return trimesh.Trimesh(vertices=v, faces=f)


def encode_mesh_to_slat(encoder, mesh: trimesh.Trimesh, device, grid_size=256):
    """Voxelize mesh and encode through shape encoder.

    Returns:
        enc_feats: (M, 32) float — SLAT features at encoder positions
        enc_coords: (M, 3) float — encoder output positions (in downsampled grid)
        n_voxels: int — number of input voxels before encoding
    """
    from trellis2.modules import sparse as sp
    from o_voxel.convert.flexible_dual_grid import _C

    verts = torch.from_numpy(mesh.vertices.astype("float32"))
    faces = torch.from_numpy(mesh.faces.astype("int32"))

    # Compute aabb and voxel_size on CPU
    min_xyz = verts.min(dim=0).values
    max_xyz = verts.max(dim=0).values
    gs = torch.tensor([grid_size] * 3, dtype=torch.int32)
    padding = (max_xyz - min_xyz) / (gs.float() - 1)
    min_xyz = min_xyz - padding * 0.5
    max_xyz = max_xyz + padding * 0.5
    aabb = torch.stack([min_xyz, max_xyz], dim=0).float()
    voxel_size = (aabb[1] - aabb[0]) / gs.float()

    vertices_shifted = verts - aabb[0].reshape(1, 3)
    grid_range = torch.stack([torch.zeros_like(gs), gs], dim=0).int()

    # Voxelize on CPU
    coords, dual_verts, intersected = _C.mesh_to_flexible_dual_grid_cpu(
        vertices_shifted, faces, voxel_size, grid_range,
        1.0, 1.0, 0.1, False,
    )

    n_voxels = coords.shape[0]
    if n_voxels == 0:
        return None, None, 0

    # Create SparseTensors on GPU
    batch_idx = torch.zeros(n_voxels, 1, dtype=torch.int32)
    coords_4d = torch.cat([batch_idx, coords.int()], dim=1).to(device)
    vst = sp.SparseTensor(feats=dual_verts.float().to(device), coords=coords_4d)
    ist = sp.SparseTensor(feats=intersected.float().to(device), coords=coords_4d)

    # Encode
    with torch.no_grad():
        z = encoder(vst, ist, sample_posterior=False)

    enc_feats = z.feats.float().cpu()          # (M, 32)
    enc_coords = z.coords[:, 1:].float().cpu() # (M, 3)

    del vst, ist, z, coords_4d
    torch.cuda.empty_cache()

    return enc_feats, enc_coords, n_voxels


def nn_project(coarse_positions, enc_coords, enc_feats):
    """Nearest-neighbor project encoder features onto coarse positions.

    Args:
        coarse_positions: (N, 3) — coarse voxel positions
        enc_coords: (M, 3) — encoder output positions
        enc_feats: (M, 32) — encoder output features

    Returns:
        projected_feats: (N, 32) — features at coarse positions
        nn_distances: (N,) — normalized distance from each coarse pos to its NN
    """
    N = coarse_positions.shape[0]
    M = enc_coords.shape[0]

    # Min-max normalize each coordinate set to [0, 1]³
    def normalize(c):
        cmin = c.min(dim=0).values
        cmax = c.max(dim=0).values
        spread = (cmax - cmin).clamp(min=1e-6)
        return (c - cmin) / spread

    coarse_norm = normalize(coarse_positions)
    enc_norm = normalize(enc_coords)

    # Compute NN
    if N * M < 50_000_000:
        dists = torch.cdist(coarse_norm.unsqueeze(0), enc_norm.unsqueeze(0)).squeeze(0)
        nn_indices = dists.argmin(dim=1)
        nn_dists = dists[torch.arange(N), nn_indices]
    else:
        nn_indices = torch.zeros(N, dtype=torch.long)
        nn_dists = torch.zeros(N)
        chunk = max(1, 50_000_000 // M)
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            d = torch.cdist(coarse_norm[s:e].unsqueeze(0), enc_norm.unsqueeze(0)).squeeze(0)
            nn_indices[s:e] = d.argmin(dim=1)
            nn_dists[s:e] = d[torch.arange(e - s), nn_indices[s:e]]

    projected_feats = enc_feats[nn_indices]
    return projected_feats, nn_dists


def main():
    parser = argparse.ArgumentParser(
        description="Gate 0D-2: Alignment Noise Severity (encoder NN projection test)"
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
    print(f"Gate 0D-2: Alignment Noise Severity")
    print(f"{'='*60}")
    print(f"  Image dir: {args.image_dir}")
    print(f"  Output: {args.output_dir}")
    print(f"  Max samples: {args.max_samples}")
    print(f"  Question: Is NN projection from encoder positions mildly noisy")
    print(f"            or fundamentally bad?")

    # Load TRELLIS.2
    print(f"\nLoading TRELLIS.2 pipeline...")
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

    # Load shape encoder separately (not part of pipeline by default)
    print(f"Loading shape encoder...")
    try:
        from safetensors.torch import load_file
        from trellis2.models.sc_vaes.fdg_vae import FlexiDualGridVaeEncoder

        enc_cfg_path = Path(args.model_dir) / "ckpts" / "shape_enc_next_dc_f16c32_fp16.json"
        enc_wt_path = Path(args.model_dir) / "ckpts" / "shape_enc_next_dc_f16c32_fp16.safetensors"

        with open(enc_cfg_path) as f:
            enc_cfg = json.load(f)
        encoder = FlexiDualGridVaeEncoder(**enc_cfg["args"])
        encoder.load_state_dict(load_file(str(enc_wt_path)))
        encoder = encoder.cuda().eval()
        print("  Shape encoder loaded")
    except Exception as e:
        print(f"  ERROR loading encoder: {e}")
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
    device = torch.device("cuda")

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

            # === Step 1: Generate coarse + fine SLAT ===
            print(f"  Generating coarse (512) + fine (1024) SLAT...")
            cond_512 = pipeline.get_cond([processed], 512)
            coords = pipeline.sample_sparse_structure(
                cond_512, 32, 1, {"steps": 12, "guidance_strength": 9.0},
            )

            shape_slat_coarse = pipeline.sample_shape_slat(
                cond_512,
                pipeline.models["shape_slat_flow_model_512"],
                coords,
                {"steps": 12, "guidance_strength": 4.5},
            )
            coarse_feats = shape_slat_coarse.feats.cpu().numpy()
            coarse_positions = shape_slat_coarse.coords[:, 1:].cpu().float()
            N = coarse_feats.shape[0]

            cond_1024 = pipeline.get_cond([processed], 1024)
            shape_slat_fine = pipeline.sample_shape_slat(
                cond_1024,
                pipeline.models["shape_slat_flow_model_1024"],
                coords,
                {"steps": 12, "guidance_strength": 4.5},
            )
            fine_feats = shape_slat_fine.feats.cpu().numpy()
            print(f"  Coarse/Fine SLAT: N={N}, dim=32")

            # Decode fine → reference mesh
            with torch.no_grad():
                meshes_fine, _ = pipeline.decode_shape_slat(shape_slat_fine, 512)
            tm_fine = extract_mesh(meshes_fine[0])
            tm_fine.export(str(sample_dir / "mesh_fine_1024.glb"))

            # Decode coarse → baseline mesh
            with torch.no_grad():
                meshes_coarse, _ = pipeline.decode_shape_slat(shape_slat_coarse, 512)
            tm_coarse = extract_mesh(meshes_coarse[0])
            tm_coarse.export(str(sample_dir / "mesh_coarse_512.glb"))

            # === Step 2: Encode the fine mesh through shape encoder ===
            print(f"  Encoding fine mesh through shape encoder...")
            enc_feats, enc_coords, n_voxels = encode_mesh_to_slat(
                encoder, tm_fine, device,
            )
            if enc_feats is None:
                print(f"  SKIP: encoder returned 0 voxels")
                results.append({"name": name, "status": "error", "error": "encoder 0 voxels"})
                continue

            M = enc_feats.shape[0]
            print(f"  Encoder: {n_voxels} input voxels → {M} output tokens")

            # === Step 3: NN-project encoder SLAT onto coarse positions ===
            print(f"  NN-projecting encoder SLAT onto coarse positions...")
            projected_feats, nn_dists = nn_project(
                coarse_positions, enc_coords, enc_feats,
            )
            print(f"  NN distances: mean={nn_dists.mean():.4f}, "
                  f"max={nn_dists.max():.4f}, "
                  f"std={nn_dists.std():.4f}")

            # Save projected SLAT
            np.save(sample_dir / "projected_slat.npy", projected_feats.numpy())
            np.save(sample_dir / "nn_distances.npy", nn_dists.numpy())

            # === Step 4: Decode projected SLAT → projected mesh ===
            print(f"  Decoding projected mesh...")
            projected_slat_obj = shape_slat_coarse.replace(
                projected_feats.to(shape_slat_coarse.feats.device).to(
                    shape_slat_coarse.feats.dtype
                )
            )
            with torch.no_grad():
                meshes_proj, _ = pipeline.decode_shape_slat(projected_slat_obj, 512)
            tm_proj = extract_mesh(meshes_proj[0])
            tm_proj.export(str(sample_dir / "mesh_projected.glb"))
            print(f"  Projected mesh: {len(tm_proj.vertices)} verts")

            # === Step 5: Compute metrics ===
            print(f"  Computing metrics...")

            # Chamfer distances
            chamfer_coarse = chamfer_distance(tm_coarse, tm_fine)
            chamfer_proj = chamfer_distance(tm_proj, tm_fine)
            chamfer_1024_oracle = 0.0  # 1024 SLAT at same positions = self
            # Direct oracle: fine SLAT decoded = fine mesh ≈ 0 Chamfer

            improvement_proj = (chamfer_coarse - chamfer_proj) / (chamfer_coarse + 1e-8) * 100

            # Feature mismatch: projected SLAT vs 1024 SLAT (both at coarse positions)
            feat_diff = projected_feats.numpy().astype(np.float32) - fine_feats.astype(np.float32)
            feat_mismatch_per_token = np.linalg.norm(feat_diff, axis=1)  # (N,)
            feat_mismatch_mean = float(feat_mismatch_per_token.mean())
            feat_mismatch_std = float(feat_mismatch_per_token.std())

            # Correlation: NN distance vs feature mismatch
            nn_dists_np = nn_dists.numpy()
            if nn_dists_np.std() > 1e-8 and feat_mismatch_per_token.std() > 1e-8:
                corr = float(np.corrcoef(nn_dists_np, feat_mismatch_per_token)[0, 1])
            else:
                corr = 0.0

            # Binned analysis: split positions into distance quintiles
            quartiles = np.percentile(nn_dists_np, [25, 50, 75])
            bins = [
                ("Q1 (nearest)", nn_dists_np <= quartiles[0]),
                ("Q2", (nn_dists_np > quartiles[0]) & (nn_dists_np <= quartiles[1])),
                ("Q3", (nn_dists_np > quartiles[1]) & (nn_dists_np <= quartiles[2])),
                ("Q4 (farthest)", nn_dists_np > quartiles[2]),
            ]
            binned_mismatch = {}
            for label, mask in bins:
                if mask.sum() > 0:
                    binned_mismatch[label] = {
                        "count": int(mask.sum()),
                        "mean_nn_dist": float(nn_dists_np[mask].mean()),
                        "mean_feat_mismatch": float(feat_mismatch_per_token[mask].mean()),
                    }

            result = {
                "name": name,
                "n_coarse": N,
                "n_encoder": M,
                "n_voxels": n_voxels,
                "nn_distance": {
                    "mean": float(nn_dists_np.mean()),
                    "std": float(nn_dists_np.std()),
                    "max": float(nn_dists_np.max()),
                    "p95": float(np.percentile(nn_dists_np, 95)),
                },
                "feature_mismatch": {
                    "mean": feat_mismatch_mean,
                    "std": feat_mismatch_std,
                    "max": float(feat_mismatch_per_token.max()),
                },
                "correlation_dist_vs_mismatch": corr,
                "binned_analysis": binned_mismatch,
                "chamfer_coarse_vs_fine": chamfer_coarse,
                "chamfer_projected_vs_fine": chamfer_proj,
                "improvement_projected_pct": round(improvement_proj, 1),
                "projected_better_than_coarse": chamfer_proj < chamfer_coarse,
            }
            results.append(result)

            print(f"  Chamfer coarse→fine:    {chamfer_coarse:.6f}")
            print(f"  Chamfer projected→fine: {chamfer_proj:.6f}")
            print(f"  Improvement:            {improvement_proj:.1f}%")
            print(f"  Feat mismatch (vs 1024): mean={feat_mismatch_mean:.3f}, std={feat_mismatch_std:.3f}")
            print(f"  Correlation (dist vs mismatch): {corr:.3f}")
            print(f"  Projected better: {'YES' if chamfer_proj < chamfer_coarse else 'NO'}")

            # Cleanup
            del shape_slat_coarse, shape_slat_fine, projected_slat_obj
            del meshes_coarse, meshes_fine, meshes_proj
            del cond_512, cond_1024, coords
            del enc_feats, enc_coords
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            results.append({"name": name, "status": "error", "error": str(e)})

    # === Summary ===
    valid = [r for r in results if "chamfer_projected_vs_fine" in r]

    n_proj_better = sum(1 for r in valid if r.get("projected_better_than_coarse", False))
    nn_dists_all = [r["nn_distance"]["mean"] for r in valid]
    corrs_all = [r["correlation_dist_vs_mismatch"] for r in valid]
    mismatches_all = [r["feature_mismatch"]["mean"] for r in valid]
    improvements_all = [r["improvement_projected_pct"] for r in valid]

    summary = {
        "gate": "0D-2",
        "description": "Alignment Noise Severity — is NN projection viable?",
        "total_samples": len(images),
        "valid_samples": len(valid),
        "projected_better_count": n_proj_better,
        "projected_better_pct": round(n_proj_better / max(len(valid), 1) * 100, 1),
        "mean_nn_distance": round(np.mean(nn_dists_all), 4) if nn_dists_all else None,
        "mean_feat_mismatch": round(np.mean(mismatches_all), 4) if mismatches_all else None,
        "mean_correlation": round(np.mean(corrs_all), 3) if corrs_all else None,
        "mean_improvement_pct": round(np.mean(improvements_all), 1) if improvements_all else None,
        "assessment": {
            "mildly_noisy": (
                np.mean(corrs_all) < 0.5 if corrs_all else False
            ),
            "fundamentally_bad": (
                n_proj_better / max(len(valid), 1) < 0.5 if valid else True
            ),
        },
        "results": results,
    }

    summary_path = output_path / "gate_0d2_results.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Gate 0D-2 Results")
    print(f"{'='*60}")
    print(f"  Total:                 {len(images)} ({len(valid)} valid)")
    print(f"  Projected better:      {n_proj_better}/{len(valid)} "
          f"({summary['projected_better_pct']}%)")
    if nn_dists_all:
        print(f"  Mean NN distance:      {np.mean(nn_dists_all):.4f}")
    if mismatches_all:
        print(f"  Mean feat mismatch:    {np.mean(mismatches_all):.4f}")
    if corrs_all:
        print(f"  Mean correlation:      {np.mean(corrs_all):.3f}")
    if improvements_all:
        print(f"  Mean improvement:      {np.mean(improvements_all):.1f}%")

    # Assessment
    if summary["assessment"]["fundamentally_bad"]:
        print(f"\n  VERDICT: NN projection is FUNDAMENTALLY BAD")
        print(f"           → Use 1024 diffusion targets (same positions, no alignment)")
    elif summary["assessment"]["mildly_noisy"]:
        print(f"\n  VERDICT: NN projection is MILDLY NOISY but viable")
        print(f"           → Could use encoder targets with some quality loss")
    else:
        print(f"\n  VERDICT: NN projection shows STRONG distance-mismatch correlation")
        print(f"           → Encoder targets need better alignment strategy")

    print(f"\n  Results saved to: {summary_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
