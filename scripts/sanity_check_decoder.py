#!/usr/bin/env python3
"""Sanity check: compare original vs fine-tuned decoder on clean and noised SLAT.

Uses GUIDED subdivisions (from original decoder on clean input) for all tests,
ensuring features are spatially aligned for fair comparison. This matches how
the fine-tuning was done (teacher subdivisions guide student).

For 5 SLAT samples:
  1. Orig decoder + clean SLAT → reference features + subdivisions
  2. FT decoder + clean SLAT (guided subs) → clean preservation check
  3. Orig decoder + noised SLAT (guided subs) → noise baseline
  4. FT decoder + noised SLAT (guided subs) → noise robustness check

Pass criteria:
  A) Clean preservation: cosine sim > 0.99, degradation ratio < 0.10
  B) Noise robustness: FT-noised closer to reference than orig-noised in ≥60%
"""

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def load_decoder(model_dir):
    pipeline_json = os.path.join(model_dir, "pipeline.json")
    with open(pipeline_json, 'r') as f:
        config = json.load(f)
    model_paths = config.get('args', {}).get('models', {})
    decoder_rel = model_paths['shape_slat_decoder']
    decoder_path = os.path.join(model_dir, decoder_rel)
    from trellis2 import models as trellis_models
    decoder = trellis_models.from_pretrained(decoder_path)
    print(f"  Loaded decoder from {decoder_rel}")
    print(f"  Type: {decoder.__class__.__name__}")
    return decoder


def make_sparse_tensor(feats_np, positions_np, sp_module, device="cuda"):
    feats = torch.from_numpy(feats_np.astype(np.float32)).to(device)
    positions = torch.from_numpy(positions_np.astype(np.float32)).to(device)
    N = feats.shape[0]
    batch_idx = torch.zeros(N, 1, dtype=torch.int32, device=device)
    coords = torch.cat([batch_idx, positions.int()], dim=1)
    return sp_module.SparseTensor(feats=feats, coords=coords)


def decoder_forward_guided(decoder, sparse_input, guide_subs=None):
    """Forward through base decoder with optional guided subdivisions.

    Replicates the fine-tuning code's _decoder_forward_guided logic.
    When guide_subs is None: collects subdivision decisions (teacher mode).
    When guide_subs is provided: uses teacher's subdivisions (student mode).

    Returns (output_feats_tensor, subs_list).
    """
    h = decoder.from_latent(sparse_input)
    h = h.type(decoder.dtype)
    subs = []
    sub_idx = 0

    for i, res in enumerate(decoder.blocks):
        for j, block in enumerate(res):
            is_upsample = (i < len(decoder.blocks) - 1 and j == len(res) - 1)
            if is_upsample:
                if guide_subs is not None:
                    # Student mode: use teacher's subdivision decisions
                    teacher_sub = guide_subs[sub_idx]
                    sub_idx += 1
                    original_to_subdiv = block.to_subdiv
                    object.__setattr__(
                        block, "to_subdiv",
                        lambda x, _s=teacher_sub: _s,
                    )
                    try:
                        h, _ = block(h)
                    finally:
                        object.__setattr__(
                            block, "to_subdiv", original_to_subdiv,
                        )
                else:
                    # Teacher mode: collect subdivision decisions
                    h, sub = block(h)
                    subs.append(sub)
            else:
                h = block(h)

    h = h.type(sparse_input.dtype)
    h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
    h = decoder.output_layer(h)
    return h.feats, subs


def compare_features(ref_feats, test_feats):
    """Compare two aligned feature tensors."""
    assert ref_feats.shape == test_feats.shape, \
        f"Shape mismatch: {ref_feats.shape} vs {test_feats.shape}"
    ref = ref_feats.float()
    test = test_feats.float()

    l2_per_token = (ref - test).norm(dim=-1)
    cos_sim = F.cosine_similarity(ref, test, dim=-1)
    mse = F.mse_loss(test, ref).item()

    # Decompose: vertices [0:3], intersection [3:6], quad_lerp [6:7]
    vertex_l2 = (ref[..., :3] - test[..., :3]).norm(dim=-1).mean().item()
    intersect_l2 = (ref[..., 3:6] - test[..., 3:6]).norm(dim=-1).mean().item()
    quad_l2 = (ref[..., 6:7] - test[..., 6:7]).norm(dim=-1).mean().item()

    ref_intersect = (ref[..., 3:6] > 0).float()
    test_intersect = (test[..., 3:6] > 0).float()
    intersect_agreement = (ref_intersect == test_intersect).float().mean().item()

    return {
        "l2_mean": round(l2_per_token.mean().item(), 6),
        "l2_std": round(l2_per_token.std().item(), 6),
        "cosine_sim": round(cos_sim.mean().item(), 6),
        "cosine_sim_min": round(cos_sim.min().item(), 6),
        "mse": round(mse, 6),
        "vertex_l2": round(vertex_l2, 6),
        "intersect_l2": round(intersect_l2, 6),
        "quad_lerp_l2": round(quad_l2, 6),
        "intersect_agreement": round(intersect_agreement, 4),
        "n_output_tokens": ref_feats.shape[0],
    }


def main():
    parser = argparse.ArgumentParser(description="Decoder fine-tuning sanity check")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--finetuned_weights", required=True)
    parser.add_argument("--trellis2_dir", default="/workspace/TRELLIS.2")
    parser.add_argument("--model_dir", default="/workspace/models/trellis2-4b")
    parser.add_argument("--max_samples", type=int, default=5)
    parser.add_argument("--noise_sigma", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Decoder Fine-Tuning Sanity Check (guided subdivisions)")
    print(f"{'='*60}")
    print(f"  Fine-tuned weights: {args.finetuned_weights}")
    print(f"  Noise σ: {args.noise_sigma}")
    print(f"  Max samples: {args.max_samples}")

    sys.path.insert(0, args.trellis2_dir)

    print(f"\nLoading decoder...")
    decoder = load_decoder(args.model_dir)
    if hasattr(decoder, "convert_to_fp32"):
        decoder.convert_to_fp32()
        decoder.use_fp16 = False
        decoder.dtype = torch.float32
        print("  Converted to fp32")
    decoder = decoder.cuda()

    from trellis2.modules.sparse import basic as sp_module

    orig_state_dict = copy.deepcopy(decoder.state_dict())

    print(f"\nLoading fine-tuned weights...")
    ft_ckpt = torch.load(args.finetuned_weights, map_location="cuda", weights_only=False)
    ft_state_dict = ft_ckpt["decoder"]
    print(f"  From step {ft_ckpt.get('step', '?')}")

    decoder.load_state_dict(ft_state_dict)
    decoder.load_state_dict(orig_state_dict)
    print("  Both weight sets verified")

    # Discover SLAT pairs
    data_dir = Path(args.data_dir)
    pairs = []
    for sd in [data_dir] + sorted(data_dir.glob("shard_*")):
        if not sd.is_dir():
            continue
        for d in sorted(sd.iterdir()):
            if not d.is_dir():
                continue
            if (d / "fine_slat.npy").exists() and (d / "positions.npy").exists():
                pairs.append(d)
    pairs = pairs[:args.max_samples]

    if not pairs:
        print(f"ERROR: No SLAT pairs in {args.data_dir}")
        sys.exit(1)
    print(f"  Found {len(pairs)} SLAT samples")

    results = []
    for i, pair_dir in enumerate(pairs):
        name = pair_dir.name
        print(f"\n[{i+1}/{len(pairs)}] {name}")

        try:
            fine_slat = np.load(pair_dir / "fine_slat.npy")
            positions = np.load(pair_dir / "positions.npy")
            n_tokens = fine_slat.shape[0]
            print(f"  Input: {n_tokens} tokens × {fine_slat.shape[1]} dims")

            clean_sparse = make_sparse_tensor(fine_slat, positions, sp_module)

            # === Test 1: Original decoder, clean SLAT → reference + subdivisions ===
            print(f"  [1/4] Original decoder, clean SLAT → reference + subs...")
            decoder.load_state_dict(orig_state_dict)
            decoder.eval()
            decoder.set_resolution(512)
            with torch.no_grad():
                ref_feats, ref_subs = decoder_forward_guided(decoder, clean_sparse, guide_subs=None)
            n_out = ref_feats.shape[0]
            n_subs = len(ref_subs)
            print(f"    Output: {n_out} tokens × {ref_feats.shape[1]} dims, {n_subs} subdivision levels")

            # === Test 2: Fine-tuned decoder, clean SLAT (guided by ref subs) ===
            print(f"  [2/4] Fine-tuned decoder, clean SLAT (guided subs)...")
            decoder.load_state_dict(ft_state_dict)
            decoder.eval()
            decoder.set_resolution(512)
            with torch.no_grad():
                ft_clean_feats, _ = decoder_forward_guided(decoder, clean_sparse, guide_subs=ref_subs)
            ft_clean_metrics = compare_features(ref_feats, ft_clean_feats)
            print(f"    L2: {ft_clean_metrics['l2_mean']:.6f}, "
                  f"Cos: {ft_clean_metrics['cosine_sim']:.6f}, "
                  f"Intersect: {ft_clean_metrics['intersect_agreement']:.4f}")

            # === Create noised SLAT ===
            torch.manual_seed(args.seed + i)
            noise = torch.randn_like(clean_sparse.feats) * args.noise_sigma
            noised_feats = clean_sparse.feats + noise
            noised_sparse = clean_sparse.replace(noised_feats)
            input_noise_l2 = noise.norm(dim=-1).mean().item()
            print(f"  Noise: σ={args.noise_sigma}, mean input L2/token={input_noise_l2:.4f}")

            # === Test 3: Original decoder, noised SLAT (guided by ref subs) ===
            print(f"  [3/4] Original decoder, noised SLAT (guided subs)...")
            decoder.load_state_dict(orig_state_dict)
            decoder.eval()
            decoder.set_resolution(512)
            with torch.no_grad():
                orig_noised_feats, _ = decoder_forward_guided(decoder, noised_sparse, guide_subs=ref_subs)
            orig_noised_metrics = compare_features(ref_feats, orig_noised_feats)
            print(f"    L2: {orig_noised_metrics['l2_mean']:.6f}, "
                  f"Cos: {orig_noised_metrics['cosine_sim']:.6f}, "
                  f"Intersect: {orig_noised_metrics['intersect_agreement']:.4f}")

            # === Test 4: Fine-tuned decoder, noised SLAT (guided by ref subs) ===
            print(f"  [4/4] Fine-tuned decoder, noised SLAT (guided subs)...")
            decoder.load_state_dict(ft_state_dict)
            decoder.eval()
            decoder.set_resolution(512)
            with torch.no_grad():
                ft_noised_feats, _ = decoder_forward_guided(decoder, noised_sparse, guide_subs=ref_subs)
            ft_noised_metrics = compare_features(ref_feats, ft_noised_feats)
            print(f"    L2: {ft_noised_metrics['l2_mean']:.6f}, "
                  f"Cos: {ft_noised_metrics['cosine_sim']:.6f}, "
                  f"Intersect: {ft_noised_metrics['intersect_agreement']:.4f}")

            # === Compute improvement ===
            l2_improvement = (orig_noised_metrics['l2_mean'] - ft_noised_metrics['l2_mean']) / \
                             (orig_noised_metrics['l2_mean'] + 1e-8) * 100
            cos_improvement = ft_noised_metrics['cosine_sim'] - orig_noised_metrics['cosine_sim']
            clean_deg_ratio = ft_clean_metrics['l2_mean'] / (orig_noised_metrics['l2_mean'] + 1e-8)

            result = {
                "name": name,
                "n_input_tokens": n_tokens,
                "n_output_tokens": n_out,
                "noise_sigma": args.noise_sigma,
                "input_noise_l2": round(input_noise_l2, 4),
                "ft_clean_vs_ref": ft_clean_metrics,
                "orig_noised_vs_ref": orig_noised_metrics,
                "ft_noised_vs_ref": ft_noised_metrics,
                "l2_noise_improvement_pct": round(l2_improvement, 1),
                "cos_noise_improvement": round(cos_improvement, 6),
                "clean_degradation_ratio": round(clean_deg_ratio, 4),
                "ft_noised_better_l2": ft_noised_metrics['l2_mean'] < orig_noised_metrics['l2_mean'],
                "ft_noised_better_cos": ft_noised_metrics['cosine_sim'] > orig_noised_metrics['cosine_sim'],
            }
            results.append(result)

            print(f"\n  --- Summary ---")
            print(f"  Clean degradation ratio: {clean_deg_ratio:.4f} (lower = better)")
            print(f"  Noise L2 improvement: {l2_improvement:+.1f}%")
            print(f"  FT-noised better (L2):  {'YES' if result['ft_noised_better_l2'] else 'NO'}")
            print(f"  FT-noised better (cos): {'YES' if result['ft_noised_better_cos'] else 'NO'}")

            del clean_sparse, noised_sparse, ref_feats, ft_clean_feats
            del orig_noised_feats, ft_noised_feats
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append({"name": name, "status": "error", "error": str(e)})

    decoder.load_state_dict(orig_state_dict)

    # === Summary ===
    valid = [r for r in results if "ft_clean_vs_ref" in r]
    if not valid:
        print("\nERROR: No valid results!")
        sys.exit(1)

    clean_cos_list = [r["ft_clean_vs_ref"]["cosine_sim"] for r in valid]
    clean_l2_list = [r["ft_clean_vs_ref"]["l2_mean"] for r in valid]
    clean_deg_ratios = [r["clean_degradation_ratio"] for r in valid]
    clean_intersect = [r["ft_clean_vs_ref"]["intersect_agreement"] for r in valid]

    n_better_l2 = sum(1 for r in valid if r["ft_noised_better_l2"])
    n_better_cos = sum(1 for r in valid if r["ft_noised_better_cos"])
    l2_improvements = [r["l2_noise_improvement_pct"] for r in valid]
    noised_orig_l2 = [r["orig_noised_vs_ref"]["l2_mean"] for r in valid]
    noised_ft_l2 = [r["ft_noised_vs_ref"]["l2_mean"] for r in valid]
    noised_orig_cos = [r["orig_noised_vs_ref"]["cosine_sim"] for r in valid]
    noised_ft_cos = [r["ft_noised_vs_ref"]["cosine_sim"] for r in valid]

    mean_clean_cos = float(np.mean(clean_cos_list))
    mean_clean_deg = float(np.mean(clean_deg_ratios))
    a_pass = mean_clean_cos > 0.99 and mean_clean_deg < 0.10
    b_pass = n_better_l2 >= len(valid) * 0.6

    summary = {
        "test": "decoder_finetune_sanity_check",
        "noise_sigma": args.noise_sigma,
        "n_samples": len(valid),
        "finetuned_weights": args.finetuned_weights,
        "note": "Uses guided subdivisions from orig decoder for fair comparison",
        "metrics": {
            "mean_clean_cosine_sim": round(mean_clean_cos, 6),
            "mean_clean_l2": round(float(np.mean(clean_l2_list)), 6),
            "mean_clean_degradation_ratio": round(mean_clean_deg, 4),
            "mean_clean_intersect_agreement": round(float(np.mean(clean_intersect)), 4),
            "mean_noise_l2_orig": round(float(np.mean(noised_orig_l2)), 6),
            "mean_noise_l2_ft": round(float(np.mean(noised_ft_l2)), 6),
            "mean_noise_cos_orig": round(float(np.mean(noised_orig_cos)), 6),
            "mean_noise_cos_ft": round(float(np.mean(noised_ft_cos)), 6),
            "mean_noise_l2_improvement_pct": round(float(np.mean(l2_improvements)), 1),
            "ft_noised_better_l2_count": f"{n_better_l2}/{len(valid)}",
            "ft_noised_better_cos_count": f"{n_better_cos}/{len(valid)}",
        },
        "criteria": {
            "A_clean_preserved": bool(a_pass),
            "B_noise_robustness": bool(b_pass),
        },
        "per_sample": results,
    }

    summary_path = output_path / "sanity_check_results.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"SANITY CHECK RESULTS (guided subdivisions)")
    print(f"{'='*60}")

    print(f"\n  Criterion A — Clean input preservation:")
    print(f"    Mean cosine sim (FT-clean vs orig-clean): {mean_clean_cos:.6f} (> 0.99 = PASS)")
    print(f"    Mean L2 (FT-clean vs orig-clean): {np.mean(clean_l2_list):.6f}")
    print(f"    Mean L2 (orig-noised vs orig-clean): {np.mean(noised_orig_l2):.6f}")
    print(f"    Mean degradation ratio: {mean_clean_deg:.4f} (< 0.10 = PASS)")
    print(f"    Mean intersection agreement: {np.mean(clean_intersect):.4f}")
    print(f"    {'PASS' if a_pass else 'FAIL'}")

    print(f"\n  Criterion B — Noise robustness:")
    print(f"    Mean L2 (orig-noised vs ref): {np.mean(noised_orig_l2):.6f}")
    print(f"    Mean L2 (FT-noised vs ref):   {np.mean(noised_ft_l2):.6f}")
    print(f"    Mean cos (orig-noised): {np.mean(noised_orig_cos):.6f}")
    print(f"    Mean cos (FT-noised):   {np.mean(noised_ft_cos):.6f}")
    print(f"    Mean L2 improvement: {np.mean(l2_improvements):+.1f}%")
    print(f"    FT better (L2): {n_better_l2}/{len(valid)} (>= 60% = PASS)")
    print(f"    FT better (cos): {n_better_cos}/{len(valid)}")
    print(f"    {'PASS' if b_pass else 'FAIL'}")

    passed = a_pass and b_pass
    print(f"\n  OVERALL: {'PASS — decoder fine-tuning validated!' if passed else 'FAIL — review needed'}")
    print(f"\n  Results: {summary_path}")
    print(f"{'='*60}\n")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
