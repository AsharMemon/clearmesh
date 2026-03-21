#!/usr/bin/env python3
"""Fast SLAT pair generation — bypasses image preprocessing entirely.

Uses saved cond_features.npy from existing pairs as conditioning input,
avoiding the broken rembg pipeline. Generates new pairs by varying the
random seed, which changes the sparse structure and SLAT generation.

For each unique model × seed combination:
  1. Load cond_features.npy → construct conditioning dict
  2. Sample sparse structure (seed-dependent)
  3. Sample coarse SLAT (512 model)
  4. Sample fine SLAT (1024 model, same positions)
  5. Save pair

Supports dual-GPU by running two processes with different --gpu flags.

Usage:
    # GPU 0: seeds 112-361
    python generate_slat_pairs_fast.py \
        --source_dir /workspace/data/slat_pairs_3k \
        --output_dir /workspace/data/slat_pairs_30k \
        --start_seed 112 --num_seeds 250 --gpu 0

    # GPU 1: seeds 362-611
    python generate_slat_pairs_fast.py \
        --source_dir /workspace/data/slat_pairs_3k \
        --output_dir /workspace/data/slat_pairs_30k \
        --start_seed 362 --num_seeds 250 --gpu 1
"""

import argparse
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


# Generation params (matching existing pairs)
SPARSE_SAMPLER_PARAMS = {"steps": 12, "guidance_strength": 9.0}
SHAPE_SAMPLER_PARAMS = {"steps": 12, "guidance_strength": 4.5}
SS_RESOLUTION = 32


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def discover_unique_models(source_dir: Path) -> dict:
    """Find unique models by extracting cond_features from existing pairs.

    Returns dict: model_hash → path_to_cond_features.npy
    """
    models = {}
    for d in sorted(source_dir.iterdir()):
        if not d.is_dir():
            continue
        cf = d / "cond_features.npy"
        if not cf.exists():
            continue
        # Extract model hash (remove _seedN suffix)
        name = d.name
        # Handle both formats: "hash_seedN" and "name_seedN"
        parts = name.rsplit("_seed", 1)
        if len(parts) == 2:
            model_key = parts[0]
        else:
            model_key = name
        if model_key not in models:
            models[model_key] = cf
    return models


def load_pipeline_models(model_dir: str, device: str):
    """Load just the generation models (no image pipeline, no rembg)."""
    from trellis2 import models as trellis_models

    pipeline_json = os.path.join(model_dir, "pipeline.json")
    with open(pipeline_json) as f:
        config = json.load(f)

    model_paths = config['args']['models']
    needed = [
        'sparse_structure_decoder',
        'sparse_structure_flow_model',
        'shape_slat_flow_model_512',
        'shape_slat_flow_model_1024',
    ]

    models = {}
    for key in needed:
        rel_path = model_paths[key]
        # Resolve path: if starts with known HF prefix, keep as-is for HF download
        # Otherwise, prepend model_dir for local loading
        if rel_path.startswith("microsoft/") or rel_path.startswith("JeffreyXiang/"):
            full_path = rel_path
        else:
            full_path = os.path.join(model_dir, rel_path)
        models[key] = trellis_models.from_pretrained(full_path)
        models[key].to(device)
        models[key].eval()
        print(f"  Loaded {key}")

    return models, config['args']


def build_cond_dict(cond_features: np.ndarray, resolution: int, device: str) -> dict:
    """Build conditioning dict from saved DINOv2 features."""
    cond = torch.from_numpy(cond_features.astype(np.float32)).unsqueeze(0).to(device)
    neg_cond = torch.zeros_like(cond)
    return {'cond': cond, 'neg_cond': neg_cond}


def create_samplers(pipeline_args: dict):
    """Create samplers matching the pipeline configuration."""
    from trellis2.pipelines import samplers as sampler_module

    ss_cfg = pipeline_args.get('sparse_structure_sampler', {})
    ss_cls = getattr(sampler_module, ss_cfg.get('name', 'FlowEulerGuidanceIntervalSampler'))
    ss_sampler = ss_cls(**ss_cfg.get('args', {'sigma_min': 1e-5}))
    ss_params = {**ss_cfg.get('params', {}), **SPARSE_SAMPLER_PARAMS}

    slat_cfg = pipeline_args.get('shape_slat_sampler', {})
    slat_cls = getattr(sampler_module, slat_cfg.get('name', 'FlowEulerGuidanceIntervalSampler'))
    slat_sampler = slat_cls(**slat_cfg.get('args', {'sigma_min': 1e-5}))
    slat_params = {**slat_cfg.get('params', {}), **SHAPE_SAMPLER_PARAMS}

    return ss_sampler, ss_params, slat_sampler, slat_params


def generate_pair(
    models: dict,
    ss_sampler, ss_params: dict,
    slat_sampler, slat_params: dict,
    cond: dict,
    seed: int,
    device: str,
) -> dict:
    """Generate a coarse/fine SLAT pair from conditioning features."""
    torch.manual_seed(seed)

    # 1. Sample sparse structure
    flow_model = models['sparse_structure_flow_model']
    reso = flow_model.resolution
    in_channels = flow_model.in_channels
    noise = torch.randn(1, in_channels, reso, reso, reso, device=device)

    z_s = ss_sampler.sample(
        flow_model, noise, **cond, **ss_params,
        verbose=False,
    ).samples

    decoder = models['sparse_structure_decoder']
    decoded = decoder(z_s) > 0
    if SS_RESOLUTION != decoded.shape[2]:
        ratio = decoded.shape[2] // SS_RESOLUTION
        decoded = torch.nn.functional.max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5
    coords = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()

    if coords.shape[0] == 0:
        return None

    # 2. Sample coarse SLAT (512)
    from trellis2.modules.sparse import basic as sp

    fm_512 = models['shape_slat_flow_model_512']
    in_ch = fm_512.in_channels
    noise_512 = torch.randn(coords.shape[0], in_ch, device=device)
    noise_sparse = sp.SparseTensor(feats=noise_512, coords=coords)

    coarse_slat = slat_sampler.sample(
        fm_512, noise_sparse, **cond, **slat_params,
        verbose=False,
    ).samples

    # 3. Sample fine SLAT (1024, same positions)
    fm_1024 = models['shape_slat_flow_model_1024']
    in_ch_1024 = fm_1024.in_channels
    noise_1024 = torch.randn(coords.shape[0], in_ch_1024, device=device)
    noise_sparse_1024 = sp.SparseTensor(feats=noise_1024, coords=coords)

    fine_slat = slat_sampler.sample(
        fm_1024, noise_sparse_1024, **cond, **slat_params,
        verbose=False,
    ).samples

    # Extract numpy arrays
    positions = coords[:, 1:].cpu().numpy().astype(np.int32)
    coarse_np = coarse_slat.feats.cpu().numpy().astype(np.float16)
    fine_np = fine_slat.feats.cpu().numpy().astype(np.float16)

    if coarse_np.shape != fine_np.shape:
        return None

    return {
        'positions': positions,
        'coarse_slat': coarse_np,
        'fine_slat': fine_np,
        'n_points': positions.shape[0],
    }


def save_pair(pair: dict, uid: str, output_dir: Path, cond_features: np.ndarray):
    out = output_dir / uid
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "positions.npy", pair['positions'])
    np.save(out / "coarse_slat.npy", pair['coarse_slat'])
    np.save(out / "fine_slat.npy", pair['fine_slat'])
    np.save(out / "cond_features.npy", cond_features)
    with open(out / "metadata.json", "w") as f:
        json.dump({"uid": uid, "n_points": int(pair['n_points'])}, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", required=True,
                        help="Dir with existing pairs (for cond_features)")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_dir", default="/workspace/models/trellis2-4b")
    parser.add_argument("--trellis2_dir", default="/workspace/TRELLIS.2")
    parser.add_argument("--start_seed", type=int, default=112)
    parser.add_argument("--num_seeds", type=int, default=250)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    sys.path.insert(0, args.trellis2_dir)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Fast SLAT Pair Generation (GPU {args.gpu})")
    print(f"{'='*60}")
    print(f"  Source: {args.source_dir}")
    print(f"  Output: {args.output_dir}")
    print(f"  Seeds: {args.start_seed} → {args.start_seed + args.num_seeds - 1}")

    # Discover unique models
    source_dir = Path(args.source_dir)
    unique_models = discover_unique_models(source_dir)
    print(f"  Unique models: {len(unique_models)}")

    seeds = list(range(args.start_seed, args.start_seed + args.num_seeds))
    total = len(unique_models) * len(seeds)
    print(f"  Total pairs planned: {len(unique_models)} × {len(seeds)} = {total}")

    # Check existing
    existing = set()
    if output_dir.exists():
        for d in output_dir.iterdir():
            if d.is_dir() and (d / "fine_slat.npy").exists():
                existing.add(d.name)
    print(f"  Existing pairs: {len(existing)}")

    # Load models
    print(f"\nLoading generation models...")
    models, pipeline_args = load_pipeline_models(args.model_dir, device)
    ss_sampler, ss_params, slat_sampler, slat_params = create_samplers(pipeline_args)
    print(f"  All models loaded on GPU {args.gpu}")
    print(f"  SS sampler: {ss_sampler.__class__.__name__}")
    print(f"  SLAT sampler: {slat_sampler.__class__.__name__}")

    success = 0
    failed = 0
    skipped = 0
    t_start = time.time()

    model_items = sorted(unique_models.items())
    pbar = tqdm(total=total, desc=f"GPU{args.gpu}")

    for model_key, cf_path in model_items:
        # Load conditioning features once per model
        cond_features = np.load(cf_path)
        cond = build_cond_dict(cond_features, 512, device)

        for seed in seeds:
            uid = f"{model_key}_seed{seed}"
            pbar.update(1)

            if uid in existing:
                skipped += 1
                continue

            try:
                pair = generate_pair(
                    models, ss_sampler, ss_params, slat_sampler, slat_params,
                    cond, seed, device,
                )
                if pair is None:
                    failed += 1
                    continue

                save_pair(pair, uid, output_dir, cond_features)
                success += 1

                if (success + failed) % 100 == 0:
                    elapsed = time.time() - t_start
                    rate = success / elapsed if elapsed > 0 else 0
                    remaining = (total - skipped - success - failed) / max(rate, 0.01)
                    pbar.set_postfix(
                        ok=success, fail=failed, skip=skipped,
                        rate=f"{rate:.1f}/s", eta=f"{remaining/60:.0f}m"
                    )

            except Exception as e:
                failed += 1
                if failed <= 5:
                    print(f"\nFAIL {uid}: {e}")
                    traceback.print_exc()

            # Periodic cleanup
            if (success + failed) % 200 == 0:
                cleanup()

        # Cleanup between models
        del cond
        cleanup()

    pbar.close()
    elapsed = time.time() - t_start

    print(f"\n{'='*60}")
    print(f"Generation Complete (GPU {args.gpu})")
    print(f"{'='*60}")
    print(f"  Success: {success}")
    print(f"  Failed:  {failed}")
    print(f"  Skipped: {skipped}")
    print(f"  Time:    {elapsed/3600:.1f}h")
    print(f"  Rate:    {success/max(elapsed,1):.2f} pairs/s")
    print(f"  Output:  {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
