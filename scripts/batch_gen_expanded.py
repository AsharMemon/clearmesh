#!/usr/bin/env python3
"""Generate expanded SLAT pairs from a large image set.

Uses all images in --image_dir with multiple seeds to create ~3000 pairs.
Resumes from existing pairs (skips already-generated UIDs).

Usage:
    python batch_gen_expanded.py \
        --image_dir /workspace/data/expanded_images \
        --output_dir /workspace/data/slat_pairs_3k \
        --seeds 28 \
        --trellis2_dir /workspace/TRELLIS.2 \
        --model_dir /workspace/models/trellis2-4b
"""

import argparse
import gc
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def setup_trellis2(trellis2_dir: str):
    if trellis2_dir not in sys.path:
        sys.path.insert(0, trellis2_dir)
    os.environ.setdefault("ATTN_BACKEND", "flash_attn")


def load_pipeline(model_dir: str, device: str = "cuda"):
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(model_dir)
    pipeline.low_vram = False
    for m in pipeline.models.values():
        if hasattr(m, "low_vram"):
            m.low_vram = False
    if hasattr(pipeline, "image_cond_model") and pipeline.image_cond_model is not None:
        if hasattr(pipeline.image_cond_model, "low_vram"):
            pipeline.image_cond_model.low_vram = False
    pipeline.to(device)
    return pipeline


def generate_pair(pipeline, image, seed=42):
    """Generate a coarse(512)/fine(1024) SLAT pair."""
    processed = pipeline.preprocess_image(image)
    torch.manual_seed(seed)

    # Get conditioning at both resolutions
    cond_512 = pipeline.get_cond([processed], 512)
    cond_1024 = pipeline.get_cond([processed], 1024)

    # Shared sparse structure
    coords = pipeline.sample_sparse_structure(
        cond_512, 32, 1, {"steps": 12, "guidance_strength": 9.0},
    )

    # Coarse SLAT (512 model)
    shape_slat_512 = pipeline.sample_shape_slat(
        cond_512,
        pipeline.models["shape_slat_flow_model_512"],
        coords,
        {"steps": 12, "guidance_strength": 4.5},
    )

    # Fine SLAT (1024 model)
    shape_slat_1024 = pipeline.sample_shape_slat(
        cond_1024,
        pipeline.models["shape_slat_flow_model_1024"],
        coords,
        {"steps": 12, "guidance_strength": 4.5},
    )

    # Extract positions
    if isinstance(coords, torch.Tensor):
        if coords.dim() == 2 and coords.shape[1] == 4:
            positions = coords[:, 1:].cpu().numpy().astype(np.int32)
        else:
            positions = coords.cpu().numpy().astype(np.int32)
    elif hasattr(coords, "coords"):
        c = coords.coords
        positions = (c[:, 1:] if c.shape[1] == 4 else c).cpu().numpy().astype(np.int32)
    else:
        raise RuntimeError(f"Cannot extract coords from {type(coords)}")

    # Extract SLAT features
    def get_feats(slat_obj):
        if hasattr(slat_obj, "feats"):
            return slat_obj.feats.float().cpu().numpy().astype(np.float16)
        elif hasattr(slat_obj, "F"):
            return slat_obj.F.float().cpu().numpy().astype(np.float16)
        elif isinstance(slat_obj, torch.Tensor):
            return slat_obj.squeeze(0).float().cpu().numpy().astype(np.float16)
        raise RuntimeError(f"Cannot extract SLAT from {type(slat_obj)}")

    coarse = get_feats(shape_slat_512)
    fine = get_feats(shape_slat_1024)

    # DINOv2 conditioning
    cond_feats = None
    if isinstance(cond_512, dict):
        ct = cond_512.get("cond", cond_512.get("image_cond"))
        if ct is None:
            ct = next(iter(cond_512.values()))
    elif isinstance(cond_512, (list, tuple)):
        ct = cond_512[0]
    else:
        ct = cond_512

    if isinstance(ct, torch.Tensor):
        ct_np = ct.squeeze(0).float().cpu().numpy() if ct.dim() == 3 else ct.float().cpu().numpy()
        cond_feats = ct_np.astype(np.float16)

    n_points = positions.shape[0]

    return {
        "positions": positions,
        "coarse_slat": coarse,
        "fine_slat": fine,
        "cond_features": cond_feats,
        "n_points": n_points,
    }


def save_pair(pair, uid, output_dir):
    out = Path(output_dir) / uid
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "positions.npy", pair["positions"])
    np.save(out / "coarse_slat.npy", pair["coarse_slat"])
    np.save(out / "fine_slat.npy", pair["fine_slat"])
    if pair["cond_features"] is not None:
        np.save(out / "cond_features.npy", pair["cond_features"])
    with open(out / "metadata.json", "w") as f:
        json.dump({"uid": uid, "n_points": int(pair["n_points"])}, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--trellis2_dir", default="/workspace/TRELLIS.2")
    parser.add_argument("--model_dir", default="/workspace/models/trellis2-4b")
    parser.add_argument("--seeds", type=int, default=28, help="Number of seeds per image")
    parser.add_argument("--start_seed", type=int, default=42)
    args = parser.parse_args()

    setup_trellis2(args.trellis2_dir)
    pipeline = load_pipeline(args.model_dir)

    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Gather images
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    images = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in exts])
    print(f"Found {len(images)} images in {image_dir}")

    # Check existing pairs
    existing = set()
    if output_dir.exists():
        for d in output_dir.iterdir():
            if d.is_dir() and (d / "coarse_slat.npy").exists():
                existing.add(d.name)
    print(f"Existing pairs: {len(existing)}")

    seeds = list(range(args.start_seed, args.start_seed + args.seeds))
    total_planned = len(images) * len(seeds)
    print(f"Plan: {len(images)} images × {len(seeds)} seeds = {total_planned} pairs")

    success = 0
    failed = 0
    skipped = 0
    t_start = time.time()

    for img_idx, img_path in enumerate(images):
        img_name = img_path.stem
        for seed in seeds:
            uid = f"{img_name}_seed{seed}"
            if uid in existing:
                skipped += 1
                continue

            try:
                image = Image.open(img_path).convert("RGBA")
                pair = generate_pair(pipeline, image, seed=seed)
                save_pair(pair, uid, output_dir)
                success += 1
                npts = pair["n_points"]

                elapsed = time.time() - t_start
                rate = (success + failed) / elapsed if elapsed > 0 else 0
                remaining = (total_planned - skipped - success - failed) / max(rate, 0.01)

                if (success + failed) % 10 == 0:
                    print(f"[{success+failed+skipped}/{total_planned}] "
                          f"{uid}: {npts} pts | "
                          f"ok={success} fail={failed} skip={skipped} | "
                          f"{rate:.1f}/s | ETA {remaining/60:.0f}m")

            except Exception as e:
                failed += 1
                print(f"FAIL {uid}: {e}")
                traceback.print_exc()

            # Periodic cleanup
            if (success + failed) % 50 == 0:
                gc.collect()
                torch.cuda.empty_cache()

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Done! {success} pairs in {elapsed/60:.1f}m")
    print(f"  Success: {success}")
    print(f"  Failed:  {failed}")
    print(f"  Skipped: {skipped}")
    print(f"  Rate:    {success/elapsed:.2f} pairs/s ({elapsed/max(success,1):.1f}s/pair)")
    print(f"  Output:  {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
