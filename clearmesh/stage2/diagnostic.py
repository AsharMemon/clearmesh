#!/usr/bin/env python3
"""Residual Stage 2 diagnostic on a handful of SLAT pairs.

Compares coarse baseline, conditioned refinement, and unconditioned refinement
so we can sanity-check whether image conditioning is helping.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from clearmesh.stage2.evaluate import (
    compute_metrics,
    find_pair_dirs,
    load_pair,
    predict_refined_slat,
)
from clearmesh.stage2.infer_slat import load_stage2_model


def main():
    parser = argparse.ArgumentParser(description="Residual Stage 2 SLAT diagnostic")
    parser.add_argument("--config", required=True, help="Training config YAML")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint .pt file")
    parser.add_argument("--data_dir", default=None, help="Override data directory")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--delta_scale", type=float, default=1.0)
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, step = load_stage2_model(config, args.checkpoint, device=device)

    data_dir = args.data_dir or config["data_dir"]
    pair_dirs = find_pair_dirs(data_dir)
    selected = pair_dirs[: min(args.num_samples, len(pair_dirs))]
    if not selected:
        raise RuntimeError(f"No valid SLAT pairs found in {data_dir}")

    print(f"\n{'=' * 80}")
    print("RESIDUAL SLAT DIAGNOSTIC")
    print(f"  Checkpoint step: {step}")
    print(f"  Data dir:        {data_dir}")
    print(f"  Samples:         {len(selected)}")
    print(f"{'=' * 80}")

    results = []
    for idx, pair_dir in enumerate(selected, start=1):
        pair = load_pair(str(pair_dir), max_tokens=args.max_tokens, seed=args.seed)
        pred_cond = predict_refined_slat(
            model,
            pair,
            device=device,
            delta_scale=args.delta_scale,
            use_conditioning=True,
        )
        pred_uncond = predict_refined_slat(
            model,
            pair,
            device=device,
            delta_scale=args.delta_scale,
            use_conditioning=False,
        )

        metrics_cond = compute_metrics(pred_cond, pair)
        metrics_uncond = compute_metrics(pred_uncond, pair)

        cond_gain = metrics_cond["improvement_pct"] - metrics_uncond["improvement_pct"]
        record = {
            "uid": pair["uid"],
            "tokens": int(pair["coarse_slat"].shape[0]),
            "cond": metrics_cond,
            "uncond": metrics_uncond,
            "conditioning_gain_pct": cond_gain,
        }
        results.append(record)

        print(f"\n[{idx}/{len(selected)}] {pair['uid']}")
        print(
            f"  coarse MAE={metrics_cond['coarse_mae']:.4f} | "
            f"cond MAE={metrics_cond['pred_mae']:.4f} ({metrics_cond['improvement_pct']:+.1f}%) | "
            f"uncond MAE={metrics_uncond['pred_mae']:.4f} ({metrics_uncond['improvement_pct']:+.1f}%)"
        )
        print(
            f"  cond corr={metrics_cond['pred_corr']:.3f} | "
            f"uncond corr={metrics_uncond['pred_corr']:.3f} | "
            f"conditioning gain={cond_gain:+.1f}%"
        )

    mean_cond = float(np.mean([r["cond"]["improvement_pct"] for r in results]))
    mean_uncond = float(np.mean([r["uncond"]["improvement_pct"] for r in results]))
    mean_gain = float(np.mean([r["conditioning_gain_pct"] for r in results]))

    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"  Mean cond improvement:   {mean_cond:+.2f}%")
    print(f"  Mean uncond improvement: {mean_uncond:+.2f}%")
    print(f"  Mean conditioning gain:  {mean_gain:+.2f}%")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved: {output_path}")


if __name__ == "__main__":
    main()
