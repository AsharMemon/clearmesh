#!/usr/bin/env python3
"""Qualitative residual inference on SLAT training pairs.

Loads one or more coarse/fine SLAT pairs, runs residual refinement, and saves
predicted refined SLAT tensors plus comparison metrics.
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


def process_pair(
    model,
    pair_dir: str,
    output_dir: str,
    max_tokens: int,
    delta_scale: float,
    device: str,
    seed: int,
):
    pair = load_pair(pair_dir, max_tokens=max_tokens, seed=seed)
    pred = predict_refined_slat(
        model,
        pair,
        device=device,
        delta_scale=delta_scale,
        use_conditioning=True,
    )
    metrics = compute_metrics(pred, pair)

    uid = pair["uid"]
    out_path = Path(output_dir) / uid
    out_path.mkdir(parents=True, exist_ok=True)

    coarse = pair["coarse_slat"].numpy()
    fine = pair["fine_slat"].numpy()
    pred_np = pred.numpy()
    delta = pred_np - coarse

    np.save(out_path / "coarse_slat.npy", coarse)
    np.save(out_path / "fine_slat.npy", fine)
    np.save(out_path / "pred_refined_slat.npy", pred_np)
    np.save(out_path / "delta.npy", delta)
    np.save(out_path / "positions.npy", pair["positions"].numpy())

    if pair.get("cond_features") is not None:
        np.save(out_path / "cond_features.npy", pair["cond_features"].numpy())

    with open(out_path / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"  {uid}")
    print(f"  Tokens: {pair['coarse_slat'].shape[0]}")
    print(f"  Refined MAE: {metrics['pred_mae']:.5f}")
    print(f"  Coarse MAE:  {metrics['coarse_mae']:.5f}")
    print(f"  Improvement: {metrics['improvement_pct']:+.2f}%")
    print(f"  Corr:        {metrics['pred_corr']:.4f}")
    print(f"  Delta L1:    {metrics['delta_l1']:.5f}")
    print(f"  Output:      {out_path}")

    return {"uid": uid, **metrics}


def main():
    parser = argparse.ArgumentParser(
        description="Qualitative inference for ClearMesh residual Stage 2"
    )
    parser.add_argument("--config", required=True, help="Training config YAML")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint .pt file")
    parser.add_argument("--pair_dir", default=None, help="Single pair directory to process")
    parser.add_argument("--data_dir", default=None, help="Data directory for batch mode")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--delta_scale", type=float, default=1.0)
    parser.add_argument("--output_dir", default="eval_qualitative")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, step = load_stage2_model(config, args.checkpoint, device=device)

    print(f"\n{'=' * 60}")
    print("ClearMesh Stage 2 Qualitative Inference")
    print(f"  Checkpoint: step {step}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Delta scale: {args.delta_scale}")
    print(f"{'=' * 60}")

    if args.pair_dir:
        process_pair(
            model,
            args.pair_dir,
            args.output_dir,
            args.max_tokens,
            args.delta_scale,
            device,
            args.seed,
        )
        return

    data_dir = args.data_dir or config["data_dir"]
    pair_dirs = find_pair_dirs(data_dir)
    n = min(args.num_samples, len(pair_dirs))
    print(f"\nFound {len(pair_dirs)} valid SLAT pairs, processing {n}")

    all_metrics = []
    for i, pair_dir in enumerate(pair_dirs[:n]):
        print(f"\n[{i+1}/{n}] Processing {pair_dir.name}...")
        metrics = process_pair(
            model,
            str(pair_dir),
            args.output_dir,
            args.max_tokens,
            args.delta_scale,
            device,
            args.seed,
        )
        all_metrics.append(metrics)

    if all_metrics:
        avg_mae = np.mean([m["pred_mae"] for m in all_metrics])
        avg_coarse = np.mean([m["coarse_mae"] for m in all_metrics])
        avg_improvement = np.mean([m["improvement_pct"] for m in all_metrics])
        avg_corr = np.mean([m["pred_corr"] for m in all_metrics])
        print(f"\n{'=' * 60}")
        print("Aggregate")
        print(f"  Refined MAE: {avg_mae:.5f}")
        print(f"  Coarse MAE:  {avg_coarse:.5f}")
        print(f"  Improvement: {avg_improvement:+.2f}%")
        print(f"  Corr:        {avg_corr:.4f}")


if __name__ == "__main__":
    main()
