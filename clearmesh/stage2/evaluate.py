#!/usr/bin/env python3
"""Evaluate a residual Stage 2 checkpoint on SLAT training pairs.

Loads coarse/fine SLAT pairs, runs residual refinement, and reports latent-space
metrics against the fine target. This is the honest evaluation path for the
current residual model family; the old diffusion/SDF evaluator no longer applies.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import zlib
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import yaml

from clearmesh.stage2.infer_slat import (
    denormalize_slat,
    load_stage2_model,
    normalize_slat,
)


def resolve_fine_slat_path(pair_path: Path) -> Path | None:
    for name in ("fine_slat_aligned.npy", "fine_slat.npy"):
        candidate = pair_path / name
        if candidate.exists():
            return candidate
    return None


def find_pair_dirs(data_dir: str) -> list[Path]:
    data_path = Path(data_dir)
    pair_dirs: list[Path] = []
    for root, _, _ in os_walk_sorted(data_path):
        pair_path = Path(root)
        if (pair_path / "coarse_slat.npy").exists() and resolve_fine_slat_path(pair_path):
            pair_dirs.append(pair_path)
    pair_dirs.sort(key=lambda p: p.name)
    return pair_dirs


def os_walk_sorted(root: Path):
    for current_root, dirs, files in os.walk(str(root)):
        dirs.sort()
        files.sort()
        yield current_root, dirs, files


def _pair_rng(uid: str, seed: int) -> np.random.Generator:
    stable = zlib.crc32(uid.encode("utf-8"))
    return np.random.default_rng(seed ^ stable)


def load_pair(pair_dir: str, max_tokens: int | None = None, seed: int = 42) -> dict:
    pair_path = Path(pair_dir)
    uid = pair_path.name
    fine_path = resolve_fine_slat_path(pair_path)
    if fine_path is None:
        raise FileNotFoundError(f"No fine SLAT target found in {pair_dir}")

    coarse = np.load(pair_path / "coarse_slat.npy").astype(np.float32)
    fine = np.load(fine_path).astype(np.float32)
    positions = np.load(pair_path / "positions.npy").astype(np.float32)

    min_len = min(coarse.shape[0], fine.shape[0], positions.shape[0])
    coarse = coarse[:min_len]
    fine = fine[:min_len]
    positions = positions[:min_len]

    if max_tokens is not None and min_len > max_tokens:
        rng = _pair_rng(uid, seed)
        idx = np.sort(rng.choice(min_len, size=max_tokens, replace=False))
        coarse = coarse[idx]
        fine = fine[idx]
        positions = positions[idx]

    cond_features = None
    cond_path = pair_path / "cond_features.npy"
    if cond_path.exists():
        cond_features = np.load(cond_path).astype(np.float32)

    return {
        "uid": uid,
        "coarse_slat": torch.from_numpy(coarse).float(),
        "fine_slat": torch.from_numpy(fine).float(),
        "positions": torch.from_numpy(positions).float(),
        "cond_features": torch.from_numpy(cond_features).float() if cond_features is not None else None,
    }


@torch.no_grad()
def predict_refined_slat(
    model,
    pair: dict,
    device: str = "cuda",
    delta_scale: float = 1.0,
    use_conditioning: bool = True,
) -> torch.Tensor:
    coarse_raw = pair["coarse_slat"].to(device)
    positions = pair["positions"].to(device)
    coarse_norm = normalize_slat(coarse_raw)

    cond_features = None
    cond_mask = None
    if use_conditioning and pair.get("cond_features") is not None:
        cond_features = pair["cond_features"].unsqueeze(0).to(device)
        cond_mask = torch.ones(
            1, cond_features.shape[1], dtype=torch.bool, device=device
        )

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.startswith("cuda")
        else nullcontext()
    )
    with autocast_ctx:
        refined_norm = model.refine_residual(
            coarse_norm.unsqueeze(0),
            positions.unsqueeze(0),
            cond_features=cond_features,
            cond_mask=cond_mask,
            delta_scale=delta_scale,
        )

    return denormalize_slat(refined_norm.float().squeeze(0)).cpu()


def compute_metrics(pred_slat: torch.Tensor, pair: dict) -> dict[str, float | list[float]]:
    pred = pred_slat.cpu().numpy().astype(np.float32)
    fine = pair["fine_slat"].cpu().numpy().astype(np.float32)
    coarse = pair["coarse_slat"].cpu().numpy().astype(np.float32)

    pred_flat = pred.reshape(-1)
    fine_flat = fine.reshape(-1)
    coarse_flat = coarse.reshape(-1)

    pred_mae = float(np.mean(np.abs(pred_flat - fine_flat)))
    pred_mse = float(np.mean((pred_flat - fine_flat) ** 2))
    coarse_mae = float(np.mean(np.abs(coarse_flat - fine_flat)))
    coarse_mse = float(np.mean((coarse_flat - fine_flat) ** 2))

    improvement_pct = 0.0
    if coarse_mae > 0:
        improvement_pct = float((coarse_mae - pred_mae) / coarse_mae * 100.0)

    pred_corr = 0.0
    if np.std(pred_flat) > 1e-8 and np.std(fine_flat) > 1e-8:
        pred_corr = float(np.corrcoef(pred_flat, fine_flat)[0, 1])

    coarse_corr = 0.0
    if np.std(coarse_flat) > 1e-8 and np.std(fine_flat) > 1e-8:
        coarse_corr = float(np.corrcoef(coarse_flat, fine_flat)[0, 1])

    delta = pred - coarse
    delta_l1 = float(np.mean(np.abs(delta)))
    delta_l2 = float(np.sqrt(np.mean(delta**2)))

    return {
        "pred_mae": pred_mae,
        "pred_mse": pred_mse,
        "coarse_mae": coarse_mae,
        "coarse_mse": coarse_mse,
        "improvement_pct": improvement_pct,
        "pred_corr": pred_corr,
        "coarse_corr": coarse_corr,
        "delta_l1": delta_l1,
        "delta_l2": delta_l2,
        "pred_range": [float(pred_flat.min()), float(pred_flat.max())],
        "fine_range": [float(fine_flat.min()), float(fine_flat.max())],
    }


@torch.no_grad()
def evaluate(
    model,
    data_dir: str,
    num_samples: int = 20,
    max_tokens: int | None = 8192,
    delta_scale: float = 1.0,
    device: str = "cuda",
    output_dir: str | None = None,
    seed: int = 42,
):
    pair_dirs = find_pair_dirs(data_dir)
    print(f"Found {len(pair_dirs)} valid SLAT pairs")
    if not pair_dirs:
        raise RuntimeError(f"No valid coarse_slat/fine_slat pairs found in {data_dir}")

    selected = pair_dirs[: min(num_samples, len(pair_dirs))]
    all_metrics = []

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(
        f"Evaluating {len(selected)} pairs | max_tokens={max_tokens} | delta_scale={delta_scale}"
    )
    print(f"{'=' * 70}\n")

    for i, pair_dir in enumerate(selected):
        pair = load_pair(str(pair_dir), max_tokens=max_tokens, seed=seed)
        t0 = time.time()
        pred = predict_refined_slat(
            model,
            pair,
            device=device,
            delta_scale=delta_scale,
            use_conditioning=True,
        )
        elapsed = time.time() - t0

        metrics = compute_metrics(pred, pair)
        metrics["uid"] = pair["uid"]
        metrics["inference_time"] = elapsed
        metrics["num_tokens"] = int(pair["coarse_slat"].shape[0])
        all_metrics.append(metrics)

        if output_dir:
            sample_dir = Path(output_dir) / pair["uid"]
            sample_dir.mkdir(parents=True, exist_ok=True)
            np.save(sample_dir / "pred_refined_slat.npy", pred.numpy())
            np.save(sample_dir / "coarse_slat.npy", pair["coarse_slat"].numpy())
            np.save(sample_dir / "fine_slat.npy", pair["fine_slat"].numpy())
            np.save(sample_dir / "positions.npy", pair["positions"].numpy())
            with open(sample_dir / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)

        print(
            f"[{i+1:3d}/{len(selected)}] {pair['uid'][:40]:40s} | "
            f"MAE={metrics['pred_mae']:.4f} "
            f"(coarse={metrics['coarse_mae']:.4f}, {metrics['improvement_pct']:+.1f}%) | "
            f"corr={metrics['pred_corr']:.3f} | {elapsed:.2f}s"
        )

    print(f"\n{'=' * 70}")
    print("AGGREGATE RESULTS")
    print(f"{'=' * 70}")
    avg = lambda key: float(np.mean([m[key] for m in all_metrics]))
    print(f"  Refined MAE:      {avg('pred_mae'):.5f}")
    print(f"  Coarse MAE:       {avg('coarse_mae'):.5f}")
    print(f"  Improvement:      {avg('improvement_pct'):+.2f}%")
    print(f"  Refined MSE:      {avg('pred_mse'):.5f}")
    print(f"  Mean corr:        {avg('pred_corr'):.5f}")
    print(f"  Mean delta L1:    {avg('delta_l1'):.5f}")
    print(f"  Mean time:        {avg('inference_time'):.2f}s")

    if output_dir:
        summary_path = Path(output_dir) / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\nSaved per-sample outputs to {output_dir}")

    return all_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ClearMesh residual Stage 2 on SLAT pairs"
    )
    parser.add_argument("--config", required=True, help="Training config YAML")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint .pt file")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--delta_scale", type=float, default=1.0)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, step = load_stage2_model(config, args.checkpoint, device=device)

    print(f"\n{'=' * 70}")
    print("ClearMesh Stage 2 Residual Evaluation")
    print(f"  Checkpoint: step {step}")
    print(f"  Data dir:   {config['data_dir']}")
    print(f"  Device:     {device}")
    print(f"{'=' * 70}")

    evaluate(
        model,
        config["data_dir"],
        num_samples=args.num_samples,
        max_tokens=args.max_tokens,
        delta_scale=args.delta_scale,
        device=device,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
