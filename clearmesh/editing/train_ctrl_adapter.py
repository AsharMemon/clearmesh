#!/usr/bin/env python3
"""Ctrl-Adapter Training Loop.

Trains the Ctrl-Adapter on 6-view RGB + normal map pairs rendered
from Objaverse meshes. The base diffusion model stays frozen —
only the adapter weights are updated.

Training data format (from render_ctrl_adapter_data.py):
    {data_dir}/{uid}/rgb_000.png ... rgb_005.png
    {data_dir}/{uid}/normal_000.png ... normal_005.png

Loss: MSE between adapter-predicted features and ground-truth
diffusion model features extracted from RGB views.

Usage:
    python -m clearmesh.editing.train_ctrl_adapter \
        --config configs/train_ctrl_adapter.yaml

    # Or direct:
    python -m clearmesh.editing.train_ctrl_adapter \
        --data_dir /workspace/data/ctrl_adapter \
        --output_dir /workspace/checkpoints/ctrl_adapter \
        --batch_size 4 \
        --learning_rate 1e-4 \
        --total_steps 20000
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from clearmesh.editing.ctrl_adapter import CtrlAdapter


class CtrlAdapterDataset(Dataset):
    """Dataset for Ctrl-Adapter training.

    Loads 6-view RGB + normal map pairs rendered from Objaverse meshes.
    Each sample is a pair of (normal_maps, rgb_views) for one object.
    """

    def __init__(
        self,
        data_dir: str,
        image_size: int = 512,
        num_views: int = 6,
        manifest_path: str | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.num_views = num_views

        # Load manifest to get list of UIDs
        if manifest_path:
            manifest = Path(manifest_path)
        else:
            manifest = self.data_dir / "manifest.json"

        if manifest.exists():
            with open(manifest) as f:
                meta = json.load(f)
            self.uids = meta["uids"]
        else:
            # Scan directory
            self.uids = sorted(
                [d.name for d in self.data_dir.iterdir() if d.is_dir()]
            )

        print(f"CtrlAdapterDataset: {len(self.uids)} objects from {data_dir}")

    def __len__(self) -> int:
        return len(self.uids)

    def __getitem__(self, idx: int) -> dict:
        uid = self.uids[idx]
        uid_dir = self.data_dir / uid

        normal_maps = []
        rgb_views = []

        for v in range(self.num_views):
            # Load normal map
            normal_path = uid_dir / f"normal_{v:03d}.png"
            normal = Image.open(normal_path).convert("RGB").resize(
                (self.image_size, self.image_size), Image.LANCZOS
            )
            normal = torch.tensor(np.array(normal), dtype=torch.float32) / 255.0
            normal = normal.permute(2, 0, 1)  # (3, H, W)
            normal_maps.append(normal)

            # Load RGB view
            rgb_path = uid_dir / f"rgb_{v:03d}.png"
            rgb = Image.open(rgb_path).convert("RGB").resize(
                (self.image_size, self.image_size), Image.LANCZOS
            )
            rgb = torch.tensor(np.array(rgb), dtype=torch.float32) / 255.0
            rgb = rgb.permute(2, 0, 1)  # (3, H, W)
            rgb_views.append(rgb)

        return {
            "uid": uid,
            "normal_maps": torch.stack(normal_maps),  # (6, 3, H, W)
            "rgb_views": torch.stack(rgb_views),  # (6, 3, H, W)
        }


def train_ctrl_adapter(
    data_dir: str,
    output_dir: str,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    total_steps: int = 20000,
    save_interval: int = 1000,
    log_interval: int = 50,
    num_workers: int = 4,
    image_size: int = 512,
    use_wandb: bool = False,
    resume_from: str | None = None,
):
    """Train the Ctrl-Adapter model.

    Args:
        data_dir: Directory with 6-view RGB + normal pairs.
        output_dir: Checkpoint output directory.
        batch_size: Training batch size.
        learning_rate: AdamW learning rate.
        weight_decay: AdamW weight decay.
        total_steps: Total training steps.
        save_interval: Save checkpoint every N steps.
        log_interval: Log losses every N steps.
        num_workers: DataLoader workers.
        image_size: Image resolution.
        use_wandb: Enable Weights & Biases logging.
        resume_from: Path to checkpoint to resume from.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize model
    model = CtrlAdapter(
        in_channels=3,
        base_channels=64,
        num_levels=4,
        blocks_per_level=2,
        num_views=6,
    ).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # LR scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=learning_rate * 0.01
    )

    # Resume from checkpoint
    start_step = 0
    if resume_from:
        print(f"Resuming from {resume_from}")
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_step = ckpt["step"]
        print(f"Resumed at step {start_step}")

    # Dataset
    dataset = CtrlAdapterDataset(data_dir, image_size=image_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # WandB
    if use_wandb:
        try:
            import wandb

            wandb.init(project="clearmesh-ctrl-adapter", config={
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "total_steps": total_steps,
                "image_size": image_size,
                "dataset_size": len(dataset),
            })
        except ImportError:
            print("wandb not installed, skipping logging")
            use_wandb = False

    # Training loop
    print(f"\n{'='*60}")
    print(f"  Ctrl-Adapter Training")
    print(f"  Dataset: {len(dataset)} objects")
    print(f"  Batch size: {batch_size}")
    print(f"  Total steps: {total_steps}")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    model.train()
    data_iter = iter(loader)
    step = start_step
    t_start = time.time()

    while step < total_steps:
        # Get batch (cycle through dataset)
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        normal_maps = batch["normal_maps"].to(device)  # (B, 6, 3, H, W)
        rgb_views = batch["rgb_views"].to(device)  # (B, 6, 3, H, W)

        # Forward pass: extract control features from normals
        control_features = model(normal_maps)

        # Training objective: control features should predict RGB appearance
        # We use a simple MSE loss on multi-scale features
        # TODO: Replace with proper diffusion-based training when ERA3D is integrated
        #   - Extract intermediate features from frozen base model on RGB views
        #   - Train adapter to match those features from normals alone
        loss = _compute_proxy_loss(control_features, rgb_views, normal_maps)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        step += 1

        # Logging
        if step % log_interval == 0:
            elapsed = time.time() - t_start
            steps_per_sec = step / max(elapsed, 1)
            eta = (total_steps - step) / max(steps_per_sec, 0.01)
            lr = scheduler.get_last_lr()[0]

            print(
                f"Step {step}/{total_steps} | "
                f"Loss: {loss.item():.4f} | "
                f"LR: {lr:.2e} | "
                f"Speed: {steps_per_sec:.1f} steps/s | "
                f"ETA: {eta/3600:.1f}h"
            )

            if use_wandb:
                wandb.log({
                    "loss": loss.item(),
                    "learning_rate": lr,
                    "step": step,
                })

        # Save checkpoint
        if step % save_interval == 0:
            ckpt_path = output_path / f"checkpoint_{step:06d}.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "step": step,
                    "config": {
                        "in_channels": 3,
                        "base_channels": 64,
                        "num_levels": 4,
                        "blocks_per_level": 2,
                        "num_views": 6,
                    },
                },
                ckpt_path,
            )
            print(f"  Saved checkpoint: {ckpt_path}")

    # Final checkpoint
    final_path = output_path / "checkpoint_final.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "step": step,
            "config": {
                "in_channels": 3,
                "base_channels": 64,
                "num_levels": 4,
                "blocks_per_level": 2,
                "num_views": 6,
            },
        },
        final_path,
    )
    print(f"\nTraining complete! Final checkpoint: {final_path}")

    if use_wandb:
        wandb.finish()


def _compute_proxy_loss(
    control_features: list[torch.Tensor],
    rgb_views: torch.Tensor,
    normal_maps: torch.Tensor,
) -> torch.Tensor:
    """Proxy training loss (pre-ERA3D integration).

    Until the base diffusion model (ERA3D) is integrated, we use a
    proxy loss: the adapter's control features should reconstruct
    multi-scale features derived from the RGB ground truth.

    This trains the adapter to extract meaningful geometric features
    from normal maps, which will transfer when swapped to the real
    diffusion-based loss.

    Args:
        control_features: Multi-scale features from adapter.
        rgb_views: Ground truth RGB (B, 6, 3, H, W).
        normal_maps: Input normal maps (B, 6, 3, H, W).

    Returns:
        Scalar loss tensor.
    """
    B, V, C, H, W = rgb_views.shape
    rgb_flat = rgb_views.view(B * V, C, H, W)

    total_loss = torch.tensor(0.0, device=rgb_views.device)

    for level, features in enumerate(control_features):
        # Downsample RGB to match feature resolution
        _, C_feat, H_feat, W_feat = features.shape
        target = F.adaptive_avg_pool2d(rgb_flat, (H_feat, W_feat))

        # Simple reconstruction: features → RGB via 1x1 conv (detached target)
        # This is a proxy — real training would use diffusion model features
        if C_feat != C:
            # Project features to RGB space for loss computation
            proj = F.adaptive_avg_pool2d(
                features[:, :3], (H_feat, W_feat)
            ) if C_feat >= 3 else features.mean(dim=1, keepdim=True).expand(-1, 3, -1, -1)
            total_loss += F.mse_loss(proj, target)
        else:
            total_loss += F.mse_loss(features, target)

    return total_loss / len(control_features)


def main():
    parser = argparse.ArgumentParser(description="Train Ctrl-Adapter")
    parser.add_argument("--config", type=str, default=None, help="YAML config file")
    parser.add_argument("--data_dir", type=str, default="/workspace/data/ctrl_adapter")
    parser.add_argument("--output_dir", type=str, default="/workspace/checkpoints/ctrl_adapter")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--total_steps", type=int, default=20000)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--resume_from", type=str, default=None)
    args = parser.parse_args()

    # Load config file if provided
    if args.config:
        import yaml

        with open(args.config) as f:
            config = yaml.safe_load(f)
        # Override with CLI args
        for key, value in vars(args).items():
            if key != "config" and value is not None:
                config.setdefault(key, value)
    else:
        config = vars(args)
        del config["config"]

    train_ctrl_adapter(**config)


if __name__ == "__main__":
    main()
