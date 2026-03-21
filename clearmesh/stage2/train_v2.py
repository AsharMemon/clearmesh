"""Training loop for Stage 2 v2 FlowMatchingDiT — rectified flow in SLAT space.

Conditional generation approach (UltraShape-like):
  - Coarse SLAT (512) = conditioning signal (concatenated with noisy target)
  - Fine SLAT (1024) = training target (same positions, no alignment needed)
  - DINOv2 image features = cross-attention conditioning

Flow matching with velocity prediction:
    x_t = (1-t)*fine_slat + t*noise, v_target = noise - fine_slat
    loss = MSE(model(x_t, coarse_slat, positions, t, cond), v_target)

Key differences from train.py (residual prediction):
  - Flow matching (velocity prediction, MSE loss) instead of direct residual (L1)
  - Actual timestep t ~ U(0,1) instead of fixed t=0
  - CFG: 10% conditioning dropout during training
  - EMA: exponential moving average of weights
  - Differential learning rates (backbone 1e-5, new layers 1e-4)
  - All parameters trainable (no backbone freezing)
  - Optional geometry regularizer (decoder-in-loop every K steps)
  - Gradient accumulation support

Usage:
    # Single GPU
    python -m clearmesh.stage2.train_v2 --config configs/train_stage2_v2.yaml

    # Multi-GPU (DDP)
    torchrun --nproc_per_node=4 -m clearmesh.stage2.train_v2 \\
        --config configs/train_stage2_v2.yaml
"""

import argparse
import json
import logging
import math
import os
import random
import signal
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from clearmesh.stage2.flow_matching import (
    estimate_x0,
    interpolate,
    sample_timestep,
    velocity_target,
)
from clearmesh.stage2.model_v2 import EMA, FlowMatchingDiT
from clearmesh.stage2.train import (
    SlatPairDataset,
    get_progressive_value,
    is_main_process,
    setup_distributed,
    slatpair_collate_fn,
)

logger = logging.getLogger(__name__)


class TrainerV2:
    """Stage 2 v2 trainer — flow matching with velocity prediction."""

    def __init__(
        self,
        config: dict,
        rank: int = 0,
        local_rank: int = 0,
        world_size: int = 1,
    ):
        self.config = config
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.is_main = is_main_process(rank)

        if world_size > 1:
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        # --- Model ---
        pretrained = config.get("pretrained_checkpoint")
        num_layers = config.get("num_layers", 30)

        model_kwargs = dict(
            voxel_dim=config.get("voxel_dim", 32),
            model_dim=config.get("model_dim", 1536),
            num_heads=config.get("num_heads", 12),
            num_layers=num_layers,
            cond_dim=config.get("cond_dim", 1024),
            mlp_ratio=config.get("mlp_ratio", 5.3334),
            use_checkpoint=config.get("use_checkpoint", True),
        )

        if pretrained and Path(pretrained).exists():
            if self.is_main:
                print(f"Loading pretrained TRELLIS.2 weights from {pretrained}")
            self.model = FlowMatchingDiT.from_pretrained(
                pretrained, num_layers=num_layers, **model_kwargs
            ).to(self.device)
        else:
            if self.is_main:
                print("No pretrained checkpoint — training from scratch")
            self.model = FlowMatchingDiT(**model_kwargs).to(self.device)

        # DDP
        if world_size > 1:
            self.model = DDP(
                self.model, device_ids=[local_rank],
                find_unused_parameters=False,
            )
            if self.is_main:
                print(f"DDP enabled: {world_size} GPUs")

        # --- Parameter counts ---
        if self.is_main:
            raw_model = self._raw_model()
            total = sum(p.numel() for p in raw_model.parameters())
            trainable = sum(
                p.numel() for p in raw_model.parameters() if p.requires_grad
            )
            print(f"Model: {total / 1e6:.1f}M total, {trainable / 1e6:.1f}M trainable")

        # --- Optimizer with differential LR ---
        raw_model = self._raw_model()
        param_groups = raw_model.param_groups(
            lr_backbone=config.get("lr_backbone", 1e-5),
            lr_new=config.get("lr_new", 1e-4),
            lr_cross_attn_kv=config.get("lr_cross_attn_kv", 5e-5),
        )

        self.optimizer = torch.optim.AdamW(
            param_groups,
            betas=tuple(config.get("betas", [0.9, 0.99])),
            weight_decay=config.get("weight_decay", 0.01),
        )

        # --- LR scheduler ---
        total_steps = config.get("total_steps", 85_000)
        warmup_steps = config.get("warmup_steps", 1000)
        self.warmup_steps = warmup_steps

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lr_lambda
        )

        # --- EMA ---
        self.ema = EMA(
            self._raw_model(), decay=config.get("ema_decay", 0.9999)
        )

        # --- CFG ---
        self.cfg_dropout = config.get("cfg_dropout", 0.1)

        # --- Gradient accumulation ---
        self.grad_accum_steps = config.get("grad_accum_steps", 4)

        # --- Progressive schedule ---
        self.progressive = config.get("progressive_schedule", [])

        # --- Geometry regularizer ---
        self.geo_loss_interval = config.get("geometry_loss_interval", 0)
        self.geo_loss_weight = config.get("geometry_loss_weight", 0.1)

        # --- Training state ---
        self.global_step = 0
        self.epoch = 0
        self._checkpoint_loaded = False
        self.output_dir = Path(config["output_dir"])
        if self.is_main:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Preemption handling
        self._emergency_save = False
        signal.signal(signal.SIGUSR1, self._handle_preemption)

        # WandB
        self.wandb = None
        if config.get("use_wandb", False) and self.is_main:
            import wandb

            wandb.init(project="clearmesh", config=config, name="stage2-v2")
            self.wandb = wandb

    def _raw_model(self) -> FlowMatchingDiT:
        return self.model.module if isinstance(self.model, DDP) else self.model

    def _handle_preemption(self, signum, frame):
        print(f"\n!!! PREEMPTION (rank {self.rank}) — saving emergency checkpoint !!!")
        self._emergency_save = True

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def _build_dataloader(self, max_tokens: int, batch_size: int) -> DataLoader:
        dataset = SlatPairDataset(
            data_dir=self.config["data_dir"],
            max_tokens=max_tokens,
            voxel_dim=self.config.get("voxel_dim", 32),
            slat_mean=self.config.get("slat_mean"),
            slat_std=self.config.get("slat_std"),
            overfit_uid=self.config.get("overfit_uid"),
        )

        sampler = None
        shuffle = True
        if self.world_size > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
            )
            shuffle = False

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.config.get("num_workers", 4),
            pin_memory=True,
            drop_last=True,
            collate_fn=slatpair_collate_fn,
        )

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, tag: str = "latest"):
        if not self.is_main:
            return
        raw_model = self._raw_model()
        ckpt = {
            "model": raw_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "ema": self.ema.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "config": self.config,
        }
        path = self.output_dir / f"checkpoint_{tag}.pt"
        torch.save(ckpt, path)
        torch.save(ckpt, self.output_dir / "checkpoint_latest.pt")
        print(f"Checkpoint saved: {path} (step {self.global_step})")

    def load_checkpoint(self, path: str | None = None):
        p = Path(path) if path else self.output_dir / "checkpoint_latest.pt"
        if not p.exists():
            if self.is_main:
                print("No checkpoint found, starting from scratch.")
            return False
        if self.is_main:
            print(f"Resuming from {p}")
        ckpt = torch.load(p, map_location=self.device, weights_only=False)
        raw_model = self._raw_model()
        raw_model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        if "ema" in ckpt:
            self.ema.load_state_dict(ckpt["ema"])
        self.global_step = ckpt["global_step"]
        self.epoch = ckpt["epoch"]
        self._checkpoint_loaded = True
        if self.is_main:
            print(f"Resumed at step {self.global_step}, epoch {self.epoch}")
        return True

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_step(self, batch: dict) -> dict[str, float]:
        self.model.train()

        coarse_slat = batch["coarse_slat"].to(self.device)  # (B, N, 32) normalized
        fine_slat = batch["fine_slat"].to(self.device)  # (B, N, 32) normalized
        positions = batch["positions"].to(self.device)  # (B, N, 3)
        B = coarse_slat.shape[0]

        # DINOv2/v3 conditioning
        cond_features = batch.get("cond_features")
        cond_mask = batch.get("cond_mask")
        if cond_features is not None:
            cond_features = cond_features.to(self.device)
        if cond_mask is not None:
            cond_mask = cond_mask.to(self.device)

        # --- CFG dropout: drop the entire condition path for this batch ---
        if cond_features is not None and random.random() < self.cfg_dropout:
            cond_features = torch.zeros_like(cond_features)
            if cond_mask is not None:
                cond_mask = torch.zeros_like(cond_mask)

        # --- Flow matching ---
        t = sample_timestep(B, self.device)
        noise = torch.randn_like(fine_slat)
        x_t = interpolate(fine_slat, noise, t)  # noisy fine SLAT
        v_target = velocity_target(fine_slat, noise)  # noise - data

        # --- Forward pass ---
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            v_pred = self.model(
                x_t,
                coarse_slat,
                positions,
                t,
                cond_features=cond_features,
                cond_mask=cond_mask,
            )

            # Primary loss: velocity MSE (unweighted)
            loss_velocity = F.mse_loss(v_pred, v_target)

            losses = {
                "velocity_mse": loss_velocity,
                "total": loss_velocity,
            }

            # --- Optional geometry regularizer ---
            if (
                self.geo_loss_interval > 0
                and self.global_step % self.geo_loss_interval == 0
            ):
                # Estimate x_0 from current prediction
                pred_x0 = estimate_x0(x_t, v_pred, t)
                # Geometry loss would be computed here by forwarding through
                # the frozen decoder. Placeholder — requires TRELLIS.2 decoder
                # to be loaded in the training loop.
                # loss_geo = compute_geometry_loss(pred_x0, positions, ...)
                # losses["geo"] = loss_geo
                # losses["total"] = loss_velocity + self.geo_loss_weight * loss_geo

        # Backward (with gradient accumulation)
        loss_scaled = losses["total"] / self.grad_accum_steps
        loss_scaled.backward()

        return {k: v.item() for k, v in losses.items()}

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self):
        total_steps = self.config.get("total_steps", 85_000)
        save_interval = self.config.get("save_interval", 2500)
        log_interval = self.config.get("log_interval", 50)

        if not self._checkpoint_loaded:
            self.load_checkpoint()

        # Current progressive values
        cur_tokens = get_progressive_value(
            self.progressive, self.global_step, "num_tokens", 4096
        )
        cur_batch = get_progressive_value(
            self.progressive,
            self.global_step,
            "batch_size",
            self.config.get("batch_size", 1),
        )
        dataloader = self._build_dataloader(cur_tokens, cur_batch)

        if self.is_main:
            eff_batch = cur_batch * self.world_size * self.grad_accum_steps
            print(f"\n{'='*60}")
            print(f"=== Training Stage 2 v2 FlowMatchingDiT ===")
            print(f"  Mode:            Flow matching (velocity prediction, MSE)")
            print(f"  Total steps:     {total_steps}")
            print(f"  Current step:    {self.global_step}")
            print(f"  Dataset size:    {len(dataloader.dataset)}")
            print(f"  Batch size:      {cur_batch} (per GPU)")
            print(f"  Grad accum:      {self.grad_accum_steps}")
            print(f"  Effective batch: {eff_batch}")
            print(f"  World size:      {self.world_size} GPU(s)")
            print(f"  Initial tokens:  {cur_tokens}")
            print(f"  CFG dropout:     {self.cfg_dropout}")
            print(f"  EMA decay:       {self.config.get('ema_decay', 0.9999)}")
            print(f"  Progressive:     {self.progressive}")
            if self.geo_loss_interval > 0:
                print(f"  Geo regularizer: every {self.geo_loss_interval} steps, "
                      f"weight={self.geo_loss_weight}")
            print(f"  bf16:            enabled")
            print(f"  Grad checkpoint: {self.config.get('use_checkpoint', True)}")
            print(f"{'='*60}\n")

        with open("/tmp/train_v2.pid", "w") as f:
            f.write(str(os.getpid()))

        pbar = tqdm(
            total=total_steps,
            initial=self.global_step,
            desc="Training v2",
            disable=not self.is_main,
        )

        accum_losses = {}
        accum_count = 0

        while self.global_step < total_steps:
            if hasattr(dataloader, "sampler") and isinstance(
                dataloader.sampler, DistributedSampler
            ):
                dataloader.sampler.set_epoch(self.epoch)

            for batch in dataloader:
                if self.global_step >= total_steps:
                    break

                # Emergency save
                if self._emergency_save:
                    self.save_checkpoint(f"emergency_{self.global_step}")
                    if self.is_main:
                        print("Emergency checkpoint saved. Exiting.")
                    if self.world_size > 1:
                        dist.destroy_process_group()
                    sys.exit(0)

                # Progressive schedule: rebuild dataloader if changed
                new_tokens = get_progressive_value(
                    self.progressive, self.global_step, "num_tokens", 4096
                )
                new_batch = get_progressive_value(
                    self.progressive,
                    self.global_step,
                    "batch_size",
                    self.config.get("batch_size", 1),
                )
                if new_tokens != cur_tokens or new_batch != cur_batch:
                    if self.is_main:
                        print(
                            f"\n>>> Progressive: tokens {cur_tokens}→{new_tokens}, "
                            f"batch {cur_batch}→{new_batch} at step {self.global_step}"
                        )
                    cur_tokens = new_tokens
                    cur_batch = new_batch
                    dataloader = self._build_dataloader(cur_tokens, cur_batch)
                    break  # restart epoch with new dataloader

                # Train step (accumulates gradients)
                losses = self.train_step(batch)
                accum_count += 1

                # Accumulate loss for logging
                for k, v in losses.items():
                    accum_losses[k] = accum_losses.get(k, 0.0) + v

                # Optimizer step (every grad_accum_steps)
                if accum_count >= self.grad_accum_steps:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.get("max_grad_norm", 1.0),
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    # EMA update
                    self.ema.update()

                    self.global_step += 1
                    pbar.update(1)

                    # Average accumulated losses
                    avg_losses = {
                        k: v / accum_count for k, v in accum_losses.items()
                    }
                    accum_losses = {}
                    accum_count = 0

                    # Logging
                    if self.is_main and self.global_step % log_interval == 0:
                        loss_str = " | ".join(
                            f"{k}: {v:.4f}" for k, v in avg_losses.items()
                        )
                        pbar.set_postfix_str(loss_str)
                        if self.wandb:
                            log_dict = {
                                f"loss/{k}": v for k, v in avg_losses.items()
                            }
                            log_dict["lr/backbone"] = self.optimizer.param_groups[0]["lr"]
                            log_dict["lr/new_layers"] = self.optimizer.param_groups[1]["lr"]
                            log_dict["tokens"] = cur_tokens
                            self.wandb.log(log_dict, step=self.global_step)

                    # Checkpoint
                    if self.global_step % save_interval == 0:
                        self.save_checkpoint(f"step_{self.global_step}")
                        if self.world_size > 1:
                            dist.barrier()

            self.epoch += 1

        pbar.close()
        self.save_checkpoint("final")
        if self.is_main:
            print(f"\nTraining complete at step {self.global_step}")
        if self.world_size > 1:
            dist.destroy_process_group()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Train Stage 2 v2 FlowMatchingDiT")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--resume_from", default=None)
    args = parser.parse_args()

    rank, local_rank, world_size = setup_distributed()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.output_dir:
        config["output_dir"] = args.output_dir

    trainer = TrainerV2(
        config, rank=rank, local_rank=local_rank, world_size=world_size
    )
    if args.resume_from:
        trainer.load_checkpoint(args.resume_from)
    trainer.train()


if __name__ == "__main__":
    main()
