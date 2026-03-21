#!/usr/bin/env python3
"""Training loop for Stage 2 RefinementDiT — direct residual prediction.

Trains the DiT to predict SLAT residuals (deltas) via single forward pass:
    refined_slat = coarse_slat + model(coarse_slat, cond_features)
    loss = L1(refined_slat, fine_slat)

Training pairs: coarse SLAT (512 model) → fine SLAT (1024 model).
At inference, refined SLAT is decoded by TRELLIS.2's frozen decoder.

Features:
  - Loads TRELLIS.2 pretrained weights (first N blocks)
  - Freezes backbone, trains only last K blocks + out_head (~50-80M params)
  - Fixed dummy timestep t=0 for AdaLN compatibility
  - bf16 mixed-precision via torch.autocast
  - Gradient checkpointing (configured in model)
  - Progressive training schedule (token count + batch size ramp)
  - SLAT normalization (zero-mean, unit-std per channel from TRELLIS.2 stats)
  - Image token masking for DINO conditioning
  - Checkpoint every N steps (Spot VM resilience)
  - SIGUSR1 handler for emergency checkpoint on preemption
  - Multi-GPU DDP support (torchrun compatible)
  - WandB logging

Usage:
    # Single GPU
    python -m clearmesh.stage2.train \\
        --config configs/train_stage2_residual.yaml

    # Multi-GPU (DDP)
    torchrun --nproc_per_node=4 -m clearmesh.stage2.train \\
        --config configs/train_stage2_residual.yaml
"""

import argparse
import json
import logging
import math
import os
import signal
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from clearmesh.stage2.model import RefinementDiT

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def setup_distributed() -> tuple[int, int, int]:
    """Initialize DDP if launched via torchrun, else return single-GPU defaults.

    Returns:
        (rank, local_rank, world_size)
    """
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    return 0, 0, 1


def is_main_process(rank: int) -> bool:
    return rank == 0


# ---------------------------------------------------------------------------
# Dataset — loads coarse/fine SLAT pairs for SLAT-space diffusion
# ---------------------------------------------------------------------------

class SlatPairDataset(Dataset):
    """Dataset of coarse/fine SLAT pairs for Stage 2 SLAT-space training.

    Each sample provides:
      - coarse_slat:    (N, voxel_dim)  coarse SLAT features (512 model)
      - fine_slat:      (N, voxel_dim)  fine SLAT features (1024 model)
      - positions:      (N, 3)          integer voxel coordinates
      - cond_features:  (M, 1024) or None — DINOv2 conditioning features
      - cond_mask:      (M,) bool or None — foreground mask for cond tokens
      - uid:            str             model identifier

    Both coarse and fine SLAT are normalized to TRELLIS.2's SLAT space
    (zero-mean, unit-std per channel) before returning.
    """

    def __init__(
        self,
        data_dir: str,
        max_tokens: int = 4096,
        voxel_dim: int = 32,
        slat_mean: list[float] = None,
        slat_std: list[float] = None,
        overfit_uid: str = None,
    ):
        self.data_dir = Path(data_dir)
        self.max_tokens = max_tokens
        self.voxel_dim = voxel_dim
        self._warned_uids: set[str] = set()

        # SLAT normalization from TRELLIS.2's pipeline.json
        if slat_mean is not None and slat_std is not None:
            self.slat_mean = torch.tensor(slat_mean, dtype=torch.float32)
            self.slat_std = torch.tensor(slat_std, dtype=torch.float32)
        else:
            # Default: TRELLIS.2 4B shape_slat_normalization
            self.slat_mean = torch.tensor([
                0.781296, 0.018091, -0.495192, -0.558457, 1.06053, 0.093252,
                1.518149, -0.933218, -0.732996, 2.604095, -0.118341, -2.143904,
                0.495076, -2.179512, -2.130751, -0.996944, 0.261421, -2.217463,
                1.260067, -0.150213, 3.790713, 1.481266, -1.046058, -1.523667,
                -0.059621, 2.22078, 1.621212, 0.87723, 0.567247, -3.175944,
                -3.186688, 1.578665,
            ], dtype=torch.float32)
            self.slat_std = torch.tensor([
                5.972266, 4.706852, 5.44501, 5.209927, 5.32022, 4.547237,
                5.020802, 5.444004, 5.226681, 5.683095, 4.831436, 5.286469,
                5.652043, 5.367606, 5.525084, 4.730578, 4.805265, 5.124013,
                5.530808, 5.619001, 5.10393, 5.41767, 5.269677, 5.547194,
                5.634698, 5.235274, 6.110351, 5.511298, 6.237273, 4.879207,
                5.347008, 5.405691,
            ], dtype=torch.float32)

        # Load or discover pairs
        manifest = self.data_dir / "pairs_manifest.json"
        if manifest.exists():
            with open(manifest) as f:
                self.pairs = json.load(f)
        else:
            self.pairs = self._discover_pairs()

        if len(self.pairs) == 0:
            raise RuntimeError(
                f"No valid SLAT pairs found in {data_dir}. "
                f"Each pair directory must contain coarse_slat.npy and fine_slat.npy."
            )

        # Filter to only valid pairs
        valid_pairs = [p for p in self.pairs if "coarse_slat" in p and "fine_slat" in p]
        if len(valid_pairs) < len(self.pairs):
            n_skipped = len(self.pairs) - len(valid_pairs)
            logger.warning(f"Skipping {n_skipped} pairs without coarse_slat/fine_slat.")
        self.pairs = valid_pairs

        if len(self.pairs) == 0:
            raise RuntimeError(
                f"No pairs with coarse_slat.npy + fine_slat.npy found in {data_dir}."
            )

        # Overfit mode: use only a single pair (repeated)
        if overfit_uid:
            overfit_pairs = [p for p in self.pairs if overfit_uid in p.get("uid", "")]
            if not overfit_pairs:
                overfit_pairs = [p for p in self.pairs if overfit_uid in p.get("coarse_slat", "")]
            if overfit_pairs:
                self.pairs = overfit_pairs
                logger.info(f"OVERFIT MODE: using {len(self.pairs)} pair(s) matching '{overfit_uid}'")
            else:
                logger.warning(f"OVERFIT MODE: no pairs match '{overfit_uid}', using all pairs")

        logger.info(f"Loaded {len(self.pairs)} SLAT training pairs from {data_dir}")

    def _discover_pairs(self) -> list[dict]:
        pairs = []
        search_dirs = [self.data_dir]
        shard_dirs = sorted(self.data_dir.glob("shard_*"))
        search_dirs.extend(shard_dirs)

        for parent_dir in search_dirs:
            if not parent_dir.is_dir():
                continue
            for d in sorted(parent_dir.iterdir()):
                if not d.is_dir():
                    continue
                if parent_dir == self.data_dir and d.name.startswith("shard_"):
                    continue
                coarse = d / "coarse_slat.npy"
                fine = d / "fine_slat.npy"
                if coarse.exists() and fine.exists():
                    entry = {
                        "uid": d.name,
                        "coarse_slat": str(coarse),
                        "fine_slat": str(fine),
                        "positions": str(d / "positions.npy"),
                    }
                    cond_path = d / "cond_features.npy"
                    if cond_path.exists():
                        entry["cond_features"] = str(cond_path)
                    pairs.append(entry)
        return pairs

    def _normalize_slat(self, slat: torch.Tensor) -> torch.Tensor:
        """Normalize SLAT features to zero-mean, unit-std per channel."""
        return (slat - self.slat_mean) / self.slat_std

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        pair = self.pairs[idx]
        uid = pair.get("uid", "unknown")

        try:
            coarse_slat = torch.from_numpy(np.load(pair["coarse_slat"])).float()
            fine_slat = torch.from_numpy(np.load(pair["fine_slat"])).float()
            positions = torch.from_numpy(np.load(pair["positions"])).float()

            # Reconcile lengths
            min_len = min(coarse_slat.shape[0], fine_slat.shape[0], positions.shape[0])
            coarse_slat = coarse_slat[:min_len]
            fine_slat = fine_slat[:min_len]
            positions = positions[:min_len]

            # Ensure correct shapes
            if coarse_slat.dim() == 1:
                coarse_slat = coarse_slat.unsqueeze(-1)
            if fine_slat.dim() == 1:
                fine_slat = fine_slat.unsqueeze(-1)

            # Normalize to TRELLIS.2's SLAT space (zero-mean, unit-std per channel)
            coarse_slat = self._normalize_slat(coarse_slat)
            fine_slat = self._normalize_slat(fine_slat)

            # --- Token subsampling ---
            N_total = coarse_slat.shape[0]
            N_target = min(self.max_tokens, N_total)

            if N_total > N_target:
                sel = torch.randperm(N_total)[:N_target]
                coarse_slat = coarse_slat[sel]
                fine_slat = fine_slat[sel]
                positions = positions[sel]

            # Pad if fewer tokens than max_tokens
            if coarse_slat.shape[0] < self.max_tokens:
                pad_n = self.max_tokens - coarse_slat.shape[0]
                coarse_slat = F.pad(coarse_slat, (0, 0, 0, pad_n))
                fine_slat = F.pad(fine_slat, (0, 0, 0, pad_n))
                positions = F.pad(positions, (0, 0, 0, pad_n))

            # --- Optional: DINOv2 conditioning ---
            cond_features = None
            cond_mask = None
            if pair.get("cond_features"):
                cond_path = Path(pair["cond_features"])
                if cond_path.exists():
                    cond_features = torch.from_numpy(np.load(str(cond_path))).float()
                    cond_mask = torch.ones(cond_features.shape[0], dtype=torch.bool)

        except Exception as e:
            if uid not in self._warned_uids:
                logger.warning(f"Failed to load SLAT pair '{uid}': {e}")
                self._warned_uids.add(uid)
            alt_idx = torch.randint(0, len(self), (1,)).item()
            if alt_idx == idx:
                alt_idx = (idx + 1) % len(self)
            return self[alt_idx]

        return {
            "coarse_slat": coarse_slat,        # (N, voxel_dim) normalized
            "fine_slat": fine_slat,             # (N, voxel_dim) normalized
            "positions": positions,             # (N, 3)
            "cond_features": cond_features,     # (M, cond_dim) or None
            "cond_mask": cond_mask,             # (M,) bool or None
            "uid": uid,
        }


def slatpair_collate_fn(batch: list[dict]) -> dict:
    """Custom collate that handles optional variable-length cond_features."""
    result = {
        "coarse_slat": torch.stack([b["coarse_slat"] for b in batch]),
        "fine_slat": torch.stack([b["fine_slat"] for b in batch]),
        "positions": torch.stack([b["positions"] for b in batch]),
        "uid": [b["uid"] for b in batch],
    }

    # DINO conditioning: mixed batches are allowed. Samples without conditioning
    # get zero features and an all-false mask, which the attention path treats
    # as an unconditional example.
    samples_with_cond = [b for b in batch if b.get("cond_features") is not None]
    if samples_with_cond:
        max_M = max(b["cond_features"].shape[0] for b in samples_with_cond)
        cond_dim = samples_with_cond[0]["cond_features"].shape[1]
        cond_feats = torch.zeros(len(batch), max_M, cond_dim)
        cond_masks = torch.zeros(len(batch), max_M, dtype=torch.bool)
        for i, b in enumerate(batch):
            if b.get("cond_features") is None:
                continue
            M = b["cond_features"].shape[0]
            cond_feats[i, :M] = b["cond_features"]
            if b.get("cond_mask") is not None:
                cond_masks[i, :M] = b["cond_mask"]
            else:
                cond_masks[i, :M] = True
        result["cond_features"] = cond_feats
        result["cond_mask"] = cond_masks
    else:
        result["cond_features"] = None
        result["cond_mask"] = None

    return result


# ---------------------------------------------------------------------------
# Progressive schedule helper
# ---------------------------------------------------------------------------

def get_progressive_value(schedule: list[dict], step: int, key: str, default):
    """Look up the active value for ``key`` at a given training step."""
    value = default
    for entry in sorted(schedule, key=lambda e: e.get("step", 0)):
        if step >= entry.get("step", 0) and key in entry:
            value = entry[key]
    return value


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Stage 2 trainer — direct residual prediction with frozen backbone."""

    def __init__(self, config: dict, rank: int = 0, local_rank: int = 0, world_size: int = 1):
        self.config = config
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.is_main = is_main_process(rank)

        if world_size > 1:
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        trainable_blocks = config.get("trainable_blocks", 3)

        # --- Model ---
        pretrained = config.get("pretrained_checkpoint")
        if pretrained and Path(pretrained).exists():
            if self.is_main:
                print(f"Loading pretrained weights from {pretrained}")
            self.model = RefinementDiT.from_pretrained(
                pretrained,
                num_layers=config.get("num_layers", 12),
                voxel_dim=config.get("voxel_dim", 32),
                model_dim=config.get("model_dim", 1536),
                num_heads=config.get("num_heads", 12),
                cond_dim=config.get("cond_dim", 1024),
                mlp_ratio=config.get("mlp_ratio", 5.3334),
                use_checkpoint=config.get("use_checkpoint", True),
            ).to(self.device)
        else:
            if self.is_main:
                print("No pretrained checkpoint — training from scratch")
            self.model = RefinementDiT(
                voxel_dim=config.get("voxel_dim", 32),
                model_dim=config.get("model_dim", 1536),
                num_heads=config.get("num_heads", 12),
                num_layers=config.get("num_layers", 12),
                cond_dim=config.get("cond_dim", 1024),
                mlp_ratio=config.get("mlp_ratio", 5.3334),
                use_checkpoint=config.get("use_checkpoint", True),
            ).to(self.device)

        # Freeze backbone — only train last K blocks + out_head
        raw_model = self.model.module if isinstance(self.model, DDP) else self.model
        raw_model.freeze_backbone(trainable_blocks)

        # Wrap in DDP if multi-GPU (find_unused_parameters for frozen params)
        if world_size > 1:
            self.model = DDP(self.model, device_ids=[local_rank], find_unused_parameters=True)
            if self.is_main:
                print(f"DDP enabled: {world_size} GPUs")

        if self.is_main:
            raw_model = self.model.module if isinstance(self.model, DDP) else self.model
            total_params = sum(p.numel() for p in raw_model.parameters())
            trainable = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
            print(f"Model params: {total_params / 1e6:.1f}M total, {trainable / 1e6:.1f}M trainable")

        # --- Optimizer (only trainable params) ---
        raw_model = self.model.module if isinstance(self.model, DDP) else self.model
        trainable_params = [p for p in raw_model.parameters() if p.requires_grad]

        base_lr = config.get("learning_rate", 1e-4)
        # Sqrt LR scaling for multi-GPU
        effective_lr = base_lr * math.sqrt(world_size) if world_size > 1 else base_lr

        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=effective_lr,
            weight_decay=config.get("weight_decay", 0.01),
        )

        total_steps = config.get("total_steps", 100_000)
        warmup_steps = config.get("warmup_steps", 200)
        self.warmup_steps = warmup_steps

        # Cosine schedule with linear warmup
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)  # linear warmup 0→1
            # Cosine decay after warmup
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lr_lambda
        )

        # --- Progressive schedule ---
        self.progressive = config.get("progressive_schedule", [])

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

        # WandB (rank 0 only)
        self.wandb = None
        if config.get("use_wandb", False) and self.is_main:
            import wandb
            wandb.init(project="clearmesh", config=config)
            self.wandb = wandb

    # ------------------------------------------------------------------

    def _handle_preemption(self, signum, frame):
        print(f"\n!!! PREEMPTION (rank {self.rank}) — saving emergency checkpoint !!!")
        self._emergency_save = True

    def _build_dataloader(self, max_tokens: int, batch_size: int) -> DataLoader:
        """Build dataloader with the given max token count and batch size."""
        dataset = SlatPairDataset(
            data_dir=self.config["data_dir"],
            max_tokens=max_tokens,
            voxel_dim=self.config.get("voxel_dim", 32),
            slat_mean=self.config.get("slat_mean"),
            slat_std=self.config.get("slat_std"),
            overfit_uid=self.config.get("overfit_uid", None),
        )

        sampler = None
        shuffle = True
        if self.world_size > 1:
            sampler = DistributedSampler(
                dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True,
            )
            shuffle = False  # sampler handles shuffling

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
        raw_model = self.model.module if isinstance(self.model, DDP) else self.model
        ckpt = {
            "model": raw_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
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
        raw_model = self.model.module if isinstance(self.model, DDP) else self.model
        raw_model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
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

        coarse_slat = batch["coarse_slat"].to(self.device)    # (B, N, voxel_dim) normalized
        fine_slat = batch["fine_slat"].to(self.device)         # (B, N, voxel_dim) normalized
        positions = batch["positions"].to(self.device)          # (B, N, 3)
        B = coarse_slat.shape[0]

        # DINOv2 conditioning (optional — works without it)
        cond_features = batch.get("cond_features")
        cond_mask = batch.get("cond_mask")
        if cond_features is not None:
            cond_features = cond_features.to(self.device)
        if cond_mask is not None:
            cond_mask = cond_mask.to(self.device)

        # Ground truth delta
        target_delta = fine_slat - coarse_slat  # (B, N, 32)

        # --- Forward pass with bf16 autocast ---
        # Fixed t=0 passed internally by model (no timestep needed)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred_delta = self.model(
                coarse_slat, positions,
                cond_features=cond_features,
                cond_mask=cond_mask,
            )

            # Primary loss: L1 on predicted delta vs ground truth delta
            delta_l1 = F.l1_loss(pred_delta, target_delta)

            # Secondary: direct reconstruction L1 (coarse + delta vs fine)
            pred_fine = coarse_slat + pred_delta
            recon_l1 = F.l1_loss(pred_fine, fine_slat)

            # Also track MSE for comparison with diffusion baseline
            delta_mse = F.mse_loss(pred_delta, target_delta)

            losses = {
                "delta_l1": delta_l1,
                "recon_l1": recon_l1,
                "delta_mse": delta_mse,
                "total": delta_l1,  # primary loss for backprop
            }

        # Backward
        self.optimizer.zero_grad()
        losses["total"].backward()

        # Only clip trainable params
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(
            trainable_params, self.config.get("max_grad_norm", 1.0)
        )
        self.optimizer.step()
        self.scheduler.step()

        return {k: v.item() for k, v in losses.items()}

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self):
        total_steps = self.config.get("total_steps", 100_000)
        save_interval = self.config.get("save_interval", 1000)
        log_interval = self.config.get("log_interval", 50)

        if not self._checkpoint_loaded:
            self.load_checkpoint()

        # Determine current progressive values
        cur_tokens = get_progressive_value(
            self.progressive, self.global_step, "num_tokens", 4096
        )
        cur_batch = get_progressive_value(
            self.progressive, self.global_step, "batch_size",
            self.config.get("batch_size", 4),
        )
        dataloader = self._build_dataloader(cur_tokens, cur_batch)

        if self.is_main:
            raw_model = self.model.module if isinstance(self.model, DDP) else self.model
            trainable = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
            print(f"\n{'='*60}")
            print(f"=== Training Stage 2 RefinementDiT (Residual) ===")
            print(f"  Mode:            Direct residual prediction (L1 loss)")
            print(f"  Trainable:       {trainable / 1e6:.1f}M params")
            print(f"  Total steps:     {total_steps}")
            print(f"  Current step:    {self.global_step}")
            print(f"  Dataset size:    {len(dataloader.dataset)}")
            print(f"  Batch size:      {cur_batch} (per GPU)")
            print(f"  World size:      {self.world_size} GPU(s)")
            print(f"  Effective batch: {cur_batch * self.world_size}")
            print(f"  Initial tokens:  {cur_tokens}")
            print(f"  Progressive:     {self.progressive}")
            print(f"  bf16:            enabled")
            print(f"  Grad checkpoint: {self.config.get('use_checkpoint', True)}")
            print(f"{'='*60}\n")

        with open("/tmp/train.pid", "w") as f:
            f.write(str(os.getpid()))

        pbar = tqdm(
            total=total_steps, initial=self.global_step, desc="Training",
            disable=not self.is_main,
        )

        while self.global_step < total_steps:
            # Set epoch for DistributedSampler
            if hasattr(dataloader, "sampler") and isinstance(dataloader.sampler, DistributedSampler):
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

                # Progressive schedule: rebuild dataloader if tokens or batch size changed
                new_tokens = get_progressive_value(
                    self.progressive, self.global_step, "num_tokens", 4096
                )
                new_batch = get_progressive_value(
                    self.progressive, self.global_step, "batch_size",
                    self.config.get("batch_size", 4),
                )
                if new_tokens != cur_tokens or new_batch != cur_batch:
                    if self.is_main:
                        print(f"\n>>> Progressive: tokens {cur_tokens}→{new_tokens}, "
                              f"batch {cur_batch}→{new_batch} at step {self.global_step}")
                    cur_tokens = new_tokens
                    cur_batch = new_batch
                    dataloader = self._build_dataloader(cur_tokens, cur_batch)
                    break  # restart epoch with new dataloader

                # Train step
                losses = self.train_step(batch)
                self.global_step += 1
                pbar.update(1)

                # Logging (rank 0 only)
                if self.is_main and self.global_step % log_interval == 0:
                    loss_str = " | ".join(f"{k}: {v:.4f}" for k, v in losses.items())
                    pbar.set_postfix_str(loss_str)
                    if self.wandb:
                        self.wandb.log(
                            {f"loss/{k}": v for k, v in losses.items()},
                            step=self.global_step,
                        )

                # Checkpoint (rank 0 only, but all ranks wait)
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
    parser = argparse.ArgumentParser(description="Train Stage 2 RefinementDiT")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--resume_from", default=None)
    args = parser.parse_args()

    rank, local_rank, world_size = setup_distributed()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.output_dir:
        config["output_dir"] = args.output_dir

    trainer = Trainer(config, rank=rank, local_rank=local_rank, world_size=world_size)
    if args.resume_from:
        trainer.load_checkpoint(args.resume_from)
    trainer.train()


if __name__ == "__main__":
    main()
