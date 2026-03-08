#!/usr/bin/env python3
"""Training loop for Stage 2 RefinementDiT.

Features:
  - Loads TRELLIS.2 pretrained weights (first N blocks)
  - bf16 mixed-precision via torch.autocast
  - Gradient checkpointing (configured in model)
  - Progressive training schedule (token count ramp)
  - Near-surface supervision point sampling
  - Image token masking for DINO conditioning
  - FlexiCubes-in-the-loop mesh extraction (every K steps)
  - Checkpoint every N steps (Spot VM resilience)
  - SIGUSR1 handler for emergency checkpoint on preemption
  - WandB logging

Usage:
    python -m clearmesh.stage2.train \\
        --config configs/train_stage2_flexicubes.yaml
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
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from clearmesh.stage2.losses import ClearMeshLoss
from clearmesh.stage2.model import RefinementDiT

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset — loads coarse/fine voxel pairs with near-surface sampling
# ---------------------------------------------------------------------------

class VoxelPairDataset(Dataset):
    """Dataset of coarse/fine O-Voxel pairs for Stage 2 training.

    Each sample provides:
      - coarse_features:  (N, voxel_dim)  features at occupied voxel positions
      - positions:        (N, 3)          integer voxel coordinates
      - gt_sdf:           (N, 1)          ground-truth SDF at those positions
      - cond_features:    (M, 1024) or None — DINOv2 conditioning features
      - cond_mask:        (M,) bool or None — foreground mask for cond tokens
      - uid:              str             model identifier

    Sampling strategy (for SDF imbalance mitigation):
      - near_surface_ratio of points sampled where |SDF| < tau
      - remaining points sampled uniformly
    """

    def __init__(
        self,
        data_dir: str,
        max_tokens: int = 4096,
        near_surface_ratio: float = 0.6,
        sdf_truncation: float = 0.1,
        voxel_dim: int = 32,
    ):
        self.data_dir = Path(data_dir)
        self.max_tokens = max_tokens
        self.near_surface_ratio = near_surface_ratio
        self.sdf_truncation = sdf_truncation
        self.voxel_dim = voxel_dim
        self._warned_uids: set[str] = set()

        # Load or discover pairs
        manifest = self.data_dir / "pairs_manifest.json"
        if manifest.exists():
            with open(manifest) as f:
                self.pairs = json.load(f)
        else:
            self.pairs = self._discover_pairs()

        if len(self.pairs) == 0:
            raise RuntimeError(
                f"No valid training pairs found in {data_dir}. "
                f"Each pair directory must contain coarse_voxels.npy and fine_sdf.npy. "
                f"Run convert_pairs_to_sdf.py first."
            )

        # Filter to only valid pairs (must have coarse_voxels and fine_sdf)
        valid_pairs = [p for p in self.pairs if "coarse_voxels" in p]
        if len(valid_pairs) < len(self.pairs):
            n_skipped = len(self.pairs) - len(valid_pairs)
            logger.warning(
                f"Skipping {n_skipped} pairs without pre-computed voxel features. "
                f"Run convert_pairs_to_sdf.py to generate missing files."
            )
        self.pairs = valid_pairs

        if len(self.pairs) == 0:
            raise RuntimeError(
                f"No pairs with coarse_voxels.npy found in {data_dir}. "
                f"Run convert_pairs_to_sdf.py first."
            )

        logger.info(f"Loaded {len(self.pairs)} training pairs from {data_dir}")

    def _discover_pairs(self) -> list[dict]:
        pairs = []
        # Check for shard subdirectories (generate_pairs.py creates shard_0/, shard_1/, etc.)
        shard_dirs = sorted(self.data_dir.glob("shard_*"))
        search_dirs = shard_dirs if shard_dirs else [self.data_dir]

        for parent_dir in search_dirs:
            if not parent_dir.is_dir():
                continue
            for d in sorted(parent_dir.iterdir()):
                if not d.is_dir():
                    continue
                coarse = d / "coarse_voxels.npy"
                fine = d / "fine_sdf.npy"
                if coarse.exists() and fine.exists():
                    entry = {
                        "uid": d.name,
                        "coarse_voxels": str(coarse),
                        "fine_sdf": str(fine),
                        "positions": str(d / "positions.npy"),
                    }
                    # Optional: DINOv2 conditioning features
                    cond_path = d / "cond_features.npy"
                    if cond_path.exists():
                        entry["cond_features"] = str(cond_path)
                    # Optional: rendered conditioning image (for deriving cond_mask)
                    rendered_path = d / "rendered.png"
                    if rendered_path.exists():
                        entry["rendered"] = str(rendered_path)
                    pairs.append(entry)
        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        pair = self.pairs[idx]
        uid = pair.get("uid", "unknown")

        try:
            features = torch.from_numpy(np.load(pair["coarse_voxels"])).float()
            gt_sdf = torch.from_numpy(np.load(pair["fine_sdf"])).float()
            positions = torch.from_numpy(np.load(pair["positions"])).float()

            # --- Near-surface sampling ---
            N_total = features.shape[0]
            N_target = min(self.max_tokens, N_total)

            if N_total > N_target and gt_sdf is not None:
                near_mask = gt_sdf.abs().squeeze(-1) < self.sdf_truncation
                near_idx = near_mask.nonzero(as_tuple=True)[0]
                far_idx = (~near_mask).nonzero(as_tuple=True)[0]

                N_near = min(int(N_target * self.near_surface_ratio), len(near_idx))
                N_far = N_target - N_near

                if len(near_idx) >= N_near and len(far_idx) >= N_far:
                    sel_near = near_idx[torch.randperm(len(near_idx))[:N_near]]
                    sel_far = far_idx[torch.randperm(len(far_idx))[:N_far]]
                    sel = torch.cat([sel_near, sel_far])
                else:
                    sel = torch.randperm(N_total)[:N_target]

                features = features[sel]
                gt_sdf = gt_sdf[sel] if gt_sdf.dim() >= 1 else gt_sdf
                positions = positions[sel]
            elif N_total > N_target:
                sel = torch.randperm(N_total)[:N_target]
                features = features[sel]
                gt_sdf = gt_sdf[sel]
                positions = positions[sel]

            # Pad if fewer tokens than max_tokens
            if features.shape[0] < self.max_tokens:
                pad_n = self.max_tokens - features.shape[0]
                features = F.pad(features, (0, 0, 0, pad_n))
                gt_sdf = F.pad(gt_sdf, (0, 0, 0, pad_n))
                positions = F.pad(positions, (0, 0, 0, pad_n))

            # Ensure gt_sdf is (N, 1)
            if gt_sdf.dim() == 1:
                gt_sdf = gt_sdf.unsqueeze(-1)

            # --- Optional: DINOv2 conditioning ---
            cond_features = None
            cond_mask = None
            if pair.get("cond_features"):
                cond_path = Path(pair["cond_features"])
                if cond_path.exists():
                    cond_features = torch.from_numpy(np.load(str(cond_path))).float()
                    # Derive foreground mask from rendered image alpha
                    rendered_path = pair.get("rendered")
                    if rendered_path and Path(rendered_path).exists():
                        try:
                            from PIL import Image
                            img = Image.open(rendered_path).convert("RGBA")
                            alpha = np.array(img)[:, :, 3]
                            M = cond_features.shape[0]
                            grid_size = int(M ** 0.5)
                            # CLS token adjustment
                            if grid_size * grid_size != M and (grid_size + 1) * (grid_size + 1) != M:
                                grid_size = int((M - 1) ** 0.5)
                                if grid_size * grid_size == M - 1:
                                    # Has CLS token — mask all True for CLS, patch mask from alpha
                                    alpha_small = np.array(
                                        Image.fromarray(alpha).resize(
                                            (grid_size, grid_size), Image.BILINEAR
                                        )
                                    )
                                    patch_mask = (alpha_small > 128).flatten()
                                    cond_mask = torch.from_numpy(
                                        np.concatenate([[True], patch_mask])
                                    )
                                else:
                                    cond_mask = torch.ones(M, dtype=torch.bool)
                            else:
                                alpha_small = np.array(
                                    Image.fromarray(alpha).resize(
                                        (grid_size, grid_size), Image.BILINEAR
                                    )
                                )
                                cond_mask = torch.from_numpy(
                                    (alpha_small > 128).flatten()
                                )
                        except Exception:
                            cond_mask = torch.ones(cond_features.shape[0], dtype=torch.bool)
                    else:
                        cond_mask = torch.ones(cond_features.shape[0], dtype=torch.bool)

        except Exception as e:
            if uid not in self._warned_uids:
                logger.warning(f"Failed to load pair '{uid}': {e}")
                self._warned_uids.add(uid)
            # Skip to a different valid sample (with recursion guard)
            alt_idx = torch.randint(0, len(self), (1,)).item()
            if alt_idx == idx:
                alt_idx = (idx + 1) % len(self)
            return self[alt_idx]

        return {
            "coarse_features": features,       # (N, voxel_dim)
            "positions": positions,             # (N, 3)
            "gt_sdf": gt_sdf,                  # (N, 1)
            "cond_features": cond_features,    # (M, cond_dim) or None
            "cond_mask": cond_mask,            # (M,) bool or None
            "uid": uid,
        }


def voxelpair_collate_fn(batch: list[dict]) -> dict:
    """Custom collate that handles optional variable-length cond_features."""
    result = {
        "coarse_features": torch.stack([b["coarse_features"] for b in batch]),
        "positions": torch.stack([b["positions"] for b in batch]),
        "gt_sdf": torch.stack([b["gt_sdf"] for b in batch]),
        "uid": [b["uid"] for b in batch],
    }

    # DINOv2 conditioning: only include if ALL samples in batch have it
    if all(b.get("cond_features") is not None for b in batch):
        max_M = max(b["cond_features"].shape[0] for b in batch)
        cond_dim = batch[0]["cond_features"].shape[1]
        cond_feats = torch.zeros(len(batch), max_M, cond_dim)
        cond_masks = torch.zeros(len(batch), max_M, dtype=torch.bool)
        for i, b in enumerate(batch):
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
# FlexiCubes Extractor
# ---------------------------------------------------------------------------

class FlexiCubesExtractor:
    """Wrapper around NVIDIA Kaolin FlexiCubes."""

    def __init__(self, resolution: int = 128, device: str = "cuda"):
        self.resolution = resolution
        self.device = device
        self._fc = None

    @property
    def fc(self):
        if self._fc is None:
            from kaolin.non_commercial import FlexiCubes
            self._fc = FlexiCubes(device=self.device)
        return self._fc

    def extract(self, sdf_grid: torch.Tensor):
        if sdf_grid.dim() == 4:
            sdf_grid = sdf_grid[0]
        R = sdf_grid.shape[0]
        x = torch.linspace(-1, 1, R, device=sdf_grid.device)
        gx, gy, gz = torch.meshgrid(x, x, x, indexing="ij")
        pts = torch.stack([gx, gy, gz], -1).reshape(-1, 3)
        return self.fc(pts, sdf_grid.reshape(-1), self._cubes(R), R)

    def _cubes(self, R: int) -> torch.Tensor:
        cubes = []
        for i in range(R - 1):
            for j in range(R - 1):
                for k in range(R - 1):
                    v0 = i * R * R + j * R + k
                    cubes.append([
                        v0, v0 + 1, v0 + R, v0 + R + 1,
                        v0 + R * R, v0 + R * R + 1,
                        v0 + R * R + R, v0 + R * R + R + 1,
                    ])
        return torch.tensor(cubes, dtype=torch.long, device=self.device)


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
    """Stage 2 trainer with pretrained init, progressive schedule, bf16."""

    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Model ---
        pretrained = config.get("pretrained_checkpoint")
        if pretrained and Path(pretrained).exists():
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

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model params: {total_params / 1e6:.1f}M total, {trainable / 1e6:.1f}M trainable")

        # --- Loss ---
        self.criterion = ClearMeshLoss(
            sdf_weight=config.get("sdf_weight", 1.0),
            eikonal_weight=config.get("eikonal_weight", 0.1),
            chamfer_weight=config.get("chamfer_weight", 1.0),
            normal_weight=config.get("normal_weight", 0.5),
            edge_weight=config.get("edge_weight", 0.3),
            watertight_weight=config.get("watertight_weight", 0.2),
            sdf_truncation=config.get("sdf_truncation", 0.1),
            sdf_surface_weight=config.get("sdf_surface_weight", 5.0),
        )

        # --- Optimizer ---
        # Lower LR for pretrained layers, higher for fresh layers
        pretrained_params, fresh_params = [], []
        fresh_names = {"sdf_proj", "out_layer"}
        for name, param in self.model.named_parameters():
            if any(fn in name for fn in fresh_names):
                fresh_params.append(param)
            else:
                pretrained_params.append(param)

        self.optimizer = torch.optim.AdamW([
            {"params": pretrained_params, "lr": config.get("learning_rate", 5e-5)},
            {"params": fresh_params, "lr": config.get("learning_rate", 5e-5) * 5},
        ], weight_decay=config.get("weight_decay", 0.01))

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.get("total_steps", 100_000)
        )

        # --- FlexiCubes ---
        fc_res = config.get("flexicubes_resolution", 0)
        if fc_res > 0:
            self.flexicubes = FlexiCubesExtractor(
                resolution=fc_res, device=str(self.device),
            )
            logger.info(f"FlexiCubes enabled at resolution {fc_res}")
        else:
            self.flexicubes = None
            logger.info("FlexiCubes disabled (flexicubes_resolution=0)")

        # --- Progressive schedule ---
        self.progressive = config.get("progressive_schedule", [])

        # --- Training state ---
        self.global_step = 0
        self.epoch = 0
        self.output_dir = Path(config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Preemption handling
        self._emergency_save = False
        signal.signal(signal.SIGUSR1, self._handle_preemption)

        # WandB
        self.wandb = None
        if config.get("use_wandb", False):
            import wandb
            wandb.init(project="clearmesh", config=config)
            self.wandb = wandb

    # ------------------------------------------------------------------

    def _handle_preemption(self, signum, frame):
        print("\n!!! PREEMPTION — saving emergency checkpoint !!!")
        self._emergency_save = True

    def _build_dataloader(self, max_tokens: int) -> DataLoader:
        """Build dataloader with the given max token count."""
        dataset = VoxelPairDataset(
            data_dir=self.config["data_dir"],
            max_tokens=max_tokens,
            near_surface_ratio=self.config.get("near_surface_ratio", 0.6),
            sdf_truncation=self.config.get("sdf_truncation", 0.1),
            voxel_dim=self.config.get("voxel_dim", 32),
        )
        return DataLoader(
            dataset,
            batch_size=self.config.get("batch_size", 4),
            shuffle=True,
            num_workers=self.config.get("num_workers", 4),
            pin_memory=True,
            drop_last=True,
            collate_fn=voxelpair_collate_fn,
        )

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, tag: str = "latest"):
        ckpt = {
            "model": self.model.state_dict(),
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
            print("No checkpoint found, starting from scratch.")
            return
        print(f"Resuming from {p}")
        ckpt = torch.load(p, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.global_step = ckpt["global_step"]
        self.epoch = ckpt["epoch"]
        print(f"Resumed at step {self.global_step}, epoch {self.epoch}")

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_step(self, batch: dict) -> dict[str, float]:
        self.model.train()

        features = batch["coarse_features"].to(self.device)   # (B, N, voxel_dim)
        positions = batch["positions"].to(self.device)         # (B, N, 3)
        gt_sdf = batch["gt_sdf"].to(self.device)              # (B, N, 1)
        B = features.shape[0]

        # DINOv2 conditioning (optional — works without it)
        cond_features = batch.get("cond_features")
        cond_mask = batch.get("cond_mask")
        if cond_features is not None:
            cond_features = cond_features.to(self.device)
        if cond_mask is not None:
            cond_mask = cond_mask.to(self.device)

        # Diffusion forward process
        t = torch.rand(B, device=self.device)
        noise = torch.randn_like(gt_sdf)
        alpha = (1 - t).view(B, 1, 1)
        noisy_sdf = alpha.sqrt() * gt_sdf + (1 - alpha).sqrt() * noise

        # --- Forward pass with bf16 autocast ---
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred_noise = self.model(
                features, positions, t,
                cond_features=cond_features,
                cond_mask=cond_mask,
                noisy_sdf=noisy_sdf,
            )

            # Reconstruct predicted SDF from noise prediction
            pred_sdf = (noisy_sdf - (1 - alpha).sqrt() * pred_noise) / alpha.sqrt()

            # Losses (pass positions for spatial eikonal regularisation)
            losses = self.criterion(pred_sdf=pred_sdf, gt_sdf=gt_sdf, positions=positions)

        # FlexiCubes mesh-space losses (every K steps)
        fc_interval = self.config.get("flexicubes_interval", 10)
        fc_res = self.config.get("flexicubes_resolution", 0)

        if (self.flexicubes is not None and fc_interval > 0 and fc_res > 0
                and self.global_step % fc_interval == 0):
            try:
                R_fc = fc_res
                R_data = self.config.get("resolution", 128)

                # Scatter sparse token predictions into a small dense grid
                # via nearest-neighbor interpolation
                pos_norm = (positions[0].float() / max(R_data - 1, 1)) * 2 - 1  # (N, 3) in [-1, 1]
                lin = torch.linspace(-1, 1, R_fc, device=self.device)
                gx, gy, gz = torch.meshgrid(lin, lin, lin, indexing="ij")
                grid_pts = torch.stack([gx, gy, gz], -1).reshape(-1, 3)  # (R_fc^3, 3)

                # Find nearest training token for each grid point
                dists = torch.cdist(grid_pts.unsqueeze(0), pos_norm.unsqueeze(0))
                nn_idx = dists.argmin(dim=2).squeeze(0)  # (R_fc^3,)

                pred_grid = pred_sdf[0, nn_idx, 0].float().view(R_fc, R_fc, R_fc)
                gt_grid = gt_sdf[0, nn_idx, 0].float().view(R_fc, R_fc, R_fc)

                verts, faces = self.flexicubes.extract(pred_grid)
                if verts is not None and verts.shape[0] > 0:
                    gt_v, gt_f = self.flexicubes.extract(gt_grid)
                    if gt_v is not None and gt_v.shape[0] > 0:
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            losses = self.criterion(
                                pred_sdf=pred_sdf, gt_sdf=gt_sdf,
                                extracted_vertices=verts,
                                extracted_faces=faces,
                                gt_vertices=gt_v, gt_faces=gt_f,
                                pred_points=verts.unsqueeze(0),
                                gt_points=gt_v.unsqueeze(0),
                            )
            except Exception as e:
                if self.global_step % (fc_interval * 100) == 0:
                    logger.warning(
                        f"FlexiCubes extraction failed at step {self.global_step}: {e}"
                    )

        # Backward
        self.optimizer.zero_grad()
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.get("max_grad_norm", 1.0)
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
        log_interval = self.config.get("log_interval", 100)

        self.load_checkpoint()

        # Determine current progressive max_tokens
        cur_tokens = get_progressive_value(
            self.progressive, self.global_step, "num_tokens", 4096
        )
        dataloader = self._build_dataloader(cur_tokens)

        print(f"\n{'='*60}")
        print(f"=== Training Stage 2 RefinementDiT ===")
        print(f"  Total steps:     {total_steps}")
        print(f"  Current step:    {self.global_step}")
        print(f"  Dataset size:    {len(dataloader.dataset)}")
        print(f"  Batch size:      {self.config.get('batch_size', 4)}")
        print(f"  Initial tokens:  {cur_tokens}")
        print(f"  Progressive:     {self.progressive}")
        print(f"  bf16:            enabled")
        print(f"  Grad checkpoint: {self.config.get('use_checkpoint', True)}")
        print(f"{'='*60}\n")

        with open("/tmp/train.pid", "w") as f:
            f.write(str(os.getpid()))

        pbar = tqdm(total=total_steps, initial=self.global_step, desc="Training")

        while self.global_step < total_steps:
            for batch in dataloader:
                if self.global_step >= total_steps:
                    break

                # Emergency save
                if self._emergency_save:
                    self.save_checkpoint(f"emergency_{self.global_step}")
                    print("Emergency checkpoint saved. Exiting.")
                    sys.exit(0)

                # Progressive schedule: rebuild dataloader if token count changed
                new_tokens = get_progressive_value(
                    self.progressive, self.global_step, "num_tokens", 4096
                )
                if new_tokens != cur_tokens:
                    print(f"\n>>> Progressive: tokens {cur_tokens} → {new_tokens} at step {self.global_step}")
                    cur_tokens = new_tokens
                    dataloader = self._build_dataloader(cur_tokens)
                    break  # restart epoch with new dataloader

                # Train step
                losses = self.train_step(batch)
                self.global_step += 1
                pbar.update(1)

                # Logging
                if self.global_step % log_interval == 0:
                    loss_str = " | ".join(f"{k}: {v:.4f}" for k, v in losses.items())
                    pbar.set_postfix_str(loss_str)
                    if self.wandb:
                        self.wandb.log(
                            {f"loss/{k}": v for k, v in losses.items()},
                            step=self.global_step,
                        )

                # Checkpoint
                if self.global_step % save_interval == 0:
                    self.save_checkpoint(f"step_{self.global_step}")

            self.epoch += 1

        pbar.close()
        self.save_checkpoint("final")
        print(f"\nTraining complete at step {self.global_step}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train Stage 2 RefinementDiT")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--resume_from", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.output_dir:
        config["output_dir"] = args.output_dir

    trainer = Trainer(config)
    if args.resume_from:
        trainer.load_checkpoint(args.resume_from)
    trainer.train()


if __name__ == "__main__":
    main()
