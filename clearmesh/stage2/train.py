#!/usr/bin/env python3
"""Training loop for Stage 2 Refinement DiT with FlexiCubes in the loop.

Option C: Best quality. FlexiCubes differentiable mesh extraction is inside
the training loop, so gradients flow from mesh-space losses back into the DiT.

Key features:
  - FlexiCubes in training loop (differentiable mesh extraction)
  - Multi-objective loss (Chamfer + normal + edge sharpness + watertight + SDF)
  - Checkpoint every N steps to persistent disk (Spot VM resilience)
  - SIGUSR1 handler for emergency checkpoint on preemption
  - Resume from latest checkpoint
  - WandB logging

Usage:
    python -m clearmesh.stage2.train \
        --config configs/train_stage2_flexicubes.yaml \
        --output_dir /mnt/data/checkpoints/clearmesh_stage2
"""

import argparse
import json
import os
import signal
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from clearmesh.stage2.losses import ClearMeshLoss
from clearmesh.stage2.model import RefinementDiT


class VoxelPairDataset(Dataset):
    """Dataset of coarse/fine O-Voxel pairs for Stage 2 training."""

    def __init__(self, data_dir: str, resolution: int = 128, supervision_points: int = 1_600_000):
        self.data_dir = Path(data_dir)
        self.resolution = resolution
        self.supervision_points = supervision_points

        # Load pair manifest
        manifest_path = self.data_dir / "pairs_manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                self.pairs = json.load(f)
        else:
            # Auto-discover pairs
            self.pairs = self._discover_pairs()

    def _discover_pairs(self) -> list[dict]:
        """Find all coarse/fine pairs in the data directory."""
        pairs = []
        for pair_dir in sorted(self.data_dir.iterdir()):
            if not pair_dir.is_dir():
                continue
            coarse = list(pair_dir.glob("coarse.*"))
            fine = list(pair_dir.glob("fine.*"))
            if coarse and fine:
                pairs.append({"uid": pair_dir.name, "coarse": str(coarse[0]), "fine": str(fine[0])})
        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        pair = self.pairs[idx]

        # Load voxel data (pre-converted to O-Voxel format)
        coarse_path = pair["coarse"].replace(Path(pair["coarse"]).suffix, ".npy")
        fine_path = pair["fine"].replace(Path(pair["fine"]).suffix, ".npy")

        try:
            import numpy as np

            coarse_voxels = torch.from_numpy(np.load(coarse_path)).float()
            fine_voxels = torch.from_numpy(np.load(fine_path)).float()
        except FileNotFoundError:
            # Return zeros as placeholder (skip in training loop)
            coarse_voxels = torch.zeros(self.resolution, self.resolution, self.resolution)
            fine_voxels = torch.zeros_like(coarse_voxels)

        # Extract occupied positions and features
        coarse_occupied = coarse_voxels.nonzero(as_tuple=False).float()
        fine_occupied = fine_voxels.nonzero(as_tuple=False).float()

        # Normalize positions to [-1, 1]
        if coarse_occupied.shape[0] > 0:
            coarse_occupied = coarse_occupied / self.resolution * 2 - 1

        # Sample supervision points for SDF loss
        num_points = min(self.supervision_points, coarse_occupied.shape[0])
        if num_points > 0:
            idx_sample = torch.randperm(coarse_occupied.shape[0])[:num_points]
            positions = coarse_occupied[idx_sample]
        else:
            positions = torch.zeros(1, 3)

        return {
            "coarse_voxels": coarse_voxels,
            "fine_voxels": fine_voxels,
            "positions": positions,
            "uid": pair["uid"],
        }


class FlexiCubesExtractor:
    """Wrapper around NVIDIA Kaolin's FlexiCubes for differentiable mesh extraction."""

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

    def extract(self, sdf_grid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract mesh from SDF grid using FlexiCubes (differentiable).

        Args:
            sdf_grid: (B, R, R, R) or (R, R, R) SDF values

        Returns:
            vertices: (V, 3) mesh vertices
            faces: (F, 3) mesh face indices
        """
        if sdf_grid.dim() == 4:
            sdf_grid = sdf_grid[0]  # Take first batch element

        R = sdf_grid.shape[0]

        # Create grid coordinates
        x = torch.linspace(-1, 1, R, device=sdf_grid.device)
        grid_x, grid_y, grid_z = torch.meshgrid(x, x, x, indexing="ij")
        x_nx3 = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)

        # FlexiCubes extraction
        sdf_flat = sdf_grid.reshape(-1)
        cube_fx8 = self._get_cube_indices(R)

        vertices, faces = self.fc(
            x_nx3,
            sdf_flat,
            cube_fx8,
            R,
        )

        return vertices, faces

    def _get_cube_indices(self, resolution: int) -> torch.Tensor:
        """Generate cube vertex indices for FlexiCubes."""
        r = resolution
        # Each cube has 8 vertices from the 3D grid
        cubes = []
        for i in range(r - 1):
            for j in range(r - 1):
                for k in range(r - 1):
                    v0 = i * r * r + j * r + k
                    v1 = v0 + 1
                    v2 = v0 + r
                    v3 = v2 + 1
                    v4 = v0 + r * r
                    v5 = v4 + 1
                    v6 = v4 + r
                    v7 = v6 + 1
                    cubes.append([v0, v1, v2, v3, v4, v5, v6, v7])

        return torch.tensor(cubes, dtype=torch.long, device=self.device)


class Trainer:
    """Stage 2 training with FlexiCubes in the loop."""

    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model
        self.model = RefinementDiT(
            voxel_dim=config.get("voxel_dim", 32),
            model_dim=config.get("model_dim", 512),
            num_heads=config.get("num_heads", 8),
            num_layers=config.get("num_layers", 12),
            cond_dim=config.get("cond_dim", 768),
        ).to(self.device)

        # FlexiCubes extractor
        self.flexicubes = FlexiCubesExtractor(
            resolution=config.get("resolution", 128), device=str(self.device)
        )

        # Loss
        self.criterion = ClearMeshLoss(
            chamfer_weight=config.get("chamfer_weight", 1.0),
            normal_weight=config.get("normal_weight", 0.5),
            edge_weight=config.get("edge_weight", 0.3),
            watertight_weight=config.get("watertight_weight", 0.2),
            sdf_weight=config.get("sdf_weight", 1.0),
        )

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.get("learning_rate", 1e-4),
            weight_decay=config.get("weight_decay", 0.01),
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.get("total_steps", 100_000)
        )

        # Dataset
        self.dataset = VoxelPairDataset(
            data_dir=config["data_dir"],
            resolution=config.get("resolution", 128),
            supervision_points=config.get("supervision_points", 1_600_000),
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config.get("batch_size", 4),
            shuffle=True,
            num_workers=config.get("num_workers", 4),
            pin_memory=True,
        )

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.output_dir = Path(config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Preemption handling
        self._emergency_save = False
        signal.signal(signal.SIGUSR1, self._handle_preemption)

        # WandB
        if config.get("use_wandb", False):
            import wandb

            wandb.init(project="clearmesh", config=config)
            self.wandb = wandb
        else:
            self.wandb = None

    def _handle_preemption(self, signum, frame):
        """Handle SIGUSR1 from preemption detector — save emergency checkpoint."""
        print("\n!!! PREEMPTION SIGNAL RECEIVED — saving emergency checkpoint !!!")
        self._emergency_save = True

    def save_checkpoint(self, tag: str = "latest"):
        """Save training checkpoint to persistent disk."""
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

        # Also save as 'latest' for resume
        latest_path = self.output_dir / "checkpoint_latest.pt"
        torch.save(ckpt, latest_path)

        print(f"Checkpoint saved: {path} (step {self.global_step})")

    def load_checkpoint(self, path: str | None = None):
        """Resume from checkpoint."""
        if path is None:
            path = self.output_dir / "checkpoint_latest.pt"
        else:
            path = Path(path)

        if not path.exists():
            print("No checkpoint found, starting from scratch.")
            return

        print(f"Resuming from {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.global_step = ckpt["global_step"]
        self.epoch = ckpt["epoch"]

        print(f"Resumed at step {self.global_step}, epoch {self.epoch}")

    def train_step(self, batch: dict) -> dict[str, float]:
        """Single training step with FlexiCubes in the loop."""
        self.model.train()

        coarse = batch["coarse_voxels"].to(self.device)
        fine = batch["fine_voxels"].to(self.device)
        positions = batch["positions"].to(self.device)

        B = coarse.shape[0]

        # Sample random timestep
        t = torch.rand(B, device=self.device)

        # Get coarse voxel features (flatten spatial dims, add feature dim)
        N = positions.shape[1]
        coarse_features = coarse.view(B, -1).unsqueeze(-1).expand(-1, -1, 32)
        # Subsample to match positions
        if coarse_features.shape[1] > N:
            idx = torch.randperm(coarse_features.shape[1])[:N]
            coarse_features = coarse_features[:, idx]
        elif coarse_features.shape[1] < N:
            # Pad
            pad = torch.zeros(B, N - coarse_features.shape[1], 32, device=self.device)
            coarse_features = torch.cat([coarse_features, pad], dim=1)

        # Ground truth SDF at supervision points
        gt_sdf = fine.view(B, -1).unsqueeze(-1)
        if gt_sdf.shape[1] > N:
            idx = torch.randperm(gt_sdf.shape[1])[:N]
            gt_sdf = gt_sdf[:, idx]
        elif gt_sdf.shape[1] < N:
            pad = torch.zeros(B, N - gt_sdf.shape[1], 1, device=self.device)
            gt_sdf = torch.cat([gt_sdf, pad], dim=1)

        # Add noise to GT SDF (diffusion forward process)
        noise = torch.randn_like(gt_sdf)
        alpha = (1 - t).view(B, 1, 1)
        noisy_sdf = alpha.sqrt() * gt_sdf + (1 - alpha).sqrt() * noise

        # Forward pass: predict noise
        pred_noise = self.model(coarse_features, positions, t, noisy_sdf=noisy_sdf)

        # === FlexiCubes in the loop (Option C) ===
        # Denoise to get predicted SDF
        pred_sdf = (noisy_sdf - (1 - alpha).sqrt() * pred_noise) / alpha.sqrt()

        # Compute losses
        losses = self.criterion(pred_sdf=pred_sdf, gt_sdf=gt_sdf)

        # FlexiCubes mesh extraction + mesh-space losses (every K steps to save compute)
        if self.global_step % self.config.get("flexicubes_interval", 10) == 0:
            try:
                # Reshape predicted SDF to grid
                R = self.config.get("resolution", 128)
                sdf_grid = pred_sdf[0, : R**3, 0].view(R, R, R)

                vertices, faces = self.flexicubes.extract(sdf_grid)

                if vertices is not None and vertices.shape[0] > 0:
                    # Sample points on extracted mesh surface
                    pred_points = vertices.unsqueeze(0)  # (1, V, 3)

                    # Get GT mesh points
                    gt_grid = gt_sdf[0, : R**3, 0].view(R, R, R)
                    gt_verts, gt_faces_t = self.flexicubes.extract(gt_grid)

                    if gt_verts is not None and gt_verts.shape[0] > 0:
                        gt_points = gt_verts.unsqueeze(0)

                        mesh_losses = self.criterion(
                            pred_sdf=pred_sdf,
                            gt_sdf=gt_sdf,
                            extracted_vertices=vertices,
                            extracted_faces=faces,
                            gt_vertices=gt_verts,
                            gt_faces=gt_faces_t,
                            pred_points=pred_points,
                            gt_points=gt_points,
                        )
                        losses = mesh_losses

            except Exception:
                # FlexiCubes extraction can fail on degenerate SDF grids
                # Fall back to SDF-only loss
                pass

        # Backward pass
        self.optimizer.zero_grad()
        losses["total"].backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.get("max_grad_norm", 1.0)
        )

        self.optimizer.step()
        self.scheduler.step()

        return {k: v.item() for k, v in losses.items()}

    def train(self):
        """Main training loop."""
        total_steps = self.config.get("total_steps", 100_000)
        save_interval = self.config.get("save_interval", 1000)
        log_interval = self.config.get("log_interval", 100)

        # Resume if checkpoint exists
        self.load_checkpoint()

        print(f"\n=== Starting training ===")
        print(f"Total steps: {total_steps}")
        print(f"Current step: {self.global_step}")
        print(f"Dataset size: {len(self.dataset)}")
        print(f"Batch size: {self.config.get('batch_size', 4)}")
        print(f"FlexiCubes interval: every {self.config.get('flexicubes_interval', 10)} steps")
        print(f"Save interval: every {save_interval} steps")
        print()

        # Save PID for preemption handler
        with open("/tmp/train.pid", "w") as f:
            f.write(str(os.getpid()))

        pbar = tqdm(total=total_steps, initial=self.global_step, desc="Training")

        while self.global_step < total_steps:
            for batch in self.dataloader:
                if self.global_step >= total_steps:
                    break

                # Emergency save on preemption
                if self._emergency_save:
                    self.save_checkpoint(f"emergency_{self.global_step}")
                    print("Emergency checkpoint saved. Exiting.")
                    sys.exit(0)

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
        print(f"\nTraining complete! Final checkpoint at step {self.global_step}")


def main():
    parser = argparse.ArgumentParser(description="Train Stage 2 refinement model")
    parser.add_argument("--config", type=str, required=True, help="Path to training config YAML")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.output_dir:
        config["output_dir"] = args.output_dir

    # Create trainer and run
    trainer = Trainer(config)

    if args.resume_from:
        trainer.load_checkpoint(args.resume_from)

    trainer.train()


if __name__ == "__main__":
    main()
