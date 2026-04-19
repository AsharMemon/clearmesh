"""Neural warm-start head for DualPrim (Tier B).

Takes a mesh representation and predicts per-primitive params that
seed a short DualPrim refinement. Target: good-enough init that 500-
1000 refine steps recover teacher-level quality.

This file defines the model architecture and dataset. Training loop
lives in scripts/dualprim/train_warmstart.py (to be written once
Gate 0.5 confirms the warm-start hypothesis).

Architecture choice:
  - Input: surface point cloud sampled from the input mesh
  - Encoder: PointNet++ style (local features → global latent)
  - Head: set transformer → K primitives × 27 params
  - Output: primitives JSON compatible with load_scene_from_json

Why PointNet++:
  - Robust baseline, proven on ShapeNet/Objaverse-style meshes
  - No dependency on TRELLIS.2 SLAT (decoupled from upstream pipeline)
  - Works on arbitrary watertight meshes
  - We can upgrade later to SLAT-features if PointNet baseline works

Why set transformer for the head:
  - Primitives have no natural ordering
  - Using MLP with fixed-order output forces arbitrary ordering
    (primitive 0 vs 1) which the model can't learn stably
  - Set transformer allows permutation-equivariant prediction

NOT YET TESTED. NOT YET TRAINED. This is the architecture skeleton
we'll fill in once Gate 0.5 feasibility is confirmed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from clearmesh.dualprim.types import DUAL_PRIM_DIM


# ---------------------------------------------------------------------
# Point cloud encoder (PointNet-style)
# ---------------------------------------------------------------------

class PointEncoder(nn.Module):
    """Extract a global feature from a point cloud.

    Input:  (B, N, 3) point cloud sampled from the mesh surface
    Output: (B, D) global feature vector

    Minimal PointNet-style: per-point MLP → max pool → global. No
    hierarchical grouping (PointNet++ style) because our shapes are
    normalized to [-1,1]^3 so scale-invariance isn't needed.
    """

    def __init__(self, feat_dim: int = 256, hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, feat_dim),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        # (B, N, 3) -> (B, N, D) -> max over N -> (B, D)
        per_point = self.mlp(points)
        return per_point.max(dim=1).values


# ---------------------------------------------------------------------
# Set transformer head
# ---------------------------------------------------------------------

class SetTransformerHead(nn.Module):
    """Predict K primitive parameter sets from a global feature.

    Uses learnable "primitive queries" (like DETR) that cross-attend
    to the global feature. Each query → one primitive's 27 params.

    Not using a full DETR stack to keep this minimal. Hungarian
    matching happens in the loss, not here.
    """

    def __init__(self, feat_dim: int = 256, k: int = 100,
                 num_heads: int = 4, num_layers: int = 3, hidden: int = 256):
        super().__init__()
        self.k = k
        self.queries = nn.Parameter(torch.randn(k, feat_dim) * 0.02)

        # Self-attention among queries (learn inter-primitive
        # relationships: "if one primitive is here, another should
        # be aligned with it").
        layer = nn.TransformerEncoderLayer(
            d_model=feat_dim, nhead=num_heads, dim_feedforward=hidden,
            batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

        # Read-out: each query → DUAL_PRIM_DIM (27) params
        self.readout = nn.Linear(feat_dim, DUAL_PRIM_DIM)

    def forward(self, global_feat: torch.Tensor) -> torch.Tensor:
        # global_feat: (B, D). Broadcast to (B, K, D) as per-query
        # conditioning. Add learnable query embeddings to break the
        # initial symmetry.
        B, D = global_feat.shape
        tokens = global_feat.unsqueeze(1) + self.queries.unsqueeze(0)  # (B, K, D)
        encoded = self.encoder(tokens)                                   # (B, K, D)
        return self.readout(encoded)                                     # (B, K, 27)


# ---------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------

class WarmStartHead(nn.Module):
    """End-to-end: mesh points -> K primitive param tensors.

    Output shape: (B, K, DUAL_PRIM_DIM) matching DualPrimScene.params.
    Consumer: save_scene_json → run_canary --resume-primitives.

    Note on param-range enforcement: we output raw values here. The
    downstream `run_canary --resume-primitives` path invokes
    `clip_to_ranges(scene, config)` at every step, which enforces the
    Table-1 ranges from the paper. So the head can emit out-of-range
    values without breaking the system — they get clamped on first
    step. Ideally we add a final sigmoid/tanh layer to output ranges
    directly, but clamping is a safety net either way.
    """

    def __init__(
        self,
        k: int = 100,
        feat_dim: int = 256,
        encoder_hidden: int = 128,
        head_hidden: int = 256,
        head_layers: int = 3,
        head_heads: int = 4,
    ):
        super().__init__()
        self.encoder = PointEncoder(feat_dim=feat_dim, hidden=encoder_hidden)
        self.head = SetTransformerHead(
            feat_dim=feat_dim, k=k, num_heads=head_heads,
            num_layers=head_layers, hidden=head_hidden,
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        global_feat = self.encoder(points)
        return self.head(global_feat)


# ---------------------------------------------------------------------
# Hungarian-matched loss
# ---------------------------------------------------------------------

def hungarian_param_loss(
    pred: torch.Tensor,       # (B, K, 27) model output
    target: torch.Tensor,     # (B, K', 27) teacher primitives (K' <= K)
    target_alive: torch.Tensor,  # (B, K') bool; teacher's live mask
) -> torch.Tensor:
    """Permutation-invariant param-matching loss.

    For each sample in the batch:
      1. Compute cost matrix: L1(pred_i, target_j) for all (i, j)
      2. Hungarian match → which predicted primitive matches which
         teacher primitive
      3. Sum matched L1 over alive targets; unmatched predictions
         get a separate "inactive" penalty to push α → 0.

    Requires `scipy.optimize.linear_sum_assignment`. Batched version
    is a Python loop over batch items — not GPU-vectorized, but
    small batch sizes at training time make this acceptable.

    TODO: replace with Sinkhorn if batch is large.
    """
    from scipy.optimize import linear_sum_assignment

    B = pred.shape[0]
    total_loss = pred.new_zeros(())
    for b in range(B):
        p = pred[b]                             # (K, 27)
        t = target[b][target_alive[b]]          # (k, 27) only alive
        if t.shape[0] == 0:
            continue
        # Cost = L1 distance per (i, j). (K, k)
        cost = (p.unsqueeze(1) - t.unsqueeze(0)).abs().sum(dim=-1)
        cost_np = cost.detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_np)
        matched = cost[row_ind, col_ind].sum() / max(t.shape[0], 1)

        # Unmatched predictions: push their α toward 0 so they stay inactive
        K = p.shape[0]
        unmatched_idx = torch.tensor(
            [i for i in range(K) if i not in row_ind],
            device=pred.device, dtype=torch.long,
        )
        alpha_unm = p[unmatched_idx, 10] if unmatched_idx.numel() else pred.new_zeros(())
        inactive_penalty = (alpha_unm ** 2).mean() if alpha_unm.numel() else pred.new_zeros(())

        total_loss = total_loss + matched + 0.1 * inactive_penalty

    return total_loss / B


# ---------------------------------------------------------------------
# Dataset: reads the teacher corpus
# ---------------------------------------------------------------------

@dataclass
class WarmStartSample:
    """A single training example."""
    mesh_path: str                 # path to input GLB
    primitives_path: str           # path to target primitives JSON
    step: int                      # which trajectory step (0 = random init,
                                   # 15000 = converged teacher; intermediate
                                   # values supervised too per friend's
                                   # guidance)
    lvis_class: Optional[str]      # metadata from objaverse manifest


class WarmStartDataset(torch.utils.data.Dataset):
    """Reads a corpus of teacher trajectory snapshots.

    Expected corpus layout (produced by autonomous_runner --trajectory):
      {out_root}/
        {exp_name}_seed0/
          input.glb  (or symlink to the ref mesh)
          metrics.json      # quality metrics; used to filter bad runs
          primitives.json   # final teacher
          trajectory/
            step_001000.json
            step_003000.json
            step_006000.json
            step_010000.json
            step_015000.json
        {exp_name}_seed1/
          ...

    Filter: only samples where metrics.json shows train_ok AND
    chamfer ≤ threshold are included. Bad teachers don't train
    good students.
    """

    def __init__(
        self,
        corpus_root: str,
        *,
        num_surface_points: int = 2048,
        include_trajectory_steps: bool = True,
        max_chamfer_x1000: float = 100.0,
        require_hole_open: bool = False,
    ):
        self.corpus_root = Path(corpus_root)
        self.num_surface_points = num_surface_points
        self.samples: list[WarmStartSample] = []
        self._index(include_trajectory_steps, max_chamfer_x1000, require_hole_open)

    def _index(
        self, include_trajectory_steps: bool,
        max_chamfer: float, require_hole: bool,
    ):
        import json
        for run_dir in sorted(self.corpus_root.iterdir()):
            if not run_dir.is_dir():
                continue
            metrics_path = run_dir / "metrics.json"
            primitives_path = run_dir / "primitives.json"
            if not (metrics_path.exists() and primitives_path.exists()):
                continue
            metrics = json.loads(metrics_path.read_text())
            if not metrics.get("ok"):
                continue
            cd = metrics.get("chamfer_x1000")
            if cd is None or cd > max_chamfer:
                continue
            if require_hole:
                # Only include runs where the teacher preserved a hole
                hole_pct = metrics.get("hole_open_pct") or 0
                if hole_pct < 10:
                    continue

            mesh_path = metrics.get("ref_glb") or str(run_dir / "input.glb")
            lvis_class = metrics.get("lvis_class")

            # Final endpoint
            self.samples.append(WarmStartSample(
                mesh_path=mesh_path,
                primitives_path=str(primitives_path),
                step=metrics["config"].get("iters", 15000),
                lvis_class=lvis_class,
            ))

            # Trajectory snapshots (friend's guidance — multi-step
            # targets teach the refinement curve, not just endpoint)
            if include_trajectory_steps:
                traj = run_dir / "trajectory"
                if traj.exists():
                    for snap in sorted(traj.glob("step_*.json")):
                        step = int(snap.stem.split("_")[1])
                        self.samples.append(WarmStartSample(
                            mesh_path=mesh_path,
                            primitives_path=str(snap),
                            step=step,
                            lvis_class=lvis_class,
                        ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        # Sample surface points from the mesh
        import trimesh
        m = trimesh.load(s.mesh_path, force="mesh")
        # Normalize to [-1, 1]^3 so predictions have a consistent frame
        m = m.copy()
        m.vertices -= m.centroid
        scale = m.extents.max()
        if scale > 0: m.vertices /= scale
        points, _ = trimesh.sample.sample_surface(m, self.num_surface_points)
        points_t = torch.from_numpy(points.astype("float32"))

        # Load target primitives
        from clearmesh.dualprim.io import _prim_dict_to_row
        import json
        payload = json.load(open(s.primitives_path))
        prims = payload.get("primitives", [])
        K_target = len(prims)
        # Pad to a fixed K for batching. Use 100 as default.
        K_pad = 100
        target = torch.zeros((K_pad, DUAL_PRIM_DIM), dtype=torch.float32)
        alive = torch.zeros(K_pad, dtype=torch.bool)
        for i, d in enumerate(prims[:K_pad]):
            target[i] = _prim_dict_to_row(d, device="cpu")
            alive[i] = True

        return {
            "points": points_t,
            "target_params": target,
            "target_alive": alive,
            "step": s.step,
            "mesh_path": s.mesh_path,
            "lvis_class": s.lvis_class or "unknown",
        }


# ---------------------------------------------------------------------
# Rough training-step sanity checker
# ---------------------------------------------------------------------

def smoke_test_forward_backward():
    """Verify shapes and gradient flow in the architecture.

    Call this once after edits to make sure nothing is wired wrong.
    No GPU needed, no data needed.
    """
    model = WarmStartHead(k=32, feat_dim=64)  # tiny config for speed
    points = torch.randn(2, 512, 3)            # batch=2, N=512 points
    out = model(points)                          # (2, 32, 27)
    assert out.shape == (2, 32, 27), f"unexpected output shape: {out.shape}"

    target = torch.randn(2, 32, 27)
    alive = torch.ones(2, 32, dtype=torch.bool)
    loss = hungarian_param_loss(out, target, alive)
    loss.backward()
    # Check gradient reached encoder
    assert model.encoder.mlp[0].weight.grad is not None
    print(f"smoke test ok: out.shape={out.shape}, loss={loss.item():.4f}")


if __name__ == "__main__":
    smoke_test_forward_backward()
