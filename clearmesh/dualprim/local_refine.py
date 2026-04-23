"""Primitive-local Stage C refinement for DualPrim.

This stage consumes the per-primitive artifacts produced by
`prepare_primitive_local_refine_artifacts(...)` and fits a tiny bounded
deformation model in each primitive's local frame.

It is intentionally conservative:
  - only operates on surviving coarse primitives
  - learns scalar displacements in a narrow band
  - preserves the DualPrim scaffold as the structural owner
  - emits explicit local summaries and point-cloud previews instead of
    silently turning into a second global reconstruction system
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh


@dataclass
class LocalRefineConfig:
    steps: int = 1500
    batch_size: int = 1024
    hidden_dim: int = 64
    layers: int = 3
    lr: float = 1e-3
    weight_decay: float = 1e-6
    smoothness_weight: float = 0.05
    regularize_weight: float = 0.01
    jitter_sigma_frac: float = 0.02
    min_sample_count: int = 256
    max_primitives: int | None = None
    target_primitive_ids: tuple[int, ...] | None = None
    normal_focus_power: float = 1.0
    distance_focus_power: float = 1.0
    boundary_focus_weight: float = 0.0
    boundary_focus_power: float = 2.0
    weight_cap: float = 6.0
    seed: int = 0


class LocalDisplacementMLP(nn.Module):
    """Tiny bounded MLP that predicts scalar displacement in local space."""

    def __init__(self, hidden_dim: int = 64, layers: int = 3):
        super().__init__()
        dims = [3]
        if layers <= 1:
            dims.append(1)
        else:
            dims.extend([hidden_dim] * (layers - 1))
            dims.append(1)
        blocks = []
        for i in range(len(dims) - 1):
            blocks.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                blocks.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(x)).squeeze(-1)


def _rotation_matrix_xyz(rot_xyz_rad: list[float] | np.ndarray) -> np.ndarray:
    ax, ay, az = [float(v) for v in rot_xyz_rad]
    cx, cy, cz = np.cos([ax, ay, az])
    sx, sy, sz = np.sin([ax, ay, az])
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    return (rz @ ry @ rx).astype(np.float32)


def _load_manifest(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _rank_primitives(primitives: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = []
    for prim in primitives:
        score = (
            float(prim["sample_count"])
            * float(prim["mean_target_to_coarse"])
            * (1.0 + float(prim["mean_normal_error_deg"]) / 90.0)
        )
        ranked.append((score, prim))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [prim for _, prim in ranked]


def _normalize_inputs(points_local: np.ndarray, support_scale: np.ndarray) -> np.ndarray:
    denom = np.clip(support_scale.reshape(1, 3), 1e-5, None)
    return (points_local / denom).astype(np.float32)


def _tensor(data: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(np.asarray(data, dtype=np.float32)).to(device)


def _fit_single_primitive(
    primitive_record: dict[str, Any],
    artifact_path: Path,
    cfg: LocalRefineConfig,
    norm_meta: dict[str, Any],
    out_dir: Path,
    device: str,
) -> dict[str, Any]:
    payload = np.load(artifact_path)
    local_points = np.asarray(payload["closest_points_local"], dtype=np.float32)
    local_normals = np.asarray(payload["closest_normals_local"], dtype=np.float32)
    target_local = np.asarray(payload["target_points_local"], dtype=np.float32)
    signed_offsets = np.asarray(payload["signed_offsets"], dtype=np.float32)
    target_to_coarse = np.asarray(payload["target_to_coarse_dist"], dtype=np.float32)
    normal_error_deg = np.asarray(payload["normal_error_deg"], dtype=np.float32)

    support_scale = np.asarray(primitive_record["support_scale_norm"], dtype=np.float32)
    max_displacement = float(primitive_record["budget"]["max_displacement"])
    if len(local_points) == 0:
        return {
            "primitive_live_index": primitive_record["primitive_live_index"],
            "status": "empty",
        }

    x = _normalize_inputs(local_points, support_scale)
    y = np.clip(signed_offsets / max(max_displacement, 1e-6), -1.0, 1.0).astype(np.float32)
    normal_term = 1.0 + np.clip(normal_error_deg / 90.0, 0.0, 2.0)
    dist_term = 1.0 + np.clip(
        target_to_coarse / max(float(np.percentile(target_to_coarse, 90)), 1e-4),
        0.0,
        2.0,
    )
    radius = np.sqrt(np.sum((local_points / np.clip(support_scale[None, :], 1e-5, None)) ** 2, axis=1))
    boundary_term = 1.0 + float(cfg.boundary_focus_weight) * (
        np.clip(radius, 0.0, 1.0) ** float(cfg.boundary_focus_power)
    )
    weights = (normal_term ** float(cfg.normal_focus_power)) * (
        dist_term ** float(cfg.distance_focus_power)
    ) * boundary_term
    weights = np.clip(weights, 0.0, float(cfg.weight_cap))
    weights = (weights / np.clip(weights.mean(), 1e-6, None)).astype(np.float32)

    x_t = _tensor(x, device)
    y_t = _tensor(y, device)
    w_t = _tensor(weights, device)

    torch.manual_seed(cfg.seed + int(primitive_record["primitive_live_index"]))
    model = LocalDisplacementMLP(hidden_dim=cfg.hidden_dim, layers=cfg.layers).to(device)
    opt = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    jitter_sigma = float(cfg.jitter_sigma_frac)
    n = x_t.shape[0]
    batch_size = min(int(cfg.batch_size), n)
    for _ in range(int(cfg.steps)):
        idx = torch.randint(0, n, (batch_size,), device=device)
        xb = x_t[idx]
        yb = y_t[idx]
        wb = w_t[idx]
        pred = model(xb)

        data_loss = (F.smooth_l1_loss(pred, yb, reduction="none") * wb).mean()
        if jitter_sigma > 0.0:
            noise = torch.randn_like(xb) * jitter_sigma
            pred_jitter = model(torch.clamp(xb + noise, -2.0, 2.0))
            smooth_loss = F.mse_loss(pred, pred_jitter)
        else:
            smooth_loss = pred.new_tensor(0.0)
        reg_loss = torch.mean(pred * pred)
        loss = data_loss + cfg.smoothness_weight * smooth_loss + cfg.regularize_weight * reg_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    with torch.no_grad():
        pred_all = model(x_t).detach().cpu().numpy().astype(np.float32)
    pred_disp = pred_all * max_displacement

    refined_local = local_points + local_normals * pred_disp[:, None]
    baseline_error = np.linalg.norm(local_points - target_local, axis=1)
    refined_error = np.linalg.norm(refined_local - target_local, axis=1)

    rot = _rotation_matrix_xyz(primitive_record["psq_rotation_rad"])
    trans_norm = np.asarray(primitive_record["psq_translation_norm"], dtype=np.float32)
    centroid = np.asarray(norm_meta["target_centroid"], dtype=np.float32)
    target_scale = float(norm_meta["target_scale"])

    refined_world_norm = (rot @ refined_local.T).T + trans_norm[None, :]
    target_world_norm = (rot @ target_local.T).T + trans_norm[None, :]
    refined_world = refined_world_norm / max(target_scale, 1e-8) + centroid[None, :]
    target_world = target_world_norm / max(target_scale, 1e-8) + centroid[None, :]

    primitive_id = int(primitive_record["primitive_live_index"])
    artifact_stem = f"primitive_{primitive_id:03d}"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": asdict(cfg),
            "primitive_record": primitive_record,
            "max_displacement": max_displacement,
        },
        out_dir / f"{artifact_stem}_refiner.pt",
    )
    np.savez_compressed(
        out_dir / f"{artifact_stem}_predictions.npz",
        refined_points_local=refined_local.astype(np.float32),
        refined_points_world=refined_world.astype(np.float32),
        target_points_local=target_local.astype(np.float32),
        target_points_world=target_world.astype(np.float32),
        closest_points_local=local_points.astype(np.float32),
        predicted_displacement=pred_disp.astype(np.float32),
        baseline_error=baseline_error.astype(np.float32),
        refined_error=refined_error.astype(np.float32),
    )

    return {
        "primitive_live_index": primitive_id,
        "artifact": artifact_path.name,
        "sample_count": int(n),
        "max_displacement": max_displacement,
        "baseline_error_mean": float(baseline_error.mean()),
        "refined_error_mean": float(refined_error.mean()),
        "baseline_error_p90": float(np.percentile(baseline_error, 90)),
        "refined_error_p90": float(np.percentile(refined_error, 90)),
        "improvement_mean": float(baseline_error.mean() - refined_error.mean()),
        "improvement_p90": float(np.percentile(baseline_error, 90) - np.percentile(refined_error, 90)),
        "alpha": float(primitive_record["alpha"]),
        "mean_target_to_coarse": float(primitive_record["mean_target_to_coarse"]),
        "mean_normal_error_deg": float(primitive_record["mean_normal_error_deg"]),
        "refiner_path": f"{artifact_stem}_refiner.pt",
        "prediction_path": f"{artifact_stem}_predictions.npz",
        "refined_points_world": refined_world.astype(np.float32),
        "target_points_world": target_world.astype(np.float32),
    }


def train_primitive_local_refiners(
    manifest_path: str | Path,
    out_dir: str | Path,
    cfg: LocalRefineConfig | None = None,
    device: str = "cuda",
) -> dict[str, Any]:
    """Fit bounded local refiners for the selected primitive artifacts."""
    cfg = cfg or LocalRefineConfig()
    manifest_path = Path(manifest_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = _load_manifest(manifest_path)
    ranked = _rank_primitives(list(manifest.get("primitives", [])))
    if cfg.target_primitive_ids:
        target_ids = {int(v) for v in cfg.target_primitive_ids}
        ranked = [p for p in ranked if int(p["primitive_live_index"]) in target_ids]
    selected = [p for p in ranked if int(p["sample_count"]) >= cfg.min_sample_count]
    if cfg.max_primitives is not None:
        selected = selected[: int(cfg.max_primitives)]

    summaries = []
    refined_clouds = []
    target_clouds = []
    for primitive_record in selected:
        artifact_path = manifest_path.parent / primitive_record["artifact"]
        summary = _fit_single_primitive(
            primitive_record,
            artifact_path,
            cfg,
            manifest["normalization"],
            out_dir,
            device,
        )
        if "refined_points_world" in summary:
            refined_clouds.append(summary.pop("refined_points_world"))
            target_clouds.append(summary.pop("target_points_world"))
        summaries.append(summary)

    aggregate = {
        "manifest_path": str(manifest_path),
        "config": asdict(cfg),
        "num_selected_primitives": len(selected),
        "summaries": summaries,
    }
    if summaries:
        aggregate["mean_improvement"] = float(np.mean([s["improvement_mean"] for s in summaries]))
        aggregate["mean_refined_error"] = float(np.mean([s["refined_error_mean"] for s in summaries]))
        aggregate["mean_baseline_error"] = float(np.mean([s["baseline_error_mean"] for s in summaries]))
    else:
        aggregate["mean_improvement"] = 0.0
        aggregate["mean_refined_error"] = 0.0
        aggregate["mean_baseline_error"] = 0.0

    if refined_clouds:
        refined_points = np.concatenate(refined_clouds, axis=0)
        target_points = np.concatenate(target_clouds, axis=0)
        trimesh.points.PointCloud(refined_points).export(out_dir / "refined_points_world.ply")
        trimesh.points.PointCloud(target_points).export(out_dir / "target_points_world.ply")
        aggregate["refined_point_count"] = int(len(refined_points))
    else:
        aggregate["refined_point_count"] = 0

    with open(out_dir / "local_refine_summary.json", "w") as f:
        json.dump(aggregate, f, indent=2)
    return aggregate
