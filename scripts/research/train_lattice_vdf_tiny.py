#!/usr/bin/env python3
"""Train a tiny LATTICE VDF regressor on lattice NPZ shards."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clearmesh.lattice.tiny_vdf import build_tiny_vdf_regressor
from clearmesh.utils.checkpoint import args_to_json_safe


@dataclass(frozen=True)
class VDFShard:
    path: Path
    points: np.ndarray
    targets: np.ndarray


def _load_shard(path: Path, use_face_vdf: bool) -> VDFShard:
    data = np.load(path)
    prefix = "face_vdf_" if use_face_vdf and "face_vdf_points" in data else "vdf_"
    points = np.asarray(data[f"{prefix}points"], dtype=np.float32)
    normals = np.asarray(data[f"{prefix}normals"], dtype=np.float32)
    displacements = np.asarray(data[f"{prefix}vertex_displacements"], dtype=np.float32)
    targets = np.concatenate([displacements.reshape(len(points), 9), normals], axis=1).astype(np.float32)
    return VDFShard(path=path, points=points, targets=targets)


def _load_dataset(dataset_dir: Path, use_face_vdf: bool, limit: int = 0) -> list[VDFShard]:
    paths = sorted(dataset_dir.glob("*.npz"))
    if limit:
        paths = paths[:limit]
    shards = [_load_shard(path, use_face_vdf=use_face_vdf) for path in paths]
    if not shards:
        raise SystemExit(f"No LATTICE shards found in {dataset_dir}")
    return shards


def _sample_batch(shards: list[VDFShard], batch_points: int, device):  # type: ignore[no-untyped-def]
    import torch

    shard = random.choice(shards)
    replace = len(shard.points) < batch_points
    indices = np.random.choice(len(shard.points), size=batch_points, replace=replace)
    x = torch.as_tensor(shard.points[indices], dtype=torch.float32, device=device)
    y = torch.as_tensor(shard.targets[indices], dtype=torch.float32, device=device)
    return x, y


def _mesh_from_vdf(points: np.ndarray, prediction: np.ndarray, weld_digits: int) -> trimesh.Trimesh:
    displacements = prediction[:, :9].reshape(-1, 3, 3)
    vertices = (points[:, None, :] + displacements).reshape(-1, 3)
    faces = np.arange(len(vertices), dtype=np.int64).reshape(-1, 3)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.merge_vertices(digits_vertex=weld_digits)
    if hasattr(mesh, "unique_faces"):
        mesh.update_faces(mesh.unique_faces())
    elif hasattr(mesh, "remove_duplicate_faces"):
        mesh.remove_duplicate_faces()
    mesh.remove_unreferenced_vertices()
    return mesh


def _vdf_error_summary(prediction: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    absolute_error = np.abs(prediction - targets)
    return {
        "vdf_max_abs_error": float(absolute_error.max()),
        "vdf_mean_abs_error": float(absolute_error.mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/lattice_vdf_tiny.pt"))
    parser.add_argument("--export-predicted", type=Path, default=None)
    parser.add_argument("--export-target", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-points", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weld-digits", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-face-vdf", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    shards = _load_dataset(args.dataset_dir, use_face_vdf=args.use_face_vdf, limit=args.limit)
    total_points = sum(len(shard.points) for shard in shards)
    summary = {
        "shards": len(shards),
        "total_points": int(total_points),
        "target_dim": int(shards[0].targets.shape[1]),
        "use_face_vdf": bool(args.use_face_vdf),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.dry_run:
        return 0

    import torch
    import torch.nn.functional as F

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    torch.manual_seed(args.seed)
    model = build_tiny_vdf_regressor(hidden_size=args.hidden_size, layers=args.layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    losses = []
    best_loss = float("inf")
    best_step = 0
    best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
    batch_points = min(args.batch_points, max(len(shard.points) for shard in shards))
    for step in range(1, args.steps + 1):
        x, y = _sample_batch(shards, batch_points=batch_points, device=device)
        pred = model(x)
        loss = F.mse_loss(pred, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if losses[-1] < best_loss:
            best_loss = losses[-1]
            best_step = step
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
        if step == 1 or step == args.steps or step % max(1, args.steps // 5) == 0:
            print(json.dumps({"step": step, "loss": losses[-1]}))

    model.load_state_dict(best_state)
    model.to(device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": best_state,
            "args": args_to_json_safe(args),
            "losses": losses,
            "best_loss": best_loss,
            "best_step": best_step,
        },
        args.output,
    )

    export_info: dict[str, str | int | bool | None] = {}
    if args.export_predicted is not None or args.export_target is not None:
        shard = shards[0]
        x = torch.as_tensor(shard.points, dtype=torch.float32, device=device)
        with torch.no_grad():
            prediction = model(x).detach().cpu().numpy()
        export_info.update(_vdf_error_summary(prediction, shard.targets))
        if args.export_predicted is not None:
            args.export_predicted.parent.mkdir(parents=True, exist_ok=True)
            mesh = _mesh_from_vdf(shard.points, prediction, weld_digits=args.weld_digits)
            mesh.export(args.export_predicted)
            export_info.update(
                {
                    "predicted_mesh": str(args.export_predicted),
                    "predicted_faces": int(len(mesh.faces)),
                    "predicted_vertices": int(len(mesh.vertices)),
                    "predicted_watertight": bool(mesh.is_watertight),
                }
            )
        if args.export_target is not None:
            args.export_target.parent.mkdir(parents=True, exist_ok=True)
            mesh = _mesh_from_vdf(shard.points, shard.targets, weld_digits=args.weld_digits)
            mesh.export(args.export_target)
            export_info.update(
                {
                    "target_mesh": str(args.export_target),
                    "target_faces": int(len(mesh.faces)),
                    "target_vertices": int(len(mesh.vertices)),
                    "target_watertight": bool(mesh.is_watertight),
                }
            )

    print(
        json.dumps(
            {
                "checkpoint": str(args.output),
                "final_loss": losses[-1],
                "best_loss": best_loss,
                "best_step": best_step,
                "device": str(device),
                **export_info,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
