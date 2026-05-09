#!/usr/bin/env python3
"""Train a tiny point-conditioned, face-level FACE decoder."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clearmesh.mesh_heads.face_arae import build_tiny_point_conditioned_face_level_decoder
from clearmesh.mesh_heads.face_topology import topology_coordinate_weights, topology_event_labels
from clearmesh.utils.checkpoint import args_to_json_safe


@dataclass(frozen=True)
class FaceLevelSample:
    path: Path
    point_features: np.ndarray
    tokens: np.ndarray


def _load_sample(path: Path) -> tuple[FaceLevelSample, int]:
    data = np.load(path)
    if "surface_points" not in data or "surface_normals" not in data:
        raise ValueError(f"{path} does not contain surface point conditioning arrays")
    num_bins = int(np.asarray(data["num_bins"]).reshape(-1)[0])
    points = np.asarray(data["surface_points"], dtype=np.float32)
    normals = np.asarray(data["surface_normals"], dtype=np.float32)
    tokens = np.asarray(data["tokens"], dtype=np.int64)
    if tokens.ndim != 2 or tokens.shape[1] != 9:
        raise ValueError(f"{path} has invalid tokens shape {tokens.shape}")
    return (
        FaceLevelSample(
            path=path,
            point_features=np.concatenate([points, normals], axis=1).astype(np.float32),
            tokens=tokens,
        ),
        num_bins,
    )


def _load_dataset(dataset_dir: Path, limit: int = 0) -> tuple[list[FaceLevelSample], int]:
    paths = sorted(dataset_dir.glob("*.npz"))
    if limit:
        paths = paths[:limit]
    samples: list[FaceLevelSample] = []
    num_bins: int | None = None
    for path in paths:
        try:
            sample, sample_bins = _load_sample(path)
        except Exception:
            continue
        if num_bins is None:
            num_bins = sample_bins
        elif num_bins != sample_bins:
            raise ValueError(f"Mixed num_bins in dataset: {num_bins} and {sample_bins}")
        samples.append(sample)
    if not samples or num_bins is None:
        raise SystemExit(f"No face-level FACE shards found in {dataset_dir}")
    return samples, num_bins


def _make_batch(
    samples: list[FaceLevelSample],
    max_faces: int,
    device,  # type: ignore[no-untyped-def]
    point_samples: int | None = None,
    reuse_vertex_loss_weight: float = 0.0,
    edge_closure_loss_weight: float = 0.0,
):
    import torch

    input_faces = torch.zeros((len(samples), max_faces, 9), dtype=torch.long, device=device)
    target_faces = torch.full((len(samples), max_faces, 9), -100, dtype=torch.long, device=device)
    target_weights = torch.zeros((len(samples), max_faces, 9), dtype=torch.float32, device=device)
    reuse_vertex_labels = torch.full((len(samples), max_faces, 3), -100, dtype=torch.long, device=device)
    edge_closure_labels = torch.full((len(samples), max_faces), -100, dtype=torch.long, device=device)
    face_counts = torch.zeros((len(samples),), dtype=torch.long, device=device)
    point_batches = []
    for row, sample in enumerate(samples):
        tokens = sample.tokens[:max_faces]
        face_count = len(tokens)
        face_counts[row] = int(face_count)
        if face_count > 0:
            target_faces[row, :face_count] = torch.as_tensor(tokens, dtype=torch.long, device=device)
            weights = topology_coordinate_weights(
                tokens,
                reuse_vertex_weight=reuse_vertex_loss_weight,
                edge_closure_weight=edge_closure_loss_weight,
            )
            topology_labels = topology_event_labels(tokens)
            target_weights[row, :face_count] = torch.as_tensor(weights, dtype=torch.float32, device=device)
            reuse_vertex_labels[row, :face_count] = torch.as_tensor(
                topology_labels["reuse_vertex"],
                dtype=torch.long,
                device=device,
            )
            edge_closure_labels[row, :face_count] = torch.as_tensor(
                topology_labels["edge_closure_count"],
                dtype=torch.long,
                device=device,
            )
        if face_count > 1:
            input_faces[row, 1:face_count] = torch.as_tensor(tokens[:-1], dtype=torch.long, device=device)
        points = sample.point_features
        if point_samples is not None and point_samples > 0:
            if len(points) >= point_samples:
                points = points[:point_samples]
            else:
                repeat = int(np.ceil(point_samples / len(points)))
                points = np.tile(points, (repeat, 1))[:point_samples]
        point_batches.append(torch.as_tensor(points, dtype=torch.float32, device=device))
    return (
        torch.stack(point_batches, dim=0),
        input_faces,
        target_faces,
        target_weights,
        reuse_vertex_labels,
        edge_closure_labels,
        face_counts,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/face_level_conditioned_tiny.pt"))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--point-samples", type=int, default=0)
    parser.add_argument("--hidden-size", type=int, default=192)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--condition-tokens", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--reuse-vertex-loss-weight", type=float, default=0.0)
    parser.add_argument("--edge-closure-loss-weight", type=float, default=0.0)
    parser.add_argument("--topology-aux-loss-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    samples, num_bins = _load_dataset(args.dataset_dir, limit=args.limit)
    max_faces = max(len(sample.tokens) for sample in samples)
    point_count = int(samples[0].point_features.shape[0] if args.point_samples <= 0 else args.point_samples)
    summary = {
        "samples": len(samples),
        "num_bins": num_bins,
        "max_faces": max_faces,
        "coordinate_tokens_per_sample": int(max_faces * 9),
        "autoregressive_steps_per_sample": int(max_faces),
        "point_features": int(samples[0].point_features.shape[1]),
        "point_samples": point_count,
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
    model = build_tiny_point_conditioned_face_level_decoder(
        num_bins=num_bins,
        max_faces=max_faces,
        point_feature_dim=6,
        hidden_size=args.hidden_size,
        layers=args.layers,
        heads=args.heads,
        condition_tokens=args.condition_tokens,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    losses = []
    best_loss = float("inf")
    best_step = 0
    best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
    for step in range(1, args.steps + 1):
        batch = random.choices(samples, k=args.batch_size)
        (
            point_features,
            input_faces,
            target_faces,
            target_weights,
            reuse_vertex_labels,
            edge_closure_labels,
            face_counts,
        ) = _make_batch(
            batch,
            max_faces=max_faces,
            device=device,
            point_samples=args.point_samples if args.point_samples > 0 else None,
            reuse_vertex_loss_weight=args.reuse_vertex_loss_weight,
            edge_closure_loss_weight=args.edge_closure_loss_weight,
        )
        aux_outputs = None
        if args.topology_aux_loss_weight > 0.0 and hasattr(model, "forward_with_aux"):
            aux_outputs = model.forward_with_aux(point_features, input_faces)
            logits = aux_outputs["coord_logits"]
        else:
            logits = model(point_features, input_faces)
        token_losses = F.cross_entropy(
            logits.reshape(-1, num_bins),
            target_faces.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).reshape_as(target_faces)
        valid_tokens = target_faces.ne(-100)
        token_loss = (
            token_losses * target_weights * valid_tokens.to(token_losses.dtype)
        ).sum() / torch.clamp((target_weights * valid_tokens.to(target_weights.dtype)).sum(), min=1.0)
        count_logits = model.predict_face_count_logits(point_features)
        count_loss = F.cross_entropy(count_logits, face_counts)
        topology_aux_loss = torch.zeros((), dtype=token_loss.dtype, device=device)
        if aux_outputs is not None:
            reuse_loss = F.cross_entropy(
                aux_outputs["reuse_vertex_logits"].reshape(-1, 2),
                reuse_vertex_labels.reshape(-1),
                ignore_index=-100,
            )
            edge_closure_loss = F.cross_entropy(
                aux_outputs["edge_closure_logits"].reshape(-1, 4),
                edge_closure_labels.reshape(-1),
                ignore_index=-100,
            )
            topology_aux_loss = reuse_loss + edge_closure_loss
        loss = token_loss + 0.05 * count_loss + float(args.topology_aux_loss_weight) * topology_aux_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if losses[-1] < best_loss:
            best_loss = losses[-1]
            best_step = step
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
        if step == 1 or step == args.steps or step % max(1, args.steps // 5) == 0:
            print(
                json.dumps(
                    {
                        "step": step,
                        "loss": losses[-1],
                        "token_loss": float(token_loss.detach().cpu()),
                        "count_loss": float(count_loss.detach().cpu()),
                        "topology_aux_loss": float(topology_aux_loss.detach().cpu()),
                    }
                )
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": best_state,
            "args": args_to_json_safe(args),
            "losses": losses,
            "num_bins": num_bins,
            "max_faces": max_faces,
            "best_loss": best_loss,
            "best_step": best_step,
            "has_count_head": True,
        },
        args.output,
    )
    print(
        json.dumps(
            {
                "checkpoint": str(args.output),
                "final_loss": losses[-1],
                "best_loss": best_loss,
                "best_step": best_step,
                "device": str(device),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
