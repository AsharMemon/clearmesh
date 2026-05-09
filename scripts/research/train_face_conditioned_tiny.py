#!/usr/bin/env python3
"""Train a tiny point-conditioned FACE decoder on FACE-token shards.

This is a practical FACE reproduction stepping stone: it consumes NPZ shards
from ``build_face_token_dataset.py`` and trains
``surface_points + surface_normals -> FACE coordinate tokens``.
"""

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

from clearmesh.mesh_heads.face_arae import build_tiny_point_conditioned_face_decoder
from clearmesh.mesh_heads.face_tiny import FaceTinyVocabulary
from clearmesh.utils.checkpoint import args_to_json_safe


@dataclass(frozen=True)
class FaceConditionedSample:
    path: Path
    point_features: np.ndarray
    input_ids: np.ndarray
    target_ids: np.ndarray


def _load_sample(path: Path) -> FaceConditionedSample:
    data = np.load(path)
    if "surface_points" not in data or "surface_normals" not in data:
        raise ValueError(f"{path} does not contain surface point conditioning arrays")
    num_bins = int(np.asarray(data["num_bins"]).reshape(-1)[0])
    vocab = FaceTinyVocabulary(num_bins)
    flat = np.asarray(data["tokens"], dtype=np.int64).reshape(-1)
    points = np.asarray(data["surface_points"], dtype=np.float32)
    normals = np.asarray(data["surface_normals"], dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError(f"{path} has invalid surface_points shape {points.shape}")
    if normals.shape != points.shape:
        raise ValueError(f"{path} has surface_normals shape {normals.shape}, expected {points.shape}")
    return FaceConditionedSample(
        path=path,
        point_features=np.concatenate([points, normals], axis=1).astype(np.float32),
        input_ids=np.concatenate([[vocab.bos], flat]).astype(np.int64),
        target_ids=np.concatenate([flat, [vocab.eos]]).astype(np.int64),
    )


def _load_dataset(dataset_dir: Path, limit: int = 0) -> tuple[list[FaceConditionedSample], int]:
    paths = sorted(dataset_dir.glob("*.npz"))
    if limit:
        paths = paths[:limit]
    samples: list[FaceConditionedSample] = []
    num_bins: int | None = None
    for path in paths:
        try:
            sample = _load_sample(path)
        except Exception:
            continue
        data = np.load(path)
        sample_bins = int(np.asarray(data["num_bins"]).reshape(-1)[0])
        if num_bins is None:
            num_bins = sample_bins
        elif num_bins != sample_bins:
            raise ValueError(f"Mixed num_bins in dataset: {num_bins} and {sample_bins}")
        samples.append(sample)
    if not samples or num_bins is None:
        raise SystemExit(f"No conditioned FACE shards found in {dataset_dir}")
    return samples, num_bins


def _make_batch(samples: list[FaceConditionedSample], device, eos_id: int, point_samples: int | None = None):  # type: ignore[no-untyped-def]
    import torch

    max_len = max(len(sample.input_ids) for sample in samples)
    input_ids = torch.full((len(samples), max_len), eos_id, dtype=torch.long, device=device)
    target_ids = torch.full((len(samples), max_len), -100, dtype=torch.long, device=device)
    point_batches = []
    for row, sample in enumerate(samples):
        input_ids[row, : len(sample.input_ids)] = torch.as_tensor(sample.input_ids, dtype=torch.long, device=device)
        target_ids[row, : len(sample.target_ids)] = torch.as_tensor(sample.target_ids, dtype=torch.long, device=device)
        points = sample.point_features
        if point_samples is not None and point_samples > 0:
            if len(points) >= point_samples:
                points = points[:point_samples]
            else:
                repeat = int(np.ceil(point_samples / len(points)))
                points = np.tile(points, (repeat, 1))[:point_samples]
        point_batches.append(torch.as_tensor(points, dtype=torch.float32, device=device))
    point_features = torch.stack(point_batches, dim=0)
    return point_features, input_ids, target_ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/face_conditioned_tiny.pt"))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--point-samples", type=int, default=0)
    parser.add_argument("--hidden-size", type=int, default=192)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--condition-tokens", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    samples, num_bins = _load_dataset(args.dataset_dir, limit=args.limit)
    vocab = FaceTinyVocabulary(num_bins)
    max_tokens = max(len(sample.input_ids) for sample in samples)
    point_count = int(samples[0].point_features.shape[0] if args.point_samples <= 0 else args.point_samples)
    summary = {
        "samples": len(samples),
        "num_bins": num_bins,
        "max_tokens": max_tokens,
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
    model = build_tiny_point_conditioned_face_decoder(
        num_bins=num_bins,
        max_tokens=max_tokens,
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
        point_features, input_ids, target_ids = _make_batch(
            batch,
            device=device,
            eos_id=vocab.eos,
            point_samples=args.point_samples if args.point_samples > 0 else None,
        )
        logits = model(point_features, input_ids)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target_ids.reshape(-1), ignore_index=-100)
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

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": best_state,
            "args": args_to_json_safe(args),
            "losses": losses,
            "num_bins": num_bins,
            "max_tokens": max_tokens,
            "best_loss": best_loss,
            "best_step": best_step,
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
