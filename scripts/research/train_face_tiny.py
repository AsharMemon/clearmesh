#!/usr/bin/env python3
"""Tiny FACE-token training smoke.

Examples:
  python scripts/research/train_face_tiny.py --dry-run
  python scripts/research/train_face_tiny.py --steps 20 --output artifacts/face_tiny_smoke.pt
  python scripts/research/train_face_tiny.py --mesh-dir data/objaverse_subset --steps 2000
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clearmesh.mesh_heads.face_tiny import build_tiny_face_autoregressor, sequence_to_autoregressive_tokens
from clearmesh.mesh_heads.face_tokens import (
    decode_face_tokens_to_mesh,
    encode_mesh_to_face_tokens,
    face_token_stats,
)
from clearmesh.utils.checkpoint import args_to_json_safe


def _load_mesh(path: Path) -> trimesh.Trimesh | None:
    loaded = trimesh.load(path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        pieces = [geom for geom in loaded.geometry.values() if isinstance(geom, trimesh.Trimesh)]
        if not pieces:
            return None
        loaded = trimesh.util.concatenate(pieces)
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
        return None
    if not loaded.is_watertight:
        loaded = loaded.copy()
    return loaded


def _load_meshes(mesh_dir: Path | None, synthetic_count: int, seed: int) -> list[trimesh.Trimesh]:
    meshes: list[trimesh.Trimesh] = []
    if mesh_dir is not None:
        suffixes = {".obj", ".ply", ".stl", ".glb", ".gltf"}
        for path in sorted(p for p in mesh_dir.rglob("*") if p.suffix.lower() in suffixes):
            mesh = _load_mesh(path)
            if mesh is not None:
                meshes.append(mesh)

    rng = np.random.default_rng(seed)
    for _ in range(max(0, synthetic_count)):
        extents = rng.uniform(0.5, 1.8, size=3)
        mesh = trimesh.creation.box(extents=extents)
        meshes.append(mesh)
    return meshes


def _encode_dataset(
    meshes: list[trimesh.Trimesh],
    num_bins: int,
    max_faces: int,
) -> list[tuple[np.ndarray, np.ndarray, dict[str, int | float]]]:
    encoded = []
    for mesh in meshes:
        if len(mesh.faces) > max_faces:
            continue
        sequence = encode_mesh_to_face_tokens(mesh, num_bins=num_bins, max_faces=max_faces)
        decoded = decode_face_tokens_to_mesh(sequence)
        stats = face_token_stats(sequence)
        stats["decoded_watertight"] = bool(decoded.is_watertight)
        input_ids, target_ids = sequence_to_autoregressive_tokens(sequence)
        encoded.append((input_ids, target_ids, stats))
    return encoded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("artifacts/face_tiny_smoke.pt"))
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--num-bins", type=int, default=128)
    parser.add_argument("--max-faces", type=int, default=256)
    parser.add_argument("--synthetic-count", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    meshes = _load_meshes(args.mesh_dir, synthetic_count=args.synthetic_count, seed=args.seed)
    dataset = _encode_dataset(meshes, num_bins=args.num_bins, max_faces=args.max_faces)
    if not dataset:
        raise SystemExit("No meshes could be encoded. Lower filters or provide a mesh directory.")

    max_tokens = max(len(input_ids) for input_ids, _, _ in dataset)
    summary = {
        "encoded_meshes": len(dataset),
        "max_tokens": max_tokens,
        "first_stats": dataset[0][2],
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
    model = build_tiny_face_autoregressor(
        num_bins=args.num_bins,
        max_tokens=max_tokens,
        hidden_size=args.hidden_size,
        layers=args.layers,
        heads=args.heads,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    losses = []
    best_loss = float("inf")
    best_step = 0
    best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
    for step in range(1, args.steps + 1):
        input_ids, target_ids, _ = random.choice(dataset)
        x = torch.as_tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
        y = torch.as_tensor(target_ids, dtype=torch.long, device=device).unsqueeze(0)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
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
