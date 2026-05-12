#!/usr/bin/env python3
"""Export a small mesh corpus with optional fixed augmentation variants.

This is a pre-scale trust-ladder utility: instead of changing augmentation every
training step, export deterministic augmented mesh variants first, tokenize them
once, and train on the finite corpus. That separates augmentation diversity from
moving-target sequence instability.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research.build_face_token_dataset import _iter_meshes


def _slug(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")
    return slug or "mesh"


def _random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    q = rng.normal(size=4)
    q /= np.linalg.norm(q) + 1e-12
    w, x, y, z = q
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _augmentation_matrix(
    rng: np.random.Generator,
    *,
    rotation: str,
    scale_min: float,
    scale_max: float,
    flip_prob: float,
) -> np.ndarray:
    if rotation == "none":
        rotation_matrix = np.eye(3, dtype=np.float64)
    elif rotation == "z":
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        rotation_matrix = np.asarray(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    elif rotation == "so3":
        rotation_matrix = _random_rotation_matrix(rng)
    else:  # pragma: no cover - argparse choices guard this.
        raise ValueError(f"unknown rotation mode {rotation!r}")
    flips = np.where(rng.random(3) < flip_prob, -1.0, 1.0).astype(np.float64)
    scales = rng.uniform(scale_min, scale_max, size=3)
    return rotation_matrix @ np.diag(flips * scales)


def _augment_mesh(mesh: trimesh.Trimesh, affine: np.ndarray) -> trimesh.Trimesh:
    out = mesh.copy()
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = affine
    out.apply_transform(matrix)
    if np.linalg.det(affine) < 0.0:
        out.invert()
    out.remove_unreferenced_vertices()
    return out


def _export_mesh(mesh: trimesh.Trimesh, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--synthetic-count", type=int, default=0)
    parser.add_argument("--synthetic-kind", choices=["boxes", "mixed", "mixed_cycle"], default="mixed_cycle")
    parser.add_argument("--variants", type=int, default=0)
    parser.add_argument("--rotation", choices=["none", "z", "so3"], default="z")
    parser.add_argument("--scale-min", type=float, default=0.95)
    parser.add_argument("--scale-max", type=float, default=1.05)
    parser.add_argument("--flip-prob", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if args.variants < 0:
        raise SystemExit("--variants must be non-negative")
    if args.scale_min <= 0.0 or args.scale_max <= 0.0 or args.scale_min > args.scale_max:
        raise SystemExit("invalid scale range")
    if not 0.0 <= args.flip_prob <= 1.0:
        raise SystemExit("--flip-prob must be in [0, 1]")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    exported = 0
    source_count = 0
    for name, mesh in _iter_meshes(args.mesh_dir, args.synthetic_count, args.seed, args.synthetic_kind):
        if args.limit and source_count >= args.limit:
            break
        source_count += 1
        base_name = f"{source_count - 1:06d}_{_slug(name)}"
        _export_mesh(mesh, args.output_dir / f"{base_name}_base.glb")
        exported += 1
        for variant in range(args.variants):
            affine = _augmentation_matrix(
                rng,
                rotation=args.rotation,
                scale_min=args.scale_min,
                scale_max=args.scale_max,
                flip_prob=args.flip_prob,
            )
            augmented = _augment_mesh(mesh, affine)
            _export_mesh(augmented, args.output_dir / f"{base_name}_aug{variant:03d}.glb")
            exported += 1
    print({"sources": source_count, "exported": exported, "output_dir": str(args.output_dir)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
