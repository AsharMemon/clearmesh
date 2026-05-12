#!/usr/bin/env python3
"""Build LATTICE geometry sample NPZ shards from meshes."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clearmesh.lattice import (
    ActiveVoxelOptions,
    build_irregular_patches,
    extract_active_surface_voxels,
    sample_edge_candidates,
    sample_vdf,
)


def _slug(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")
    return slug or "mesh"


def _load_mesh(path: Path) -> trimesh.Trimesh | None:
    loaded = trimesh.load(path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        pieces = [geom for geom in loaded.geometry.values() if isinstance(geom, trimesh.Trimesh)]
        if not pieces:
            return None
        loaded = trimesh.util.concatenate(pieces)
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
        return None
    return loaded


def _iter_meshes(mesh_dir: Path | None, synthetic_count: int, seed: int):
    if mesh_dir is not None:
        suffixes = {".obj", ".ply", ".stl", ".glb", ".gltf"}
        for path in sorted(p for p in mesh_dir.rglob("*") if p.suffix.lower() in suffixes):
            mesh = _load_mesh(path)
            if mesh is not None:
                yield path.stem, mesh

    rng = np.random.default_rng(seed)
    for idx in range(max(0, synthetic_count)):
        extents = rng.uniform(0.5, 1.8, size=3)
        yield f"synthetic_box_{idx:05d}", trimesh.creation.box(extents=extents)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--synthetic-count", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--surface-samples", type=int, default=20_000)
    parser.add_argument("--vdf-samples", type=int, default=8192)
    parser.add_argument("--patch-surface-samples", type=int, default=4096)
    parser.add_argument("--anchor-count", type=int, default=512)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--negative-edges", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.jsonl"
    written = 0
    skipped = 0

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for name, mesh in _iter_meshes(args.mesh_dir, args.synthetic_count, args.seed):
            if args.limit and written >= args.limit:
                break
            try:
                voxels = extract_active_surface_voxels(
                    mesh,
                    ActiveVoxelOptions(
                        resolution=args.resolution,
                        surface_samples=args.surface_samples,
                        seed=args.seed + written,
                    ),
                )
                vdf = sample_vdf(mesh, count=args.vdf_samples, seed=args.seed + written)
                face_points = np.asarray(mesh.triangles_center, dtype=np.float32)
                face_vertices = np.asarray(mesh.vertices[mesh.faces], dtype=np.float32)
                face_displacements = face_vertices - face_points[:, None, :]
                face_normals = np.asarray(mesh.face_normals, dtype=np.float32)
                edges = sample_edge_candidates(
                    mesh,
                    random_negative_count=args.negative_edges or None,
                    seed=args.seed + written,
                )
                patch_points, _ = trimesh.sample.sample_surface(
                    mesh,
                    args.patch_surface_samples,
                    seed=args.seed + written,
                )
                patches = build_irregular_patches(
                    patch_points,
                    anchor_count=args.anchor_count,
                    patch_size=args.patch_size,
                    seed=args.seed + written,
                )
            except Exception:
                skipped += 1
                continue

            stem = f"{written:07d}_{_slug(name)}"
            out_path = args.output_dir / f"{stem}.npz"
            np.savez_compressed(
                out_path,
                voxel_indices=np.asarray(voxels.indices, dtype=np.int16),
                voxel_centers=np.asarray(voxels.centers, dtype=np.float32),
                vdf_points=np.asarray(vdf.points, dtype=np.float32),
                vdf_normals=np.asarray(vdf.normals, dtype=np.float32),
                vdf_vertex_displacements=np.asarray(vdf.vertex_displacements, dtype=np.float32),
                face_vdf_points=face_points,
                face_vdf_normals=face_normals,
                face_vdf_vertex_displacements=face_displacements,
                edge_positive=np.asarray(edges.positive_edges, dtype=np.int32),
                edge_negative=np.asarray(edges.negative_edges, dtype=np.int32),
                patch_points=np.asarray(patch_points, dtype=np.float32),
                patch_anchor_indices=np.asarray(patches.anchor_indices, dtype=np.int32),
                patch_indices=np.asarray(patches.patch_indices, dtype=np.int32),
            )
            row = {
                "path": str(out_path),
                "source_name": name,
                "source_faces": int(len(mesh.faces)),
                "active_voxels": int(voxels.count),
                "vdf_samples": int(args.vdf_samples),
                "positive_edges": int(edges.positive_count),
                "negative_edges": int(edges.negative_count),
                "anchors": int(patches.anchor_count),
                "patch_size": int(patches.patch_size),
            }
            manifest.write(json.dumps(row, sort_keys=True) + "\n")
            written += 1

    print(json.dumps({"written": written, "skipped": skipped, "manifest": str(manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
