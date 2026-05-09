#!/usr/bin/env python3
"""Build FACE-token NPZ shards from meshes.

This is the bridge from "tokenizer works" to "train on a real corpus":
each output shard contains canonical FACE coordinate tokens plus the affine
normalization transform needed to decode/edit/debug examples later.
"""

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

from clearmesh.mesh_heads.face_indexed import (
    decode_indexed_face_tokens_to_mesh,
    encode_mesh_to_indexed_face_tokens,
    indexed_face_closure_counts,
    indexed_face_stats,
    indexed_to_coordinate_tokens,
)
from clearmesh.mesh_heads.face_tokens import (
    decode_face_tokens_to_mesh,
    decode_paper_face_tokens_to_mesh,
    encode_mesh_to_face_tokens,
    encode_mesh_to_paper_face_tokens,
    face_token_stats,
)
from clearmesh.mesh_heads.face_topology import face_token_topology_report


def _slug(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")
    return slug or "mesh"


def _load_mesh(path: Path) -> trimesh.Trimesh | None:
    loaded = trimesh.load(path, force="mesh", process=False, skip_materials=True)
    if isinstance(loaded, trimesh.Scene):
        pieces = [geom for geom in loaded.geometry.values() if isinstance(geom, trimesh.Trimesh)]
        if not pieces:
            return None
        loaded = trimesh.util.concatenate(pieces)
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
        return None
    return loaded


def _random_transform(mesh: trimesh.Trimesh, rng: np.random.Generator) -> trimesh.Trimesh:
    mesh = mesh.copy()
    mesh.apply_scale(rng.uniform(0.55, 1.65, size=3))
    return mesh


def _synthetic_mesh(index: int, rng: np.random.Generator, kind: str) -> tuple[str, trimesh.Trimesh]:
    if kind == "boxes":
        extents = rng.uniform(0.5, 1.8, size=3)
        return f"synthetic_box_{index:05d}", trimesh.creation.box(extents=extents)

    choices = ["box", "cylinder", "cone", "icosphere", "capsule", "torus"]
    shape = choices[index % len(choices)] if kind == "mixed_cycle" else str(rng.choice(choices))
    if shape == "box":
        mesh = trimesh.creation.box(extents=rng.uniform(0.5, 1.8, size=3))
    elif shape == "cylinder":
        mesh = trimesh.creation.cylinder(
            radius=float(rng.uniform(0.35, 0.9)),
            height=float(rng.uniform(0.6, 1.8)),
            sections=int(rng.choice([8, 12, 16])),
        )
        mesh = _random_transform(mesh, rng)
    elif shape == "cone":
        mesh = trimesh.creation.cone(
            radius=float(rng.uniform(0.35, 0.9)),
            height=float(rng.uniform(0.6, 1.8)),
            sections=int(rng.choice([8, 12, 16])),
        )
        mesh = _random_transform(mesh, rng)
    elif shape == "icosphere":
        mesh = trimesh.creation.icosphere(
            subdivisions=int(rng.choice([1, 2])),
            radius=float(rng.uniform(0.45, 0.95)),
        )
        mesh = _random_transform(mesh, rng)
    elif shape == "capsule":
        mesh = trimesh.creation.capsule(
            height=float(rng.uniform(0.6, 1.6)),
            radius=float(rng.uniform(0.25, 0.55)),
            count=[8, 8],
        )
        mesh = _random_transform(mesh, rng)
    elif shape == "torus":
        mesh = trimesh.creation.torus(
            major_radius=float(rng.uniform(0.45, 0.8)),
            minor_radius=float(rng.uniform(0.12, 0.25)),
            major_sections=int(rng.choice([12, 16])),
            minor_sections=8,
        )
        mesh = _random_transform(mesh, rng)
    else:  # pragma: no cover - guarded by choices above.
        raise ValueError(f"unknown synthetic shape {shape}")
    return f"synthetic_{shape}_{index:05d}", mesh


def _iter_manifest_rows(manifest: Path):
    if manifest.suffix.lower() == ".jsonl":
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                yield json.loads(line)
        return
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "records" in data:
        for row in data["records"]:
            if row.get("accepted", row.get("status") == "accepted"):
                yield row
        return
    if isinstance(data, list):
        yield from data
        return
    raise ValueError(f"unsupported manifest format: {manifest}")


def _manifest_mesh_path(row: dict) -> Path | None:
    for key in ("target_path", "path", "local_path", "source_path"):
        value = row.get(key)
        if value:
            path = Path(str(value))
            if path.suffix.lower() in {".obj", ".ply", ".stl", ".glb", ".gltf"}:
                return path
    return None


def _manifest_mesh_name(row: dict, path: Path) -> str:
    for key in ("uid", "source_name", "name"):
        value = row.get(key)
        if value:
            return str(value)
    return path.stem


def _iter_meshes(mesh_dir: Path | None, manifest: Path | None, synthetic_count: int, seed: int, synthetic_kind: str):
    if manifest is not None:
        for row in _iter_manifest_rows(manifest):
            path = _manifest_mesh_path(row)
            if path is None:
                continue
            mesh = _load_mesh(path)
            if mesh is not None:
                yield _manifest_mesh_name(row, path), mesh

    if mesh_dir is not None:
        suffixes = {".obj", ".ply", ".stl", ".glb", ".gltf"}
        for path in sorted(p for p in mesh_dir.rglob("*") if p.suffix.lower() in suffixes):
            mesh = _load_mesh(path)
            if mesh is not None:
                yield path.stem, mesh

    rng = np.random.default_rng(seed)
    for idx in range(max(0, synthetic_count)):
        yield _synthetic_mesh(idx, rng, synthetic_kind)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-dir", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None, help="JSON/JSONL manifest with target_path/path rows.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-bins", "--quantization-bins", dest="num_bins", type=int, default=128)
    parser.add_argument("--max-faces", type=int, default=4096)
    parser.add_argument("--point-samples", type=int, default=8192)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--synthetic-count", type=int, default=0)
    parser.add_argument("--synthetic-kind", choices=["boxes", "mixed", "mixed_cycle"], default="boxes")
    parser.add_argument(
        "--paper-within-face-order",
        choices=["preserve", "rotate_min_zyx", "sort_zyx"],
        default="preserve",
        help="FACE paper leaves within-triangle vertex ordering under-specified; expose it for ablations.",
    )
    parser.add_argument(
        "--indexed-face-order",
        choices=["lex", "boundary_growth"],
        default="lex",
        help="Order indexed topology targets lexicographically or as a boundary-growing shelling sequence.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.mesh_dir is None and args.manifest is None and args.synthetic_count <= 0:
        parser.error("one of --mesh-dir, --manifest, or --synthetic-count is required")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.jsonl"
    written = 0
    skipped = 0

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for name, mesh in _iter_meshes(args.mesh_dir, args.manifest, args.synthetic_count, args.seed, args.synthetic_kind):
            if args.limit and written >= args.limit:
                break
            if len(mesh.faces) > args.max_faces:
                skipped += 1
                continue
            try:
                sequence = encode_mesh_to_face_tokens(mesh, num_bins=args.num_bins, max_faces=args.max_faces)
                paper_sequence = encode_mesh_to_paper_face_tokens(
                    mesh,
                    num_bins=args.num_bins,
                    max_faces=args.max_faces,
                    within_face_order=args.paper_within_face_order,
                )
                indexed_sequence = encode_mesh_to_indexed_face_tokens(
                    mesh,
                    num_bins=args.num_bins,
                    max_faces=args.max_faces,
                    face_order=args.indexed_face_order,
                )
                decoded = decode_face_tokens_to_mesh(sequence)
                paper_decoded = decode_paper_face_tokens_to_mesh(paper_sequence)
                indexed_decoded = decode_indexed_face_tokens_to_mesh(indexed_sequence)
                if args.point_samples > 0:
                    surface_points, face_indices = trimesh.sample.sample_surface(
                        mesh,
                        args.point_samples,
                        seed=args.seed + written,
                    )
                    surface_points = sequence.transform.normalize(surface_points).astype(np.float32)
                    surface_normals = np.asarray(mesh.face_normals[face_indices], dtype=np.float32)
                else:
                    surface_points = np.zeros((0, 3), dtype=np.float32)
                    surface_normals = np.zeros((0, 3), dtype=np.float32)
            except Exception:
                skipped += 1
                continue

            stem = f"{written:07d}_{_slug(name)}"
            out_path = args.output_dir / f"{stem}.npz"
            stats = face_token_stats(sequence)
            topology = face_token_topology_report(sequence.tokens)
            paper_topology = face_token_topology_report(paper_sequence.tokens)
            indexed_topology = face_token_topology_report(indexed_to_coordinate_tokens(indexed_sequence))
            indexed_stats = indexed_face_stats(indexed_sequence)
            indexed_closures = indexed_face_closure_counts(indexed_sequence.faces)
            stats["source_name"] = name
            stats["source_faces"] = int(len(mesh.faces))
            stats["decoded_watertight"] = bool(decoded.is_watertight)
            stats["token_watertight_edge_graph"] = bool(topology.watertight_edge_graph)
            stats["token_boundary_edge_count"] = int(topology.boundary_edge_count)
            stats["token_nonmanifold_edge_count"] = int(topology.nonmanifold_edge_count)
            stats["token_edge_pairing_ratio"] = float(topology.edge_pairing_ratio)
            stats["paper_decoded_watertight"] = bool(paper_decoded.is_watertight)
            stats["paper_token_watertight_edge_graph"] = bool(paper_topology.watertight_edge_graph)
            stats["paper_token_boundary_edge_count"] = int(paper_topology.boundary_edge_count)
            stats["paper_token_nonmanifold_edge_count"] = int(paper_topology.nonmanifold_edge_count)
            stats["paper_token_edge_pairing_ratio"] = float(paper_topology.edge_pairing_ratio)
            stats["indexed_vertices"] = int(indexed_stats["vertices"])
            stats["indexed_faces"] = int(indexed_stats["faces"])
            stats["indexed_index_tokens"] = int(indexed_stats["index_tokens"])
            stats["indexed_compression_vs_xyz_face_tokens"] = float(indexed_stats["compression_vs_xyz_face_tokens"])
            stats["indexed_decoded_watertight"] = bool(indexed_decoded.is_watertight)
            stats["indexed_token_watertight_edge_graph"] = bool(indexed_topology.watertight_edge_graph)
            stats["indexed_token_boundary_edge_count"] = int(indexed_topology.boundary_edge_count)
            stats["indexed_token_nonmanifold_edge_count"] = int(indexed_topology.nonmanifold_edge_count)
            stats["indexed_token_edge_pairing_ratio"] = float(indexed_topology.edge_pairing_ratio)
            stats["indexed_face_order"] = args.indexed_face_order
            stats["indexed_zero_closure_after_first"] = int(np.sum(indexed_closures[1:] == 0)) if len(indexed_closures) > 1 else 0
            stats["indexed_zero_closure_after_first_ratio"] = (
                float(np.mean(indexed_closures[1:] == 0)) if len(indexed_closures) > 1 else 0.0
            )
            stats["point_samples"] = int(len(surface_points))
            np.savez_compressed(
                out_path,
                tokens=np.asarray(sequence.tokens, dtype=np.int16),
                paper_tokens=np.asarray(paper_sequence.tokens, dtype=np.int16),
                indexed_vertices=np.asarray(indexed_sequence.vertices, dtype=np.int16),
                indexed_faces=np.asarray(indexed_sequence.faces, dtype=np.int32),
                center=np.asarray(sequence.transform.center, dtype=np.float32),
                scale=np.asarray([sequence.transform.scale], dtype=np.float32),
                num_bins=np.asarray([sequence.num_bins], dtype=np.int32),
                paper_within_face_order=np.asarray([args.paper_within_face_order]),
                surface_points=surface_points,
                surface_normals=surface_normals,
            )
            manifest.write(json.dumps({"path": str(out_path), **stats}, sort_keys=True) + "\n")
            written += 1

    print(json.dumps({"written": written, "skipped": skipped, "manifest": str(manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
