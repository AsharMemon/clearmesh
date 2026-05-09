#!/usr/bin/env python3
"""Prepare a small real-proxy mesh corpus for FACE-level smoke tests.

This is intentionally conservative: it never mutates source artifacts, it only
loads meshes, applies light optional hygiene, filters by face count, and exports
accepted meshes into a standalone fixture directory that can be uploaded to a
GPU worker.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import trimesh

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clearmesh.eval.mesh_quality import evaluate_mesh
from clearmesh.mesh.cleanup import CleanupOptions, cleanup_mesh

MESH_SUFFIXES = {".glb", ".gltf", ".obj", ".ply", ".stl"}


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.-")
    return slug or "mesh"


def _iter_inputs(inputs: list[Path], roots: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for path in inputs:
        if path.is_file() and path.suffix.lower() in MESH_SUFFIXES:
            paths.append(path)
    for root in roots:
        if root.is_file() and root.suffix.lower() in MESH_SUFFIXES:
            paths.append(root)
        elif root.is_dir():
            paths.extend(sorted(p for p in root.rglob("*") if p.suffix.lower() in MESH_SUFFIXES))

    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def _load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        pieces = [geom for geom in loaded.geometry.values() if isinstance(geom, trimesh.Trimesh) and len(geom.faces) > 0]
        if not pieces:
            raise ValueError("scene contains no mesh geometry")
        return trimesh.util.concatenate(pieces)
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
        raise ValueError(f"unsupported or empty mesh: {type(loaded).__name__}")
    return loaded


def _clean_for_fixture(mesh: trimesh.Trimesh, *, fill_holes: bool, keep_largest_components: int | None) -> trimesh.Trimesh:
    cleaned, _ = cleanup_mesh(
        mesh,
        CleanupOptions(
            min_component_faces=1,
            keep_largest_components=keep_largest_components,
            fill_holes=fill_holes,
            merge_vertices=True,
            fix_normals=True,
        ),
    )
    return cleaned


def _record(status: str, path: Path, **values: Any) -> dict[str, Any]:
    return {
        "status": status,
        "source_path": str(path),
        **values,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", default=[], help="A specific mesh file to consider.")
    parser.add_argument("--root", type=Path, action="append", default=[], help="A directory tree of candidate mesh files.")
    parser.add_argument("--input-list", type=Path, default=None, help="Newline-delimited list of mesh paths.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-faces", type=int, default=2048)
    parser.add_argument("--min-faces", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--decimate-to-faces",
        type=int,
        default=0,
        help="If set, quadric-decimate meshes above this face count before filtering.",
    )
    parser.add_argument("--decimation-aggression", type=int, default=7)
    parser.add_argument(
        "--max-decimation-source-faces",
        type=int,
        default=0,
        help="Skip meshes above this source face count before decimation. 0 disables this guard.",
    )
    parser.add_argument("--fill-holes", action="store_true")
    parser.add_argument("--keep-largest-components", type=int, default=0)
    parser.add_argument("--manifest-name", default="fixture_manifest.json")
    args = parser.parse_args()

    inputs = list(args.input)
    if args.input_list is not None:
        for line in args.input_list.read_text(encoding="utf-8").splitlines():
            value = line.strip()
            if value and not value.startswith("#"):
                inputs.append(Path(value))

    candidate_paths = _iter_inputs(inputs, args.root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir = args.output_dir / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    accepted = 0
    records: list[dict[str, Any]] = []
    for path in candidate_paths:
        if args.limit and accepted >= args.limit:
            records.append(_record("not_attempted_limit_reached", path))
            continue
        if not path.exists():
            records.append(_record("skipped_missing", path))
            continue
        try:
            mesh = _load_mesh(path)
            input_faces = int(len(mesh.faces))
            input_vertices = int(len(mesh.vertices))
            decimated = False
            if (
                args.decimate_to_faces > 0
                and args.max_decimation_source_faces > 0
                and input_faces > args.max_decimation_source_faces
            ):
                records.append(
                    _record(
                        "skipped_too_many_source_faces_for_decimation",
                        path,
                        input_faces=input_faces,
                        input_vertices=input_vertices,
                        max_decimation_source_faces=int(args.max_decimation_source_faces),
                    )
                )
                continue
            if args.decimate_to_faces > 0 and len(mesh.faces) > args.decimate_to_faces:
                mesh = mesh.simplify_quadric_decimation(
                    face_count=int(args.decimate_to_faces),
                    aggression=int(args.decimation_aggression),
                )
                decimated = True
            if args.keep_largest_components > 0 or args.fill_holes:
                mesh = _clean_for_fixture(
                    mesh,
                    fill_holes=bool(args.fill_holes),
                    keep_largest_components=args.keep_largest_components or None,
                )
            output_faces = int(len(mesh.faces))
            output_vertices = int(len(mesh.vertices))
            if output_faces < args.min_faces:
                records.append(
                    _record(
                        "skipped_too_few_faces",
                        path,
                        input_faces=input_faces,
                        input_vertices=input_vertices,
                        output_faces=output_faces,
                        output_vertices=output_vertices,
                        decimated=decimated,
                    )
                )
                continue
            if output_faces > args.max_faces:
                records.append(
                    _record(
                        "skipped_too_many_faces",
                        path,
                        input_faces=input_faces,
                        input_vertices=input_vertices,
                        output_faces=output_faces,
                        output_vertices=output_vertices,
                        decimated=decimated,
                    )
                )
                continue

            stem = f"{accepted:04d}_{_slug(path.parent.name)}_{_slug(path.stem)}"
            out_path = mesh_dir / f"{stem}.glb"
            mesh.export(out_path)
            metrics = evaluate_mesh(out_path)
            records.append(
                _record(
                    "accepted",
                    path,
                    output_path=str(out_path),
                    input_faces=input_faces,
                    input_vertices=input_vertices,
                    output_faces=output_faces,
                    output_vertices=output_vertices,
                    decimated=decimated,
                    metrics=metrics,
                )
            )
            accepted += 1
        except Exception as exc:  # noqa: BLE001 - fixture prep should report and keep scanning.
            records.append(_record("skipped_error", path, error=f"{type(exc).__name__}: {exc}"))

    summary = {
        "candidate_count": len(candidate_paths),
        "accepted": accepted,
        "mesh_dir": str(mesh_dir),
        "max_faces": int(args.max_faces),
        "min_faces": int(args.min_faces),
        "decimate_to_faces": int(args.decimate_to_faces),
        "decimation_aggression": int(args.decimation_aggression),
        "max_decimation_source_faces": int(args.max_decimation_source_faces),
        "fill_holes": bool(args.fill_holes),
        "keep_largest_components": int(args.keep_largest_components),
        "records": records,
    }
    manifest_path = args.output_dir / args.manifest_name
    manifest_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=2, sort_keys=True))
    print(f"Wrote {manifest_path}")
    return 0 if accepted > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
