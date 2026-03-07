#!/usr/bin/env python3
"""Prepare a normalized, texture-stripped mesh asset in an isolated process."""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys

import numpy as np
import trimesh

FAIL_FILE_NOT_FOUND = "FILE_NOT_FOUND"
FAIL_MESH_LOAD = "MESH_LOAD_FAIL"
FAIL_MESH_EMPTY = "MESH_EMPTY"
FAIL_MESH_DEGENERATE = "MESH_DEGENERATE"
FAIL_MESH_TOO_LARGE = "MESH_FILE_TOO_LARGE"
FAIL_MESH_TOO_COMPLEX = "MESH_TOO_COMPLEX"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--max-file-bytes", type=int, required=True)
    parser.add_argument("--max-faces", type=int, required=True)
    parser.add_argument("--memory-limit-gb", type=float, default=0.0)
    return parser.parse_args()


def write_meta(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def apply_memory_limit(memory_limit_gb: float) -> None:
    if memory_limit_gb <= 0:
        return
    limit_bytes = int(memory_limit_gb * (1024 ** 3))
    resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))


def prepare_render_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    render_mesh = mesh.copy()
    face_color = np.tile(np.array([[200, 200, 200, 255]], dtype=np.uint8), (len(render_mesh.faces), 1))
    try:
        render_mesh.visual = trimesh.visual.ColorVisuals(mesh=render_mesh, face_colors=face_color)
    except Exception:
        pass
    return render_mesh


def load_mesh_robust(mesh_path: str, max_file_bytes: int, max_faces: int):
    if not os.path.exists(mesh_path):
        return None, FAIL_FILE_NOT_FOUND

    try:
        file_size = os.path.getsize(mesh_path)
        if file_size > max_file_bytes:
            return None, FAIL_MESH_TOO_LARGE
    except OSError:
        file_size = None

    try:
        loaded = trimesh.load(mesh_path, process=False)
    except Exception:
        return None, FAIL_MESH_LOAD

    if isinstance(loaded, trimesh.Scene):
        meshes = [
            g for g in loaded.geometry.values()
            if isinstance(g, trimesh.Trimesh) and len(g.faces) > 0
        ]
        if not meshes:
            return None, FAIL_MESH_EMPTY
        if len(meshes) == 1:
            loaded = meshes[0]
        else:
            total_faces = sum(len(m.faces) for m in meshes)
            try:
                if total_faces > max_faces:
                    loaded = max(meshes, key=lambda m: len(m.faces))
                else:
                    loaded = trimesh.util.concatenate(meshes)
            except Exception:
                loaded = max(meshes, key=lambda m: len(m.faces))

    if not isinstance(loaded, trimesh.Trimesh):
        try:
            loaded = trimesh.load(mesh_path, force="mesh", process=False)
        except Exception:
            return None, FAIL_MESH_LOAD

    if len(loaded.vertices) == 0 or len(loaded.faces) == 0:
        return None, FAIL_MESH_EMPTY
    if len(loaded.faces) > max_faces:
        return None, FAIL_MESH_TOO_COMPLEX
    if np.any(~np.isfinite(loaded.vertices)):
        return None, FAIL_MESH_DEGENERATE

    if len(loaded.faces) < 100_000:
        try:
            components = loaded.split(only_watertight=False)
            if len(components) > 1:
                largest = max(components, key=lambda c: len(c.faces))
                if len(largest.faces) >= 4:
                    loaded = largest
        except Exception:
            pass

    loaded.vertices -= loaded.centroid
    extents = loaded.extents
    max_extent = extents.max()
    if max_extent < 1e-8:
        return None, FAIL_MESH_DEGENERATE
    loaded.vertices /= max_extent
    return prepare_render_mesh(loaded), None


def main() -> int:
    args = parse_args()
    try:
        apply_memory_limit(args.memory_limit_gb)
        mesh, fail_reason = load_mesh_robust(args.mesh, args.max_file_bytes, args.max_faces)
        if fail_reason:
            write_meta(args.meta, {"reason": fail_reason, "mesh": args.mesh})
            return 1
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        mesh.export(args.output)
        write_meta(
            args.meta,
            {
                "reason": None,
                "mesh": args.mesh,
                "output": args.output,
                "vertices": int(len(mesh.vertices)),
                "faces": int(len(mesh.faces)),
            },
        )
        return 0
    except MemoryError:
        write_meta(args.meta, {"reason": FAIL_MESH_TOO_COMPLEX, "mesh": args.mesh})
        return 1
    except Exception as exc:
        write_meta(
            args.meta,
            {
                "reason": FAIL_MESH_LOAD,
                "mesh": args.mesh,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
