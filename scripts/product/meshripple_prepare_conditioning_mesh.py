#!/usr/bin/env python3
"""Prepare a TRELLIS/proxy mesh for MeshRipple-style conditioning.

The first version used `mesh.split()`, which materializes thousands of submeshes
for fragmented TRELLIS output. This version labels connected face components and
filters by face mask, which is much faster and less memory-hungry.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import trimesh


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="scene")
    if isinstance(loaded, trimesh.Scene):
        meshes = [geom for geom in loaded.geometry.values() if isinstance(geom, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"no mesh geometry in {path}")
        return trimesh.util.concatenate(meshes)
    if isinstance(loaded, trimesh.Trimesh):
        return loaded
    raise ValueError(f"unsupported mesh type: {type(loaded).__name__}")


def face_component_labels(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    """Return a label per face and face counts per component."""

    face_count = len(mesh.faces)
    if face_count == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    adjacency = np.asarray(mesh.face_adjacency, dtype=np.int64)
    if len(adjacency) == 0:
        labels = np.arange(face_count, dtype=np.int64)
        counts = np.ones(face_count, dtype=np.int64)
        return labels, counts

    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    rows = np.concatenate([adjacency[:, 0], adjacency[:, 1]])
    cols = np.concatenate([adjacency[:, 1], adjacency[:, 0]])
    graph = coo_matrix((np.ones(len(rows), dtype=np.uint8), (rows, cols)), shape=(face_count, face_count)).tocsr()
    component_count, labels = connected_components(graph, directed=False, return_labels=True)
    counts = np.bincount(labels, minlength=component_count)
    return labels.astype(np.int64), counts.astype(np.int64)


def filter_components(mesh: trimesh.Trimesh, *, keep_largest: int, min_faces: int) -> tuple[trimesh.Trimesh, dict]:
    labels, counts = face_component_labels(mesh)
    if len(labels) == 0:
        raise ValueError("mesh has no faces")
    component_ids = np.arange(len(counts))
    component_ids = component_ids[counts >= min_faces]
    if len(component_ids) == 0:
        raise ValueError("no components survived filtering")
    component_ids = component_ids[np.argsort(counts[component_ids])[::-1]]
    if keep_largest > 0:
        component_ids = component_ids[:keep_largest]
    keep_mask = np.isin(labels, component_ids)
    filtered = mesh.submesh([keep_mask], append=True, repair=False)
    filtered.remove_unreferenced_vertices()
    report = {
        "input_components": int(len(counts)),
        "kept_components": int(len(component_ids)),
        "largest_component_faces": int(counts.max()),
        "kept_faces_before_cleanup": int(keep_mask.sum()),
    }
    return filtered, report


def simplify_mesh(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    if target_faces <= 0 or len(mesh.faces) <= target_faces:
        return mesh
    try:
        import open3d as o3d

        o3d_mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(mesh.vertices),
            triangles=o3d.utility.Vector3iVector(mesh.faces),
        )
        simplified = o3d_mesh.simplify_quadric_decimation(target_faces)
        return trimesh.Trimesh(vertices=simplified.vertices, faces=simplified.triangles, process=False)
    except Exception:
        simplified = mesh.simplify_quadric_decimation(target_faces)
        return simplified if simplified is not None else mesh


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--keep-largest-components", type=int, default=20)
    parser.add_argument("--min-component-faces", type=int, default=32)
    parser.add_argument("--target-faces", type=int, default=5000)
    args = parser.parse_args()

    mesh = load_mesh(args.input)
    raw = {"faces": int(len(mesh.faces)), "vertices": int(len(mesh.vertices))}
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    after_basic = {"faces": int(len(mesh.faces)), "vertices": int(len(mesh.vertices))}

    mesh, component_report = filter_components(
        mesh,
        keep_largest=args.keep_largest_components,
        min_faces=args.min_component_faces,
    )
    before_simplify = {"faces": int(len(mesh.faces)), "vertices": int(len(mesh.vertices)), **component_report}
    mesh = simplify_mesh(mesh, args.target_faces)

    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()

    labels, counts = face_component_labels(mesh)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(args.output)
    report = {
        "input": str(args.input),
        "output": str(args.output),
        "raw": raw,
        "after_basic_cleanup": after_basic,
        "before_simplify": before_simplify,
        "final": {
            "faces": int(len(mesh.faces)),
            "vertices": int(len(mesh.vertices)),
            "components": int(len(counts)),
            "largest_component_faces": int(counts.max()) if len(counts) else 0,
            "watertight": bool(mesh.is_watertight),
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
