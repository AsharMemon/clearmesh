"""Mesh passport diagnostics for the unified ClearMesh pipeline.

The passport is a small, deterministic contract between arbitrary 3D generators
and downstream artist-mesh/refinement heads. It turns messy mesh metrics into a
few product decisions without exposing separate "pathways" to users.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from clearmesh.eval.mesh_quality import count_nonmanifold_vertices, load_mesh


@dataclass(frozen=True)
class MeshPassport:
    mesh_path: str
    metrics: dict[str, Any]
    normalized_surface_required: bool
    mesh_head_ready: bool
    high_resolution_ready: bool
    risk_level: str
    notes: list[str]
    recommended_operator: str
    estimated_preview_seconds: tuple[int, int]
    estimated_high_resolution_seconds: tuple[int, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def create_mesh_passport(mesh_path: str | Path) -> MeshPassport:
    """Create a production-facing diagnostic profile for a mesh.

    The thresholds are deliberately conservative. MeshRipple-style heads assume
    coherent conditioning geometry; highly fragmented generator meshes should be
    normalized before the expensive autoregressive stage sees them.
    """

    metrics = fast_passport_metrics(mesh_path)
    if not metrics.get("ok"):
        return MeshPassport(
            mesh_path=str(mesh_path),
            metrics=metrics,
            normalized_surface_required=True,
            mesh_head_ready=False,
            high_resolution_ready=False,
            risk_level="blocked",
            notes=["mesh could not be evaluated"],
            recommended_operator="manual_inspection",
            estimated_preview_seconds=(0, 0),
            estimated_high_resolution_seconds=(0, 0),
        )

    face_count = int(metrics.get("face_count") or 0)
    component_count = int(metrics.get("connected_components") or 0)
    tiny_count = int(metrics.get("tiny_component_count") or 0)
    boundary_loops = int(metrics.get("boundary_loop_count") or 0)
    nonmanifold_edges = int(metrics.get("nonmanifold_edge_count") or 0)
    watertight = bool(metrics.get("watertight"))

    notes: list[str] = []
    if component_count > 50:
        notes.append("high component count; likely shredded conditioning surface")
    if tiny_count > max(10, component_count // 2):
        notes.append("many tiny components; debris removal alone may not be enough")
    if boundary_loops > 25:
        notes.append("many open boundaries; surface reconstruction should run before mesh heads")
    if nonmanifold_edges > 100:
        notes.append("non-manifold edge count is high")
    if not watertight:
        notes.append("not watertight")
    if face_count > 150_000:
        notes.append("very dense input; normalize to bounded control mesh before generation")

    normalized_surface_required = (
        component_count > 20
        or tiny_count > 20
        or boundary_loops > 20
        or nonmanifold_edges > 100
        or face_count > 120_000
        or not watertight
    )
    mesh_head_ready = (
        component_count <= 20
        and tiny_count <= 10
        and boundary_loops <= 20
        and nonmanifold_edges <= 100
        and face_count <= 80_000
    )
    high_resolution_ready = mesh_head_ready and component_count <= 10 and nonmanifold_edges <= 40

    if not mesh_head_ready and normalized_surface_required:
        risk_level = "high"
        recommended_operator = "normalize_control_surface"
    elif high_resolution_ready:
        risk_level = "low"
        recommended_operator = "direct_refine"
    else:
        risk_level = "medium"
        recommended_operator = "normalize_then_refine"

    return MeshPassport(
        mesh_path=str(mesh_path),
        metrics=metrics,
        normalized_surface_required=normalized_surface_required,
        mesh_head_ready=mesh_head_ready,
        high_resolution_ready=high_resolution_ready,
        risk_level=risk_level,
        notes=notes,
        recommended_operator=recommended_operator,
        estimated_preview_seconds=(60, 240),
        estimated_high_resolution_seconds=(300, 1500),
    )


def fast_passport_metrics(mesh_path: str | Path) -> dict[str, Any]:
    """Compute passport metrics without materializing component submeshes.

    `trimesh.split()` can be painfully slow on generator failures with thousands
    of tiny islands. The passport has to be safe to run on exactly those meshes,
    so connected components are counted on sparse face adjacency instead.
    """

    path = Path(mesh_path)
    try:
        mesh = load_mesh(path)
        if not mesh.faces.size:
            raise ValueError("mesh has no faces")

        edge_counts = np.bincount(mesh.edges_unique_inverse, minlength=len(mesh.edges_unique))
        boundary_edges = mesh.edges_unique[edge_counts == 1]
        nonmanifold_edges = mesh.edges_unique[edge_counts != 2]
        component_count, labels = _face_connected_components(mesh)
        face_total = max(int(len(mesh.faces)), 1)
        component_faces = np.bincount(labels, minlength=component_count) if len(labels) else np.array([], dtype=np.int64)
        tiny_components = int(np.sum(component_faces / face_total < 0.01)) if component_faces.size else 0

        return {
            "path": str(path),
            "ok": True,
            "error": None,
            "vertex_count": int(len(mesh.vertices)),
            "face_count": int(len(mesh.faces)),
            "connected_components": int(component_count),
            "tiny_component_count": tiny_components,
            "watertight": bool(np.all(edge_counts == 2)) if len(edge_counts) else False,
            "winding_consistent": bool(mesh.is_winding_consistent),
            "euler_number": int(mesh.euler_number),
            "genus_estimate": None,
            "boundary_edge_count": int(len(boundary_edges)),
            "boundary_loop_count": int(_boundary_loop_count(boundary_edges)),
            "nonmanifold_edge_count": int(len(nonmanifold_edges)),
            "nonmanifold_vertex_count": int(count_nonmanifold_vertices(mesh)),
            "surface_area": float(mesh.area),
            "volume": float(mesh.volume) if bool(np.all(edge_counts == 2)) and len(edge_counts) else None,
        }
    except Exception as exc:  # noqa: BLE001 - passport should report, not crash callers.
        return {
            "path": str(path),
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _face_connected_components(mesh) -> tuple[int, np.ndarray]:
    face_count = int(len(mesh.faces))
    if face_count == 0:
        return 0, np.array([], dtype=np.int64)
    adjacency = np.asarray(mesh.face_adjacency, dtype=np.int64)
    if adjacency.size == 0:
        return face_count, np.arange(face_count, dtype=np.int64)
    rows = np.concatenate([adjacency[:, 0], adjacency[:, 1]])
    cols = np.concatenate([adjacency[:, 1], adjacency[:, 0]])
    graph = coo_matrix((np.ones(len(rows), dtype=np.uint8), (rows, cols)), shape=(face_count, face_count))
    count, labels = connected_components(graph, directed=False, return_labels=True)
    return int(count), labels.astype(np.int64, copy=False)


def _boundary_loop_count(boundary_edges: np.ndarray) -> int:
    if len(boundary_edges) == 0:
        return 0
    adjacency: dict[int, set[int]] = {}
    for a, b in boundary_edges:
        adjacency.setdefault(int(a), set()).add(int(b))
        adjacency.setdefault(int(b), set()).add(int(a))

    seen: set[int] = set()
    loops = 0
    for start in adjacency:
        if start in seen:
            continue
        loops += 1
        stack = [start]
        seen.add(start)
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
    return loops
