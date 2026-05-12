"""Conservative mesh cleanup for generated artist meshes.

This stage is intentionally separate from evaluation: it removes obvious debris
and degenerate geometry while preserving the raw mesh-head output as an asset for
bake-off analysis.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from clearmesh.eval.mesh_quality import load_mesh


@dataclass(frozen=True)
class CleanupOptions:
    min_component_faces: int = 8
    min_component_face_ratio: float = 0.0
    keep_largest_components: int | None = None
    dominant_component_face_ratio: float | None = None
    split_nonmanifold_vertices: bool = False
    fill_holes: bool = False
    fix_normals: bool = True
    merge_vertices: bool = True


@dataclass
class CleanupReport:
    input_path: str
    output_path: str
    input_vertices: int
    input_faces: int
    output_vertices: int
    output_faces: int
    input_components: int
    output_components: int
    removed_components: int
    removed_faces: int
    options: dict[str, Any]
    split_vertices_added: int = 0
    split_vertex_groups: int = 0


def cleanup_mesh_file(input_path: str | Path, output_path: str | Path, options: CleanupOptions | None = None) -> CleanupReport:
    options = options or CleanupOptions()
    mesh = load_mesh(input_path)
    cleaned, report = cleanup_mesh(mesh, options=options)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.export(output_path)
    report.input_path = str(input_path)
    report.output_path = str(output_path)
    return report


def cleanup_mesh(mesh: trimesh.Trimesh, options: CleanupOptions | None = None) -> tuple[trimesh.Trimesh, CleanupReport]:
    options = options or CleanupOptions()
    original = mesh.copy()
    mesh = mesh.copy()
    if len(mesh.faces) == 0:
        raise ValueError("mesh has no faces")

    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    if options.merge_vertices:
        mesh.merge_vertices()

    component_count, labels = _face_connected_components(mesh)
    component_faces = np.bincount(labels, minlength=component_count) if len(labels) else np.array([], dtype=np.int64)
    face_total = max(int(len(mesh.faces)), 1)

    keep_labels: list[int] = []
    dominant_ratio = options.dominant_component_face_ratio
    if dominant_ratio is not None and component_faces.size:
        largest_label = int(np.argmax(component_faces))
        if float(component_faces[largest_label]) / face_total >= float(dominant_ratio):
            keep_labels = [largest_label]

    if not keep_labels:
        for label, face_count in enumerate(component_faces):
            if int(face_count) < int(options.min_component_faces):
                continue
            if float(face_count) / face_total < float(options.min_component_face_ratio):
                continue
            keep_labels.append(int(label))

    if not keep_labels and component_faces.size:
        keep_labels = [int(np.argmax(component_faces))]

    if options.keep_largest_components is not None:
        keep_labels = sorted(keep_labels, key=lambda label: int(component_faces[label]), reverse=True)[
            : max(1, int(options.keep_largest_components))
        ]

    face_mask = np.isin(labels, np.asarray(keep_labels, dtype=np.int64)) if len(labels) else np.ones(len(mesh.faces), dtype=bool)
    if not np.any(face_mask):
        face_mask = np.ones(len(mesh.faces), dtype=bool)

    cleaned = mesh.copy()
    cleaned.update_faces(face_mask)
    cleaned.remove_unreferenced_vertices()
    if options.merge_vertices:
        cleaned.merge_vertices()
    split_vertices_added = 0
    split_vertex_groups = 0
    if options.split_nonmanifold_vertices:
        cleaned, split_report = split_pinched_vertices(cleaned)
        split_vertices_added = split_report["split_vertices_added"]
        split_vertex_groups = split_report["split_vertex_groups"]
    if options.fix_normals:
        cleaned.fix_normals()
    if options.fill_holes:
        cleaned.fill_holes()

    output_component_count, _ = _face_connected_components(cleaned)
    report = CleanupReport(
        input_path="",
        output_path="",
        input_vertices=int(len(original.vertices)),
        input_faces=int(len(original.faces)),
        output_vertices=int(len(cleaned.vertices)),
        output_faces=int(len(cleaned.faces)),
        input_components=int(component_count),
        output_components=int(output_component_count),
        removed_components=int(max(0, component_count - output_component_count)),
        removed_faces=int(max(0, len(original.faces) - len(cleaned.faces))),
        options=asdict(options),
        split_vertices_added=int(split_vertices_added),
        split_vertex_groups=int(split_vertex_groups),
    )
    return cleaned, report


def split_pinched_vertices(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, dict[str, int]]:
    """Duplicate vertices whose incident faces form multiple disconnected fans.

    This is the inverse of an over-aggressive weld. It preserves exact geometry
    while separating bow-tie vertex links into independently editable manifold
    fans, which is safer than smoothing or global remeshing at this stage.
    """

    source = mesh.copy()
    faces = np.asarray(source.faces, dtype=np.int64).copy()
    vertices = np.asarray(source.vertices, dtype=np.float64).copy()
    if len(vertices) == 0 or len(faces) == 0:
        return source, {"split_vertices_added": 0, "split_vertex_groups": 0}

    incident: list[list[tuple[int, int, int]]] = [[] for _ in range(len(vertices))]
    for face_idx, face in enumerate(faces.reshape(-1, 3)):
        a, b, c = (int(value) for value in face)
        incident[a].append((face_idx, b, c))
        incident[b].append((face_idx, c, a))
        incident[c].append((face_idx, a, b))

    vertex_list = [row.copy() for row in vertices]
    split_vertices_added = 0
    split_vertex_groups = 0
    for vertex, entries in enumerate(incident):
        if len(entries) <= 1:
            continue
        components = _vertex_link_face_components(entries)
        if len(components) <= 1:
            continue
        split_vertex_groups += len(components)
        for component in components[1:]:
            new_vertex = len(vertex_list)
            vertex_list.append(vertices[vertex].copy())
            split_vertices_added += 1
            for face_idx in component:
                faces[face_idx, faces[face_idx] == vertex] = new_vertex

    if split_vertices_added == 0:
        return source, {"split_vertices_added": 0, "split_vertex_groups": 0}
    cleaned = trimesh.Trimesh(vertices=np.asarray(vertex_list, dtype=np.float64), faces=faces, process=False)
    cleaned.remove_unreferenced_vertices()
    return cleaned, {
        "split_vertices_added": int(split_vertices_added),
        "split_vertex_groups": int(split_vertex_groups),
    }


def _vertex_link_face_components(entries: list[tuple[int, int, int]]) -> list[list[int]]:
    parent: dict[int, int] = {}

    def find(value: int) -> int:
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for _, left, right in entries:
        union(int(left), int(right))

    groups: dict[int, list[int]] = {}
    for face_idx, left, _ in entries:
        groups.setdefault(find(int(left)), []).append(int(face_idx))
    return [groups[root] for root in sorted(groups, key=lambda item: (len(groups[item]), item), reverse=True)]


def _face_connected_components(mesh: trimesh.Trimesh) -> tuple[int, np.ndarray]:
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


def cleanup_options_from_metadata(metadata: dict[str, Any]) -> CleanupOptions:
    return CleanupOptions(
        min_component_faces=int(metadata.get("cleanup_min_component_faces", 8)),
        min_component_face_ratio=float(metadata.get("cleanup_min_component_face_ratio", 0.0)),
        keep_largest_components=_optional_int(metadata.get("cleanup_keep_largest_components")),
        dominant_component_face_ratio=_optional_float(metadata.get("cleanup_dominant_component_face_ratio")),
        split_nonmanifold_vertices=_bool(metadata.get("cleanup_split_nonmanifold_vertices", True)),
        fill_holes=_bool(metadata.get("cleanup_fill_holes", False)),
        fix_normals=_bool(metadata.get("cleanup_fix_normals", True)),
        merge_vertices=_bool(metadata.get("cleanup_merge_vertices", True)),
    )


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}
