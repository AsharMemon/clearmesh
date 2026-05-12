"""Constrained shrink-wrap projection for ClearMesh control surfaces.

Shrink-wrap is useful when the topology already comes from a clean control mesh.
This module projects that control mesh toward a visual target while smoothing the
motion, so we preserve editability instead of inheriting target triangle soup.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
import trimesh

from clearmesh.eval.mesh_quality import evaluate_mesh, evaluate_mesh_pair, load_mesh


@dataclass(frozen=True)
class ShrinkwrapOptions:
    sample_points: int = 100_000
    iterations: int = 6
    attraction: float = 0.65
    smoothing: float = 0.12
    max_step_ratio: float = 0.08
    seed: int = 0


@dataclass
class ShrinkwrapReport:
    source_path: str
    target_path: str
    output_path: str
    options: dict[str, Any]
    source_metrics: dict[str, Any]
    output_metrics: dict[str, Any]
    pair_metrics: dict[str, Any] | None


def shrinkwrap_file(
    source_path: str | Path,
    target_path: str | Path,
    output_path: str | Path,
    options: ShrinkwrapOptions | None = None,
) -> ShrinkwrapReport:
    options = options or ShrinkwrapOptions()
    source = load_mesh(source_path)
    target = load_mesh(target_path)
    wrapped = shrinkwrap_mesh(source, target, options)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wrapped.export(output_path)
    pair_metrics = None
    try:
        pair_metrics = evaluate_mesh_pair(output_path, target_path, samples=10_000, seed=options.seed)
    except Exception as exc:  # noqa: BLE001 - distance is informative, not required.
        pair_metrics = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return ShrinkwrapReport(
        source_path=str(source_path),
        target_path=str(target_path),
        output_path=str(output_path),
        options=asdict(options),
        source_metrics=evaluate_mesh(source_path),
        output_metrics=evaluate_mesh(output_path),
        pair_metrics=pair_metrics,
    )


def shrinkwrap_obj_vertices_file(
    source_path: str | Path,
    target_path: str | Path,
    output_path: str | Path,
    options: ShrinkwrapOptions | None = None,
) -> ShrinkwrapReport:
    """Project OBJ vertices while preserving polygon face topology.

    Trimesh triangulates OBJ quads on load. For our quad lane, projection must
    move vertices only and keep the remesher's `f ...` arity intact.
    """

    options = options or ShrinkwrapOptions()
    source_path = Path(source_path)
    vertices, faces = _read_obj_vertices_faces(source_path)
    if len(vertices) == 0 or not faces:
        return shrinkwrap_file(source_path, target_path, output_path, options)
    target = load_mesh(target_path)
    projected = _project_vertices(vertices, faces, target, options)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_obj_vertices_faces(output_path, projected, faces)
    pair_metrics = None
    try:
        pair_metrics = evaluate_mesh_pair(output_path, target_path, samples=10_000, seed=options.seed)
    except Exception as exc:  # noqa: BLE001 - distance is informative, not required.
        pair_metrics = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return ShrinkwrapReport(
        source_path=str(source_path),
        target_path=str(target_path),
        output_path=str(output_path),
        options=asdict(options),
        source_metrics=evaluate_mesh(source_path),
        output_metrics=evaluate_mesh(output_path),
        pair_metrics=pair_metrics,
    )


def shrinkwrap_mesh(source: trimesh.Trimesh, target: trimesh.Trimesh, options: ShrinkwrapOptions) -> trimesh.Trimesh:
    if len(source.vertices) == 0 or len(source.faces) == 0:
        raise ValueError("source mesh must have vertices and faces")
    if len(target.vertices) == 0 or len(target.faces) == 0:
        raise ValueError("target mesh must have vertices and faces")

    sample_count = max(1_000, int(options.sample_points))
    points, _ = trimesh.sample.sample_surface(target, sample_count, seed=options.seed)
    tree = cKDTree(points)
    vertices = np.asarray(source.vertices, dtype=float).copy()
    neighbors = _vertex_neighbors(source)
    diag = float(np.linalg.norm(source.bounds[1] - source.bounds[0])) or 1.0
    max_step = max(1e-6, diag * float(options.max_step_ratio))
    attraction = float(np.clip(options.attraction, 0.0, 1.0))
    smoothing = float(np.clip(options.smoothing, 0.0, 1.0))

    for _ in range(max(1, int(options.iterations))):
        _, nearest = tree.query(vertices, k=1)
        desired = points[nearest]
        delta = desired - vertices
        lengths = np.linalg.norm(delta, axis=1)
        scale = np.minimum(1.0, max_step / np.maximum(lengths, 1e-12))
        vertices = vertices + attraction * delta * scale[:, None]
        if smoothing > 0:
            vertices = (1.0 - smoothing) * vertices + smoothing * _neighbor_means(vertices, neighbors)

    wrapped = trimesh.Trimesh(vertices=vertices, faces=np.asarray(source.faces), process=True)
    wrapped.fix_normals()
    return wrapped


def _project_vertices(vertices: np.ndarray, faces: list[list[int]], target: trimesh.Trimesh, options: ShrinkwrapOptions) -> np.ndarray:
    sample_count = max(1_000, int(options.sample_points))
    points, _ = trimesh.sample.sample_surface(target, sample_count, seed=options.seed)
    tree = cKDTree(points)
    projected = np.asarray(vertices, dtype=float).copy()
    neighbors = _polygon_vertex_neighbors(len(projected), faces)
    diag = float(np.linalg.norm(np.ptp(projected, axis=0))) or 1.0
    max_step = max(1e-6, diag * float(options.max_step_ratio))
    attraction = float(np.clip(options.attraction, 0.0, 1.0))
    smoothing = float(np.clip(options.smoothing, 0.0, 1.0))
    for _ in range(max(1, int(options.iterations))):
        _, nearest = tree.query(projected, k=1)
        desired = points[nearest]
        delta = desired - projected
        lengths = np.linalg.norm(delta, axis=1)
        scale = np.minimum(1.0, max_step / np.maximum(lengths, 1e-12))
        projected = projected + attraction * delta * scale[:, None]
        if smoothing > 0:
            projected = (1.0 - smoothing) * projected + smoothing * _neighbor_means(projected, neighbors)
    return projected


def shrinkwrap_options_from_metadata(metadata: dict[str, Any]) -> ShrinkwrapOptions:
    return ShrinkwrapOptions(
        sample_points=int(metadata.get("shrinkwrap_sample_points", 100_000)),
        iterations=int(metadata.get("shrinkwrap_iterations", 6)),
        attraction=float(metadata.get("shrinkwrap_attraction", 0.65)),
        smoothing=float(metadata.get("shrinkwrap_smoothing", 0.12)),
        max_step_ratio=float(metadata.get("shrinkwrap_max_step_ratio", 0.08)),
        seed=int(metadata.get("seed", metadata.get("shrinkwrap_seed", 0))),
    )


def _vertex_neighbors(mesh: trimesh.Trimesh) -> list[np.ndarray]:
    neighbors: list[set[int]] = [set() for _ in range(len(mesh.vertices))]
    for a, b in np.asarray(mesh.edges_unique, dtype=np.int64):
        neighbors[int(a)].add(int(b))
        neighbors[int(b)].add(int(a))
    return [np.fromiter(values, dtype=np.int64) if values else np.array([index], dtype=np.int64) for index, values in enumerate(neighbors)]


def _polygon_vertex_neighbors(vertex_count: int, faces: list[list[int]]) -> list[np.ndarray]:
    neighbors: list[set[int]] = [set() for _ in range(vertex_count)]
    for face in faces:
        for index, vertex in enumerate(face):
            if not 0 <= vertex < vertex_count:
                continue
            prev_vertex = face[index - 1]
            next_vertex = face[(index + 1) % len(face)]
            if 0 <= prev_vertex < vertex_count:
                neighbors[vertex].add(prev_vertex)
            if 0 <= next_vertex < vertex_count:
                neighbors[vertex].add(next_vertex)
    return [np.fromiter(values, dtype=np.int64) if values else np.array([index], dtype=np.int64) for index, values in enumerate(neighbors)]


def _neighbor_means(vertices: np.ndarray, neighbors: list[np.ndarray]) -> np.ndarray:
    means = np.empty_like(vertices)
    for index, item in enumerate(neighbors):
        means[index] = vertices[item].mean(axis=0)
    return means


def _read_obj_vertices_faces(path: Path) -> tuple[np.ndarray, list[list[int]]]:
    vertices: list[tuple[float, float, float]] = []
    faces: list[list[int]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.strip().split()
                if len(parts) >= 4:
                    vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith("f "):
                face: list[int] = []
                for token in line.strip().split()[1:]:
                    raw = token.split("/", 1)[0]
                    if not raw:
                        continue
                    index = int(raw)
                    if index < 0:
                        index = len(vertices) + index + 1
                    face.append(index - 1)
                if len(face) >= 3:
                    faces.append(face)
    return np.asarray(vertices, dtype=float), faces


def _write_obj_vertices_faces(path: Path, vertices: np.ndarray, faces: list[list[int]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# ClearMesh feature projection output\n")
        for vertex in np.asarray(vertices, dtype=float):
            handle.write(f"v {vertex[0]:.9f} {vertex[1]:.9f} {vertex[2]:.9f}\n")
        for face in faces:
            handle.write("f " + " ".join(str(index + 1) for index in face) + "\n")


def report_to_dict(report: ShrinkwrapReport) -> dict[str, Any]:
    return asdict(report)
