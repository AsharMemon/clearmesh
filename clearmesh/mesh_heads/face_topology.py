"""Token-level topology diagnostics for FACE-style triangle sequences.

FACE tokens are quantized triangle coordinates. That means we can inspect many
topology failure modes before decoding to a floating-point mesh: repeated
faces, degenerate triangles, boundary edges, and edges used by too many faces.
These helpers are intentionally deterministic so they can be used in training
logs, evaluation reports, and conservative pre-decode repair.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np


QuantizedVertex = tuple[int, int, int]
QuantizedFace = tuple[QuantizedVertex, QuantizedVertex, QuantizedVertex]
QuantizedEdge = tuple[QuantizedVertex, QuantizedVertex]


@dataclass(frozen=True)
class FaceTokenTopologyReport:
    face_count: int
    unique_vertex_count: int
    unique_edge_count: int
    degenerate_face_count: int
    duplicate_face_count: int
    boundary_edge_count: int
    nonmanifold_edge_count: int
    manifold_edge_count: int
    max_edge_use: int
    edge_pairing_ratio: float
    watertight_edge_graph: bool
    euler_number: int | None
    genus_estimate: float | None

    def to_dict(self) -> dict[str, int | float | bool | None]:
        return asdict(self)


@dataclass(frozen=True)
class FaceTokenRepairReport:
    input_faces: int
    output_faces: int
    dropped_degenerate_faces: int
    dropped_duplicate_faces: int
    dropped_nonmanifold_faces: int
    filled_triangle_holes: int
    input_topology: dict[str, int | float | bool | None]
    output_topology: dict[str, int | float | bool | None]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def face_token_topology_report(tokens: np.ndarray) -> FaceTokenTopologyReport:
    """Return edge-pairing diagnostics for ``(F, 9)`` FACE tokens."""

    faces = _tokens_to_faces(tokens)
    nondegenerate = [face for face in faces if not _is_degenerate_face(face)]
    unique_vertices = {vertex for face in nondegenerate for vertex in face}
    face_keys = [_face_key(face) for face in nondegenerate]
    duplicate_face_count = len(face_keys) - len(set(face_keys))
    edge_counts = _edge_counts(nondegenerate)

    boundary_edge_count = sum(1 for count in edge_counts.values() if count == 1)
    nonmanifold_edge_count = sum(1 for count in edge_counts.values() if count > 2)
    manifold_edge_count = sum(1 for count in edge_counts.values() if count == 2)
    max_edge_use = max(edge_counts.values(), default=0)
    total_edge_uses = sum(edge_counts.values())
    paired_edge_uses = sum(count for count in edge_counts.values() if count == 2)
    edge_pairing_ratio = float(paired_edge_uses / max(1, total_edge_uses))
    watertight = bool(edge_counts) and boundary_edge_count == 0 and nonmanifold_edge_count == 0

    euler_number = None
    genus_estimate = None
    if watertight:
        euler_number = int(len(unique_vertices) - len(edge_counts) + len(nondegenerate))
        components = _face_component_count(nondegenerate)
        genus_estimate = float((2 * max(components, 1) - euler_number) / 2)

    return FaceTokenTopologyReport(
        face_count=int(len(faces)),
        unique_vertex_count=int(len(unique_vertices)),
        unique_edge_count=int(len(edge_counts)),
        degenerate_face_count=int(len(faces) - len(nondegenerate)),
        duplicate_face_count=int(duplicate_face_count),
        boundary_edge_count=int(boundary_edge_count),
        nonmanifold_edge_count=int(nonmanifold_edge_count),
        manifold_edge_count=int(manifold_edge_count),
        max_edge_use=int(max_edge_use),
        edge_pairing_ratio=edge_pairing_ratio,
        watertight_edge_graph=watertight,
        euler_number=euler_number,
        genus_estimate=genus_estimate,
    )


def repair_face_tokens(tokens: np.ndarray, mode: str = "dedupe") -> tuple[np.ndarray, FaceTokenRepairReport]:
    """Conservatively repair FACE tokens before mesh decoding.

    Modes:
    - ``none``: return the input unchanged, with diagnostics.
    - ``dedupe``: drop degenerate and duplicate faces, then fill triangle holes.
    - ``manifold``: additionally skip faces that would make an edge nonmanifold.
    """

    mode = str(mode).lower()
    if mode not in {"none", "dedupe", "manifold"}:
        raise ValueError(f"unsupported FACE token repair mode: {mode}")

    input_tokens = np.asarray(tokens, dtype=np.int64)
    input_topology = face_token_topology_report(input_tokens)
    if mode == "none":
        return input_tokens.copy(), FaceTokenRepairReport(
            input_faces=int(len(input_tokens)),
            output_faces=int(len(input_tokens)),
            dropped_degenerate_faces=0,
            dropped_duplicate_faces=0,
            dropped_nonmanifold_faces=0,
            filled_triangle_holes=0,
            input_topology=input_topology.to_dict(),
            output_topology=input_topology.to_dict(),
        )

    faces = _tokens_to_faces(input_tokens)
    kept: list[QuantizedFace] = []
    seen_faces: set[tuple[QuantizedVertex, QuantizedVertex, QuantizedVertex]] = set()
    edge_counts: Counter[QuantizedEdge] = Counter()
    dropped_degenerate = 0
    dropped_duplicate = 0
    dropped_nonmanifold = 0

    for face in faces:
        if _is_degenerate_face(face):
            dropped_degenerate += 1
            continue
        key = _face_key(face)
        if key in seen_faces:
            dropped_duplicate += 1
            continue
        edges = _face_edges(face)
        if mode == "manifold" and any(edge_counts[edge] >= 2 for edge in edges):
            dropped_nonmanifold += 1
            continue
        kept.append(face)
        seen_faces.add(key)
        for edge in edges:
            edge_counts[edge] += 1

    filled = _fill_triangle_holes(kept, seen_faces, edge_counts)
    repaired = _faces_to_tokens(kept)
    output_topology = face_token_topology_report(repaired)
    report = FaceTokenRepairReport(
        input_faces=int(len(input_tokens)),
        output_faces=int(len(repaired)),
        dropped_degenerate_faces=int(dropped_degenerate),
        dropped_duplicate_faces=int(dropped_duplicate),
        dropped_nonmanifold_faces=int(dropped_nonmanifold),
        filled_triangle_holes=int(filled),
        input_topology=input_topology.to_dict(),
        output_topology=output_topology.to_dict(),
    )
    return repaired, report


def topology_coordinate_weights(
    tokens: np.ndarray,
    *,
    reuse_vertex_weight: float = 0.0,
    edge_closure_weight: float = 0.0,
) -> np.ndarray:
    """Return per-coordinate training weights for topology-critical tokens.

    The tiny FACE decoder predicts independent coordinate-bin logits. Exact
    vertex reuse matters: a one-bin miss can turn a closed edge into a boundary
    edge. This helper lets teacher-forced training emphasize coordinates that
    should reuse prior vertices or close prior boundary edges.
    """

    arr = np.asarray(tokens, dtype=np.int64)
    faces = _tokens_to_faces(arr)
    weights = np.ones(arr.shape, dtype=np.float32)
    reuse_vertex_weight = float(reuse_vertex_weight)
    edge_closure_weight = float(edge_closure_weight)
    if reuse_vertex_weight == 0.0 and edge_closure_weight == 0.0:
        return weights

    seen_vertices: set[QuantizedVertex] = set()
    edge_counts: Counter[QuantizedEdge] = Counter()
    for face_index, face in enumerate(faces):
        if _is_degenerate_face(face):
            continue
        for vertex_index, vertex in enumerate(face):
            start = vertex_index * 3
            end = start + 3
            if reuse_vertex_weight and vertex in seen_vertices:
                weights[face_index, start:end] += reuse_vertex_weight

        for edge in _face_edges(face):
            if not edge_closure_weight or edge_counts[edge] != 1:
                continue
            for vertex in edge:
                for vertex_index, face_vertex in enumerate(face):
                    if face_vertex == vertex:
                        start = vertex_index * 3
                        weights[face_index, start : start + 3] += edge_closure_weight

        seen_vertices.update(face)
        edge_counts.update(_face_edges(face))
    return weights


def topology_event_labels(tokens: np.ndarray) -> dict[str, np.ndarray]:
    """Return teacher-forcing labels for explicit topology auxiliary heads."""

    faces = _tokens_to_faces(tokens)
    reuse_vertex = np.zeros((len(faces), 3), dtype=np.int64)
    edge_closure_count = np.zeros((len(faces),), dtype=np.int64)
    seen_vertices: set[QuantizedVertex] = set()
    edge_counts: Counter[QuantizedEdge] = Counter()
    for face_index, face in enumerate(faces):
        if _is_degenerate_face(face):
            continue
        for vertex_index, vertex in enumerate(face):
            reuse_vertex[face_index, vertex_index] = int(vertex in seen_vertices)
        closures = sum(1 for edge in _face_edges(face) if edge_counts[edge] == 1)
        edge_closure_count[face_index] = int(min(3, closures))
        seen_vertices.update(face)
        edge_counts.update(_face_edges(face))
    return {
        "reuse_vertex": reuse_vertex,
        "edge_closure_count": edge_closure_count,
    }


def _tokens_to_faces(tokens: np.ndarray) -> list[QuantizedFace]:
    arr = np.asarray(tokens, dtype=np.int64)
    if arr.ndim != 2 or arr.shape[1] != 9:
        raise ValueError(f"tokens must have shape (F, 9), got {arr.shape}")
    q_faces = arr.reshape(-1, 3, 3)
    return [
        (
            _vertex_tuple(face[0]),
            _vertex_tuple(face[1]),
            _vertex_tuple(face[2]),
        )
        for face in q_faces
    ]


def _faces_to_tokens(faces: Iterable[QuantizedFace]) -> np.ndarray:
    rows: list[list[int]] = []
    for face in faces:
        rows.append([coord for vertex in face for coord in vertex])
    if not rows:
        return np.zeros((0, 9), dtype=np.int64)
    return np.asarray(rows, dtype=np.int64)


def _vertex_tuple(vertex: np.ndarray) -> QuantizedVertex:
    return tuple(int(value) for value in vertex.tolist())  # type: ignore[return-value]


def _is_degenerate_face(face: QuantizedFace) -> bool:
    return len(set(face)) != 3


def _face_key(face: QuantizedFace) -> tuple[QuantizedVertex, QuantizedVertex, QuantizedVertex]:
    return tuple(sorted(face))  # type: ignore[return-value]


def _edge_key(a: QuantizedVertex, b: QuantizedVertex) -> QuantizedEdge:
    return (a, b) if a <= b else (b, a)


def _face_edges(face: QuantizedFace) -> tuple[QuantizedEdge, QuantizedEdge, QuantizedEdge]:
    return (
        _edge_key(face[0], face[1]),
        _edge_key(face[1], face[2]),
        _edge_key(face[2], face[0]),
    )


def _edge_counts(faces: Iterable[QuantizedFace]) -> Counter[QuantizedEdge]:
    counts: Counter[QuantizedEdge] = Counter()
    for face in faces:
        if _is_degenerate_face(face):
            continue
        counts.update(_face_edges(face))
    return counts


def _face_component_count(faces: list[QuantizedFace]) -> int:
    if not faces:
        return 0
    vertex_to_faces: dict[QuantizedVertex, list[int]] = {}
    for index, face in enumerate(faces):
        for vertex in face:
            vertex_to_faces.setdefault(vertex, []).append(index)

    seen: set[int] = set()
    components = 0
    for start in range(len(faces)):
        if start in seen:
            continue
        components += 1
        stack = [start]
        seen.add(start)
        while stack:
            current = stack.pop()
            for vertex in faces[current]:
                for neighbor in vertex_to_faces.get(vertex, []):
                    if neighbor not in seen:
                        seen.add(neighbor)
                        stack.append(neighbor)
    return components


def _fill_triangle_holes(
    faces: list[QuantizedFace],
    seen_faces: set[tuple[QuantizedVertex, QuantizedVertex, QuantizedVertex]],
    edge_counts: Counter[QuantizedEdge],
) -> int:
    """Fill only obvious holes bounded by exactly three boundary edges."""

    boundary_edges = [edge for edge, count in edge_counts.items() if count == 1]
    adjacency: dict[QuantizedVertex, set[QuantizedVertex]] = {}
    for a, b in boundary_edges:
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)

    candidates: set[tuple[QuantizedVertex, QuantizedVertex, QuantizedVertex]] = set()
    for a, neighbors in adjacency.items():
        ordered = sorted(neighbors)
        for idx, b in enumerate(ordered):
            for c in ordered[idx + 1 :]:
                if _edge_key(b, c) in edge_counts and edge_counts[_edge_key(b, c)] == 1:
                    candidates.add(tuple(sorted((a, b, c))))  # type: ignore[arg-type]

    filled = 0
    for key in sorted(candidates):
        if key in seen_faces:
            continue
        face = (key[0], key[1], key[2])
        edges = _face_edges(face)
        if any(edge_counts[edge] >= 2 for edge in edges):
            continue
        faces.append(face)
        seen_faces.add(key)
        for edge in edges:
            edge_counts[edge] += 1
        filled += 1
    return filled
