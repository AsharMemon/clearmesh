"""Topology-indexed FACE-lite representation.

The original FACE-lite scaffold predicts each triangle as nine coordinate bins.
That is compact, but graph closure is only implicit: two vertices are shared only
when all three coordinate bins match exactly. This module adds a stricter v2
contract for research smokes:

- canonical quantized vertex table,
- triangle faces as indices into that table,
- deterministic decode through explicit vertex reuse,
- topology diagnostics inherited from the coordinate-token path.

This is not the final paper-scale model. It is the representation-level fix we
need before scaling: the decoder can be trained to emit mesh connectivity rather
than rediscovering vertex welding from raw XYZ bins.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from itertools import permutations, product
from typing import Any

import numpy as np
import trimesh

from .face_tokens import (
    FaceTokenTransform,
    dequantize_normalized_points,
    fit_face_token_transform,
    quantize_normalized_points,
)


@dataclass(frozen=True)
class FaceIndexedSequence:
    """Canonical vertex table plus triangle index tokens."""

    vertices: np.ndarray
    faces: np.ndarray
    num_bins: int
    transform: FaceTokenTransform

    @property
    def vertex_count(self) -> int:
        return int(np.asarray(self.vertices).shape[0])

    @property
    def face_count(self) -> int:
        return int(np.asarray(self.faces).shape[0])

    @property
    def coordinate_token_count(self) -> int:
        return int(np.asarray(self.vertices).size)

    @property
    def index_token_count(self) -> int:
        return int(np.asarray(self.faces).size)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "vertices": np.asarray(self.vertices, dtype=np.int64).tolist(),
            "faces": np.asarray(self.faces, dtype=np.int64).tolist(),
            "num_bins": int(self.num_bins),
            "transform": {
                "center": list(self.transform.center),
                "scale": float(self.transform.scale),
            },
        }


@dataclass(frozen=True)
class IndexedBoundaryFillReport:
    input_faces: int
    output_faces: int
    input_boundary_edges: int
    output_boundary_edges: int
    output_nonmanifold_edges: int
    filled_loops: int
    filled_faces: int
    skipped_loops: int

    def to_dict(self) -> dict[str, int]:
        return {
            "input_faces": int(self.input_faces),
            "output_faces": int(self.output_faces),
            "input_boundary_edges": int(self.input_boundary_edges),
            "output_boundary_edges": int(self.output_boundary_edges),
            "output_nonmanifold_edges": int(self.output_nonmanifold_edges),
            "filled_loops": int(self.filled_loops),
            "filled_faces": int(self.filled_faces),
            "skipped_loops": int(self.skipped_loops),
        }


IndexedEdge = tuple[int, int]
IndexedFaceKey = tuple[int, int, int]


@dataclass
class IndexedDecodeState:
    """Mutable edge state used by constrained indexed decoding."""

    edge_counts: Counter[IndexedEdge]
    seen_faces: set[IndexedFaceKey]
    accepted_faces: int = 0
    vertex_links: dict[int, list[tuple[int, int]]] = field(default_factory=dict)

    @classmethod
    def empty(cls) -> "IndexedDecodeState":
        return cls(edge_counts=Counter(), seen_faces=set(), accepted_faces=0)

    @classmethod
    def from_faces(cls, faces: np.ndarray) -> "IndexedDecodeState":
        state = cls.empty()
        for face in np.asarray(faces, dtype=np.int64).reshape(-1, 3):
            state.add_face(face)
        return state

    @property
    def boundary_edge_count(self) -> int:
        return int(sum(1 for count in self.edge_counts.values() if count == 1))

    @property
    def boundary_edges(self) -> list[IndexedEdge]:
        return sorted(edge for edge, count in self.edge_counts.items() if count == 1)

    def add_face(self, face: np.ndarray | list[int] | tuple[int, int, int]) -> None:
        values = tuple(int(value) for value in face)
        if len(set(values)) != 3:
            return
        self.seen_faces.add(_indexed_face_key(values))
        for edge in _indexed_face_edges(values):
            self.edge_counts[edge] += 1
        for vertex, left, right in (
            (values[0], values[1], values[2]),
            (values[1], values[2], values[0]),
            (values[2], values[0], values[1]),
        ):
            self.vertex_links.setdefault(int(vertex), []).append((int(left), int(right)))
        self.accepted_faces += 1


def encode_mesh_to_indexed_face_tokens(
    mesh: trimesh.Trimesh,
    num_bins: int = 128,
    normalize: bool = True,
    max_faces: int | None = None,
    max_vertices: int | None = None,
    padding: float = 1.0,
    face_order: str = "lex",
) -> FaceIndexedSequence:
    """Encode a triangular mesh into explicit vertex-table/face-index tokens."""

    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"mesh must be a trimesh.Trimesh, got {type(mesh)!r}")
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError("mesh must contain vertices and triangular faces")
    if max_faces is not None and len(mesh.faces) > int(max_faces):
        raise ValueError(f"mesh has {len(mesh.faces)} faces, above max_faces={max_faces}")

    transform = fit_face_token_transform(mesh.vertices, padding=padding) if normalize else FaceTokenTransform(
        center=(0.0, 0.0, 0.0),
        scale=1.0,
    )
    vertices = transform.normalize(mesh.vertices) if normalize else np.asarray(mesh.vertices, dtype=np.float64)
    q_vertices = quantize_normalized_points(vertices, num_bins=num_bins)
    unique_q, inverse = np.unique(q_vertices, axis=0, return_inverse=True)
    indexed_faces = inverse[np.asarray(mesh.faces, dtype=np.int64)]
    indexed_faces = _rotate_faces_to_min_vertex(indexed_faces)
    indexed_faces = _drop_degenerate_faces(indexed_faces)
    if len(indexed_faces) == 0:
        raise ValueError("all faces became degenerate after vertex-table quantization")

    # Re-sort vertex table in ZYX order, then remap faces into that canonical table.
    order = np.lexsort((unique_q[:, 0], unique_q[:, 1], unique_q[:, 2]))
    remap = np.empty_like(order)
    remap[order] = np.arange(len(order), dtype=np.int64)
    canonical_vertices = unique_q[order].astype(np.int64)
    canonical_faces = remap[indexed_faces]
    canonical_faces = _rotate_faces_to_min_vertex(canonical_faces)
    canonical_faces = _drop_duplicate_faces(canonical_faces)
    lex_order = np.lexsort((canonical_faces[:, 2], canonical_faces[:, 1], canonical_faces[:, 0]))
    canonical_faces = canonical_faces[lex_order]
    if face_order not in {"lex", "boundary_growth"}:
        raise ValueError(f"unknown indexed face_order={face_order!r}")
    if face_order == "boundary_growth":
        canonical_faces = order_indexed_faces_boundary_growth(canonical_faces)

    if max_vertices is not None and len(canonical_vertices) > int(max_vertices):
        raise ValueError(f"mesh has {len(canonical_vertices)} quantized vertices, above max_vertices={max_vertices}")

    return FaceIndexedSequence(
        vertices=canonical_vertices.astype(np.int64),
        faces=canonical_faces.astype(np.int64),
        num_bins=int(num_bins),
        transform=transform,
    )


def decode_indexed_face_tokens_to_mesh(
    sequence: FaceIndexedSequence,
    denormalize: bool = True,
    process: bool = False,
) -> trimesh.Trimesh:
    """Decode explicit vertex-table/face-index tokens into a mesh."""

    q_vertices = np.asarray(sequence.vertices, dtype=np.int64)
    faces = np.asarray(sequence.faces, dtype=np.int64)
    if q_vertices.ndim != 2 or q_vertices.shape[1] != 3:
        raise ValueError(f"vertices must have shape (V, 3), got {q_vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must have shape (F, 3), got {faces.shape}")
    if np.any(faces < 0) or np.any(faces >= len(q_vertices)):
        raise ValueError("face indices must reference the vertex table")
    vertices = dequantize_normalized_points(q_vertices, num_bins=sequence.num_bins)
    if denormalize:
        vertices = sequence.transform.denormalize(vertices)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=process)
    if len(mesh.faces):
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.remove_unreferenced_vertices()
    return mesh


def indexed_to_coordinate_tokens(sequence: FaceIndexedSequence) -> np.ndarray:
    """Expand indexed tokens back to FACE-style ``(F, 9)`` coordinate tokens."""

    vertices = np.asarray(sequence.vertices, dtype=np.int64)
    faces = np.asarray(sequence.faces, dtype=np.int64)
    if len(faces) == 0:
        return np.zeros((0, 9), dtype=np.int64)
    return vertices[faces].reshape(-1, 9).astype(np.int64)


def drop_geometric_degenerate_indexed_faces(sequence: FaceIndexedSequence) -> tuple[FaceIndexedSequence, int]:
    """Drop indexed faces that collapse to zero area in quantized space.

    ``decode_indexed_face_tokens_to_mesh`` removes geometrically degenerate
    triangles before export. If we wait until that late stage, a token graph can
    look watertight and then reopen after decode. Dropping those faces before
    boundary filling makes the topology state match the actual exported mesh.
    """

    vertices = np.asarray(sequence.vertices, dtype=np.int64)
    faces = np.asarray(sequence.faces, dtype=np.int64).reshape(-1, 3)
    if len(faces) == 0:
        return sequence, 0
    keep = np.asarray(
        [_quantized_face_area2(vertices, tuple(int(value) for value in face)) > 0.0 for face in faces],
        dtype=bool,
    )
    dropped = int(np.sum(~keep))
    if dropped == 0:
        return sequence, 0
    return (
        FaceIndexedSequence(
            vertices=vertices,
            faces=faces[keep].astype(np.int64),
            num_bins=sequence.num_bins,
            transform=sequence.transform,
        ),
        dropped,
    )


def coordinate_tokens_to_indexed(
    tokens: np.ndarray,
    *,
    num_bins: int,
    transform: FaceTokenTransform,
    max_vertices: int | None = None,
) -> FaceIndexedSequence:
    """Build an explicit vertex table from existing coordinate FACE tokens."""

    arr = np.asarray(tokens, dtype=np.int64)
    if arr.ndim != 2 or arr.shape[1] != 9:
        raise ValueError(f"tokens must have shape (F, 9), got {arr.shape}")
    flat = arr.reshape(-1, 3)
    vertices, inverse = np.unique(flat, axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3)
    faces = _drop_degenerate_faces(_rotate_faces_to_min_vertex(faces))
    faces = _drop_duplicate_faces(faces)
    order = np.lexsort((vertices[:, 0], vertices[:, 1], vertices[:, 2]))
    remap = np.empty_like(order)
    remap[order] = np.arange(len(order), dtype=np.int64)
    vertices = vertices[order]
    faces = remap[faces]
    faces = _rotate_faces_to_min_vertex(faces)
    if len(faces):
        face_order = np.lexsort((faces[:, 2], faces[:, 1], faces[:, 0]))
        faces = faces[face_order]
    if max_vertices is not None and len(vertices) > int(max_vertices):
        raise ValueError(f"token sequence has {len(vertices)} vertices, above max_vertices={max_vertices}")
    return FaceIndexedSequence(vertices=vertices, faces=faces, num_bins=int(num_bins), transform=transform)


def indexed_face_stats(sequence: FaceIndexedSequence) -> dict[str, int | float]:
    """Return compact representation diagnostics."""

    faces = np.asarray(sequence.faces, dtype=np.int64)
    vertices = np.asarray(sequence.vertices, dtype=np.int64)
    return {
        "faces": int(len(faces)),
        "vertices": int(len(vertices)),
        "coordinate_tokens": int(vertices.size),
        "index_tokens": int(faces.size),
        "num_bins": int(sequence.num_bins),
        "compression_vs_xyz_face_tokens": float((vertices.size + faces.size) / max(1, len(faces) * 9)),
    }


def indexed_face_closure_counts(faces: np.ndarray) -> np.ndarray:
    """Return edge-closure counts induced by a face sequence.

    Each value is the number of currently open boundary edges consumed by the
    next face. A good shelling order for a connected watertight component has
    exactly one zero at the component seed; extra zeros mean the autoregressive
    sequence jumps to a disconnected frontier.
    """

    edge_counts: Counter[IndexedEdge] = Counter()
    labels: list[int] = []
    for face_arr in np.asarray(faces, dtype=np.int64).reshape(-1, 3):
        face = tuple(int(value) for value in face_arr)
        edges = _indexed_face_edges(face)
        labels.append(int(sum(1 for edge in edges if edge_counts[edge] == 1)))
        for edge in edges:
            edge_counts[edge] += 1
    return np.asarray(labels, dtype=np.int64)


def fill_indexed_boundary_loops(
    sequence: FaceIndexedSequence,
    *,
    max_loop_edges: int = 128,
    strategy: str = "fan",
) -> tuple[FaceIndexedSequence, IndexedBoundaryFillReport]:
    """Fill simple boundary loops with triangles.

    ``strategy="fan"`` preserves the original behavior: triangulate each loop
    from an existing boundary vertex. ``strategy="centroid"`` adds one quantized
    center vertex per loop, then caps each boundary edge against that center.
    The centroid cap is more robust for skinny holes because it avoids long
    diagonals that can collapse into zero-area triangles during mesh decode.
    """

    if strategy not in {"fan", "centroid"}:
        raise ValueError(f"unknown indexed boundary fill strategy={strategy!r}")

    vertices = np.asarray(sequence.vertices, dtype=np.int64)
    faces = np.asarray(sequence.faces, dtype=np.int64).reshape(-1, 3)
    edge_counts = _indexed_edge_counts(faces)
    input_boundary = [edge for edge, count in edge_counts.items() if count == 1]
    if not input_boundary:
        report = IndexedBoundaryFillReport(
            input_faces=int(len(faces)),
            output_faces=int(len(faces)),
            input_boundary_edges=0,
            output_boundary_edges=0,
            output_nonmanifold_edges=sum(1 for count in edge_counts.values() if count > 2),
            filled_loops=0,
            filled_faces=0,
            skipped_loops=0,
        )
        return sequence, report

    seen_faces = {_indexed_face_key(tuple(int(v) for v in face)) for face in faces}
    filled: list[tuple[int, int, int]] = []
    filled_loops = 0
    skipped_loops = 0
    for loop in _boundary_loops(input_boundary):
        if len(loop) < 3 or len(loop) > int(max_loop_edges):
            skipped_loops += 1
            continue
        loop_faces: list[tuple[int, int, int]]
        next_vertices = vertices
        if strategy == "centroid":
            center = _choose_boundary_loop_center(vertices, loop, sequence.num_bins)
            if center is None:
                skipped_loops += 1
                continue
            center_index = int(len(next_vertices))
            loop_faces = [
                (int(loop[idx]), int(loop[(idx + 1) % len(loop)]), center_index)
                for idx in range(len(loop))
            ]
        else:
            anchor = int(loop[0])
            loop_faces = [
                (anchor, int(loop[idx]), int(loop[idx + 1]))
                for idx in range(1, len(loop) - 1)
            ]
        loop_ok = True
        for face in loop_faces:
            if len(set(face)) != 3:
                loop_ok = False
                break
            key = _indexed_face_key(face)
            if key in seen_faces:
                loop_ok = False
                break
            if _quantized_face_area2(next_vertices, face, extra_vertex=center if strategy == "centroid" else None) <= 0:
                loop_ok = False
                break
        if not loop_ok:
            skipped_loops += 1
            continue
        if strategy == "centroid":
            vertices = np.concatenate([vertices, np.asarray([center], dtype=np.int64)], axis=0)
        trial_counts = edge_counts.copy()
        for face in loop_faces:
            for edge in _indexed_face_edges(face):
                trial_counts[edge] += 1
        if any(count > 2 for count in trial_counts.values()):
            skipped_loops += 1
            continue
        for face in loop_faces:
            seen_faces.add(_indexed_face_key(face))
            filled.append(face)
            for edge in _indexed_face_edges(face):
                edge_counts[edge] += 1
        filled_loops += 1

    if filled:
        output_faces = np.concatenate([faces, np.asarray(filled, dtype=np.int64)], axis=0)
    else:
        output_faces = faces.copy()
    output_counts = _indexed_edge_counts(output_faces)
    report = IndexedBoundaryFillReport(
        input_faces=int(len(faces)),
        output_faces=int(len(output_faces)),
        input_boundary_edges=int(len(input_boundary)),
        output_boundary_edges=int(sum(1 for count in output_counts.values() if count == 1)),
        output_nonmanifold_edges=int(sum(1 for count in output_counts.values() if count > 2)),
        filled_loops=int(filled_loops),
        filled_faces=int(len(filled)),
        skipped_loops=int(skipped_loops),
    )
    return (
        FaceIndexedSequence(
            vertices=vertices,
            faces=output_faces.astype(np.int64),
            num_bins=sequence.num_bins,
            transform=sequence.transform,
        ),
        report,
    )


def order_indexed_faces_boundary_growth(faces: np.ndarray) -> np.ndarray:
    """Order faces so autoregressive decoding grows from open boundaries.

    Lexicographic FACE ordering is deterministic, but it can jump around the
    surface. That is hostile to constrained decoding because inference is then
    forced to close boundary edges while training targets sometimes start a new
    frontier. This greedy shelling keeps determinism while preferring faces that
    consume the current boundary.
    """

    arr = np.asarray(faces, dtype=np.int64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"faces must have shape (F, 3), got {arr.shape}")
    if len(arr) <= 1:
        return arr.copy()

    remaining: set[int] = set(range(len(arr)))
    ordered: list[int] = []
    state = IndexedDecodeState.empty()
    lex_keys = [tuple(int(value) for value in row) for row in arr]
    lex_tiebreakers = [tuple(-value for value in key) for key in lex_keys]
    face_edges = [_indexed_face_edges(tuple(int(value) for value in row)) for row in arr]

    while remaining:
        best_idx: int | None = None
        best_rank: tuple[int, int, tuple[int, int, int]] | None = None
        for idx in remaining:
            edges = face_edges[idx]
            closures = sum(1 for edge in edges if state.edge_counts[edge] == 1)
            nonmanifold_hits = sum(1 for edge in edges if state.edge_counts[edge] >= 2)
            if state.accepted_faces and closures == 0:
                # Only start a new component/frontier if no boundary candidate exists.
                rank = (-1, -nonmanifold_hits, lex_tiebreakers[idx])
            else:
                rank = (closures, -nonmanifold_hits, lex_tiebreakers[idx])
            if best_rank is None or rank > best_rank:
                best_rank = rank
                best_idx = idx

        if best_idx is None:  # pragma: no cover - defensive against impossible set state.
            break

        # If the provisional winner does not touch the boundary, first check
        # whether any remaining face does. This keeps disconnected components
        # supported without prematurely fragmenting connected meshes.
        if state.accepted_faces and best_rank is not None and best_rank[0] <= 0:
            boundary_best: int | None = None
            boundary_rank: tuple[int, int, tuple[int, int, int]] | None = None
            for idx in remaining:
                edges = face_edges[idx]
                closures = sum(1 for edge in edges if state.edge_counts[edge] == 1)
                if closures <= 0:
                    continue
                nonmanifold_hits = sum(1 for edge in edges if state.edge_counts[edge] >= 2)
                rank = (closures, -nonmanifold_hits, lex_tiebreakers[idx])
                if boundary_rank is None or rank > boundary_rank:
                    boundary_rank = rank
                    boundary_best = idx
            if boundary_best is not None:
                best_idx = boundary_best

        remaining.remove(best_idx)
        ordered.append(best_idx)
        state.add_face(arr[best_idx])

    return arr[np.asarray(ordered, dtype=np.int64)]


def select_constrained_indexed_face(
    logits: np.ndarray,
    state: IndexedDecodeState,
    *,
    vertex_count: int,
    vertices: np.ndarray | None = None,
    top_k: int = 12,
    closure_bonus: float = 2.0,
    new_edge_penalty: float = 0.15,
    edge_length_penalty: float = 0.0,
    aspect_penalty: float = 0.0,
    nonmanifold_penalty: float = 1000.0,
    require_boundary_closure_after: int = 0,
    closure_target_scores: np.ndarray | None = None,
    closure_target_bonus: float = 0.0,
    enforce_vertex_link_manifold: bool = False,
) -> np.ndarray:
    """Select the next face while respecting current edge-use state.

    ``logits`` has shape ``(3, V)``. The model still proposes independent corner
    distributions, but this selector scores candidate triples as faces: it skips
    degenerate/duplicate triangles, avoids edges already used by two faces, and
    rewards candidates that close existing boundary edges.
    """

    scores = np.asarray(logits, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[0] != 3:
        raise ValueError(f"logits must have shape (3, V), got {scores.shape}")
    vertex_count = max(0, min(int(vertex_count), int(scores.shape[1])))
    if vertex_count < 3:
        raise ValueError(f"vertex_count must be at least 3, got {vertex_count}")
    top_k = max(3, min(int(top_k), vertex_count))
    top_by_corner = [np.argsort(scores[corner, :vertex_count])[-top_k:][::-1].tolist() for corner in range(3)]

    strict = _best_indexed_candidate(
        scores,
        state,
        top_by_corner,
        closure_bonus=closure_bonus,
        new_edge_penalty=new_edge_penalty,
        edge_length_penalty=edge_length_penalty,
        aspect_penalty=aspect_penalty,
        nonmanifold_penalty=nonmanifold_penalty,
        require_boundary_closure_after=require_boundary_closure_after,
        closure_target_scores=closure_target_scores,
        closure_target_bonus=closure_target_bonus,
        enforce_vertex_link_manifold=enforce_vertex_link_manifold,
        vertices=vertices,
        strict_manifold=True,
    )
    if strict is not None:
        return strict

    relaxed = _best_indexed_candidate(
        scores,
        state,
        top_by_corner,
        closure_bonus=closure_bonus,
        new_edge_penalty=new_edge_penalty,
        edge_length_penalty=edge_length_penalty,
        aspect_penalty=aspect_penalty,
        nonmanifold_penalty=nonmanifold_penalty,
        require_boundary_closure_after=0,
        closure_target_scores=closure_target_scores,
        closure_target_bonus=closure_target_bonus,
        enforce_vertex_link_manifold=enforce_vertex_link_manifold,
        vertices=vertices,
        strict_manifold=False,
    )
    if relaxed is not None:
        return relaxed

    # Last-ditch fallback: pick the best nondegenerate corner-wise proposal.
    for candidate in product(*top_by_corner):
        face = tuple(int(value) for value in candidate)
        if len(set(face)) == 3 and (
            not enforce_vertex_link_manifold or _candidate_preserves_vertex_links(state, face)
        ):
            return np.asarray(candidate, dtype=np.int64)
    return np.asarray([0, 1, 2], dtype=np.int64)


def select_boundary_edge_action_face(
    logits: np.ndarray,
    state: IndexedDecodeState,
    *,
    vertex_count: int,
    vertices: np.ndarray | None = None,
    top_k: int = 24,
    boundary_edge_top_k: int | None = None,
    local_candidate_neighbors: int = 0,
    closure_bonus: float = 2.0,
    new_edge_penalty: float = 0.15,
    edge_length_penalty: float = 0.0,
    aspect_penalty: float = 0.0,
    nonmanifold_penalty: float = 1000.0,
    require_boundary_closure_after: int = 0,
    closure_target_scores: np.ndarray | None = None,
    closure_target_bonus: float = 0.0,
    enforce_vertex_link_manifold: bool = False,
) -> np.ndarray:
    """Select the next face as a boundary-edge completion action.

    FACE predicts face tokens, but production topology wants a stronger action
    space than three independent vertex picks. This selector keeps the model's
    per-corner logits, then searches faces that include one already-open
    boundary edge plus one high-probability third vertex. If no valid boundary
    action exists, it falls back to the standard constrained triple selector.
    """

    scores = np.asarray(logits, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[0] != 3:
        raise ValueError(f"logits must have shape (3, V), got {scores.shape}")
    vertex_count = max(0, min(int(vertex_count), int(scores.shape[1])))
    if vertex_count < 3:
        raise ValueError(f"vertex_count must be at least 3, got {vertex_count}")

    boundary_edges = state.boundary_edges
    should_force_boundary = bool(
        boundary_edges
        and (not require_boundary_closure_after or state.accepted_faces >= int(require_boundary_closure_after))
    )
    if not should_force_boundary:
        return select_constrained_indexed_face(
            scores,
            state,
            vertex_count=vertex_count,
            top_k=top_k,
            closure_bonus=closure_bonus,
            new_edge_penalty=new_edge_penalty,
            edge_length_penalty=edge_length_penalty,
            aspect_penalty=aspect_penalty,
            nonmanifold_penalty=nonmanifold_penalty,
            require_boundary_closure_after=require_boundary_closure_after,
            closure_target_scores=closure_target_scores,
            closure_target_bonus=closure_target_bonus,
            enforce_vertex_link_manifold=enforce_vertex_link_manifold,
            vertices=vertices,
        )

    top_k = max(3, min(int(top_k), vertex_count))
    top_vertices = np.argsort(scores[:, :vertex_count].max(axis=0))[-top_k:][::-1].tolist()
    edge_limit = len(boundary_edges) if boundary_edge_top_k is None else max(1, int(boundary_edge_top_k))
    edge_limit = min(edge_limit, len(boundary_edges))
    ranked_edges = sorted(
        boundary_edges,
        key=lambda edge: _boundary_edge_logit_score(scores, edge),
        reverse=True,
    )[:edge_limit]

    best_face: tuple[int, int, int] | None = None
    best_score = float("-inf")
    for edge in ranked_edges:
        edge_vertices = set(edge)
        third_candidates = list(top_vertices)
        third_candidates.extend(
            _edge_local_candidate_vertices(vertices, edge, vertex_count=vertex_count, count=local_candidate_neighbors)
        )
        for third in _dedupe_ints(third_candidates):
            third = int(third)
            if third in edge_vertices:
                continue
            for face in permutations((int(edge[0]), int(edge[1]), third), 3):
                model_score = float(scores[0, face[0]] + scores[1, face[1]] + scores[2, face[2]])
                score = score_indexed_face_candidate(
                    face,
                    model_score,
                    state,
                    closure_bonus=closure_bonus,
                    new_edge_penalty=new_edge_penalty,
                    edge_length_penalty=edge_length_penalty,
                    aspect_penalty=aspect_penalty,
                    nonmanifold_penalty=nonmanifold_penalty,
                    require_boundary_closure_after=require_boundary_closure_after,
                    closure_target_scores=closure_target_scores,
                    closure_target_bonus=closure_target_bonus,
                    enforce_vertex_link_manifold=enforce_vertex_link_manifold,
                    vertices=vertices,
                    strict_manifold=True,
                )
                if score is None:
                    continue
                if score > best_score:
                    best_score = score
                    best_face = face
    if best_face is not None:
        return np.asarray(best_face, dtype=np.int64)

    return select_constrained_indexed_face(
        scores,
        state,
        vertex_count=vertex_count,
        top_k=top_k,
        closure_bonus=closure_bonus,
        new_edge_penalty=new_edge_penalty,
        edge_length_penalty=edge_length_penalty,
        aspect_penalty=aspect_penalty,
        nonmanifold_penalty=nonmanifold_penalty,
        require_boundary_closure_after=require_boundary_closure_after,
        closure_target_scores=closure_target_scores,
        closure_target_bonus=closure_target_bonus,
        enforce_vertex_link_manifold=enforce_vertex_link_manifold,
        vertices=vertices,
    )


def _best_indexed_candidate(
    scores: np.ndarray,
    state: IndexedDecodeState,
    top_by_corner: list[list[int]],
    *,
    closure_bonus: float,
    new_edge_penalty: float,
    edge_length_penalty: float,
    aspect_penalty: float,
    nonmanifold_penalty: float,
    require_boundary_closure_after: int,
    closure_target_scores: np.ndarray | None,
    closure_target_bonus: float,
    enforce_vertex_link_manifold: bool,
    vertices: np.ndarray | None,
    strict_manifold: bool,
) -> np.ndarray | None:
    best_face: tuple[int, int, int] | None = None
    best_score = float("-inf")
    for candidate in product(*top_by_corner):
        face = tuple(int(value) for value in candidate)
        model_score = float(scores[0, face[0]] + scores[1, face[1]] + scores[2, face[2]])
        score = score_indexed_face_candidate(
            face,
            model_score,
            state,
            closure_bonus=closure_bonus,
            new_edge_penalty=new_edge_penalty,
            edge_length_penalty=edge_length_penalty,
            aspect_penalty=aspect_penalty,
            nonmanifold_penalty=nonmanifold_penalty,
            require_boundary_closure_after=require_boundary_closure_after,
            closure_target_scores=closure_target_scores,
            closure_target_bonus=closure_target_bonus,
            enforce_vertex_link_manifold=enforce_vertex_link_manifold,
            vertices=vertices,
            strict_manifold=strict_manifold,
        )
        if score is None:
            continue
        if score > best_score:
            best_score = score
            best_face = face
    if best_face is None:
        return None
    return np.asarray(best_face, dtype=np.int64)


def score_indexed_face_candidate(
    face: tuple[int, int, int],
    model_score: float,
    state: IndexedDecodeState,
    *,
    closure_bonus: float = 2.0,
    new_edge_penalty: float = 0.15,
    edge_length_penalty: float = 0.0,
    aspect_penalty: float = 0.0,
    nonmanifold_penalty: float = 1000.0,
    require_boundary_closure_after: int = 0,
    closure_target_scores: np.ndarray | None = None,
    closure_target_bonus: float = 0.0,
    enforce_vertex_link_manifold: bool = False,
    vertices: np.ndarray | None = None,
    strict_manifold: bool = True,
) -> float | None:
    """Score a candidate triangle against the current indexed topology state."""

    if len(set(face)) != 3:
        return None
    if _indexed_face_key(face) in state.seen_faces:
        return None
    edges = _indexed_face_edges(face)
    edge_uses = [state.edge_counts[edge] for edge in edges]
    if strict_manifold and any(count >= 2 for count in edge_uses):
        return None
    if enforce_vertex_link_manifold and not _candidate_preserves_vertex_links(state, face):
        return None
    closures = sum(1 for count in edge_uses if count == 1)
    must_close = bool(require_boundary_closure_after and state.accepted_faces >= require_boundary_closure_after and state.boundary_edge_count)
    if must_close and closures == 0:
        return None
    nonmanifold_hits = sum(1 for count in edge_uses if count >= 2)
    new_edges = sum(1 for count in edge_uses if count == 0)
    closure_target_term = 0.0
    if closure_target_scores is not None and float(closure_target_bonus) != 0.0:
        target_scores = np.asarray(closure_target_scores, dtype=np.float64).reshape(-1)
        if len(target_scores) >= 4:
            closure_target_term = float(closure_target_bonus) * float(target_scores[min(3, closures)])
    geometry_term = _indexed_face_geometry_term(
        vertices,
        face,
        edge_length_penalty=edge_length_penalty,
        aspect_penalty=aspect_penalty,
    )
    return (
        float(model_score)
        + float(closure_bonus) * closures
        - float(new_edge_penalty) * new_edges
        - float(nonmanifold_penalty) * nonmanifold_hits
        + closure_target_term
        + geometry_term
    )


def _candidate_preserves_vertex_links(state: IndexedDecodeState, face: tuple[int, int, int]) -> bool:
    """Reject candidate faces that create bow-tie/pinched vertex links."""

    additions = (
        (int(face[0]), int(face[1]), int(face[2])),
        (int(face[1]), int(face[2]), int(face[0])),
        (int(face[2]), int(face[0]), int(face[1])),
    )
    for vertex, left, right in additions:
        link_edges = list(state.vertex_links.get(vertex, ()))
        link_edges.append((left, right))
        if not _is_single_path_or_cycle_link(link_edges):
            return False
    return True


def _indexed_face_geometry_term(
    vertices: np.ndarray | None,
    face: tuple[int, int, int],
    *,
    edge_length_penalty: float,
    aspect_penalty: float,
) -> float:
    if vertices is None or (edge_length_penalty == 0.0 and aspect_penalty == 0.0):
        return 0.0
    q_vertices = np.asarray(vertices, dtype=np.float64)
    if len(q_vertices) == 0:
        return 0.0
    points = q_vertices[np.asarray(face, dtype=np.int64)]
    lengths = np.asarray(
        [
            np.linalg.norm(points[1] - points[0]),
            np.linalg.norm(points[2] - points[1]),
            np.linalg.norm(points[0] - points[2]),
        ],
        dtype=np.float64,
    )
    diag = float(np.linalg.norm(np.max(q_vertices, axis=0) - np.min(q_vertices, axis=0))) or 1.0
    longest = float(np.max(lengths) / diag)
    shortest = float(max(np.min(lengths), 1e-12))
    aspect = float(np.max(lengths) / shortest)
    return -float(edge_length_penalty) * longest - float(aspect_penalty) * float(np.log(max(aspect, 1.0)))


def _edge_local_candidate_vertices(
    vertices: np.ndarray | None,
    edge: IndexedEdge,
    *,
    vertex_count: int,
    count: int,
) -> list[int]:
    if vertices is None or count <= 0:
        return []
    q_vertices = np.asarray(vertices, dtype=np.float64)[: int(vertex_count)]
    if len(q_vertices) == 0:
        return []
    a, b = int(edge[0]), int(edge[1])
    if a >= len(q_vertices) or b >= len(q_vertices):
        return []
    midpoint = (q_vertices[a] + q_vertices[b]) * 0.5
    distances = np.linalg.norm(q_vertices - midpoint, axis=1)
    order = np.argsort(distances)
    return [int(index) for index in order[: max(0, int(count)) + 2] if int(index) not in {a, b}]


def _dedupe_ints(values: list[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for value in values:
        item = int(value)
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _is_single_path_or_cycle_link(link_edges: list[tuple[int, int]]) -> bool:
    if not link_edges:
        return True
    degrees: dict[int, int] = {}
    adjacency: dict[int, set[int]] = {}
    for raw_a, raw_b in link_edges:
        a, b = int(raw_a), int(raw_b)
        if a == b:
            return False
        degrees[a] = degrees.get(a, 0) + 1
        degrees[b] = degrees.get(b, 0) + 1
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    start = next(iter(adjacency))
    stack = [start]
    visited: set[int] = set()
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        stack.extend(neighbor for neighbor in adjacency.get(current, set()) if neighbor not in visited)
    if len(visited) != len(adjacency):
        return False
    degree_values = list(degrees.values())
    if any(degree not in {1, 2} for degree in degree_values):
        return False
    degree_one_count = sum(1 for degree in degree_values if degree == 1)
    return degree_one_count in {0, 2}


def _rotate_faces_to_min_vertex(faces: np.ndarray) -> np.ndarray:
    rotated = np.asarray(faces, dtype=np.int64).copy()
    for idx, face in enumerate(rotated):
        offset = int(np.argmin(face))
        rotated[idx] = np.roll(face, -offset)
    return rotated


def _drop_degenerate_faces(faces: np.ndarray) -> np.ndarray:
    if len(faces) == 0:
        return faces.reshape(0, 3)
    keep = np.array([len(set(map(int, face))) == 3 for face in faces], dtype=bool)
    return faces[keep]


def _drop_duplicate_faces(faces: np.ndarray) -> np.ndarray:
    if len(faces) == 0:
        return faces.reshape(0, 3)
    seen: set[tuple[int, int, int]] = set()
    keep = []
    for idx, face in enumerate(faces):
        key = tuple(sorted(int(v) for v in face))
        if key in seen:
            continue
        seen.add(key)
        keep.append(idx)
    return faces[np.asarray(keep, dtype=np.int64)]


def _indexed_edge_key(a: int, b: int) -> IndexedEdge:
    left = int(a)
    right = int(b)
    return (left, right) if left <= right else (right, left)


def _indexed_face_key(face: tuple[int, int, int]) -> IndexedFaceKey:
    return tuple(sorted(int(value) for value in face))  # type: ignore[return-value]


def _indexed_face_edges(face: tuple[int, int, int]) -> tuple[IndexedEdge, IndexedEdge, IndexedEdge]:
    return (
        _indexed_edge_key(face[0], face[1]),
        _indexed_edge_key(face[1], face[2]),
        _indexed_edge_key(face[2], face[0]),
    )


def _indexed_edge_counts(faces: np.ndarray) -> Counter[IndexedEdge]:
    counts: Counter[IndexedEdge] = Counter()
    for face_arr in np.asarray(faces, dtype=np.int64).reshape(-1, 3):
        face = tuple(int(value) for value in face_arr)
        if len(set(face)) != 3:
            continue
        counts.update(_indexed_face_edges(face))
    return counts


def _choose_boundary_loop_center(vertices: np.ndarray, loop: list[int], num_bins: int) -> np.ndarray | None:
    """Pick a quantized cap vertex that avoids degenerate loop triangles."""

    q_vertices = np.asarray(vertices, dtype=np.int64)
    loop_vertices = q_vertices[np.asarray(loop, dtype=np.int64)]
    center_float = np.mean(loop_vertices.astype(np.float64), axis=0)
    center_round = np.rint(center_float).astype(np.int64)
    used = {tuple(int(value) for value in row) for row in q_vertices.tolist()}
    offsets = [(0, 0, 0)]
    for radius in range(1, 4):
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    if max(abs(dx), abs(dy), abs(dz)) == radius:
                        offsets.append((dx, dy, dz))

    best: tuple[float, float, tuple[int, int, int]] | None = None
    for offset in offsets:
        candidate = np.clip(center_round + np.asarray(offset, dtype=np.int64), 0, int(num_bins) - 1)
        candidate_key = tuple(int(value) for value in candidate.tolist())
        if candidate_key in used:
            continue
        areas = [
            _quantized_face_area2(
                q_vertices,
                (int(loop[idx]), int(loop[(idx + 1) % len(loop)]), len(q_vertices)),
                extra_vertex=candidate,
            )
            for idx in range(len(loop))
        ]
        min_area = float(min(areas)) if areas else 0.0
        if min_area <= 0.0:
            continue
        distance = float(np.linalg.norm(candidate.astype(np.float64) - center_float))
        rank = (min_area, -distance, candidate_key)
        if best is None or rank > best:
            best = rank
    if best is None:
        return None
    return np.asarray(best[2], dtype=np.int64)


def _quantized_face_area2(
    vertices: np.ndarray,
    face: tuple[int, int, int],
    *,
    extra_vertex: np.ndarray | None = None,
) -> float:
    """Return squared doubled-area in quantized coordinate space."""

    q_vertices = np.asarray(vertices, dtype=np.int64)
    points = []
    for index in face:
        if int(index) == len(q_vertices) and extra_vertex is not None:
            points.append(np.asarray(extra_vertex, dtype=np.float64))
        else:
            points.append(q_vertices[int(index)].astype(np.float64))
    a, b, c = points
    cross = np.cross(b - a, c - a)
    return float(np.dot(cross, cross))


def _boundary_loops(boundary_edges: list[IndexedEdge]) -> list[list[int]]:
    adjacency: dict[int, list[int]] = {}
    for a, b in boundary_edges:
        adjacency.setdefault(int(a), []).append(int(b))
        adjacency.setdefault(int(b), []).append(int(a))
    if any(len(neighbors) != 2 for neighbors in adjacency.values()):
        cycles = _boundary_cycle_basis(boundary_edges)
        if cycles:
            return cycles
    unused = {_indexed_edge_key(a, b) for a, b in boundary_edges}
    loops: list[list[int]] = []
    while unused:
        start_edge = min(unused)
        start, nxt = int(start_edge[0]), int(start_edge[1])
        loop = [start, nxt]
        unused.remove(start_edge)
        prev, current = start, nxt
        closed = False
        while True:
            candidates = sorted(
                neighbor
                for neighbor in adjacency.get(current, [])
                if neighbor != prev and _indexed_edge_key(current, neighbor) in unused
            )
            if not candidates:
                if _indexed_edge_key(current, start) in unused:
                    unused.remove(_indexed_edge_key(current, start))
                    closed = True
                break
            following = int(candidates[0])
            unused.remove(_indexed_edge_key(current, following))
            if following == start:
                closed = True
                break
            loop.append(following)
            prev, current = current, following
            if len(loop) > len(boundary_edges) + 1:
                break
        if closed and len(loop) >= 3:
            loops.append(loop)
    return loops


def _boundary_cycle_basis(boundary_edges: list[IndexedEdge]) -> list[list[int]]:
    """Return edge-disjoint simple cycles for pinched boundary graphs."""

    try:
        import networkx as nx
    except Exception:  # pragma: no cover - optional fallback for minimal envs.
        return []
    graph = nx.Graph()
    graph.add_edges_from((int(a), int(b)) for a, b in boundary_edges)
    loops: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for component in nx.connected_components(graph):
        subgraph = graph.subgraph(component).copy()
        if any(degree % 2 != 0 for _, degree in subgraph.degree()):
            continue
        circuit = list(nx.eulerian_circuit(subgraph, source=min(component)))
        if not circuit:
            continue
        vertices = [int(circuit[0][0])] + [int(right) for _, right in circuit]
        for cycle in _split_eulerian_vertices_into_cycles(vertices):
            if len(cycle) < 3:
                continue
            loop = _canonical_boundary_loop([int(value) for value in cycle])
            key = tuple(loop)
            if key in seen:
                continue
            seen.add(key)
            loops.append(loop)
    loops.sort(key=lambda loop: (len(loop), tuple(loop)))
    return loops


def _split_eulerian_vertices_into_cycles(vertices: list[int]) -> list[list[int]]:
    """Split an Eulerian vertex walk into edge-disjoint simple cycles."""

    stack: list[int] = []
    positions: dict[int, int] = {}
    cycles: list[list[int]] = []
    for raw_vertex in vertices:
        vertex = int(raw_vertex)
        if vertex in positions:
            start = positions[vertex]
            cycle = stack[start:]
            if len(cycle) >= 3:
                cycles.append(cycle)
            for removed in stack[start + 1 :]:
                positions.pop(removed, None)
            stack = stack[: start + 1]
            positions[vertex] = start
        else:
            positions[vertex] = len(stack)
            stack.append(vertex)
    return cycles


def _canonical_boundary_loop(loop: list[int]) -> list[int]:
    """Rotate/reverse a simple cycle for deterministic fill order."""

    values = [int(value) for value in loop]
    if not values:
        return values
    min_pos = min(range(len(values)), key=lambda idx: values[idx])
    forward = values[min_pos:] + values[:min_pos]
    reversed_values = list(reversed(values))
    min_pos_reversed = min(range(len(reversed_values)), key=lambda idx: reversed_values[idx])
    backward = reversed_values[min_pos_reversed:] + reversed_values[:min_pos_reversed]
    return forward if tuple(forward) <= tuple(backward) else backward


def _boundary_edge_logit_score(scores: np.ndarray, edge: IndexedEdge) -> float:
    a, b = edge
    best = float("-inf")
    for left_corner, right_corner in permutations(range(3), 2):
        best = max(best, float(scores[left_corner, a] + scores[right_corner, b]))
        best = max(best, float(scores[left_corner, b] + scores[right_corner, a]))
    return best
