"""Vector-displacement and edge-supervision helpers.

These are the local, testable geometry contracts behind the LATO/FACE-adjacent
sidecar we want for LATTICE: sample a point on a face, represent its three
incident vertices as displacement vectors, and provide edge/non-edge supervision
for topology.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh


@dataclass(frozen=True)
class VDFSamples:
    """Vector-displacement field samples from mesh surface triangles."""

    points: np.ndarray
    normals: np.ndarray
    face_indices: np.ndarray
    face_vertices: np.ndarray
    vertex_displacements: np.ndarray

    @property
    def features(self) -> np.ndarray:
        """Concatenate point, three vertex displacements, and normal: 15 dims."""

        return np.concatenate(
            [
                self.points,
                self.vertex_displacements.reshape(len(self.points), 9),
                self.normals,
            ],
            axis=1,
        )


@dataclass(frozen=True)
class EdgeCandidates:
    """Positive mesh edges and sampled negative non-edge pairs."""

    positive_edges: np.ndarray
    negative_edges: np.ndarray

    @property
    def positive_count(self) -> int:
        return int(self.positive_edges.shape[0])

    @property
    def negative_count(self) -> int:
        return int(self.negative_edges.shape[0])


def _validate_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"mesh must be a trimesh.Trimesh, got {type(mesh)!r}")
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError("mesh must contain vertices and faces")
    return mesh


def sample_vdf(mesh: trimesh.Trimesh, count: int, seed: int | None = 0) -> VDFSamples:
    """Sample points and per-face vector displacements.

    For each sampled surface point ``p`` on triangle ``(v0, v1, v2)``, the VDF
    stores ``v0 - p``, ``v1 - p``, and ``v2 - p``. Reconstructing face vertices
    from ``p + displacement`` is therefore an exact sanity check independent of
    any neural model.
    """

    mesh = _validate_mesh(mesh)
    count = int(count)
    if count <= 0:
        raise ValueError(f"count must be positive, got {count}")

    points, face_indices = trimesh.sample.sample_surface(mesh, count, seed=seed)
    face_indices = np.asarray(face_indices, dtype=np.int64)
    face_vertices = np.asarray(mesh.vertices[mesh.faces[face_indices]], dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    vertex_displacements = face_vertices - points[:, None, :]
    normals = np.asarray(mesh.face_normals[face_indices], dtype=np.float64)
    return VDFSamples(
        points=points,
        normals=normals,
        face_indices=face_indices,
        face_vertices=face_vertices,
        vertex_displacements=vertex_displacements,
    )


def unique_mesh_edges(mesh: trimesh.Trimesh) -> np.ndarray:
    """Return sorted unique undirected mesh edges as ``(E, 2)`` vertex pairs."""

    mesh = _validate_mesh(mesh)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    edge_pairs = faces[:, [(0, 1), (1, 2), (2, 0)]].reshape(-1, 2)
    edge_pairs.sort(axis=1)
    return np.unique(edge_pairs, axis=0)


def _edge_set(edges: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, edge)) for edge in np.asarray(edges, dtype=np.int64)}


def sample_edge_candidates(
    mesh: trimesh.Trimesh,
    random_negative_count: int | None = None,
    seed: int | None = 0,
) -> EdgeCandidates:
    """Sample edge/non-edge pairs for topology supervision.

    Negatives are random undirected vertex pairs that are not true mesh edges.
    This deliberately does not enumerate all pairs on large meshes.
    """

    mesh = _validate_mesh(mesh)
    positives = unique_mesh_edges(mesh)
    positive_set = _edge_set(positives)
    vertex_count = int(len(mesh.vertices))
    if vertex_count < 2:
        raise ValueError("mesh must contain at least two vertices")

    if random_negative_count is None:
        random_negative_count = len(positives)
    random_negative_count = int(random_negative_count)
    if random_negative_count < 0:
        raise ValueError(f"random_negative_count must be non-negative, got {random_negative_count}")

    max_pairs = vertex_count * (vertex_count - 1) // 2
    max_negatives = max_pairs - len(positive_set)
    target = min(random_negative_count, max_negatives)
    rng = np.random.default_rng(seed)
    negatives: set[tuple[int, int]] = set()

    attempts = 0
    max_attempts = max(1_000, target * 100)
    while len(negatives) < target and attempts < max_attempts:
        a, b = rng.choice(vertex_count, size=2, replace=False)
        edge = (int(min(a, b)), int(max(a, b)))
        if edge not in positive_set:
            negatives.add(edge)
        attempts += 1

    if len(negatives) < target:
        for a in range(vertex_count):
            for b in range(a + 1, vertex_count):
                edge = (a, b)
                if edge not in positive_set:
                    negatives.add(edge)
                    if len(negatives) == target:
                        break
            if len(negatives) == target:
                break

    negative_edges = np.asarray(sorted(negatives), dtype=np.int64).reshape(-1, 2)
    return EdgeCandidates(positive_edges=positives, negative_edges=negative_edges)
