"""FACE-style artist-mesh tokenization.

This is not the full FACE model. It is the geometry-critical tokenizer and
roundtrip contract we need before training a FACE-like ARAE/decoder:

- normalize vertices to a centered cube,
- quantize coordinates to a fixed bin range,
- sort vertices in ZYX order,
- canonicalize triangle rotation without flipping winding,
- sort faces deterministically,
- decode by welding identical quantized vertices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import trimesh


@dataclass(frozen=True)
class FaceTokenTransform:
    """Affine transform from original coordinates to normalized coordinates."""

    center: tuple[float, float, float]
    scale: float

    def normalize(self, vertices: np.ndarray) -> np.ndarray:
        return (np.asarray(vertices, dtype=np.float64) - np.asarray(self.center, dtype=np.float64)) * self.scale

    def denormalize(self, vertices: np.ndarray) -> np.ndarray:
        return np.asarray(vertices, dtype=np.float64) / self.scale + np.asarray(self.center, dtype=np.float64)


@dataclass(frozen=True)
class FaceTokenSequence:
    """Quantized FACE-style triangle tokens with enough metadata to decode."""

    tokens: np.ndarray
    num_bins: int
    transform: FaceTokenTransform

    @property
    def face_count(self) -> int:
        return int(self.tokens.shape[0])

    @property
    def coordinate_token_count(self) -> int:
        return int(self.tokens.size)

    def as_flat_tokens(self) -> np.ndarray:
        return np.asarray(self.tokens, dtype=np.int64).reshape(-1)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "tokens": np.asarray(self.tokens, dtype=np.int64).tolist(),
            "num_bins": int(self.num_bins),
            "transform": {
                "center": list(self.transform.center),
                "scale": float(self.transform.scale),
            },
        }


def _validate_num_bins(num_bins: int) -> int:
    num_bins = int(num_bins)
    if num_bins < 2:
        raise ValueError(f"num_bins must be at least 2, got {num_bins}")
    return num_bins


def _validate_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"mesh must be a trimesh.Trimesh, got {type(mesh)!r}")
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError("mesh must contain vertices and triangular faces")
    return mesh


def fit_face_token_transform(vertices: np.ndarray, padding: float = 1.0) -> FaceTokenTransform:
    """Fit a centered uniform scale into approximately ``[-1, 1]``."""

    verts = np.asarray(vertices, dtype=np.float64)
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError(f"vertices must have shape (N, 3), got {verts.shape}")
    vmin = verts.min(axis=0)
    vmax = verts.max(axis=0)
    extent = vmax - vmin
    max_extent = float(np.max(extent))
    if max_extent <= 0.0:
        raise ValueError("vertices must span a non-zero extent")
    padding = float(padding)
    if padding <= 0.0:
        raise ValueError(f"padding must be positive, got {padding}")
    center = tuple(((vmin + vmax) * 0.5).tolist())
    scale = (2.0 / max_extent) / padding
    return FaceTokenTransform(center=center, scale=scale)


def quantize_normalized_points(points: np.ndarray, num_bins: int = 128) -> np.ndarray:
    """Quantize normalized ``[-1, 1]`` points to integer coordinate bins."""

    num_bins = _validate_num_bins(num_bins)
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {pts.shape}")
    normalized = np.clip((pts + 1.0) * 0.5, 0.0, 1.0)
    quantized = np.floor(normalized * num_bins).astype(np.int64)
    return np.clip(quantized, 0, num_bins - 1)


def dequantize_normalized_points(quantized: np.ndarray, num_bins: int = 128) -> np.ndarray:
    """Decode integer coordinate bins to bin-center points in ``[-1, 1]``."""

    num_bins = _validate_num_bins(num_bins)
    q = np.asarray(quantized, dtype=np.int64)
    if q.ndim != 2 or q.shape[1] != 3:
        raise ValueError(f"quantized must have shape (N, 3), got {q.shape}")
    if np.any(q < 0) or np.any(q >= num_bins):
        raise ValueError("quantized coordinates must be inside [0, num_bins)")
    return ((q.astype(np.float64) + 0.5) / num_bins) * 2.0 - 1.0


def _sort_vertices_zyx(quantized_vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted vertices and an old-to-new index map."""

    q = np.asarray(quantized_vertices, dtype=np.int64)
    order = np.lexsort((q[:, 0], q[:, 1], q[:, 2]))
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order), dtype=np.int64)
    return q[order], inverse


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


def _non_colinear_quantized_faces(q_faces: np.ndarray) -> np.ndarray:
    """Return a mask for quantized triangles with non-zero integer area."""

    q = np.asarray(q_faces, dtype=np.int64)
    if q.ndim != 3 or q.shape[1:] != (3, 3):
        raise ValueError(f"q_faces must have shape (F, 3, 3), got {q.shape}")
    if len(q) == 0:
        return np.zeros((0,), dtype=bool)
    ab = q[:, 1] - q[:, 0]
    ac = q[:, 2] - q[:, 0]
    cross = np.cross(ab, ac)
    return np.any(cross != 0, axis=1)


def canonicalize_mesh_faces(
    vertices: np.ndarray,
    faces: np.ndarray,
    num_bins: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Quantize and return canonical quantized triangle coordinates."""

    q_vertices = quantize_normalized_points(vertices, num_bins=num_bins)
    sorted_q_vertices, old_to_new = _sort_vertices_zyx(q_vertices)
    sorted_faces = old_to_new[np.asarray(faces, dtype=np.int64)]
    sorted_faces = _drop_degenerate_faces(_rotate_faces_to_min_vertex(sorted_faces))
    sorted_faces = sorted_faces[_non_colinear_quantized_faces(sorted_q_vertices[sorted_faces])]
    if len(sorted_faces) == 0:
        raise ValueError("all faces became degenerate after quantization")

    face_order = np.lexsort((sorted_faces[:, 2], sorted_faces[:, 1], sorted_faces[:, 0]))
    sorted_faces = sorted_faces[face_order]
    tokens = sorted_q_vertices[sorted_faces].reshape(-1, 9)
    return tokens.astype(np.int64), sorted_faces.astype(np.int64)


def canonicalize_mesh_faces_paper_zyx(
    vertices: np.ndarray,
    faces: np.ndarray,
    num_bins: int = 128,
    within_face_order: str = "preserve",
) -> tuple[np.ndarray, np.ndarray]:
    """Quantize faces in the ordering described by the FACE paper.

    This path is intentionally separate from the older ClearMesh tokenizer.
    FACE orders faces by the lexicographic ``ZYX`` coordinate of each face's
    minimum-coordinate vertex, and Fig. 2 shows each vertex emitted as
    ``z, y, x``. The paper does not specify how to order the three vertices
    inside a face, so this function exposes the choice for ablation.
    """

    within_face_order = str(within_face_order).strip().lower()
    if within_face_order not in {"preserve", "rotate_min_zyx", "sort_zyx"}:
        raise ValueError(f"unknown within_face_order: {within_face_order}")
    q_vertices_xyz = quantize_normalized_points(vertices, num_bins=num_bins)
    raw_faces = np.asarray(faces, dtype=np.int64)
    raw_faces = _drop_degenerate_faces(raw_faces)
    if len(raw_faces) == 0:
        raise ValueError("all faces became degenerate after quantization")

    face_vertex_zyx = q_vertices_xyz[raw_faces][:, :, [2, 1, 0]]
    quantized_non_duplicate = np.asarray(
        [len({tuple(map(int, vertex)) for vertex in face}) == 3 for face in face_vertex_zyx],
        dtype=bool,
    )
    raw_faces = raw_faces[quantized_non_duplicate]
    face_vertex_zyx = face_vertex_zyx[quantized_non_duplicate]
    raw_faces = raw_faces[_non_colinear_quantized_faces(q_vertices_xyz[raw_faces])]
    face_vertex_zyx = q_vertices_xyz[raw_faces][:, :, [2, 1, 0]]
    if len(raw_faces) == 0:
        raise ValueError("all faces became degenerate after quantization")
    min_offsets = np.asarray(
        [np.lexsort((face[:, 2], face[:, 1], face[:, 0]))[0] for face in face_vertex_zyx],
        dtype=np.int64,
    )
    min_vertices = face_vertex_zyx[np.arange(len(raw_faces)), min_offsets]
    ordered_faces = raw_faces.copy()
    if within_face_order == "rotate_min_zyx":
        for idx, offset in enumerate(min_offsets):
            ordered_faces[idx] = np.roll(ordered_faces[idx], -int(offset))
    elif within_face_order == "sort_zyx":
        for idx, face_zyx in enumerate(face_vertex_zyx):
            ordered_faces[idx] = ordered_faces[idx][np.lexsort((face_zyx[:, 2], face_zyx[:, 1], face_zyx[:, 0]))]
    ordered_face_vertex_zyx = q_vertices_xyz[ordered_faces][:, :, [2, 1, 0]]
    flat_tie = ordered_face_vertex_zyx.reshape(len(raw_faces), 9)
    order = np.lexsort(tuple([flat_tie[:, idx] for idx in range(8, -1, -1)] + [min_vertices[:, 2], min_vertices[:, 1], min_vertices[:, 0]]))
    sorted_faces = ordered_faces[order]
    tokens = q_vertices_xyz[sorted_faces][:, :, [2, 1, 0]].reshape(-1, 9)
    return tokens.astype(np.int64), sorted_faces.astype(np.int64)


def encode_mesh_to_face_tokens(
    mesh: trimesh.Trimesh,
    num_bins: int = 128,
    normalize: bool = True,
    max_faces: int | None = None,
    padding: float = 1.0,
) -> FaceTokenSequence:
    """Encode a triangular mesh into deterministic FACE-style face tokens."""

    mesh = _validate_mesh(mesh)
    num_bins = _validate_num_bins(num_bins)
    if max_faces is not None and len(mesh.faces) > int(max_faces):
        raise ValueError(f"mesh has {len(mesh.faces)} faces, above max_faces={max_faces}")

    transform = fit_face_token_transform(mesh.vertices, padding=padding) if normalize else FaceTokenTransform(
        center=(0.0, 0.0, 0.0),
        scale=1.0,
    )
    vertices = transform.normalize(mesh.vertices) if normalize else np.asarray(mesh.vertices, dtype=np.float64)
    tokens, _ = canonicalize_mesh_faces(vertices, mesh.faces, num_bins=num_bins)
    return FaceTokenSequence(tokens=tokens, num_bins=num_bins, transform=transform)


def encode_mesh_to_paper_face_tokens(
    mesh: trimesh.Trimesh,
    num_bins: int = 128,
    normalize: bool = True,
    max_faces: int | None = None,
    padding: float = 1.0,
    within_face_order: str = "preserve",
) -> FaceTokenSequence:
    """Encode a mesh with the FACE paper's coordinate and face-order contract."""

    mesh = _validate_mesh(mesh)
    num_bins = _validate_num_bins(num_bins)
    if max_faces is not None and len(mesh.faces) > int(max_faces):
        raise ValueError(f"mesh has {len(mesh.faces)} faces, above max_faces={max_faces}")

    transform = fit_face_token_transform(mesh.vertices, padding=padding) if normalize else FaceTokenTransform(
        center=(0.0, 0.0, 0.0),
        scale=1.0,
    )
    vertices = transform.normalize(mesh.vertices) if normalize else np.asarray(mesh.vertices, dtype=np.float64)
    tokens, _ = canonicalize_mesh_faces_paper_zyx(vertices, mesh.faces, num_bins=num_bins, within_face_order=within_face_order)
    return FaceTokenSequence(tokens=tokens, num_bins=num_bins, transform=transform)


def decode_face_tokens_to_mesh(
    sequence: FaceTokenSequence,
    denormalize: bool = True,
    process: bool = False,
) -> trimesh.Trimesh:
    """Decode tokens to a triangle mesh, welding repeated quantized vertices."""

    tokens = np.asarray(sequence.tokens, dtype=np.int64)
    if tokens.ndim != 2 or tokens.shape[1] != 9:
        raise ValueError(f"tokens must have shape (F, 9), got {tokens.shape}")
    q_faces = tokens.reshape(-1, 3, 3)
    flat_q = q_faces.reshape(-1, 3)
    unique_q, inverse = np.unique(flat_q, axis=0, return_inverse=True)
    vertices = dequantize_normalized_points(unique_q, num_bins=sequence.num_bins)
    if denormalize:
        vertices = sequence.transform.denormalize(vertices)
    faces = inverse.reshape(-1, 3)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=process)
    if len(mesh.faces):
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.remove_unreferenced_vertices()
    return mesh


def decode_paper_face_tokens_to_mesh(
    sequence: FaceTokenSequence,
    denormalize: bool = True,
    process: bool = False,
) -> trimesh.Trimesh:
    """Decode FACE-paper ``z,y,x`` coordinate tokens into a welded mesh."""

    tokens = np.asarray(sequence.tokens, dtype=np.int64)
    if tokens.ndim != 2 or tokens.shape[1] != 9:
        raise ValueError(f"tokens must have shape (F, 9), got {tokens.shape}")
    q_faces_zyx = tokens.reshape(-1, 3, 3)
    q_faces_xyz = q_faces_zyx[:, :, [2, 1, 0]]
    flat_q = q_faces_xyz.reshape(-1, 3)
    unique_q, inverse = np.unique(flat_q, axis=0, return_inverse=True)
    vertices = dequantize_normalized_points(unique_q, num_bins=sequence.num_bins)
    if denormalize:
        vertices = sequence.transform.denormalize(vertices)
    faces = inverse.reshape(-1, 3)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=process)
    if len(mesh.faces):
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.remove_unreferenced_vertices()
    return mesh


def face_token_stats(sequence: FaceTokenSequence) -> dict[str, int | float]:
    """Return compact diagnostics for logging and tests."""

    tokens = np.asarray(sequence.tokens, dtype=np.int64)
    unique_vertices = np.unique(tokens.reshape(-1, 3), axis=0)
    duplicate_coordinate_ratio = 1.0 - (len(unique_vertices) / max(1, tokens.shape[0] * 3))
    return {
        "faces": int(tokens.shape[0]),
        "coordinate_tokens": int(tokens.size),
        "num_bins": int(sequence.num_bins),
        "unique_quantized_vertices": int(len(unique_vertices)),
        "duplicate_coordinate_ratio": float(duplicate_coordinate_ratio),
    }
