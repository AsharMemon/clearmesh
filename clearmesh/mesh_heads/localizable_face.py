"""Voxel-anchored FACE patch helpers.

This module is the first small, testable step toward "localizable FACE":
instead of asking one autoregressive decoder to emit an entire mesh as one
global face stream, we partition indexed FACE targets into local patches tied
to known 3D anchors. The anchors can come from a coarse TRELLIS/LATTICE voxel
support at inference time; for supervised corpus prep we derive them from the
ground-truth indexed face centroids.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LocalFacePatch:
    """A local indexed-FACE patch anchored to one voxel cell."""

    anchor: np.ndarray
    vertices: np.ndarray
    faces: np.ndarray
    global_vertex_indices: np.ndarray
    source_face_indices: np.ndarray
    point_indices: np.ndarray

    @property
    def face_count(self) -> int:
        return int(np.asarray(self.faces).shape[0])

    @property
    def vertex_count(self) -> int:
        return int(np.asarray(self.vertices).shape[0])


def face_centroid_anchors(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    num_bins: int,
    voxel_resolution: int,
) -> np.ndarray:
    """Assign each indexed face to a quantized voxel anchor.

    ``vertices`` are FACE quantized coordinates in ``[0, num_bins)``. We map
    triangle centroids into a coarser voxel lattice and clip for numerical
    robustness. The output has shape ``(F, 3)`` and integer coordinates in
    ``[0, voxel_resolution)``.
    """

    verts = _validate_vertices(vertices)
    fcs = _validate_faces(faces, len(verts))
    bins = _positive_int(num_bins, "num_bins")
    resolution = _positive_int(voxel_resolution, "voxel_resolution")
    if len(fcs) == 0:
        return np.zeros((0, 3), dtype=np.int16)

    centroids = verts[fcs].astype(np.float64).mean(axis=1)
    normalized = (centroids + 0.5) / float(bins)
    anchors = np.floor(normalized * float(resolution)).astype(np.int64)
    return np.clip(anchors, 0, resolution - 1).astype(np.int16)


def vertex_anchor_coords(
    vertices: np.ndarray,
    *,
    num_bins: int,
    voxel_resolution: int,
) -> np.ndarray:
    """Map each quantized vertex-table row into the local voxel lattice."""

    verts = _validate_vertices(vertices)
    bins = _positive_int(num_bins, "num_bins")
    resolution = _positive_int(voxel_resolution, "voxel_resolution")
    normalized = (verts.astype(np.float64) + 0.5) / float(bins)
    anchors = np.floor(normalized * float(resolution)).astype(np.int64)
    return np.clip(anchors, 0, resolution - 1).astype(np.int16)


def surface_point_anchors(
    points: np.ndarray,
    *,
    voxel_resolution: int,
) -> np.ndarray:
    """Map normalized surface points in roughly ``[-1, 1]`` to voxel anchors."""

    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {pts.shape}")
    resolution = _positive_int(voxel_resolution, "voxel_resolution")
    normalized = np.clip((pts + 1.0) * 0.5, 0.0, np.nextafter(1.0, 0.0))
    anchors = np.floor(normalized * float(resolution)).astype(np.int64)
    return np.clip(anchors, 0, resolution - 1).astype(np.int16)


def build_local_face_patches(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    num_bins: int,
    voxel_resolution: int = 32,
    max_faces_per_patch: int = 128,
    surface_points: np.ndarray | None = None,
    point_samples_per_patch: int = 0,
) -> list[LocalFacePatch]:
    """Partition indexed FACE targets into voxel-local patches.

    The partition is lossless with respect to face membership: every input face
    appears in exactly one output patch, unless validation fails before patching.
    Local faces reference a compact per-patch vertex table, while
    ``global_vertex_indices`` preserves the mapping back to the source mesh.
    """

    verts = _validate_vertices(vertices)
    fcs = _validate_faces(faces, len(verts))
    _positive_int(num_bins, "num_bins")
    resolution = _positive_int(voxel_resolution, "voxel_resolution")
    max_faces = _positive_int(max_faces_per_patch, "max_faces_per_patch")
    if len(fcs) == 0:
        return []

    anchors = face_centroid_anchors(
        verts,
        fcs,
        num_bins=num_bins,
        voxel_resolution=resolution,
    )
    point_anchor_map = None
    points_arr = None
    if surface_points is not None and point_samples_per_patch > 0:
        points_arr = np.asarray(surface_points, dtype=np.float32)
        point_anchor_map = surface_point_anchors(points_arr, voxel_resolution=resolution)

    patches: list[LocalFacePatch] = []
    anchor_keys = [tuple(int(value) for value in row) for row in anchors]
    grouped: dict[tuple[int, int, int], list[int]] = {}
    for face_index, key in enumerate(anchor_keys):
        grouped.setdefault(key, []).append(face_index)

    for key in sorted(grouped):
        indices = grouped[key]
        for start in range(0, len(indices), max_faces):
            source_face_indices = np.asarray(indices[start : start + max_faces], dtype=np.int64)
            patch_faces_global = fcs[source_face_indices]
            global_vertices = np.unique(patch_faces_global.reshape(-1))
            remap = np.empty(len(verts), dtype=np.int64)
            remap.fill(-1)
            remap[global_vertices] = np.arange(len(global_vertices), dtype=np.int64)
            patch_faces = remap[patch_faces_global]
            if np.any(patch_faces < 0):  # pragma: no cover - defensive.
                raise RuntimeError("local face remap failed")
            point_indices = _select_patch_point_indices(
                key,
                point_anchor_map=point_anchor_map,
                points=points_arr,
                max_points=int(point_samples_per_patch),
                voxel_resolution=resolution,
            )
            patches.append(
                LocalFacePatch(
                    anchor=np.asarray(key, dtype=np.int16),
                    vertices=verts[global_vertices].astype(np.int16),
                    faces=patch_faces.astype(np.int32),
                    global_vertex_indices=global_vertices.astype(np.int32),
                    source_face_indices=source_face_indices.astype(np.int32),
                    point_indices=point_indices.astype(np.int32),
                )
            )
    return patches


def assert_patch_face_coverage(patches: list[LocalFacePatch], face_count: int) -> None:
    """Raise if local patches do not cover every source face exactly once."""

    face_count = int(face_count)
    if face_count < 0:
        raise ValueError(f"face_count must be non-negative, got {face_count}")
    if face_count == 0:
        if patches:
            raise ValueError("patches were provided for an empty source mesh")
        return
    seen = np.zeros(face_count, dtype=np.int64)
    for patch in patches:
        indices = np.asarray(patch.source_face_indices, dtype=np.int64)
        if np.any(indices < 0) or np.any(indices >= face_count):
            raise ValueError("patch source_face_indices are out of range")
        np.add.at(seen, indices, 1)
    missing = np.flatnonzero(seen == 0)
    duplicated = np.flatnonzero(seen > 1)
    if len(missing) or len(duplicated):
        raise ValueError(
            "patches must cover every source face exactly once "
            f"(missing={len(missing)}, duplicated={len(duplicated)})"
        )


def reconstruct_source_faces_from_patches(
    patches: list[LocalFacePatch],
    *,
    face_count: int,
) -> np.ndarray:
    """Rebuild source-order global face indices from local patches.

    This is a losslessness diagnostic for localizable FACE patching: every local
    face should map back to the exact original indexed-face row via the patch's
    ``global_vertex_indices`` and ``source_face_indices`` arrays.
    """

    assert_patch_face_coverage(patches, face_count)
    reconstructed = np.full((int(face_count), 3), -1, dtype=np.int64)
    for patch in patches:
        local_faces = np.asarray(patch.faces, dtype=np.int64)
        global_vertices = np.asarray(patch.global_vertex_indices, dtype=np.int64)
        source_indices = np.asarray(patch.source_face_indices, dtype=np.int64)
        if len(local_faces) != len(source_indices):
            raise ValueError("patch faces/source_face_indices length mismatch")
        if np.any(local_faces < 0) or np.any(local_faces >= len(global_vertices)):
            raise ValueError("patch local faces reference vertices outside the local table")
        reconstructed[source_indices] = global_vertices[local_faces]
    if np.any(reconstructed < 0):  # pragma: no cover - assert_patch_face_coverage should catch this.
        raise ValueError("some source faces were not reconstructed")
    return reconstructed


def reconstruct_source_faces_from_packed_arrays(
    *,
    patch_faces_flat: np.ndarray,
    patch_face_offsets: np.ndarray,
    global_vertex_indices_flat: np.ndarray,
    patch_vertex_offsets: np.ndarray,
    source_face_indices_flat: np.ndarray,
    face_count: int,
) -> np.ndarray:
    """Rebuild source-order global face indices from a packed patch NPZ."""

    face_offsets = np.asarray(patch_face_offsets, dtype=np.int64)
    vertex_offsets = np.asarray(patch_vertex_offsets, dtype=np.int64)
    patch_faces = np.asarray(patch_faces_flat, dtype=np.int64)
    global_vertices = np.asarray(global_vertex_indices_flat, dtype=np.int64)
    source_face_indices = np.asarray(source_face_indices_flat, dtype=np.int64)
    if len(face_offsets) != len(vertex_offsets):
        raise ValueError("face and vertex offset arrays must have matching patch counts")
    patch_count = len(face_offsets) - 1
    reconstructed = np.full((int(face_count), 3), -1, dtype=np.int64)
    seen = np.zeros(int(face_count), dtype=np.int64)
    for patch_index in range(patch_count):
        f0, f1 = int(face_offsets[patch_index]), int(face_offsets[patch_index + 1])
        v0, v1 = int(vertex_offsets[patch_index]), int(vertex_offsets[patch_index + 1])
        local_faces = patch_faces[f0:f1]
        source_indices = source_face_indices[f0:f1]
        local_global_vertices = global_vertices[v0:v1]
        if len(local_faces) != len(source_indices):
            raise ValueError("packed patch faces/source indices length mismatch")
        if np.any(source_indices < 0) or np.any(source_indices >= int(face_count)):
            raise ValueError("packed source_face_indices are out of range")
        if np.any(local_faces < 0) or np.any(local_faces >= len(local_global_vertices)):
            raise ValueError("packed local faces reference vertices outside the local table")
        reconstructed[source_indices] = local_global_vertices[local_faces]
        np.add.at(seen, source_indices, 1)
    missing = np.flatnonzero(seen == 0)
    duplicated = np.flatnonzero(seen > 1)
    if len(missing) or len(duplicated):
        raise ValueError(
            "packed patches must cover every source face exactly once "
            f"(missing={len(missing)}, duplicated={len(duplicated)})"
        )
    if np.any(reconstructed < 0):  # pragma: no cover - seen should catch this.
        raise ValueError("some packed source faces were not reconstructed")
    return reconstructed


def patch_summary(patches: list[LocalFacePatch]) -> dict[str, int | float]:
    """Return compact diagnostics for a set of local FACE patches."""

    if not patches:
        return {
            "patches": 0,
            "faces": 0,
            "vertices": 0,
            "max_faces_per_patch": 0,
            "mean_faces_per_patch": 0.0,
            "anchors": 0,
        }
    face_counts = np.asarray([patch.face_count for patch in patches], dtype=np.int64)
    vertex_counts = np.asarray([patch.vertex_count for patch in patches], dtype=np.int64)
    anchors = {tuple(int(value) for value in patch.anchor) for patch in patches}
    return {
        "patches": int(len(patches)),
        "faces": int(face_counts.sum()),
        "vertices": int(vertex_counts.sum()),
        "max_faces_per_patch": int(face_counts.max()),
        "mean_faces_per_patch": float(face_counts.mean()),
        "anchors": int(len(anchors)),
    }


def _select_patch_point_indices(
    anchor: tuple[int, int, int],
    *,
    point_anchor_map: np.ndarray | None,
    points: np.ndarray | None,
    max_points: int,
    voxel_resolution: int,
) -> np.ndarray:
    if point_anchor_map is None or points is None or max_points <= 0 or len(points) == 0:
        return np.zeros((0,), dtype=np.int64)
    anchor_arr = np.asarray(anchor, dtype=np.int16)
    matches = np.flatnonzero(np.all(point_anchor_map == anchor_arr[None, :], axis=1))
    if len(matches) >= max_points:
        return matches[:max_points].astype(np.int64)
    if len(matches) > 0:
        return matches.astype(np.int64)

    # Fallback for tiny patches: choose nearest normalized surface points to the
    # anchor center so every training patch can still carry conditioning.
    center = ((anchor_arr.astype(np.float32) + 0.5) / max(1, int(voxel_resolution))) * 2.0 - 1.0
    dist2 = np.sum((points.astype(np.float32) - center[None, :]) ** 2, axis=1)
    count = min(max_points, len(points))
    return np.argsort(dist2)[:count].astype(np.int64)


def _positive_int(value: int, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _validate_vertices(vertices: np.ndarray) -> np.ndarray:
    arr = np.asarray(vertices, dtype=np.int64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"vertices must have shape (V, 3), got {arr.shape}")
    if len(arr) == 0:
        raise ValueError("vertices must not be empty")
    return arr


def _validate_faces(faces: np.ndarray, vertex_count: int) -> np.ndarray:
    arr = np.asarray(faces, dtype=np.int64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"faces must have shape (F, 3), got {arr.shape}")
    if len(arr) and (np.any(arr < 0) or np.any(arr >= int(vertex_count))):
        raise ValueError("faces reference vertices outside the source vertex table")
    return arr
