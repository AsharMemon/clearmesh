"""Surface-active voxel extraction for LATTICE-style query sets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from .queries import Bounds3D, quantize_points_to_indices, voxel_indices_to_centers


@dataclass(frozen=True)
class ActiveVoxelOptions:
    """Options for extracting a sparse active voxel set from a mesh surface."""

    resolution: int = 64
    surface_samples: int = 20_000
    include_face_centroids: bool = True
    bounds: Bounds3D = Bounds3D()
    seed: int | None = 0


@dataclass(frozen=True)
class ActiveVoxelSet:
    """Sparse surface voxel set and its query centers."""

    indices: np.ndarray
    centers: np.ndarray
    resolution: int
    bounds: Bounds3D

    @property
    def count(self) -> int:
        return int(self.indices.shape[0])


def _as_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"mesh must be a trimesh.Trimesh, got {type(mesh)!r}")
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError("mesh must contain vertices and triangular faces")
    return mesh


def extract_active_surface_voxels(
    mesh: trimesh.Trimesh,
    options: ActiveVoxelOptions | None = None,
) -> ActiveVoxelSet:
    """Extract unique voxel indices touched by sampled mesh surface points."""

    mesh = _as_mesh(mesh)
    options = options or ActiveVoxelOptions()

    samples: list[np.ndarray] = []
    if options.include_face_centroids:
        samples.append(np.asarray(mesh.triangles_center, dtype=np.float64))
    if options.surface_samples > 0:
        points, _ = trimesh.sample.sample_surface(mesh, int(options.surface_samples), seed=options.seed)
        samples.append(np.asarray(points, dtype=np.float64))
    if not samples:
        raise ValueError("At least one of include_face_centroids or surface_samples must be enabled")

    points = np.concatenate(samples, axis=0)
    indices = quantize_points_to_indices(points, options.resolution, options.bounds)
    unique_indices = np.unique(indices, axis=0)
    centers = voxel_indices_to_centers(unique_indices, options.resolution, options.bounds)
    return ActiveVoxelSet(
        indices=unique_indices,
        centers=centers,
        resolution=int(options.resolution),
        bounds=options.bounds,
    )
