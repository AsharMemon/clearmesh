"""Voxel-query helpers for the LATTICE reproduction path."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Bounds3D:
    """Axis-aligned bounds used by quantization and voxel centers."""

    minimum: tuple[float, float, float] = (-1.0, -1.0, -1.0)
    maximum: tuple[float, float, float] = (1.0, 1.0, 1.0)

    @property
    def min_array(self) -> np.ndarray:
        return np.asarray(self.minimum, dtype=np.float64)

    @property
    def max_array(self) -> np.ndarray:
        return np.asarray(self.maximum, dtype=np.float64)

    @property
    def span(self) -> np.ndarray:
        span = self.max_array - self.min_array
        if np.any(span <= 0.0):
            raise ValueError(f"Bounds maximum must be greater than minimum, got {self}")
        return span


def _rng(seed: int | np.random.Generator | None) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def _validate_resolution(resolution: int) -> int:
    resolution = int(resolution)
    if resolution <= 0:
        raise ValueError(f"resolution must be positive, got {resolution}")
    return resolution


def quantize_points_to_indices(
    points: np.ndarray,
    resolution: int,
    bounds: Bounds3D = Bounds3D(),
) -> np.ndarray:
    """Map xyz points into integer voxel indices in ``[0, resolution)``.

    Points outside the bounds are clipped. This mirrors the robust behavior we
    need when TRELLIS/UltraShape proxies are slightly outside the training cube.
    """

    resolution = _validate_resolution(resolution)
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {pts.shape}")

    normalized = (pts - bounds.min_array) / bounds.span
    indices = np.floor(normalized * resolution).astype(np.int64)
    return np.clip(indices, 0, resolution - 1)


def voxel_indices_to_centers(
    indices: np.ndarray,
    resolution: int,
    bounds: Bounds3D = Bounds3D(),
) -> np.ndarray:
    """Return xyz centers for integer voxel indices."""

    resolution = _validate_resolution(resolution)
    idx = np.asarray(indices, dtype=np.int64)
    if idx.ndim != 2 or idx.shape[1] != 3:
        raise ValueError(f"indices must have shape (N, 3), got {idx.shape}")
    if np.any(idx < 0) or np.any(idx >= resolution):
        raise ValueError("indices must be inside [0, resolution)")

    return bounds.min_array + ((idx.astype(np.float64) + 0.5) / resolution) * bounds.span


def jitter_queries(
    queries: np.ndarray,
    resolution: int,
    bounds: Bounds3D = Bounds3D(),
    seed: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Jitter voxel-center queries by at most half a voxel per axis.

    LATTICE uses jittered active-voxel queries during training to avoid a brittle
    lattice memorization problem. Keeping this helper explicit makes that data
    contract testable before we wire a learned decoder around it.
    """

    resolution = _validate_resolution(resolution)
    q = np.asarray(queries, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 3:
        raise ValueError(f"queries must have shape (N, 3), got {q.shape}")

    half_cell = bounds.span / (2.0 * resolution)
    offsets = _rng(seed).uniform(-1.0, 1.0, size=q.shape) * half_cell
    return q + offsets
