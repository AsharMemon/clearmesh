"""Irregular point-patch sampling inspired by 3DILG-style local tokens."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class IrregularPatches:
    """Patch anchors and KNN membership over an unordered point cloud."""

    anchor_indices: np.ndarray
    patch_indices: np.ndarray

    @property
    def anchor_count(self) -> int:
        return int(self.anchor_indices.shape[0])

    @property
    def patch_size(self) -> int:
        if self.patch_indices.ndim != 2:
            return 0
        return int(self.patch_indices.shape[1])


def farthest_point_indices(
    points: np.ndarray,
    count: int,
    seed: int | None = 0,
) -> np.ndarray:
    """Dependency-light farthest-point sampling for small/medium clouds."""

    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {pts.shape}")
    if len(pts) == 0:
        raise ValueError("points must not be empty")

    count = int(count)
    if count <= 0:
        raise ValueError(f"count must be positive, got {count}")
    if count >= len(pts):
        return np.arange(len(pts), dtype=np.int64)

    rng = np.random.default_rng(seed)
    selected = np.empty(count, dtype=np.int64)
    selected[0] = int(rng.integers(0, len(pts)))
    min_dist2 = np.full(len(pts), np.inf, dtype=np.float64)

    for i in range(1, count):
        current = pts[selected[i - 1]]
        dist2 = np.sum((pts - current) ** 2, axis=1)
        min_dist2 = np.minimum(min_dist2, dist2)
        selected[i] = int(np.argmax(min_dist2))

    return selected


def knn_patch_indices(points: np.ndarray, centers: np.ndarray, k: int) -> np.ndarray:
    """Return KNN indices for each center point."""

    pts = np.asarray(points, dtype=np.float64)
    ctr = np.asarray(centers, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {pts.shape}")
    if ctr.ndim != 2 or ctr.shape[1] != 3:
        raise ValueError(f"centers must have shape (M, 3), got {ctr.shape}")
    if len(pts) == 0:
        raise ValueError("points must not be empty")

    k = min(int(k), len(pts))
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    _, indices = cKDTree(pts).query(ctr, k=k)
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim == 1:
        indices = indices[:, None]
    return indices


def build_irregular_patches(
    points: np.ndarray,
    anchor_count: int = 512,
    patch_size: int = 32,
    seed: int | None = 0,
) -> IrregularPatches:
    """Create FPS anchors and local KNN patches for unordered point features."""

    pts = np.asarray(points, dtype=np.float64)
    anchor_indices = farthest_point_indices(pts, anchor_count, seed=seed)
    patch_indices = knn_patch_indices(pts, pts[anchor_indices], patch_size)
    return IrregularPatches(anchor_indices=anchor_indices, patch_indices=patch_indices)
