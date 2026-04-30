"""Structured DualPrim initialization from calibrated multi-view masks.

This is a paper-driven augmented alternative to blind random placement:
build a coarse visual hull from the same RGB/mask/camera supervision
used by the renderer, split that occupied volume into local regions,
and seed positive superquadrics from those boxes. It is intentionally
not paper-parity initialization.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from clearmesh.dualprim.params import DualPrimConfig
from clearmesh.dualprim.types import (
    DualPrimScene,
    IDX_ALPHA,
    IDX_COLOR,
    IDX_NSQ_ROTATION,
    IDX_NSQ_SCALE,
    IDX_NSQ_SHAPE,
    IDX_NSQ_TRANSLATION,
    IDX_PSQ_ROTATION,
    IDX_PSQ_SCALE,
    IDX_PSQ_SHAPE,
    IDX_PSQ_TRANSLATION,
    IDX_THETA,
)


def apply_visual_hull_init(
    scene: DualPrimScene,
    views_dir: str | Path,
    config: DualPrimConfig,
    *,
    grid_res: int = 48,
    min_points_per_region: int = 24,
    scale_margin: float = 1.15,
    region_method: str | None = None,
) -> dict:
    """Seed primitive locations/scales from a visual hull.

    The initializer uses only ``views.json`` plus ``*_mask.png`` files,
    so it is valid for both synthetic canaries and real paper-mode
    multi-view inputs. It does not change the loss; it only replaces
    the initial primitive soup with local AABB proposals.
    """

    views_dir = Path(views_dir)
    occupancy = _visual_hull_occupancy(views_dir, grid_res=grid_res)
    points = _occupancy_points(occupancy)
    method = region_method or getattr(config, "visual_hull_region_method", "recursive")
    if method == "watershed":
        regions = _watershed_regions(occupancy, scene.K)
        if len(regions) < scene.K:
            regions = _split_regions_to_count(
                regions, scene.K, min_points_per_region=min_points_per_region,
            )
    elif method == "recursive":
        regions = _split_regions(points, scene.K, min_points_per_region=min_points_per_region)
    else:
        raise ValueError(f"unknown visual hull region method: {method}")
    if len(regions) == 0:
        raise ValueError(f"{views_dir}: visual hull is empty; check masks/cameras")
    regions = sorted(regions, key=len, reverse=True)

    rows, anchors, scales, active = _regions_to_rows(
        regions, scene.K, config, scale_margin=scale_margin,
    )
    rows_t = torch.tensor(rows, dtype=scene.params.dtype, device=scene.params.device)
    anchors_t = torch.tensor(anchors, dtype=scene.params.dtype, device=scene.params.device)
    scales_t = torch.tensor(scales, dtype=scene.params.dtype, device=scene.params.device)
    active_t = torch.tensor(active, dtype=torch.bool, device=scene.params.device)
    with torch.no_grad():
        scene.params[: rows_t.shape[0]] = rows_t
        scene.alive[:] = rows_t[:, IDX_ALPHA] > 0.0
        active_start = int(getattr(config, "visual_hull_active_start", 0) or 0)
        if active_start > 0:
            active_start = min(active_start, int(active_t.sum().item()))
            queued = torch.arange(scene.K, device=scene.params.device) >= active_start
            queued = queued & active_t
            scene.params[queued, IDX_ALPHA] = 0.0
            scene.alive[queued] = False
    scene.region_anchors = anchors_t
    scene.region_scales = scales_t
    scene.region_active = active_t

    return {
        "num_regions": len(regions),
        "num_occupied_voxels": int(points.shape[0]),
        "grid_res": int(grid_res),
        "scale_margin": float(scale_margin),
        "region_method": method,
        "num_initially_alive": int(scene.alive.sum().item()),
    }


def activate_visual_hull_regions(
    scene: DualPrimScene,
    count: int,
    *,
    scores: torch.Tensor | None = None,
    min_score: float | None = None,
    return_details: bool = False,
) -> int | tuple[int, list[int], list[float]]:
    """Activate the next queued visual-hull primitive slots.

    This is the cheap, differentiable-rendering equivalent of
    residual primitive birth: rows are initialized up front from local
    visual-hull supports, but only a subset competes initially.
    """

    region_active = getattr(scene, "region_active", None)
    if region_active is None or count <= 0:
        return (0, [], []) if return_details else 0
    valid = region_active.to(scene.alive.device)
    queued = valid & ~scene.alive
    idx = queued.nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return (0, [], []) if return_details else 0
    if scores is not None:
        score_values = scores.detach().to(device=scene.alive.device, dtype=scene.params.dtype)
        if score_values.numel() != scene.K:
            raise ValueError(f"expected {scene.K} visual-hull birth scores, got {score_values.numel()}")
        idx_scores = score_values[idx]
        if min_score is not None:
            keep = idx_scores >= float(min_score)
            idx = idx[keep]
            idx_scores = idx_scores[keep]
            if idx.numel() == 0:
                return (0, [], []) if return_details else 0
        order = torch.argsort(idx_scores, descending=True)
        idx = idx[order]
        idx_scores = idx_scores[order]
    else:
        idx_scores = torch.zeros_like(idx, dtype=scene.params.dtype)
    idx = idx[: int(count)]
    idx_scores = idx_scores[: int(count)]
    with torch.no_grad():
        scene.alive[idx] = True
        scene.params[idx, IDX_ALPHA] = 1.0
    details = (
        int(idx.numel()),
        [int(i) for i in idx.detach().cpu().tolist()],
        [float(s) for s in idx_scores.detach().cpu().tolist()],
    )
    return details if return_details else details[0]


def _visual_hull_occupancy(views_dir: Path, *, grid_res: int) -> np.ndarray:
    with open(views_dir / "views.json") as f:
        meta = json.load(f)
    views = meta["views"]
    resolution = int(meta["camera"]["resolution"])
    yfov = float(meta["camera"]["yfov_rad"])
    fx = fy = 0.5 * resolution / math.tan(yfov / 2.0)

    masks = []
    poses = []
    for i, view in enumerate(views):
        mask_path = views_dir / f"{i:02d}_mask.png"
        if not mask_path.exists():
            raise FileNotFoundError(mask_path)
        mask = np.asarray(Image.open(mask_path).convert("L")) > 127
        masks.append(mask)
        poses.append(np.asarray(view["pose_world_from_camera"], dtype=np.float32))
    masks = np.stack(masks, axis=0)
    poses = np.stack(poses, axis=0)

    lin = np.linspace(-1.0, 1.0, grid_res, dtype=np.float32)
    gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
    points = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)
    keep = np.ones(points.shape[0], dtype=bool)

    for v, pose in enumerate(poses):
        rot = pose[:3, :3]
        origin = pose[:3, 3]
        pc = (points - origin) @ rot
        z = pc[:, 2]
        in_front = z < -1e-4
        x = (pc[:, 0] / np.clip(-z, 1e-6, None)) * fx + resolution / 2.0
        y = -(pc[:, 1] / np.clip(-z, 1e-6, None)) * fy + resolution / 2.0
        xi = np.rint(x).astype(np.int32)
        yi = np.rint(y).astype(np.int32)
        in_frame = (
            in_front
            & (xi >= 0) & (xi < resolution)
            & (yi >= 0) & (yi < resolution)
        )
        hit = np.zeros(points.shape[0], dtype=bool)
        valid = np.flatnonzero(in_frame)
        hit[valid] = masks[v, yi[valid], xi[valid]]
        keep &= hit
        if not keep.any():
            break
    return keep.reshape(grid_res, grid_res, grid_res)


def _occupancy_points(occupancy: np.ndarray) -> np.ndarray:
    grid_res = int(occupancy.shape[0])
    lin = np.linspace(-1.0, 1.0, grid_res, dtype=np.float32)
    coords = np.argwhere(occupancy)
    if coords.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return lin[coords]


def _watershed_regions(occupancy: np.ndarray, target_count: int) -> list[np.ndarray]:
    if not occupancy.any():
        return []
    try:
        from scipy import ndimage
        from skimage.feature import peak_local_max
        from skimage.segmentation import watershed
    except ImportError:
        return _split_regions(
            _occupancy_points(occupancy),
            target_count,
            min_points_per_region=24,
        )

    distance = ndimage.distance_transform_edt(occupancy)
    peaks = peak_local_max(
        distance,
        labels=occupancy,
        min_distance=2,
        exclude_border=False,
        num_peaks=max(target_count * 2, target_count),
    )
    markers = np.zeros_like(distance, dtype=np.int32)
    if len(peaks) == 0:
        seed = np.unravel_index(int(np.argmax(distance)), distance.shape)
        peaks = np.asarray([seed])
    for i, p in enumerate(peaks, start=1):
        markers[tuple(p)] = i
    labels = watershed(-distance, markers, mask=occupancy)
    lin = np.linspace(-1.0, 1.0, occupancy.shape[0], dtype=np.float32)
    regions: list[np.ndarray] = []
    for label in range(1, labels.max() + 1):
        coords = np.argwhere(labels == label)
        if len(coords) == 0:
            continue
        regions.append(lin[coords])
    return sorted(regions, key=len, reverse=True)[:target_count]


def _split_regions_to_count(
    regions: list[np.ndarray],
    target_count: int,
    *,
    min_points_per_region: int,
) -> list[np.ndarray]:
    regions = list(regions)
    while len(regions) < target_count:
        splittable = [
            i for i, r in enumerate(regions)
            if len(r) >= max(2 * min_points_per_region, 2)
        ]
        if not splittable:
            break
        idx = max(splittable, key=lambda i: len(regions[i]))
        region = regions.pop(idx)
        regions.extend(
            _split_regions(region, 2, min_points_per_region=min_points_per_region)
        )
    return regions


def _split_regions(
    points: np.ndarray,
    target_count: int,
    *,
    min_points_per_region: int,
) -> list[np.ndarray]:
    regions = [points]
    while len(regions) < target_count:
        sizes = np.array([len(r) for r in regions])
        splittable = [
            i for i, r in enumerate(regions)
            if len(r) >= max(2 * min_points_per_region, 2)
        ]
        if not splittable:
            break
        idx = max(splittable, key=lambda i: sizes[i])
        region = regions.pop(idx)
        extents = region.max(axis=0) - region.min(axis=0)
        axis = int(np.argmax(extents))
        order = np.argsort(region[:, axis])
        mid = len(order) // 2
        left = region[order[:mid]]
        right = region[order[mid:]]
        if len(left) == 0 or len(right) == 0:
            regions.append(region)
            break
        regions.extend([left, right])
    return regions


def _regions_to_rows(
    regions: list[np.ndarray],
    K: int,
    config: DualPrimConfig,
    *,
    scale_margin: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from clearmesh.dualprim.types import DUAL_PRIM_DIM

    rows = np.zeros((K, DUAL_PRIM_DIM), dtype=np.float32)
    anchors = np.zeros((K, 3), dtype=np.float32)
    scales = np.ones((K, 3), dtype=np.float32)
    active = np.zeros((K,), dtype=bool)
    s_lo, s_hi = config.scale_range
    shape_lo, shape_hi = config.shape_range
    n = min(K, len(regions))
    for i, region in enumerate(regions[:n]):
        lo = region.min(axis=0)
        hi = region.max(axis=0)
        center = (lo + hi) * 0.5
        scale = np.maximum((hi - lo) * 0.5 * scale_margin, s_lo)
        scale = np.clip(scale, s_lo, s_hi)

        rows[i, IDX_PSQ_SCALE] = scale
        nsq_init = str(getattr(config, "visual_hull_nsq_init", "knife"))
        nsq_scale_fraction = float(getattr(config, "visual_hull_nsq_scale_fraction", 0.70))
        nsq_scale = np.clip(scale * nsq_scale_fraction, s_lo, s_hi)
        nsq_center = center.copy()
        if nsq_init == "knife":
            # Centered NSQs make hollow blobs. Offset them toward a region
            # face so they begin as Boolean cutters, matching DualPrim's
            # visible crescent-like early parts in the paper video.
            axis = int(i % 3)
            sign = -1.0 if ((i // 3) % 2) else 1.0
            offset_fraction = float(getattr(config, "visual_hull_nsq_offset_fraction", 0.45))
            nsq_center[axis] += sign * scale[axis] * offset_fraction
        elif nsq_init != "centered":
            raise ValueError(f"unknown visual_hull_nsq_init: {nsq_init}")
        rows[i, IDX_NSQ_SCALE] = nsq_scale
        rows[i, IDX_PSQ_SHAPE] = np.clip([0.35, 0.35], shape_lo, shape_hi)
        rows[i, IDX_NSQ_SHAPE] = np.clip([0.6, 0.6], shape_lo, shape_hi)
        rows[i, IDX_ALPHA] = 1.0
        rows[i, IDX_THETA] = 0.5
        rows[i, IDX_PSQ_TRANSLATION] = np.clip(center, *config.translation_range)
        rows[i, IDX_NSQ_TRANSLATION] = np.clip(nsq_center, *config.translation_range)
        rows[i, IDX_PSQ_ROTATION] = 0.0
        rows[i, IDX_NSQ_ROTATION] = 0.0
        rows[i, IDX_COLOR] = 0.5
        anchors[i] = np.clip(center, *config.translation_range)
        scales[i] = np.maximum((hi - lo) * 0.5, s_lo)
        active[i] = True

    if n < K:
        rows[n:] = rows[:1]
        rows[n:, IDX_ALPHA] = 0.0
        anchors[n:] = anchors[:1]
        scales[n:] = scales[:1]
    return rows, anchors, scales, active
