"""Hole-specific metrics for DualPrim quality evaluation.

Friend's framing: Chamfer is too coarse — a scattered-blob mesh that
roughly fills the bounding box can have moderate CD while completely
missing the hole. We need a metric that asks "did the model learn
the hole?" directly.

Three metrics computed:

  mask IoU
    Per-view binary IoU between rendered-pred-mask and rendered-ref-mask.
    Tells us how well the silhouette matches.

  hole-area error
    Per-view |pixels-where-ref-is-bg-AND-pred-is-fg| (false-positive
    fill of the hole). High = the pred filled the hole that should
    be empty.

  through-hole open %
    Of the views where the ref shows a clear hole (background pixels
    inside the projected silhouette), what fraction of those views
    has the pred ALSO showing a hole there. 100% = pred preserves the
    hole everywhere; 0% = pred fills it everywhere.

Usage:
    python scripts/dualprim/hole_metric.py \\
        --ref /workspace/test_box_hole.glb \\
        --pred /workspace/dualprim_quality/refit.glb \\
        --resolution 192
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

# Path setup
for _r in ("/workspace/clearmesh", "/root/clearmesh",
           str(Path(__file__).resolve().parents[2])):
    if os.path.isdir(_r) and _r not in sys.path:
        sys.path.insert(0, _r)

import numpy as np
import trimesh

# Lazy import pyrender at function scope
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


def render_mask(mesh: trimesh.Trimesh, pose: np.ndarray, resolution: int, yfov: float) -> np.ndarray:
    """Return a binary uint8 mask of the mesh from the given camera pose."""
    import pyrender
    sc = pyrender.Scene(ambient_light=(0.0, 0.0, 0.0), bg_color=(0, 0, 0, 0))
    sc.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))
    cam = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=1.0)
    sc.add(cam, pose=pose)
    rr = pyrender.OffscreenRenderer(resolution, resolution)
    _, depth = rr.render(sc)
    rr.delete()
    return (depth > 0).astype(np.uint8)


def fibonacci_sphere(n: int) -> list:
    points = []
    phi_g = math.pi * (math.sqrt(5.0) - 1.0)
    for i in range(n):
        y = 1.0 - ((i + 0.5) / float(n)) * 2.0
        r = math.sqrt(max(1.0 - y * y, 0.0))
        theta = phi_g * i
        x = math.cos(theta) * r
        z = math.sin(theta) * r
        points.append((x, y, z))
    return points


def detect_hole_axis(ref_mesh: trimesh.Trimesh, probe_res: int = 256) -> int:
    """Auto-detect which axis (0=X, 1=Y, 2=Z) has the dominant through-hole.

    Rationale: for each axis, render the mesh silhouette looking along
    that axis and count "hole pixels" = (binary_fill_holes(mask) AND
    NOT mask). The axis with the highest hole-pixel count is the
    through-hole axis. This beats fibonacci_sphere sampling which
    might spend <10% of views looking down the axis that actually
    exposes the hole.
    """
    from scipy.ndimage import binary_fill_holes
    yfov = math.radians(40.0)
    best_axis, best_count = 0, -1
    for axis in range(3):
        d = [0.0, 0.0, 0.0]; d[axis] = 1.0
        pose = camera_pose(d, distance=2.0)
        mask = render_mask(ref_mesh, pose, probe_res, yfov)
        filled = binary_fill_holes(mask > 0).astype(np.uint8)
        hole_pixels = int(((filled > 0) & (mask == 0)).sum())
        if hole_pixels > best_count:
            best_count, best_axis = hole_pixels, axis
    return best_axis


def hole_axis_views(hole_axis: int, n_ring: int = 12, tilt_deg: float = 15.0) -> list:
    """Generate camera directions clustered around a hole axis.

    Returns `n_ring` directions on a small tilted-cone around the
    +axis direction, plus `n_ring` on a cone around the -axis
    direction. Total = 2*n_ring views, all of which see through the
    hole. Catches the hole visually and gives us statistical power
    (12-24 views instead of 2 in fibonacci_sphere).

    The tilt is small enough that the through-hole is still visible
    but large enough to get view diversity — prevents degenerate
    "all views look identical" collapse.
    """
    tilt = math.radians(tilt_deg)
    dirs = []
    for sign in (1.0, -1.0):
        for i in range(n_ring):
            phi = 2.0 * math.pi * i / n_ring
            # On a cone of half-angle `tilt` around axis `hole_axis`:
            # principal component = sign * cos(tilt), perpendicular = sin(tilt) * (cos(phi), sin(phi))
            d = [0.0, 0.0, 0.0]
            d[hole_axis] = sign * math.cos(tilt)
            # Fill the two perpendicular axes
            perp_axes = [a for a in range(3) if a != hole_axis]
            d[perp_axes[0]] = math.sin(tilt) * math.cos(phi)
            d[perp_axes[1]] = math.sin(tilt) * math.sin(phi)
            dirs.append(tuple(d))
    return dirs


def camera_pose(direction, distance=2.0):
    direction = np.asarray(direction, dtype=np.float32)
    direction = direction / np.linalg.norm(direction)
    eye = direction * distance
    forward = -direction
    up_world = np.array([0.0, 1.0, 0.0])
    if abs(forward @ up_world) > 0.99:
        up_world = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, up_world); right /= np.linalg.norm(right)
    up = np.cross(right, forward); up /= np.linalg.norm(up)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 0] = right; pose[:3, 1] = up; pose[:3, 2] = -forward
    pose[:3, 3] = eye
    return pose


def normalize(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    m = mesh.copy()
    m.vertices -= m.centroid
    s = m.extents.max()
    if s > 0:
        m.vertices /= s
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--resolution", type=int, default=192)
    ap.add_argument("--n-views", type=int, default=24,
                    help="Sphere views for mask-IoU coverage")
    ap.add_argument("--yfov-deg", type=float, default=40.0)
    ap.add_argument("--hole-axis", default="auto",
                    choices=["auto", "x", "y", "z", "none"],
                    help="Which axis exposes the through-hole. 'auto' "
                         "detects by rendering the ref along each axis "
                         "and picking the one with most hole pixels. "
                         "'none' reverts to pure fibonacci sphere.")
    ap.add_argument("--n-hole-views", type=int, default=12,
                    help="Per-side view count for the hole-axis ring. "
                         "Total hole-ring = 2 * n_hole_views.")
    ap.add_argument("--hole-tilt-deg", type=float, default=15.0,
                    help="Half-angle of the hole-ring cone. Small = "
                         "lots of through-hole visibility, big = more "
                         "view diversity. 15° is a reasonable compromise.")
    args = ap.parse_args()

    ref = normalize(trimesh.load(args.ref, force="mesh"))
    pred = normalize(trimesh.load(args.pred, force="mesh"))
    ref.fix_normals(); pred.fix_normals()

    yfov = math.radians(args.yfov_deg)
    res = args.resolution

    # Build the view set: a fibonacci sphere for IoU coverage PLUS a
    # hole-axis ring for statistically-reliable through-hole measurement.
    # With 2/24 hole-visible views (phase 1 signal) the metric had
    # catastrophic variance; with 24 hole-visible views it's usable.
    sphere_dirs = fibonacci_sphere(args.n_views)
    if args.hole_axis == "auto":
        hole_axis = detect_hole_axis(ref, probe_res=res)
        print(f"[hole_metric] auto-detected hole axis: "
              f"{'XYZ'[hole_axis]}")
    elif args.hole_axis == "none":
        hole_axis = None
    else:
        hole_axis = {"x": 0, "y": 1, "z": 2}[args.hole_axis]

    ring_dirs = []
    if hole_axis is not None:
        ring_dirs = hole_axis_views(
            hole_axis, n_ring=args.n_hole_views,
            tilt_deg=args.hole_tilt_deg,
        )
        print(f"[hole_metric] sphere views: {len(sphere_dirs)}, "
              f"hole-axis ring views: {len(ring_dirs)}")
    # The sphere views and ring views serve different purposes:
    # sphere = IoU signal, ring = hole-preservation signal. We compute
    # IoU across BOTH (more is better) and hole-recall only across the
    # RING (where ref-hole is actually visible).
    directions = sphere_dirs + ring_dirs
    n_sphere = len(sphere_dirs)

    iou_per_view = []
    hole_pixels_per_view = []
    pred_hole_pixels_per_view = []

    for i, d in enumerate(directions):
        pose = camera_pose(d, distance=2.0)
        ref_mask = render_mask(ref, pose, res, yfov)
        pred_mask = render_mask(pred, pose, res, yfov)

        # IoU
        inter = ((ref_mask > 0) & (pred_mask > 0)).sum()
        union = ((ref_mask > 0) | (pred_mask > 0)).sum()
        iou = inter / max(union, 1)
        iou_per_view.append(float(iou))

        # Hole detection: a "hole" pixel is one INSIDE the convex hull
        # of the ref silhouette but OUTSIDE the ref mask itself.
        # cv2.fillPoly would be the proper convex-hull fill; cheap proxy:
        # use scipy binary_fill_holes then XOR with original.
        from scipy.ndimage import binary_fill_holes
        ref_filled = binary_fill_holes(ref_mask > 0).astype(np.uint8)
        ref_holes = (ref_filled > 0) & (ref_mask == 0)        # holes IN ref
        pred_filled = binary_fill_holes(pred_mask > 0).astype(np.uint8)
        pred_holes = (pred_filled > 0) & (pred_mask == 0)     # holes IN pred

        ref_hole_count = int(ref_holes.sum())
        # Of the ref's hole pixels, how many are also "hole" in pred
        # (i.e. pred preserved the carve at that pixel)?
        if ref_hole_count > 0:
            preserved = int((ref_holes & pred_holes).sum())
        else:
            preserved = 0
        hole_pixels_per_view.append(ref_hole_count)
        pred_hole_pixels_per_view.append(preserved)

    iou_per_view = np.array(iou_per_view)
    hole_pixels_per_view = np.array(hole_pixels_per_view)
    pred_hole_pixels_per_view = np.array(pred_hole_pixels_per_view)

    # Aggregate metrics
    mean_iou = float(iou_per_view.mean())
    # Through-hole-open % — only count views where ref ACTUALLY shows a hole
    has_hole_view = hole_pixels_per_view > 5    # ignore tiny blips
    n_hole_views = int(has_hole_view.sum())
    if n_hole_views > 0:
        # Fraction of ref-hole-pixels that were preserved as hole in pred
        hole_recall_per_view = pred_hole_pixels_per_view / np.maximum(hole_pixels_per_view, 1)
        hole_recall_per_view = hole_recall_per_view[has_hole_view]
        mean_hole_recall = float(hole_recall_per_view.mean())
        n_views_open = int((hole_recall_per_view > 0.3).sum())
        pct_open = 100.0 * n_views_open / n_hole_views
    else:
        mean_hole_recall = 0.0
        pct_open = 0.0
        n_views_open = 0

    # Also compute sphere-only and ring-only breakdowns so we can
    # see if the ring-view signal is meaningfully different from
    # sphere coverage (it should be, if the hole axis is real).
    if n_sphere > 0 and n_sphere < len(directions):
        iou_sphere = float(iou_per_view[:n_sphere].mean())
        iou_ring = float(iou_per_view[n_sphere:].mean())
        ring_mask = has_hole_view[n_sphere:]
        n_ring_hole = int(ring_mask.sum())
        if n_ring_hole > 0:
            ring_recall = (pred_hole_pixels_per_view[n_sphere:][ring_mask]
                           / np.maximum(hole_pixels_per_view[n_sphere:][ring_mask], 1))
            ring_open_pct = 100.0 * (ring_recall > 0.3).sum() / n_ring_hole
            ring_mean_recall = float(ring_recall.mean())
        else:
            ring_open_pct = 0.0
            ring_mean_recall = 0.0
    else:
        iou_sphere = iou_ring = None
        ring_open_pct = ring_mean_recall = None
        n_ring_hole = 0

    print(f"REF:  {args.ref}  ({len(ref.vertices):,}v)")
    print(f"PRED: {args.pred}  ({len(pred.vertices):,}v)")
    print(f"Views evaluated: {len(directions)} "
          f"({n_sphere} sphere + {len(directions) - n_sphere} hole-ring)")
    print()
    print(f"Mean mask IoU (all views):      {mean_iou:.4f}")
    if iou_sphere is not None:
        print(f"  sphere-only IoU:              {iou_sphere:.4f}")
        print(f"  hole-ring-only IoU:           {iou_ring:.4f}")
    print(f"Views with hole in ref:         {n_hole_views}/{len(directions)}")
    print(f"Mean hole recall (all hole-views): {mean_hole_recall:.4f}  "
          f"(1.0 = pred carved every ref-hole pixel)")
    print(f"Through-hole open % (all):      {pct_open:.1f}%  "
          f"({n_views_open}/{n_hole_views} views with recall > 30%)")
    if ring_open_pct is not None:
        print(f"  ring-only open %:             {ring_open_pct:.1f}%  "
              f"(n={n_ring_hole} hole-visible ring views)")
        print(f"  ring-only mean recall:        {ring_mean_recall:.4f}")


if __name__ == "__main__":
    main()
