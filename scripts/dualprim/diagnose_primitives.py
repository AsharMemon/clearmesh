"""Diagnose a DualPrim primitives.json: is any NSQ actually carving?

Answers these questions from a saved primitives.json:
  - How many primitives survived pruning (alive after α + scale)?
  - How many of the survivors have a meaningfully-sized NSQ?
  - For each primitive, is the NSQ center INSIDE the PSQ bbox?
    (if not, it can't carve the PSQ)
  - Which primitive pairs are positioned near a target axis/location
    (e.g. the hole axis)?

Run post-training, no GPU required. Output is a compact table + a
summary paragraph with "is any primitive configured to carve the
target region?" verdict.

Usage:
    python scripts/dualprim/diagnose_primitives.py \\
        --primitives /workspace/dualprim_phase2/hole/primitives.json \\
        --ref /workspace/test_box_hole.glb \\
        --hole-axis auto
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


def load_prims(path: str) -> list[dict]:
    data = json.load(open(path))
    return data.get("primitives", [])


def diagnose_one(p: dict) -> dict:
    """Per-primitive analysis."""
    psq_scale = np.array(p["psq_scale"])
    nsq_scale = np.array(p["nsq_scale"])
    psq_tr = np.array(p["psq_translation"])
    nsq_tr = np.array(p["nsq_translation"])

    # NSQ-center-in-PSQ-bbox: centroid-to-centroid distance vs PSQ half-size
    offset = nsq_tr - psq_tr
    psq_half = psq_scale  # conservative: PSQ half-extents along axes
    # Component-wise: is each offset within PSQ's extent along that axis?
    inside = (np.abs(offset) <= psq_half).all()
    # How deeply is the NSQ center inside? (1.0 = coincident, 0.0 = at boundary)
    depth = float(1.0 - (np.abs(offset) / np.maximum(psq_half, 1e-6)).max())

    return {
        "alpha": float(p["alpha"]),
        "psq_scale_norm": float(np.linalg.norm(psq_scale)),
        "nsq_scale_norm": float(np.linalg.norm(nsq_scale)),
        "psq_tr": psq_tr.tolist(),
        "nsq_tr": nsq_tr.tolist(),
        "offset_norm": float(np.linalg.norm(offset)),
        "nsq_inside_psq": bool(inside),
        "nsq_penetration_depth": depth,
        "theta": float(p["theta"]),
    }


def carves_significantly(d: dict, *,
                          min_alpha: float = 0.5,
                          min_nsq_scale: float = 0.03,
                          min_penetration: float = 0.2) -> bool:
    """Would this primitive actually carve anything visible?

    Conservative: all three must hold. A primitive with α=0.5 but
    NSQ scale = 1e-4 does not carve. One with α=0.9 and NSQ inside
    but at shallow penetration barely does.
    """
    return (d["alpha"] >= min_alpha
            and d["nsq_scale_norm"] >= min_nsq_scale
            and d["nsq_penetration_depth"] >= min_penetration)


def detect_hole_axis_from_mesh(ref_path: str) -> int:
    """Return 0/1/2 for X/Y/Z as the hole axis (maximum through-hole
    pixel count via silhouette along that axis).

    Reuse the same logic as hole_metric.py (duplicated to keep this
    script zero-dependency on pyrender if the user runs it without
    --ref). If pyrender/egl isn't available, return None.
    """
    try:
        import os
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        import trimesh
        import pyrender
        from scipy.ndimage import binary_fill_holes
        m = trimesh.load(ref_path, force="mesh").copy()
        m.vertices -= m.centroid
        s = m.extents.max()
        if s > 0: m.vertices /= s
        yfov = math.radians(40.0)
        best_axis, best_count = 0, -1
        for axis in range(3):
            d = np.zeros(3); d[axis] = 1.0
            eye = d * 2.0; fwd = -d
            uw = np.array([0, 1, 0], dtype=np.float32)
            if abs(fwd @ uw) > 0.99: uw = np.array([0, 0, 1], dtype=np.float32)
            r = np.cross(fwd, uw); r /= np.linalg.norm(r)
            u = np.cross(r, fwd); u /= np.linalg.norm(u)
            pose = np.eye(4, dtype=np.float32)
            pose[:3, 0] = r; pose[:3, 1] = u; pose[:3, 2] = -fwd; pose[:3, 3] = eye
            sc = pyrender.Scene(ambient_light=(0, 0, 0), bg_color=(0, 0, 0, 0))
            sc.add(pyrender.Mesh.from_trimesh(m, smooth=False))
            sc.add(pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=1.0), pose=pose)
            rr = pyrender.OffscreenRenderer(256, 256)
            _, depth = rr.render(sc); rr.delete()
            mask = (depth > 0)
            filled = binary_fill_holes(mask)
            hole_pixels = int((filled & ~mask).sum())
            if hole_pixels > best_count:
                best_count, best_axis = hole_pixels, axis
        return best_axis
    except Exception as e:
        print(f"[warn] auto hole-axis detection failed: {e}", file=sys.stderr)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--primitives", required=True)
    ap.add_argument("--ref", default=None,
                    help="Reference mesh — used to auto-detect hole axis "
                         "for the 'near target' primitive filter.")
    ap.add_argument("--hole-axis", default="auto",
                    choices=["auto", "x", "y", "z", "none"])
    ap.add_argument("--export-alpha", type=float, default=0.5,
                    help="Alpha threshold used at export (paper T_export).")
    args = ap.parse_args()

    prims = load_prims(args.primitives)
    print(f"=== Diagnostics: {args.primitives} ===")
    print(f"Total primitives in JSON: {len(prims)}")

    diags = [diagnose_one(p) for p in prims]

    alive_export = [d for d in diags if d["alpha"] >= args.export_alpha]
    print(f"Above α={args.export_alpha} (export-alive): {len(alive_export)}")

    carvers = [d for d in alive_export if carves_significantly(d)]
    print(f"Carves-significantly (α≥{args.export_alpha} AND "
          f"‖NSQ‖≥0.03 AND NSQ deeply inside PSQ): {len(carvers)}")
    print()

    # If we can determine the hole axis, filter carvers for proximity
    hole_axis = None
    if args.hole_axis == "auto" and args.ref:
        hole_axis = detect_hole_axis_from_mesh(args.ref)
        if hole_axis is not None:
            print(f"Hole axis (auto): {'XYZ'[hole_axis]}")
    elif args.hole_axis in ("x", "y", "z"):
        hole_axis = {"x": 0, "y": 1, "z": 2}[args.hole_axis]
        print(f"Hole axis (arg): {'XYZ'[hole_axis]}")

    if hole_axis is not None:
        # Primitives whose NSQ center is within 0.3 of the hole-axis
        # line (through origin). These are positioned to carve the
        # through-hole.
        near_axis = []
        for d in carvers:
            nsq = np.array(d["nsq_tr"])
            perp_axes = [a for a in range(3) if a != hole_axis]
            perp_dist = float(np.linalg.norm(nsq[perp_axes]))
            if perp_dist < 0.3:
                d["perp_dist_from_hole_axis"] = perp_dist
                near_axis.append(d)
        print(f"Carvers positioned near hole-axis (perp < 0.3): "
              f"{len(near_axis)} / {len(carvers)}")

        if near_axis:
            print()
            print("  α   | ‖PSQ‖ ‖NSQ‖ | NSQ_inside | penetration | perp_d | pos")
            print("  " + "-" * 76)
            for d in near_axis:
                print(f"  {d['alpha']:.2f} | "
                      f"{d['psq_scale_norm']:.3f}  {d['nsq_scale_norm']:.3f} | "
                      f"{'✓' if d['nsq_inside_psq'] else '✗':10s} | "
                      f"{d['nsq_penetration_depth']:.2f}        | "
                      f"{d['perp_dist_from_hole_axis']:.2f}   | "
                      f"[{d['nsq_tr'][0]:+.2f},{d['nsq_tr'][1]:+.2f},{d['nsq_tr'][2]:+.2f}]")

    print()
    print("=== Summary ===")
    alpha_vals = [d["alpha"] for d in diags]
    print(f"α distribution: "
          f"min={min(alpha_vals):.2f} "
          f"median={sorted(alpha_vals)[len(alpha_vals)//2]:.2f} "
          f"max={max(alpha_vals):.2f}")

    nsq_scales = [d["nsq_scale_norm"] for d in alive_export]
    if nsq_scales:
        print(f"NSQ scale (export-alive): "
              f"min={min(nsq_scales):.3f} "
              f"median={sorted(nsq_scales)[len(nsq_scales)//2]:.3f} "
              f"max={max(nsq_scales):.3f}")

    penetrations = [d["nsq_penetration_depth"] for d in alive_export]
    if penetrations:
        n_inside = sum(1 for d in alive_export if d["nsq_inside_psq"])
        print(f"NSQs-inside-their-PSQ: {n_inside} / {len(alive_export)}")

    # Verdict
    print()
    if not carvers:
        print("VERDICT: no primitives are configured to carve ANY"
              " geometry — supervision pressure failed to push NSQs "
              "into their PSQs at meaningful scale.")
        print("         Round-3 tuning options: higher λ_mask, "
              "longer warmup before θ-curriculum anneals, "
              "explicit NSQ-in-PSQ positional loss.")
    elif hole_axis is not None and not near_axis:
        print("VERDICT: some primitives carve something, but NONE are "
              "positioned near the hole axis. Supervision pushed NSQs "
              "around but not toward the hole.")
        print("         Likely fix: add hole-axis-biased training views "
              "so supervision gradient specifically pressures NSQs "
              "toward the through-hole.")
    else:
        print("VERDICT: primitives ARE carving near the hole axis in "
              "the field — if rendered mesh still fills the hole, the "
              "failure is in MESH EXPORT (marching-cubes res, Boolean "
              "union smoothing), not optimization.")
        print("         Likely fix: bump tessellation_resolution, "
              "inspect the raw SDF field along the hole axis.")


if __name__ == "__main__":
    main()
