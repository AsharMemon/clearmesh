"""Symmetric Chamfer distance between two meshes.

Used for paper comparison: paper Table 2 reports CD↓ on a per-object
benchmark. Their "Ours" (DualPrim) line is 7.94. We want to be at or
below that.

Both meshes are normalised to fit in the unit cube before sampling
to give a stable Chamfer regardless of input scale.

Usage:
    python scripts/dualprim/eval_chamfer.py \\
        --ref /workspace/test_box_hole.glb \\
        --pred /workspace/dualprim_quality/refit.glb \\
        --n-samples 30000
"""

from __future__ import annotations

import argparse

import numpy as np
import trimesh


def normalise_unit_cube(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    m = mesh.copy()
    m.vertices -= m.centroid
    s = m.extents.max()
    if s > 0:
        m.vertices /= s
    return m


def chamfer_l2(a_pts: np.ndarray, b_pts: np.ndarray) -> tuple[float, float, float]:
    """Symmetric L2 Chamfer in original distance units.

    CD(A, B) = mean_{a in A} min_{b in B} ||a-b||_2 + mean_{b in B} min_{a in A} ||b-a||_2
    """
    from scipy.spatial import cKDTree
    tree_a = cKDTree(a_pts)
    tree_b = cKDTree(b_pts)
    d_ab, _ = tree_b.query(a_pts, k=1)
    d_ba, _ = tree_a.query(b_pts, k=1)
    return float(d_ab.mean()), float(d_ba.mean()), float(d_ab.mean() + d_ba.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="Ground-truth mesh")
    ap.add_argument("--pred", required=True, help="Predicted mesh")
    ap.add_argument("--n-samples", type=int, default=30000)
    ap.add_argument("--scale-factor", type=float, default=1000.0,
                    help="Multiply Chamfer by this for paper-style units (paper uses CD x 1000)")
    args = ap.parse_args()

    ref = trimesh.load(args.ref, force="mesh")
    pred = trimesh.load(args.pred, force="mesh")
    ref = normalise_unit_cube(ref)
    pred = normalise_unit_cube(pred)

    rng = np.random.default_rng(0)
    np.random.seed(0)
    a_pts, _ = trimesh.sample.sample_surface(ref, args.n_samples)
    b_pts, _ = trimesh.sample.sample_surface(pred, args.n_samples)

    d_ab, d_ba, cd = chamfer_l2(np.asarray(a_pts), np.asarray(b_pts))

    print(f"REF:  {args.ref}")
    print(f"      {len(ref.vertices):,}v / {len(ref.faces):,}f")
    print(f"PRED: {args.pred}")
    print(f"      {len(pred.vertices):,}v / {len(pred.faces):,}f")
    print()
    print(f"Symmetric Chamfer (raw):       {cd:.6f}")
    print(f"Symmetric Chamfer x{int(args.scale_factor)}:    {cd * args.scale_factor:.4f}")
    print(f"  pred -> ref:  {d_ab * args.scale_factor:.4f}")
    print(f"  ref -> pred:  {d_ba * args.scale_factor:.4f}")
    print()
    print(f"Compactness:")
    print(f"  pred verts:  {len(pred.vertices):,}  (paper Ours: ~1,540)")
    print(f"  pred faces:  {len(pred.faces):,}  (paper Ours: ~790)")
    print()
    paper_cd = 7.94
    print(f"Paper DualPrim CD x1000 = {paper_cd}")
    cd_scaled = cd * args.scale_factor
    delta_pct = (cd_scaled - paper_cd) / paper_cd * 100
    sign = "WORSE" if cd_scaled > paper_cd else "BETTER"
    print(f"Our CD x1000 = {cd_scaled:.4f} ({sign} by {abs(delta_pct):.1f}%)")


if __name__ == "__main__":
    main()
