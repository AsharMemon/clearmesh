"""Gate 0.1 — iter × resolution sweep on the hole canary.

User mandate: maximize quality. Friend's adjustment: don't assume
15k iters / 192px is good enough; measure. This script runs the
controlled comparison to pick the (iters, resolution) config that
delivers paper-level quality at minimum compute cost.

Design:
  - single canary (hole) — cheap, reveals topology-preservation clearly
  - iter sweep: {5000, 10000, 15000, 30000} at fixed res=192
  - res sweep: {128, 192, 256} at fixed iters=15000
  - same seed across all runs so differences are attributable to the
    varied parameter, not random init
  - measure: chamfer×1000, hole IoU, through-hole-open %, wall time
  - writes gate01_results.json + gate01_results.md report

Wall-time budget at 400ms/step:
  iter sweep: (5+10+15+30)k * 400ms = 60k * 0.4s = 24000s ≈ 6.7hrs
  res sweep:  3 * 15k * ~(r/192)^2 scaling * 400ms ≈ ~5hrs
  total:      ~12hrs on 1 GPU
  cost at A100_SXM4 $0.89/hr: ~$11

Cheaper variant via --quick: halve iter counts, skip res=256.

Usage (runs locally or on pod):
    python scripts/dualprim/gate01_sweep.py \\
        --ref /workspace/test_box_hole.glb \\
        --out /workspace/gate01_sweep \\
        --repo-root /workspace/clearmesh
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


_CANDIDATE_ROOTS = ["/workspace/clearmesh",
                    str(Path(__file__).resolve().parents[2])]
for _r in _CANDIDATE_ROOTS:
    if os.path.isdir(_r) and _r not in sys.path:
        sys.path.insert(0, _r)


def run_one(
    *, ref: str, out_dir: Path, iters: int, resolution: int,
    k: int = 30, rays: int = 768, seed: int = 0,
    fg_bias: float = 0.7, nsq_init: str = "independent",
) -> dict:
    """Launch run_canary.py as a subprocess and parse its outputs.

    Returns {ok, iters, resolution, wall_time_s, chamfer_x1000,
             hole_iou, through_hole_open_pct, hole_recall,
             final_loss_total, scene_verts, scene_faces, out_dir}.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "train.log"
    env = os.environ.copy()
    env["PYOPENGL_PLATFORM"] = "egl"
    cmd = [
        sys.executable, "-u", "scripts/dualprim/run_canary.py",
        "--input", ref,
        "--out", str(out_dir),
        "--mode", "mesh_rendered_views",
        "--k", str(k),
        "--iters", str(iters),
        "--rays", str(rays),
        "--resolution", str(resolution),
        "--seed", str(seed),
        "--nsq-init", nsq_init,
        "--fg-bias", str(fg_bias),
        "--union-export",
    ]
    print(f"\n[gate01] {out_dir.name}: iters={iters} res={resolution}")
    t0 = time.time()
    with open(log_file, "w") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
    wall = time.time() - t0
    ok = proc.returncode == 0 and (out_dir / "refit.glb").exists()
    print(f"[gate01] {out_dir.name}: done in {wall/60:.1f}min ok={ok}")

    result = {
        "ok": ok, "iters": iters, "resolution": resolution,
        "wall_time_s": wall, "out_dir": str(out_dir),
    }
    if not ok:
        return result

    # Chamfer
    try:
        from scipy.spatial import cKDTree
        import trimesh
        ref_mesh = trimesh.load(ref, force="mesh")
        pred_mesh = trimesh.load(str(out_dir / "refit.glb"), force="mesh")
        for m in (ref_mesh, pred_mesh):
            m.vertices -= m.centroid
            s = m.extents.max()
            if s > 0:
                m.vertices /= s
        ref_pts, _ = trimesh.sample.sample_surface(ref_mesh, 30000)
        pred_pts, _ = trimesh.sample.sample_surface(pred_mesh, 30000)
        d_pr, _ = cKDTree(ref_pts).query(pred_pts, k=1)
        d_rp, _ = cKDTree(pred_pts).query(ref_pts, k=1)
        result["chamfer_x1000"] = float((d_pr.mean() + d_rp.mean()) * 1000.0)
    except Exception as e:
        print(f"[gate01] chamfer failed: {e}")

    # Hole metric
    try:
        import numpy as np
        from PIL import Image
        os.environ["PYOPENGL_PLATFORM"] = "egl"
        import math
        import trimesh
        import pyrender
        from scipy.ndimage import binary_fill_holes

        def normalize(m):
            m = m.copy()
            m.vertices -= m.centroid
            s = m.extents.max()
            if s > 0:
                m.vertices /= s
            return m

        def fibonacci_sphere(n):
            pts = []
            phi_g = math.pi * (math.sqrt(5.0) - 1.0)
            for i in range(n):
                y = 1.0 - ((i + 0.5) / float(n)) * 2.0
                r = math.sqrt(max(1.0 - y * y, 0.0))
                th = phi_g * i
                pts.append((math.cos(th) * r, y, math.sin(th) * r))
            return pts

        def cam_pose(d):
            d = np.asarray(d, dtype=np.float32); d = d / np.linalg.norm(d)
            eye = d * 2.0
            fwd = -d
            uw = np.array([0, 1, 0], dtype=np.float32)
            if abs(fwd @ uw) > 0.99:
                uw = np.array([0, 0, 1], dtype=np.float32)
            r = np.cross(fwd, uw); r /= np.linalg.norm(r)
            u = np.cross(r, fwd); u /= np.linalg.norm(u)
            p = np.eye(4, dtype=np.float32)
            p[:3, 0] = r; p[:3, 1] = u; p[:3, 2] = -fwd; p[:3, 3] = eye
            return p

        def render_mask(m, pose, res, yfov):
            sc = pyrender.Scene(ambient_light=(0, 0, 0), bg_color=(0, 0, 0, 0))
            sc.add(pyrender.Mesh.from_trimesh(m, smooth=False))
            cam = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=1.0)
            sc.add(cam, pose=pose)
            rr = pyrender.OffscreenRenderer(res, res)
            _, depth = rr.render(sc); rr.delete()
            return (depth > 0).astype(np.uint8)

        ref_n = normalize(trimesh.load(ref, force="mesh"))
        pred_n = normalize(trimesh.load(str(out_dir / "refit.glb"), force="mesh"))
        ref_n.fix_normals(); pred_n.fix_normals()
        yfov = math.radians(40.0)
        dirs = fibonacci_sphere(24)
        ious, ref_hole_pix, pred_preserved = [], [], []
        for d in dirs:
            pose = cam_pose(d)
            rm = render_mask(ref_n, pose, 192, yfov)
            pm = render_mask(pred_n, pose, 192, yfov)
            inter = ((rm > 0) & (pm > 0)).sum()
            union = ((rm > 0) | (pm > 0)).sum()
            ious.append(inter / max(union, 1))
            rf = binary_fill_holes(rm > 0)
            ref_h = rf & (rm == 0)
            pf = binary_fill_holes(pm > 0)
            pred_h = pf & (pm == 0)
            ref_hole_pix.append(int(ref_h.sum()))
            pred_preserved.append(int((ref_h & pred_h).sum()))
        ious = np.array(ious)
        ref_hp = np.array(ref_hole_pix)
        pred_p = np.array(pred_preserved)
        mask = ref_hp > 5
        result["hole_iou"] = float(ious.mean())
        result["hole_views_n"] = int(mask.sum())
        if mask.sum() > 0:
            recall = pred_p[mask] / np.maximum(ref_hp[mask], 1)
            result["hole_recall"] = float(recall.mean())
            result["through_hole_open_pct"] = float(100.0 * (recall > 0.3).sum() / mask.sum())
        else:
            result["hole_recall"] = 0.0
            result["through_hole_open_pct"] = 0.0
    except Exception as e:
        print(f"[gate01] hole metric failed: {e}")

    # Mesh stats
    try:
        import trimesh
        m = trimesh.load(str(out_dir / "refit.glb"), force="mesh")
        result["scene_verts"] = len(m.vertices)
        result["scene_faces"] = len(m.faces)
    except Exception:
        pass

    return result


def write_report(results: list[dict], out_path: Path):
    lines = [
        "# Gate 0.1 — iter × resolution sweep",
        "",
        "Measure (iters, resolution) vs quality on the hole canary. "
        "Winner is the config with paper-level quality at minimum wall "
        "time. User mandate: maximize quality.",
        "",
        "## Iter sweep (res=192 fixed)",
        "",
        "| iters | wall (min) | CD×1000 | hole IoU | hole recall | through-hole open % | K alive |",
        "|---|---|---|---|---|---|---|",
    ]
    iter_runs = [r for r in results if r.get("ok") and r["resolution"] == 192]
    iter_runs.sort(key=lambda r: r["iters"])
    for r in iter_runs:
        lines.append(
            f"| {r['iters']:,} | {r['wall_time_s']/60:.1f} | "
            f"{r.get('chamfer_x1000', '—'):.2f} | "
            f"{r.get('hole_iou', '—'):.3f} | "
            f"{r.get('hole_recall', '—'):.3f} | "
            f"{r.get('through_hole_open_pct', '—'):.1f}% | "
            f"— |"
        )
    lines.append("")
    lines.append("## Resolution sweep (iters=15k fixed)")
    lines.append("")
    lines.append("| resolution | wall (min) | CD×1000 | hole IoU | hole recall | through-hole open % |")
    lines.append("|---|---|---|---|---|---|")
    res_runs = [r for r in results if r.get("ok") and r["iters"] == 15000]
    res_runs.sort(key=lambda r: r["resolution"])
    for r in res_runs:
        lines.append(
            f"| {r['resolution']} | {r['wall_time_s']/60:.1f} | "
            f"{r.get('chamfer_x1000', '—'):.2f} | "
            f"{r.get('hole_iou', '—'):.3f} | "
            f"{r.get('hole_recall', '—'):.3f} | "
            f"{r.get('through_hole_open_pct', '—'):.1f}% |"
        )
    lines.append("")
    lines.append("## Decision criteria")
    lines.append("")
    lines.append("Pick the (iters, res) pair that:")
    lines.append("  1. Achieves `through-hole-open % ≥ 30` AND")
    lines.append("  2. Achieves `hole IoU ≥ 0.85` AND")
    lines.append("  3. Minimizes wall-time among configs that satisfy 1 & 2.")
    lines.append("")
    lines.append("If no config in this sweep satisfies 1-2: supervision is "
                 "still broken; do not proceed to Gate 1.")
    out_path.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="reference mesh for sweep")
    ap.add_argument("--out", required=True, help="output root dir")
    ap.add_argument("--repo-root", default="/workspace/clearmesh")
    ap.add_argument("--quick", action="store_true",
                    help="Half iters, skip res=256. Runs in ~3 hrs instead "
                         "of ~12 hrs. Use for first-pass shakedown.")
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_root = Path(args.out); out_root.mkdir(parents=True, exist_ok=True)
    results_path = out_root / "gate01_results.json"
    report_path = out_root / "gate01_results.md"

    if args.quick:
        iter_configs = [(5000, 192), (10000, 192), (15000, 192)]
        res_configs = [(15000, 128), (15000, 192)]
    else:
        iter_configs = [(5000, 192), (10000, 192), (15000, 192), (30000, 192)]
        res_configs = [(15000, 128), (15000, 192), (15000, 256)]

    # Dedup (iters=15000, res=192) appears in both lists
    specs = list(set(iter_configs + res_configs))
    specs.sort(key=lambda x: (x[0], x[1]))

    print(f"[gate01] {len(specs)} configs to run on {args.ref}")
    for iters, res in specs:
        print(f"    iters={iters:,} res={res}")

    results: list[dict] = []
    for iters, res in specs:
        run_dir = out_root / f"i{iters}_r{res}"
        if (run_dir / "refit.glb").exists():
            print(f"[gate01] skipping {run_dir.name} (already exists)")
            # Still want its metrics — re-evaluate cheaply
            r = run_one(
                ref=args.ref, out_dir=run_dir,
                iters=iters, resolution=res, k=args.k, seed=args.seed,
            )
        else:
            r = run_one(
                ref=args.ref, out_dir=run_dir,
                iters=iters, resolution=res, k=args.k, seed=args.seed,
            )
        results.append(r)
        # Checkpoint results after each run — don't lose data on a crash
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        write_report(results, report_path)
        # Commit progress
        try:
            repo = Path(args.repo_root)
            subprocess.run(
                ["git", "-C", str(repo), "add", str(report_path.relative_to(repo))
                 if str(report_path).startswith(str(repo)) else str(report_path)],
                check=False,
            )
            subprocess.run(
                ["git", "-C", str(repo), "commit", "-m",
                 f"gate01: {run_dir.name} complete"],
                check=False,
            )
            subprocess.run(
                ["git", "-C", str(repo), "push", "origin", "HEAD"],
                check=False,
            )
        except Exception as e:
            print(f"[gate01] commit failed: {e}")

    print("\n" + "=" * 70)
    print("GATE 0.1 SWEEP DONE")
    print("=" * 70)
    for r in results:
        if r.get("ok"):
            print(f"  iters={r['iters']:>6,} res={r['resolution']:>3} "
                  f"wall={r['wall_time_s']/60:.1f}m "
                  f"CD={r.get('chamfer_x1000', '—'):>6.2f} "
                  f"IoU={r.get('hole_iou', 0):.3f} "
                  f"open%={r.get('through_hole_open_pct', 0):.1f}")


if __name__ == "__main__":
    main()
