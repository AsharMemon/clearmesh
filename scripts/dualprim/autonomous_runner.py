"""Autonomous experiment runner.

Chains: train canary → render output → compute Chamfer → write
markdown report → git commit → next experiment.

Designed to run unattended for hours. Each experiment is independent;
a failure on one doesn't stop the next. Exit conditions:
  - all experiments completed
  - VAST/Thunder credit exhausted (we'd see SSH/disk failures)
  - explicit kill

Reports go to ``docs/dualprim_runs/`` in the repo so they survive
pod loss.

Run on pod (or local with --no-ssh):
    python scripts/dualprim/autonomous_runner.py \\
        --experiments stool dumbbell camera window_box \\
        --k 30 --iters 5000 --resolution 192
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

# Make clearmesh + scripts importable
_CANDIDATE_ROOTS = ["/workspace/clearmesh", "/root/clearmesh",
                     str(Path(__file__).resolve().parents[2])]
for _r in _CANDIDATE_ROOTS:
    if os.path.isdir(_r) and _r not in sys.path:
        sys.path.insert(0, _r)


# ---------------------------------------------------------------------
# Experiment specs
# ---------------------------------------------------------------------

# Each experiment specifies:
#   name      slug used for paths and report headings
#   ref_glb   ground-truth mesh (must already exist on pod)
#   summary   one-line description for the report
EXPERIMENT_LIBRARY = {
    "hole":       ("box-with-through-hole",       "/workspace/test_box_hole.glb"),
    "stool":      ("4-leg stool",                  "/workspace/test_stool.glb"),
    "dumbbell":   ("two-sphere dumbbell",          "/workspace/test_dumbbell.glb"),
    "camera":     ("box + cylindrical lens",       "/workspace/test_camera.glb"),
    "window_box": ("box with rectangular cut-out", "/workspace/test_window_box.glb"),
}


# ---------------------------------------------------------------------
# Per-experiment driver
# ---------------------------------------------------------------------

def run_experiment(
    name: str,
    ref_glb: str,
    summary: str,
    out_root: str,
    *,
    k: int,
    iters: int,
    resolution: int,
    rays: int,
    seed: int = 0,
    nsq_init: str = "coupled",
    union_export: bool = False,
    fg_bias: float = 0.7,
    trajectory: bool = False,
    run_label: str = None,
) -> dict:
    """Train, render, evaluate. Returns a result dict.

    If ``trajectory=True``, passes --trajectory-dir to run_canary so
    log-spaced JSON snapshots are written to {out_dir}/trajectory/ —
    these become training points for the warm-start predictor.

    ``run_label`` (e.g. "seed0") is appended to the output dir name so
    multiple seeds of the same experiment don't collide.
    """
    dir_name = f"{name}_{run_label}" if run_label else name
    out_dir = Path(out_root) / dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "train.log"
    print(f"\n{'=' * 70}\n[{name}] {summary}\n{'=' * 70}")

    t0 = time.time()
    # Render views if not cached
    views_dir = out_dir / "views"
    if not (views_dir / "views.json").exists():
        print(f"[{name}] rendering 26 views @ {resolution}px")
        env = os.environ.copy()
        env["PYOPENGL_PLATFORM"] = "egl"
        subprocess.run(
            [sys.executable, "scripts/dualprim/render_views.py",
             "--input", ref_glb, "--out", str(views_dir),
             "--resolution", str(resolution)],
            check=False, env=env,
        )

    # Train
    print(f"[{name}] training (K={k}, iters={iters}, rays={rays}, res={resolution})")
    env = os.environ.copy()
    env["PYOPENGL_PLATFORM"] = "egl"
    train_cmd = [
        sys.executable, "-u", "scripts/dualprim/run_canary.py",
        "--input", ref_glb,
        "--out", str(out_dir),
        "--mode", "mesh_rendered_views",
        "--k", str(k),
        "--iters", str(iters),
        "--rays", str(rays),
        "--resolution", str(resolution),
        "--seed", str(seed),
        "--nsq-init", nsq_init,
        "--fg-bias", str(fg_bias),
    ]
    if union_export:
        train_cmd.append("--union-export")
    if trajectory:
        train_cmd.extend(["--trajectory-dir", str(out_dir / "trajectory")])
    with open(log_file, "w") as lf:
        proc = subprocess.run(train_cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
    train_dt = time.time() - t0
    train_ok = proc.returncode == 0 and (out_dir / "refit.glb").exists()
    print(f"[{name}] training done in {train_dt/60:.1f} min, ok={train_ok}")

    # Render the export
    render_paths = []
    if train_ok:
        from PIL import Image
        import numpy as np, trimesh
        os.environ["PYOPENGL_PLATFORM"] = "egl"
        import pyrender
        try:
            mesh = trimesh.load(str(out_dir / "refit.glb"), force="mesh")
            mesh.vertices -= mesh.centroid
            s = mesh.extents.max()
            if s > 0:
                mesh.vertices /= s
            for ang_deg in (0, 45, 90, 135):
                theta = math.radians(ang_deg)
                eye = np.array([2 * math.sin(theta), 0.3, 2 * math.cos(theta)], dtype=np.float32)
                fwd = -eye / np.linalg.norm(eye)
                up_w = np.array([0, 1, 0], dtype=np.float32)
                r = np.cross(fwd, up_w); r /= np.linalg.norm(r)
                u = np.cross(r, fwd)
                pose = np.eye(4, dtype=np.float32)
                pose[:3, 0] = r; pose[:3, 1] = u; pose[:3, 2] = -fwd; pose[:3, 3] = eye
                sc = pyrender.Scene(ambient_light=(0.3, 0.3, 0.3), bg_color=(255, 255, 255, 255))
                sc.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))
                cam = pyrender.PerspectiveCamera(yfov=math.radians(40), aspectRatio=1.0)
                sc.add(cam, pose=pose)
                sc.add(pyrender.DirectionalLight(color=np.ones(3), intensity=4.0), pose=pose)
                rr = pyrender.OffscreenRenderer(720, 720)
                color, _ = rr.render(sc); rr.delete()
                p = out_dir / f"refit_{ang_deg:03d}.png"
                Image.fromarray(color[..., :3]).save(p)
                render_paths.append(p.name)
        except Exception as e:
            print(f"[{name}] render failed: {e}")

    # Chamfer
    cd_scaled = None
    if train_ok:
        try:
            from scipy.spatial import cKDTree
            import trimesh
            ref_mesh = trimesh.load(ref_glb, force="mesh")
            pred_mesh = trimesh.load(str(out_dir / "refit.glb"), force="mesh")
            for m in (ref_mesh, pred_mesh):
                m.vertices -= m.centroid
                s = m.extents.max()
                if s > 0:
                    m.vertices /= s
            ref_pts, _ = trimesh.sample.sample_surface(ref_mesh, 30000)
            pred_pts, _ = trimesh.sample.sample_surface(pred_mesh, 30000)
            tr = cKDTree(ref_pts)
            tp = cKDTree(pred_pts)
            d_pr, _ = tr.query(pred_pts, k=1)
            d_rp, _ = tp.query(ref_pts, k=1)
            cd_raw = float(d_pr.mean() + d_rp.mean())
            cd_scaled = cd_raw * 1000.0
        except Exception as e:
            print(f"[{name}] chamfer failed: {e}")

    # Final mesh stats
    n_v, n_f = 0, 0
    if train_ok:
        import trimesh
        m = trimesh.load(str(out_dir / "refit.glb"), force="mesh")
        n_v, n_f = len(m.vertices), len(m.faces)

    # Last log line for primitive count + final loss
    final_log = ""
    try:
        with open(log_file) as f:
            tail = f.readlines()[-30:]
        for line in tail:
            if "CANARY DONE" in line or "K final" in line or "alive" in line:
                final_log += line
    except Exception:
        pass

    result = {
        "name": name,
        "summary": summary,
        "dir_name": dir_name,
        "seed": seed,
        "run_label": run_label,
        "config": {
            "k": k, "iters": iters, "resolution": resolution, "rays": rays,
            "nsq_init": nsq_init, "fg_bias": fg_bias,
            "union_export": union_export, "trajectory": trajectory,
        },
        "ok": train_ok,
        "train_minutes": train_dt / 60,
        "scene_verts": n_v,
        "scene_faces": n_f,
        "chamfer_x1000": cd_scaled,
        "renders": render_paths,
        "final_log_tail": final_log,
        "ref_glb": ref_glb,
    }

    # Write per-run metrics.json so the downstream dataset loader can
    # filter by quality without re-reading train logs. Dataset builder
    # only keeps runs where {chamfer ≤ threshold, hole_metric ≥
    # threshold, train_ok} — a failed optimization is NOT valid
    # training data.
    try:
        with open(out_dir / "metrics.json", "w") as f:
            json.dump(result, f, indent=2, default=str)
    except Exception as e:
        print(f"[{name}] metrics.json write failed: {e}")

    return result


def write_report(out_path: Path, results: list[dict]):
    """Markdown summary committed to git."""
    paper_cd = 7.94
    lines = []
    lines.append("# DualPrim autonomous run results")
    lines.append("")
    lines.append("Per-target Chamfer (×1000) versus paper Table 2 baseline.")
    lines.append("")
    lines.append(f"Paper DualPrim CD×1000 = **{paper_cd}** (averaged over 180 ShapeNet objects).")
    lines.append("")
    lines.append("| Target | K | iters | Verts | Faces | Time (min) | CD×1000 | vs paper |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in results:
        cd = r.get("chamfer_x1000")
        cd_s = f"{cd:.2f}" if cd is not None else "—"
        if cd is not None:
            delta = (cd - paper_cd) / paper_cd * 100
            sign = "✓" if cd <= paper_cd * 1.2 else "✗"
            vs = f"{sign} {('+' if delta > 0 else '')}{delta:.0f}%"
        else:
            vs = "—"
        lines.append(
            f"| {r['name']} | {r['config']['k']} | {r['config']['iters']} | "
            f"{r['scene_verts']:,} | {r['scene_faces']:,} | "
            f"{r['train_minutes']:.1f} | {cd_s} | {vs} |"
        )
    lines.append("")
    for r in results:
        lines.append(f"## {r['name']}")
        lines.append(f"_{r['summary']}_")
        lines.append("")
        if r.get("renders"):
            lines.append("Renders (in same folder as this report):")
            for rp in r["renders"]:
                lines.append(f"- `{r['name']}/{rp}`")
            lines.append("")
        if r.get("final_log_tail"):
            lines.append("Training tail:")
            lines.append("```")
            lines.append(r["final_log_tail"])
            lines.append("```")
            lines.append("")

    out_path.write_text("\n".join(lines))


def commit_results(repo_root: Path, results_dir: Path, msg: str):
    """git add + commit + push the results dir."""
    try:
        # Stage everything under the results dir + the report
        subprocess.run(["git", "-C", str(repo_root), "add", str(results_dir)],
                       check=True)
        # Also stage the markdown report
        subprocess.run(
            ["git", "-C", str(repo_root), "commit", "-m", msg,
             "--author", "ClearMesh Autonomous <noreply@anthropic.com>"],
            check=False,
        )
        subprocess.run(["git", "-C", str(repo_root), "push", "origin", "HEAD"],
                       check=False)
    except Exception as e:
        print(f"[commit] failed: {e}")


def main():
    ap = argparse.ArgumentParser()
    # Two mutually-exclusive mesh-source modes:
    #   --experiments: named canaries from EXPERIMENT_LIBRARY (phase 1/2 style)
    #   --mesh-list: JSON manifest (e.g. from objaverse_sampler) for
    #     teacher data collection at scale
    ap.add_argument("--experiments", nargs="+",
                    choices=list(EXPERIMENT_LIBRARY.keys()),
                    help="Named canary meshes from EXPERIMENT_LIBRARY. "
                         "Mutually exclusive with --mesh-list.")
    ap.add_argument("--mesh-list", default=None,
                    help="Path to a JSON manifest (as written by "
                         "objaverse_sampler.py) listing {uid, path, "
                         "lvis_class} records. Runs autonomous training "
                         "on each listed mesh. Mutually exclusive with "
                         "--experiments.")
    ap.add_argument("--out-root", default="/workspace/dualprim_auto")
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--resolution", type=int, default=192)
    ap.add_argument("--rays", type=int, default=768)
    ap.add_argument("--repo-root", default="/workspace/clearmesh",
                    help="Path to git repo for committing results")
    ap.add_argument("--nsq-init", default="coupled",
                    choices=["coupled", "independent"])
    ap.add_argument("--union-export", action="store_true")
    ap.add_argument("--fg-bias", type=float, default=0.7,
                    help="Foreground+boundary ray sampling fraction; "
                         "0.0 = uniform (paper default), 0.7 = friend's "
                         "recommended silhouette-pressure setting.")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0],
                    help="Run each experiment with these seeds. "
                         "Multi-seed teaches the downstream predictor "
                         "the DISTRIBUTION of valid primitive decompositions "
                         "for the same mesh, not a single (arbitrary) one. "
                         "Friend's guidance: 2-3 seeds per mesh.")
    ap.add_argument("--trajectory", action="store_true",
                    help="Save log-spaced primitives snapshots during "
                         "training. Each run produces ~5 training points "
                         "instead of 1 endpoint — critical for training a "
                         "warm-start predictor that outputs partially-"
                         "refined states.")
    args = ap.parse_args()

    # Validate mesh-source flags: exactly one must be set.
    if bool(args.experiments) == bool(args.mesh_list):
        ap.error("specify exactly one of --experiments or --mesh-list")

    repo_root = Path(args.repo_root)
    docs_dir = repo_root / "docs" / "dualprim_runs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    # Build the (exp_key, ref_glb, summary) tuples we'll iterate.
    # Named canaries and Objaverse manifest items unify into the same
    # shape, so the main loop below doesn't branch on source type.
    mesh_specs: list[tuple[str, str, str]] = []
    if args.experiments:
        for exp_key in args.experiments:
            summary, ref_glb = EXPERIMENT_LIBRARY[exp_key]
            mesh_specs.append((exp_key, ref_glb, summary))
    else:
        manifest = json.load(open(args.mesh_list))
        meshes = manifest.get("meshes", [])
        for m in meshes:
            # Slugify the Objaverse UID (first 8 chars) + lvis_class for
            # a stable, human-readable experiment key.
            uid_short = m["uid"][:8]
            lvis = m.get("lvis_class", "unknown").replace(" ", "_")
            exp_key = f"{lvis}_{uid_short}"
            summary = f"Objaverse {m['uid']} ({m.get('lvis_class', '?')})"
            mesh_specs.append((exp_key, m["path"], summary))
        print(f"[runner] loaded {len(mesh_specs)} meshes from {args.mesh_list}")

    # Cross-product of {meshes} × {seeds}. Multi-seed is the
    # primary source of dataset richness — same mesh with different
    # random inits converges to different (valid) primitive configs,
    # which is exactly the signal a warm-start predictor needs.
    run_specs = [
        (ms, seed)
        for ms in mesh_specs
        for seed in args.seeds
    ]
    print(f"[runner] {len(run_specs)} runs: "
          f"{len(mesh_specs)} meshes × {len(args.seeds)} seeds")

    for (exp_key, ref_glb, summary), seed in run_specs:
        if not Path(ref_glb).exists():
            print(f"[skip] {exp_key}: ref mesh {ref_glb} not found")
            continue
        # Only add a seed suffix when running >1 seed — keeps single-seed
        # output layout backward-compatible with phase 1 / phase 2 runs.
        run_label = f"seed{seed}" if len(args.seeds) > 1 else None
        try:
            res = run_experiment(
                exp_key, ref_glb, summary, args.out_root,
                k=args.k, iters=args.iters,
                resolution=args.resolution, rays=args.rays,
                seed=seed,
                nsq_init=args.nsq_init, union_export=args.union_export,
                fg_bias=args.fg_bias,
                trajectory=args.trajectory,
                run_label=run_label,
            )
        except Exception as e:
            print(f"[error] {exp_key}/seed{seed} failed: {e}")
            res = {"name": exp_key, "summary": summary, "seed": seed,
                   "ok": False, "error": str(e),
                   "config": {"k": args.k, "iters": args.iters,
                              "resolution": args.resolution, "rays": args.rays},
                   "train_minutes": 0, "scene_verts": 0, "scene_faces": 0}
        results.append(res)

        # Copy renders into the docs dir for git commit
        if res.get("ok") and res.get("renders"):
            doc_subdir = res.get("dir_name", exp_key)
            target = docs_dir / doc_subdir
            target.mkdir(parents=True, exist_ok=True)
            src_dir = Path(args.out_root) / doc_subdir
            for rp in res["renders"]:
                try:
                    import shutil
                    shutil.copy(src_dir / rp, target / rp)
                except Exception as e:
                    print(f"[copy] {rp} failed: {e}")

        # Update the markdown report after each experiment
        write_report(docs_dir / "results.md", results)
        # Commit + push (might no-op if no changes)
        commit_results(
            repo_root, docs_dir,
            msg=f"dualprim_runs: {res.get('dir_name', exp_key)} complete "
                f"(CD={res.get('chamfer_x1000', 'na')})",
        )

    # Final summary
    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS DONE")
    print("=" * 70)
    for r in results:
        cd = r.get("chamfer_x1000")
        cd_s = f"{cd:.2f}" if cd is not None else "—"
        print(f"  {r['name']:14s} v={r.get('scene_verts', 0):>7,}  "
              f"f={r.get('scene_faces', 0):>7,}  "
              f"CD×1000={cd_s}  ok={r.get('ok')}")


if __name__ == "__main__":
    main()
