"""One-object DualPrim canary run.

Modes (matches DualPrimConfig.mode):

  mesh_fit
    Just fit dual-primitives to the mesh's TSDF. No renderer, no
    images. Sanity-checks the SQ math, P_E gate, losses wiring. Fast
    (~5 min CPU). Expected: captures geometry but loses parts.

  mesh_rendered_views
    Render our mesh to 26 views with analytic normals, then run the
    full paper objective (RGB + mask + normal + regularizers) with
    that as GT. ~1 hr GPU. Expected: recovers part decomposition.

  paper
    Use the user's source images + StableNormal + paper's multi-view
    objective. Requires multi-view input — typically not available for
    a single-image pipeline.

Usage:
    python scripts/dualprim/run_canary.py \\
        --input /workspace/demo_e2e_compass_v4/03_polished.glb \\
        --out /workspace/dualprim_canary_compass \\
        --mode mesh_rendered_views \\
        --iters 10000

Outputs:
    views/           26-view GT renders (for the _views modes)
    logs.json        per-log-step loss curves
    primitives.json  final primitive parameters
    refit.glb        exported scene mesh
    per_prim_*.glb   each dual-primitive's Boolean difference
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Make clearmesh importable on both laptop and pod
_CANDIDATE_ROOTS = [
    "/workspace/clearmesh",
    str(Path(__file__).resolve().parents[2]),
]
for _root in _CANDIDATE_ROOTS:
    if os.path.isdir(_root) and _root not in sys.path:
        sys.path.insert(0, _root)

import numpy as np
import torch
import trimesh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="path to mesh (GLB/OBJ) — used as GT geometry")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="mesh_rendered_views",
                    choices=["mesh_fit", "mesh_rendered_views", "paper"])
    ap.add_argument("--k", type=int, default=100,
                    help="initial number of dual-primitives")
    ap.add_argument("--iters", type=int, default=10_000)
    ap.add_argument("--rays", type=int, default=1024)
    ap.add_argument("--resolution", type=int, default=256,
                    help="view render resolution")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nsq-init", default="coupled",
                    choices=["coupled", "independent"],
                    help="NSQ init strategy: 'coupled' (NSQ near PSQ) or "
                         "'independent' (NSQ random, paper-faithful)")
    ap.add_argument("--union-export", action="store_true",
                    help="Boolean-union all primitives at export. "
                         "Slow (~minutes for K>=20) but produces a single "
                         "watertight mesh ~10-100x more compact than concatenate.")
    ap.add_argument("--fg-bias", type=float, default=0.7,
                    help="Fraction of rays drawn from foreground+boundary "
                         "pixels (rest uniform). 0.0 = paper's default "
                         "uniform sampling; 0.7 = friend's recommended "
                         "silhouette-pressure boost.")
    ap.add_argument("--trajectory-dir", default=None,
                    help="If set, write primitives_step_{N}.json snapshots "
                         "at canonical training iters (~log-spaced). Used "
                         "to build a warm-start training corpus: each "
                         "snapshot is one (mesh, seed, step) tuple for the "
                         "downstream feedforward predictor. Default off.")
    ap.add_argument("--resume-primitives", default=None,
                    help="Path to a primitives JSON (endpoint or trajectory "
                         "snapshot) to warm-start from. Skips random init. "
                         "Useful for measuring 'how few iters do I need "
                         "from a good starting point?' — the Gate 3 "
                         "feasibility check.")
    ap.add_argument("--lambda-mask", type=float, default=None,
                    help="Override lambda_mask in config. Friend's initial "
                         "tuning bumped this 1->3; round 4 fallback may "
                         "try 5+ if round 3 still fails.")
    ap.add_argument("--pruning-interval", type=int, default=None,
                    help="Override pruning_interval (how often to kill "
                         "weak primitives). Default 1000. Increase to "
                         "2000+ to give primitives more time to find "
                         "positions before being pruned.")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Config
    from clearmesh.dualprim import DualPrimConfig, init_scene, train, export_scene

    config = DualPrimConfig(
        num_primitives_init=args.k,
        num_iterations=args.iters,
        view_resolution=args.resolution,
        mode=args.mode,
        seed=args.seed,
        nsq_init_strategy=args.nsq_init,
    )
    # CLI overrides for round-4 tuning levers
    if args.lambda_mask is not None:
        config.lambda_mask = args.lambda_mask
    if args.pruning_interval is not None:
        config.pruning_interval = args.pruning_interval
    # Write the effective config for reproducibility
    from dataclasses import asdict
    with open(out_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)

    # ----- Initialize scene -----
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[canary] device={device}")
    print(f"[canary] mode={args.mode}, K={config.num_primitives_init}, "
          f"iters={config.num_iterations}")
    if args.resume_primitives:
        from clearmesh.dualprim.io import load_scene_from_json
        print(f"[canary] WARM START from {args.resume_primitives}")
        scene = load_scene_from_json(
            args.resume_primitives, config,
            device=device, pad_to_K=config.num_primitives_init,
        )
        print(f"[canary] loaded {scene.num_alive}/{scene.K} live primitives")
    else:
        scene = init_scene(config, device=device)

    # ----- Build the ray sampler for this mode -----
    if args.mode == "mesh_fit":
        # mesh_fit skips the renderer entirely. Precompute a TSDF and
        # use the dedicated train_mesh_fit() path below. We bypass the
        # ray sampler / train() flow.
        sampler = None
    elif args.mode == "mesh_rendered_views":
        views_dir = out_dir / "views"
        if not (views_dir / "views.json").exists():
            print(f"[canary] rendering 26 views to {views_dir}")
            from scripts.dualprim.render_views import render_views
            render_views(
                args.input, str(views_dir),
                resolution=config.view_resolution,
                render_normals=True,
            )
        sampler = _build_views_sampler(views_dir, device=device, fg_bias=args.fg_bias)
    elif args.mode == "paper":
        raise NotImplementedError(
            "mode=paper requires real source images + StableNormal — "
            "run mode=mesh_rendered_views first to validate the code path"
        )

    # ----- Train -----
    print(f"[canary] training for {config.num_iterations} iters")
    log_path = out_dir / "logs.json"
    log_rows = []

    if args.mode == "mesh_fit":
        # Precompute target TSDF on a grid, use train_mesh_fit
        from clearmesh.dualprim.optimize_scene import train_mesh_fit
        query_points, target_sdf = _build_mesh_fit_tsdf(
            args.input, device=device, resolution=64,
        )
        print(f"[canary] TSDF: {query_points.shape[0]:,} samples, "
              f"target range [{target_sdf.min():.3f}, {target_sdf.max():.3f}]")

        def log_fn(it, parts):
            log_rows.append(parts)
            print(f"[canary] it={it:6d} total={parts['total']:.4f} "
                  f"tsdf={parts['tsdf']:.4f} alive={parts['alive']}")
            with open(log_path, "w") as f:
                json.dump(log_rows, f, indent=2)

        t0 = time.time()
        state = train_mesh_fit(
            scene, query_points, target_sdf, config,
            samples_per_batch=args.rays * 4,   # re-use --rays as batch size
            device=device,
            log_fn=log_fn,
            checkpoint_path=str(out_dir / "checkpoints"),
        )
    else:
        def log_fn(it, parts):
            log_rows.append(parts)
            # Core loss + prune health
            line = (
                f"[canary] it={it:6d} total={parts['total']:.4f} "
                f"rgb={parts['rgb']:.3f} mask={parts['mask']:.3f} "
                f"norm={parts['norm']:.3f} alive={parts['alive']}"
            )
            # NSQ-health diagnostics (added in review round 3)
            if "theta_p50" in parts:
                line += (f"  θ[{parts['theta_p10']:.2f}/{parts['theta_p50']:.2f}/"
                         f"{parts['theta_p90']:.2f}]")
            if "theta_min_eff" in parts:
                line += f" θ_min_eff={parts['theta_min_eff']:.2f}"
            if "nsq_overlap_pct" in parts:
                line += f" NSQ∩PSQ={parts['nsq_overlap_pct']:.0f}%"
            if "pe_mean_fg" in parts:
                line += f" P_E_fg={parts['pe_mean_fg']:.3f}/{parts['pe_max_fg']:.2f}"
            if "t_render_ms" in parts:
                # Stage timings — per-iter average for this window. Friend's
                # debug recommendation: catches "one iteration takes 30s"
                # cost-cliff failure modes (NaN-skip storms, etc).
                line += (
                    f"  t[r{parts['t_render_ms']:.0f}/l{parts['t_loss_ms']:.0f}"
                    f"/s{parts['t_step_ms']:.0f}/p{parts['t_prune_ms']:.0f}]ms"
                )
            print(line)
            with open(log_path, "w") as f:
                json.dump(log_rows, f, indent=2)

        t0 = time.time()
        state = train(
            scene, sampler, config,
            rays_per_batch=args.rays,
            device=device,
            log_fn=log_fn,
            checkpoint_path=str(out_dir / "checkpoints"),
            trajectory_dir=args.trajectory_dir,
        )

    train_dt = time.time() - t0
    print(f"[canary] training done in {train_dt/60:.1f} min "
          f"— {scene.num_alive}/{config.num_primitives_init} alive")

    # ----- Export -----
    print(f"[canary] exporting (α ≥ {config.export_alpha_threshold})")
    scene_mesh, per_prim = export_scene(scene, config, union_all=args.union_export)
    scene_mesh.export(out_dir / "refit.glb")
    for i, m in enumerate(per_prim):
        m.export(out_dir / f"per_prim_{i:03d}.glb")

    # Save primitive params (same format as trajectory snapshots for
    # corpus-uniformity — downstream dataset loaders can treat the
    # final primitives.json as just another snapshot keyed at iter==N).
    from clearmesh.dualprim.io import save_scene_json
    save_scene_json(
        scene, out_dir / "primitives.json",
        iteration=config.num_iterations,
        extra_metadata={"training_s": train_dt},
    )

    print()
    print("=" * 60)
    print(f"CANARY DONE")
    print(f"  mode:     {args.mode}")
    print(f"  K final:  {scene.num_alive}/{config.num_primitives_init}")
    print(f"  iters:    {config.num_iterations}")
    print(f"  train_s:  {train_dt:.1f}")
    print(f"  scene_v:  {len(scene_mesh.vertices):,}")
    print(f"  scene_f:  {len(scene_mesh.faces):,}")
    print(f"  out:      {out_dir}")
    print("=" * 60)


# ---------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------

def _build_mesh_fit_tsdf(
    mesh_path: str, device: str, resolution: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a TSDF of the reference mesh on a regular grid in [-1, 1]^3.

    Returns (query_points, target_sdf) — both torch tensors on ``device``.
    Used as supervision for the mesh_fit debug path. Tries mesh2sdf
    (fast, CUDA) first and falls back to trimesh contains + EDT if not
    available — same strategy as clearmesh.refit.light_sq.build_tsdf.
    """
    import numpy as np

    mesh = trimesh.load(mesh_path, force="mesh")
    # Normalize to [-1+1/N, 1-1/N]^3 (same frame as init_scene)
    mesh = mesh.copy()
    mesh.vertices -= mesh.centroid
    s = mesh.extents.max()
    if s > 0:
        mesh.vertices *= (2.0 / s) * 0.98    # 2% margin

    lin = np.linspace(-1.0 + 1.0 / resolution, 1.0 - 1.0 / resolution, resolution)
    gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
    coords = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3).astype(np.float32)

    try:
        import mesh2sdf
        verts = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        # mesh2sdf already returns negative-inside (matches paper).
        # Verified via scripts/dualprim/check_conventions.py group [4]:
        # raw centre-of-sphere = -0.976, raw corner-of-cube = +0.648.
        sdf = mesh2sdf.compute(
            verts, faces, size=resolution,
            fix=False, level=2.0 / resolution, return_mesh=False,
        ).astype(np.float32).reshape(-1)
    except ImportError:
        from scipy.ndimage import distance_transform_edt
        occupied = np.zeros(len(coords), dtype=bool)
        chunk = 100_000
        for i in range(0, len(coords), chunk):
            occupied[i:i + chunk] = mesh.contains(coords[i:i + chunk])
        occ_grid = occupied.reshape(resolution, resolution, resolution)
        voxel = 2.0 / resolution
        d_out = distance_transform_edt(~occ_grid) * voxel
        d_in = distance_transform_edt(occ_grid) * voxel
        sdf = np.where(occ_grid, -d_in, d_out).astype(np.float32).reshape(-1)

    return (
        torch.from_numpy(coords).to(device),
        torch.from_numpy(sdf).to(device),
    )


def _build_views_sampler(views_dir: Path, device: str, fg_bias: float = 0.7):
    """Sample rays from the 26 rendered views in `views_dir`.

    Friend's tuning input: uniform pixel sampling wastes most rays on
    background. For shape supervision (especially for hole/concavity
    detection), the foreground silhouette and its boundary are where
    the geometry signal lives. Bias sampling toward foreground +
    boundary pixels.

    fg_bias: fraction of rays drawn from "interesting" pixels (mask
    foreground OR mask-boundary). Remaining (1 - fg_bias) drawn
    uniformly across the full image.

    Pre-computes per-view boundary masks once (mask XOR eroded mask).
    """
    from PIL import Image
    from clearmesh.dualprim import RaySampleBatch

    with open(views_dir / "views.json") as f:
        meta = json.load(f)
    V = len(meta["views"])
    H = W = meta["camera"]["resolution"]

    rgbs = np.zeros((V, H, W, 3), dtype=np.float32)
    masks = np.zeros((V, H, W), dtype=np.float32)
    normals = np.zeros((V, H, W, 3), dtype=np.float32)
    poses = np.zeros((V, 4, 4), dtype=np.float32)
    for i in range(V):
        rgbs[i] = np.asarray(Image.open(views_dir / f"{i:02d}_rgb.png").convert("RGB")) / 255.0
        m = np.asarray(Image.open(views_dir / f"{i:02d}_mask.png").convert("L"))
        masks[i] = (m > 127).astype(np.float32)
        n = np.asarray(Image.open(views_dir / f"{i:02d}_normal.png").convert("RGB")) / 255.0
        normals[i] = n * 2.0 - 1.0
        poses[i] = np.asarray(meta["views"][i]["pose_world_from_camera"], dtype=np.float32)

    yfov = meta["camera"]["yfov_rad"]
    # Precompute per-pixel ray directions in camera space
    fx = fy = 0.5 * H / math.tan(yfov / 2.0)
    ys, xs = np.meshgrid(
        np.arange(H, dtype=np.float32),
        np.arange(W, dtype=np.float32),
        indexing="ij",
    )
    cam_dirs = np.stack([
        (xs - W / 2.0) / fx,
        -(ys - H / 2.0) / fy,   # +Y up
        -np.ones_like(xs),
    ], axis=-1)
    cam_dirs = cam_dirs / np.linalg.norm(cam_dirs, axis=-1, keepdims=True)

    rgbs_t = torch.from_numpy(rgbs).to(device)
    masks_t = torch.from_numpy(masks).to(device)
    normals_t = torch.from_numpy(normals).to(device)
    poses_t = torch.from_numpy(poses).to(device)
    cam_dirs_t = torch.from_numpy(cam_dirs).to(device)

    # Pre-compute "interesting" pixel index sets per view: foreground
    # (mask>0.5) PLUS a few-pixel-wide boundary band around the mask
    # (extracted via 1-px erosion XOR mask). The boundary captures
    # silhouette + hole edges, which are where the geometric pressure
    # for carving lives.
    from scipy import ndimage
    interesting_per_view = []
    for v in range(V):
        m = masks[v] > 0.5
        eroded = ndimage.binary_erosion(m, iterations=2)
        boundary = m & ~eroded
        # Also add the inverse boundary (just-outside-mask) so silhouette
        # rays actually hit empty space too.
        outer = ndimage.binary_dilation(m, iterations=2) & ~m
        interesting = m | boundary | outer
        # Flatten to indices
        idx = np.flatnonzero(interesting.ravel())
        interesting_per_view.append(idx)
    # Pad to same length so we can stack (use max length, sample with
    # replacement if needed — fine since pool is large).
    max_len = max(len(ix) for ix in interesting_per_view)
    interesting_padded = np.zeros((V, max_len), dtype=np.int64)
    for v in range(V):
        ix = interesting_per_view[v]
        # Cycle to fill
        if len(ix) < max_len:
            reps = (max_len + len(ix) - 1) // len(ix)
            ix = np.tile(ix, reps)[:max_len]
        interesting_padded[v] = ix
    interesting_padded_t = torch.from_numpy(interesting_padded).to(device)

    rng = torch.Generator(device=device).manual_seed(42)

    def sampler(n_rays: int) -> RaySampleBatch:
        n_fg = int(n_rays * fg_bias)
        n_uniform = n_rays - n_fg

        # Foreground+boundary biased rays
        vi_f = torch.randint(0, V, (n_fg,), generator=rng, device=device)
        # Pick from each view's interesting pool
        col = torch.randint(0, max_len, (n_fg,), generator=rng, device=device)
        flat = interesting_padded_t[vi_f, col]
        yi_f = flat // W
        xi_f = flat % W

        # Uniform rays
        vi_u = torch.randint(0, V, (n_uniform,), generator=rng, device=device)
        yi_u = torch.randint(0, H, (n_uniform,), generator=rng, device=device)
        xi_u = torch.randint(0, W, (n_uniform,), generator=rng, device=device)

        # Concatenate
        vi = torch.cat([vi_f, vi_u])
        yi = torch.cat([yi_f, yi_u])
        xi = torch.cat([xi_f, xi_u])

        rgb = rgbs_t[vi, yi, xi]                              # (R, 3)
        mask = masks_t[vi, yi, xi]                             # (R,)
        normal = normals_t[vi, yi, xi]                         # (R, 3)

        cam_dir = cam_dirs_t[yi, xi]
        rot = poses_t[vi, :3, :3]
        world_dir = torch.einsum("rij,rj->ri", rot, cam_dir)
        origin = poses_t[vi, :3, 3]

        return RaySampleBatch(
            origins=origin, dirs=world_dir,
            rgb_gt=rgb, mask_gt=mask, normals_gt=normal,
        )
    return sampler


# scripts/dualprim/render_views.py imports math; keep it here too
import math


if __name__ == "__main__":
    main()
