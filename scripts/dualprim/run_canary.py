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
                    choices=["coupled", "coupled_axial", "independent"],
                    help="NSQ init strategy: 'coupled' (NSQ near PSQ), "
                         "'coupled_axial' (coupled + 3x elongated on random "
                         "axis, round-10 addition), or "
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
    ap.add_argument("--lambda-open-ray", type=float, default=None,
                    help="Weight on the open-ray loss (round-7 addition). "
                         "Penalizes predicted mask > 0 on rays passing "
                         "through GT holes, STRONGER than the global BCE "
                         "mask loss. Specifically rewards NSQ carving. "
                         "Default off (0). Round 7 recipe: 5.0. "
                         "NOTE: superseded by --hole-ray-oversample which "
                         "avoids the NaN-grad cascade this loss triggers.")
    ap.add_argument("--hole-ray-oversample", type=float, default=0.0,
                    help="Fraction of rays per batch drawn specifically "
                         "from GT hole pixels. Upweights the existing "
                         "BCE mask loss at hole pixels WITHOUT adding a "
                         "new loss path — avoids round-7's NaN-grad "
                         "cascade. Round 8 recipe: 0.3 (30% of rays).")
    ap.add_argument("--detect-anomaly", action="store_true",
                    help="Enable torch.autograd.detect_anomaly for the "
                         "training loop. Slow but useful for identifying "
                         "the exact backward op that first creates NaN.")
    ap.add_argument("--abort-on-nan-grad", action="store_true",
                    help="Abort immediately on the first non-finite "
                         "gradient instead of continuing with skip logic.")
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
    # CLI overrides for round-4+ tuning levers
    if args.lambda_mask is not None:
        config.lambda_mask = args.lambda_mask
    if args.pruning_interval is not None:
        config.pruning_interval = args.pruning_interval
    if args.lambda_open_ray is not None:
        config.lambda_open_ray = args.lambda_open_ray
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
        sampler = _build_views_sampler(
            views_dir, device=device,
            fg_bias=args.fg_bias,
            hole_ray_oversample=args.hole_ray_oversample,
        )
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
            # Open-ray loss (round 7+): only show if >0
            if parts.get("open", 0) > 0:
                line += f" open={parts['open']:.4f}"
            if parts.get("nan_grad_skip"):
                line += " [nan_grad]"
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
            if "nan_grad_summary" in parts:
                line += f" offender={parts['nan_grad_summary']}"
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
            detect_anomaly=args.detect_anomaly,
            abort_on_nan_grad=args.abort_on_nan_grad,
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


def _build_views_sampler(views_dir: Path, device: str,
                          fg_bias: float = 0.7,
                          hole_ray_oversample: float = 0.0):
    """Sample rays from the rendered views in `views_dir`.

    fg_bias: fraction of rays drawn from "interesting" pixels (mask
    foreground OR mask-boundary OR near-silhouette hole pixels).
    Remaining (1 - fg_bias) drawn uniformly.

    hole_ray_oversample: fraction of rays specifically drawn from
    GT hole pixels (binary_fill_holes(mask) & ~mask). These pixels
    lie inside the projected silhouette but are empty in the ref —
    i.e. where the ref mesh has a hole. Oversampling them
    effectively upweights the existing BCE mask loss at hole pixels
    WITHOUT adding a new loss term (avoids the NaN-grad cascade
    seen when adding loss_open_ray). Friend's point 3: "weight
    hole/boundary rays higher in sampling, not in loss weight."

    Final ratio: n_hole + n_fg + n_uniform = n_rays, where
    n_hole = n_rays * hole_ray_oversample, n_fg = remaining * fg_bias.
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

    # Pre-compute per-view HOLE masks = binary_fill_holes(mask) & ~mask.
    # These are the pixels that lie INSIDE the silhouette's convex hull
    # but OUTSIDE the actual mask — i.e. places where the ref mesh has
    # a hole. For round 7's open-ray loss we mark rays hitting these
    # pixels as "hole rays" and reward low predicted opacity on them.
    from scipy import ndimage
    hole_masks = np.zeros((V, H, W), dtype=bool)
    for v in range(V):
        m = masks[v] > 0.5
        filled = ndimage.binary_fill_holes(m)
        hole_masks[v] = filled & ~m
    hole_masks_t = torch.from_numpy(hole_masks).to(device)
    n_hole_pixels_total = int(hole_masks.sum())
    print(f"[sampler] total hole pixels across {V} views: {n_hole_pixels_total:,} "
          f"({100.0 * n_hole_pixels_total / (V * H * W):.2f}%)")

    # Pre-compute "interesting" pixel index sets per view: foreground
    # (mask>0.5) PLUS a few-pixel-wide boundary band around the mask
    # (extracted via 1-px erosion XOR mask). Friend's round-7 addition:
    # ALSO include hole pixels in the "interesting" pool so fg-biased
    # sampling actually hits holes with reasonable frequency.
    interesting_per_view = []
    for v in range(V):
        m = masks[v] > 0.5
        eroded = ndimage.binary_erosion(m, iterations=2)
        boundary = m & ~eroded
        # Also add the inverse boundary (just-outside-mask) so silhouette
        # rays actually hit empty space too.
        outer = ndimage.binary_dilation(m, iterations=2) & ~m
        interesting = m | boundary | outer | hole_masks[v]
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

    # Dedicated hole-ray pool: views with ANY hole pixels get a
    # padded index array of those pixels only. Views without hole
    # pixels are excluded from this pool. hole_ray_oversample fraction
    # of rays are drawn from this pool uniformly across valid views.
    hole_views = [v for v in range(V) if hole_masks[v].any()]
    hole_pool_padded = None
    hole_pool_max_len = 0
    hole_views_t = None
    if hole_views and hole_ray_oversample > 0:
        hole_per_view = []
        for v in hole_views:
            idx = np.flatnonzero(hole_masks[v].ravel())
            hole_per_view.append(idx)
        hole_pool_max_len = max(len(ix) for ix in hole_per_view)
        padded = np.zeros((len(hole_views), hole_pool_max_len), dtype=np.int64)
        for i, ix in enumerate(hole_per_view):
            if len(ix) < hole_pool_max_len:
                reps = (hole_pool_max_len + len(ix) - 1) // len(ix)
                ix = np.tile(ix, reps)[:hole_pool_max_len]
            padded[i] = ix
        hole_pool_padded = torch.from_numpy(padded).to(device)
        hole_views_t = torch.tensor(hole_views, dtype=torch.long, device=device)
        print(f"[sampler] {len(hole_views)}/{V} views have hole pixels; "
              f"max pool size {hole_pool_max_len}. "
              f"hole_ray_oversample={hole_ray_oversample:.2f}")

    rng = torch.Generator(device=device).manual_seed(42)

    def sampler(n_rays: int) -> RaySampleBatch:
        # Split ray budget: hole → fg → uniform
        if hole_pool_padded is not None and hole_ray_oversample > 0:
            n_hole = int(n_rays * hole_ray_oversample)
        else:
            n_hole = 0
        remaining = n_rays - n_hole
        n_fg = int(remaining * fg_bias)
        n_uniform = remaining - n_fg

        # Hole rays — oversample from GT hole pixels
        if n_hole > 0:
            vi_h_idx = torch.randint(0, hole_views_t.shape[0], (n_hole,),
                                      generator=rng, device=device)
            vi_h = hole_views_t[vi_h_idx]
            col_h = torch.randint(0, hole_pool_max_len, (n_hole,),
                                   generator=rng, device=device)
            flat_h = hole_pool_padded[vi_h_idx, col_h]
            yi_h = flat_h // W
            xi_h = flat_h % W
        else:
            vi_h = torch.empty(0, dtype=torch.long, device=device)
            yi_h = torch.empty(0, dtype=torch.long, device=device)
            xi_h = torch.empty(0, dtype=torch.long, device=device)

        # Foreground+boundary biased rays
        vi_f = torch.randint(0, V, (n_fg,), generator=rng, device=device)
        col = torch.randint(0, max_len, (n_fg,), generator=rng, device=device)
        flat = interesting_padded_t[vi_f, col]
        yi_f = flat // W
        xi_f = flat % W

        # Uniform rays
        vi_u = torch.randint(0, V, (n_uniform,), generator=rng, device=device)
        yi_u = torch.randint(0, H, (n_uniform,), generator=rng, device=device)
        xi_u = torch.randint(0, W, (n_uniform,), generator=rng, device=device)

        # Concatenate — hole rays first, then fg, then uniform
        vi = torch.cat([vi_h, vi_f, vi_u])
        yi = torch.cat([yi_h, yi_f, yi_u])
        xi = torch.cat([xi_h, xi_f, xi_u])

        rgb = rgbs_t[vi, yi, xi]                              # (R, 3)
        mask = masks_t[vi, yi, xi]                             # (R,)
        normal = normals_t[vi, yi, xi]                         # (R, 3)
        hole_ray = hole_masks_t[vi, yi, xi]                    # (R,) bool

        cam_dir = cam_dirs_t[yi, xi]
        rot = poses_t[vi, :3, :3]
        world_dir = torch.einsum("rij,rj->ri", rot, cam_dir)
        origin = poses_t[vi, :3, 3]

        return RaySampleBatch(
            origins=origin, dirs=world_dir,
            rgb_gt=rgb, mask_gt=mask, normals_gt=normal,
            hole_ray_gt=hole_ray,
        )
    return sampler


# scripts/dualprim/render_views.py imports math; keep it here too
import math


if __name__ == "__main__":
    main()
