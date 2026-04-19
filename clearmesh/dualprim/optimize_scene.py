"""Per-scene Adam optimization with adaptive pruning (§4.2).

Workflow:
  1. Initialize K=100 dual-primitives randomly in [-1, 1]
  2. Randomly pick rays across all N views + pixel positions
  3. Render rays through the scene (renderer.py) → RenderOutput
  4. Compute total_loss vs GT RGB/mask/normal (losses.py)
  5. Adam step on scene.params (and lighting MLP)
  6. Every ``pruning_interval`` iterations:
       - Kill primitives with α < 0.02 (paper §4.2)
       - Kill primitives with min-scale < 0.01
       - View-dependent filter: kill primitives with negligible
         rendering weight across all viewpoints
  7. After N_iterations, export via export.py

This is the training loop SKELETON — it is paper-faithful on loss
construction and ordering but exercises paper-unspecified knobs via
DualPrimConfig (mu, lambda_*, lr, etc.).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch

from clearmesh.dualprim.params import DualPrimConfig
from clearmesh.dualprim.types import (
    DualPrimScene,
    DUAL_PRIM_DIM,
    IDX_PSQ_SCALE,
    IDX_NSQ_SCALE,
    IDX_PSQ_SHAPE,
    IDX_NSQ_SHAPE,
    IDX_ALPHA,
    IDX_THETA,
    IDX_PSQ_TRANSLATION,
    IDX_NSQ_TRANSLATION,
    IDX_PSQ_ROTATION,
    IDX_NSQ_ROTATION,
    IDX_COLOR,
)
from clearmesh.dualprim.renderer import LightingMLP, render_rays
from clearmesh.dualprim.losses import total_loss, total_loss_tsdf


# ---------------------------------------------------------------------
# Scene initialization (§4.2: "initialize with a dense set of
# randomly distributed dual-primitives")
# ---------------------------------------------------------------------

def init_scene(config: DualPrimConfig, device="cuda") -> DualPrimScene:
    K = config.num_primitives_init
    g = torch.Generator(device=device).manual_seed(config.seed)
    params = torch.zeros(K, DUAL_PRIM_DIM, device=device)

    def _uniform(lo, hi, shape):
        return torch.rand(*shape, generator=g, device=device) * (hi - lo) + lo

    # PSQ / NSQ scale — from config range, biased small initially
    s_lo, s_hi = config.scale_range
    init_s_hi = min(s_hi, 0.3)  # start compact so they don't cover the whole cube
    params[:, IDX_PSQ_SCALE] = _uniform(s_lo, init_s_hi, (K, 3))
    # NSQ starts ~0.7x of PSQ so subtraction lands inside
    params[:, IDX_NSQ_SCALE] = params[:, IDX_PSQ_SCALE] * 0.7

    # Shape — start at rounded cuboid (paper's implicit default)
    params[:, IDX_PSQ_SHAPE] = _uniform(0.5, 1.2, (K, 2))
    params[:, IDX_NSQ_SHAPE] = _uniform(0.5, 1.2, (K, 2))

    # α — start small (sparse) so the sparsity loss has room to work
    params[:, IDX_ALPHA] = _uniform(0.3, 0.5, (K,))

    # θ (render sharpness) — mid-range
    params[:, IDX_THETA] = _uniform(0.3, 0.7, (K,))

    # Translation — paper: "randomly in [-1, 1] space"
    t_lo, t_hi = config.init_space
    params[:, IDX_PSQ_TRANSLATION] = _uniform(t_lo, t_hi, (K, 3))
    # NSQ translation near its PSQ partner so they overlap by default
    params[:, IDX_NSQ_TRANSLATION] = (
        params[:, IDX_PSQ_TRANSLATION]
        + _uniform(-0.05, 0.05, (K, 3))
    )

    # Rotation — uniform over full range, in radians
    import math
    params[:, IDX_PSQ_ROTATION] = _uniform(-math.pi, math.pi, (K, 3))
    params[:, IDX_NSQ_ROTATION] = params[:, IDX_PSQ_ROTATION].clone()

    # Color — mid-grey
    params[:, IDX_COLOR] = _uniform(0.4, 0.6, (K, 3))

    params.requires_grad_(True)
    mlp = LightingMLP(
        hidden=config.lighting_mlp_hidden,
        layers=config.lighting_mlp_layers,
    ).to(device)
    return DualPrimScene(params=params, lighting_mlp=mlp)


# ---------------------------------------------------------------------
# Parameter clipping after each step
# ---------------------------------------------------------------------

def clip_to_ranges(scene: DualPrimScene, config: DualPrimConfig):
    """Project each per-primitive field back into its Table 1 range.

    Box-constraint vs soft-regularizer philosophy:

      α:
        NOT clamped here. L_max (Eq 17 = ReLU(α − 1)) is a soft cap
        that only fires if α is allowed to exceed 1 during the forward
        pass. Hard-clamping α each step would make α never exceed 1,
        making L_max's gradient identically zero. We clamp α strictly
        only at export time.

      shape (ε1, ε2):
        CLAMPED here. There is no soft regularizer for shape in the
        paper's six losses, so without post-step projection the raw
        parameter can drift outside [0.05, 2.0] permanently — and once
        it's outside, ``sq_implicit`` clamps it internally BEFORE
        using it in the forward pass, which produces a zero gradient
        through the clamp. So without explicit projection, a shape
        param that steps outside the valid range has no way back in.
        Projecting post-step keeps the raw param in-range so the
        unclipped forward always sees it.

        (Review round 1 removed this clamp in an attempt to mirror α's
        soft-regularizer philosophy. Review round 2 caught that shape
        has no matching soft loss, so projection has to come back.)
    """
    with torch.no_grad():
        scene.params[:, IDX_PSQ_SCALE].clamp_(*config.scale_range)
        scene.params[:, IDX_NSQ_SCALE].clamp_(*config.scale_range)
        scene.params[:, IDX_PSQ_SHAPE].clamp_(*config.shape_range)
        scene.params[:, IDX_NSQ_SHAPE].clamp_(*config.shape_range)
        # α intentionally NOT clamped — see docstring above.
        scene.params[:, IDX_THETA].clamp_(*config.sharpness_range)
        scene.params[:, IDX_PSQ_TRANSLATION].clamp_(*config.translation_range)
        scene.params[:, IDX_NSQ_TRANSLATION].clamp_(*config.translation_range)
        # rotations unbounded
        scene.params[:, IDX_COLOR].clamp_(*config.color_range)


# ---------------------------------------------------------------------
# Adaptive pruning (§4.2)
# ---------------------------------------------------------------------

def prune(
    scene: DualPrimScene,
    config: DualPrimConfig,
    verbose: bool = False,
) -> int:
    """Kill dual-primitives with α < 0.02 OR min-scale < 0.01.

    Returns the number of primitives killed in this call.
    """
    with torch.no_grad():
        alpha = scene.alpha()
        min_scale = torch.minimum(
            scene.psq_scale().min(dim=-1).values,
            scene.nsq_scale().min(dim=-1).values,
        )
        kill = scene.alive & (
            (alpha < config.prune_alpha_threshold)
            | (min_scale < config.prune_scale_threshold)
        )
        n_killed = int(kill.sum().item())
        scene.alive &= ~kill
        # Zero out α so dead slots contribute nothing
        scene.params[kill, IDX_ALPHA] = 0.0
    if verbose and n_killed > 0:
        print(f"[prune] killed {n_killed} primitives (α<{config.prune_alpha_threshold} "
              f"or scale<{config.prune_scale_threshold}); {scene.num_alive} alive")
    return n_killed


def _theta_min_curriculum(config: DualPrimConfig, iteration: int) -> float:
    """θ-curriculum schedule.

    Returns the EFFECTIVE theta_min for the current training step,
    linearly annealing from config.theta_curriculum_start down to
    config.theta_min over the first config.theta_curriculum_fraction
    of training, then holding at config.theta_min.
    """
    tot = max(config.num_iterations, 1)
    frac = iteration / tot
    stop = config.theta_curriculum_fraction
    if frac >= stop:
        return config.theta_min
    alpha = frac / max(stop, 1e-9)
    return (1.0 - alpha) * config.theta_curriculum_start + alpha * config.theta_min


def _dp_diagnostics(
    scene: DualPrimScene,
    ray_sampler: Optional["RaySampler"] = None,
    n_probe: int = 1024,
    theta_min_eff: float = 0.01,
    mu: float = 0.0,
) -> dict:
    """Diagnostics to catch P_E-gate collapse and NSQ inactivity.

    Without these, failure modes like "NSQ drifted out of PSQ" only
    show up via post-hoc parameter dumps. Everything here runs under
    ``torch.no_grad`` and is cheap.

    Returns (all scalars or lists of floats):
      theta_p10/50/90       percentiles of θ across alive primitives
      nsq_overlap_pct       % of alive primitives whose NSQ AABB
                            overlaps its PSQ AABB (cheap proxy for
                            "P_E can be nonzero for this primitive")
      pe_mean_fg            mean P_E at foreground probe-ray samples
                            (populated only if ray_sampler provided
                            and it returns mask_gt > 0 rays)
    """
    with torch.no_grad():
        out: dict = {}
        alive = scene.alive
        alive_idx = alive.nonzero(as_tuple=True)[0]
        if len(alive_idx) == 0:
            return {"alive": 0}

        theta_alive = scene.theta()[alive_idx].float()
        out["theta_p10"] = theta_alive.quantile(0.1).item()
        out["theta_p50"] = theta_alive.quantile(0.5).item()
        out["theta_p90"] = theta_alive.quantile(0.9).item()

        psq_t = scene.psq_translation()[alive_idx]
        psq_s = scene.psq_scale()[alive_idx]
        nsq_t = scene.nsq_translation()[alive_idx]
        nsq_s = scene.nsq_scale()[alive_idx]
        # Axis-aligned bounding boxes ignore rotation, but give a fast
        # and sufficient necessary-condition for "primitives might
        # overlap somewhere" (no overlap here ⇒ definitely no P_E > 0).
        psq_lo, psq_hi = psq_t - psq_s, psq_t + psq_s
        nsq_lo, nsq_hi = nsq_t - nsq_s, nsq_t + nsq_s
        aabb_overlap = torch.all(
            (psq_lo <= nsq_hi) & (psq_hi >= nsq_lo), dim=-1,
        )
        out["nsq_overlap_pct"] = float(aabb_overlap.float().mean().item() * 100)

        # Optional: mean P_E at foreground probe rays. Requires a
        # ray_sampler that returns mask_gt so we can filter to fg.
        if ray_sampler is not None:
            from clearmesh.dualprim.renderer import sample_ray_points
            from clearmesh.dualprim.superquadric import (
                sq_implicit, effectiveness_probability,
            )
            batch = ray_sampler(n_probe)
            fg = batch.mask_gt > 0.5
            if int(fg.sum()) > 32:
                o = batch.origins[fg]
                d = batch.dirs[fg]
                pts, _, _ = sample_ray_points(
                    o, d, near=0.1, far=4.0, N=16,
                    perturb=False, device=scene.params.device,
                )
                f_psq = sq_implicit(
                    pts, scene.psq_translation(), scene.psq_rotation(),
                    scene.psq_scale(), scene.psq_shape(),
                )
                f_nsq = sq_implicit(
                    pts, scene.nsq_translation(), scene.nsq_rotation(),
                    scene.nsq_scale(), scene.nsq_shape(),
                )
                p_e = effectiveness_probability(
                    f_psq, f_nsq, scene.theta(),
                    mu=mu, theta_min=theta_min_eff,
                )
                out["pe_mean_fg"] = float(p_e.mean().item())
                out["pe_max_fg"] = float(p_e.max().item())
        return out


def prune_view_dependent(
    scene: DualPrimScene,
    ray_sampler: "RaySampler",
    config: DualPrimConfig,
    *,
    num_probe_rays: int = 8192,
    weight_threshold: float = 1e-3,
    foreground_only: bool = True,
    min_foreground_rays: int = 256,
    verbose: bool = False,
) -> int:
    """Kill primitives with negligible rendering weight across viewpoints.

    Paper §4.2: "We prune primitives with negligible rendering weights
    across all viewpoints." A primitive is redundant if no ray through
    any view ever accumulates significant contribution from it.

    Normalization note (fixed in review round 2):
      We normalize per-primitive contribution by the FOREGROUND-HIT
      ray count, not the total probe-ray count. If probes are sampled
      uniformly over the image (most render_views-based samplers),
      most rays miss the object entirely — including them in the
      denominator makes the threshold depend on silhouette area and
      disproportionately punishes thin primitives (slats, rings, small
      protrusions — exactly what DualPrim's NSQ is designed to keep).

    Args:
      foreground_only: if True (default), filter the probe batch to
          rays with mask_gt > 0.5 before accumulating. Requires the
          sampler to provide mask_gt; falls back to using rays whose
          rendered mask is > 0.5 if mask_gt is all zero (legacy
          sanity samplers).
      min_foreground_rays: if fewer than this many rays actually hit
          the object, skip this prune cycle entirely (not enough
          signal to trust the threshold).

    Runs under ``torch.no_grad``.
    """
    from clearmesh.dualprim.renderer import (
        sample_ray_points,
        _scene_field,
        density_from_field,
        _accumulated_transmittance,
    )

    with torch.no_grad():
        batch = ray_sampler(num_probe_rays)

        # Foreground filter — the critical fix. Without this, thin
        # primitives in pixel-sparse regions get pruned just because
        # most probe rays miss them.
        if foreground_only:
            fg_mask = batch.mask_gt > 0.5
            n_fg = int(fg_mask.sum().item())
            if n_fg < min_foreground_rays:
                if verbose:
                    print(f"[prune/view] only {n_fg} foreground rays "
                          f"(< {min_foreground_rays}); skipping prune cycle")
                return 0
            origins = batch.origins[fg_mask]
            dirs = batch.dirs[fg_mask]
            R_eff = n_fg
        else:
            origins = batch.origins
            dirs = batch.dirs
            R_eff = num_probe_rays

        points, t_vals, deltas = sample_ray_points(
            origins, dirs,
            near=config.near_plane, far=config.far_plane,
            N=config.num_samples_per_ray,
            perturb=False, device=scene.params.device,
        )

        dp = dirs.unsqueeze(1) * 0.01  # same default Δp
        f_fwd = _scene_field(
            scene, points + dp, config.mu_gate_offset, config.theta_min,
        )
        f_bwd = _scene_field(
            scene, points - dp, config.mu_gate_offset, config.theta_min,
        )
        sigma_k = density_from_field(
            f_fwd, f_bwd, scene.theta(), theta_min=config.theta_min,
        )

        alive = scene.alive.to(sigma_k.dtype)
        # NOTE: α is unclamped during training (see clip_to_ranges comment).
        # Clamp locally only for computing composited weights, so a
        # temporarily-above-1 α doesn't artificially inflate contribution.
        alpha_k = scene.alpha().clamp(0.0, 1.0)
        sigma_k_weighted = sigma_k * (alpha_k * alive).view(1, 1, -1)

        sigma = sigma_k_weighted.sum(dim=-1)                 # (R_eff, N)
        alpha_ray = 1.0 - torch.exp(-sigma * deltas)         # (R_eff, N)
        trans = _accumulated_transmittance(alpha_ray)        # (R_eff, N)
        w_ray = alpha_ray * trans                            # (R_eff, N)

        denom = sigma.unsqueeze(-1) + 1e-8
        per_prim_contrib = (
            (sigma_k_weighted / denom) * w_ray.unsqueeze(-1)
        ).sum(dim=(0, 1))                                     # (K,)

        # Normalize by foreground-hit-ray count, NOT total probe rays.
        per_prim_contrib = per_prim_contrib / max(R_eff, 1)

        kill = scene.alive & (per_prim_contrib < weight_threshold)
        n_killed = int(kill.sum().item())
        scene.alive &= ~kill
        scene.params[kill, IDX_ALPHA] = 0.0

    if verbose and n_killed > 0:
        print(f"[prune/view] killed {n_killed} primitives "
              f"(contribution<{weight_threshold}, over {R_eff} fg rays); "
              f"{scene.num_alive} alive")
    return n_killed


# ---------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------

@dataclass
class TrainingState:
    scene: DualPrimScene
    iteration: int
    loss_history: list[dict]
    timings: dict[str, float]


RaySampler = Callable[[int], "RaySampleBatch"]


class RaySampleBatch:
    """Typed batch of training rays.

    All tensors must live on the same device as the scene.
    """
    def __init__(
        self,
        origins: torch.Tensor,      # (R, 3)
        dirs: torch.Tensor,         # (R, 3)
        rgb_gt: torch.Tensor,       # (R, 3)
        mask_gt: torch.Tensor,      # (R,)
        normals_gt: torch.Tensor,   # (R, 3)
    ):
        self.origins = origins
        self.dirs = dirs
        self.rgb_gt = rgb_gt
        self.mask_gt = mask_gt
        self.normals_gt = normals_gt


def train(
    scene: DualPrimScene,
    ray_sampler: RaySampler,
    config: DualPrimConfig,
    *,
    rays_per_batch: int = 1024,
    device: str = "cuda",
    log_fn: Optional[Callable[[int, dict], None]] = None,
    checkpoint_path: Optional[str] = None,
) -> TrainingState:
    """Run per-scene optimization.

    ray_sampler(n) returns a RaySampleBatch of n rays with RGB/mask/
    normal GT. Caller is responsible for building this — see
    scripts/dualprim/run_canary.py for the mesh-rendered-views
    implementation.
    """
    opt_params = [scene.params]
    if scene.lighting_mlp is not None:
        opt_params += list(scene.lighting_mlp.parameters())
    optimizer = torch.optim.Adam(opt_params, lr=config.learning_rate)

    scheduler = None
    if config.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.num_iterations, eta_min=config.learning_rate * 0.1,
        )

    loss_history: list[dict] = []
    timings = {"total": 0.0, "render": 0.0, "loss": 0.0, "step": 0.0, "prune": 0.0}
    # Stage-timing baseline (per-window deltas in the log_fn block)
    prev_log_it = 0
    prev_t_render = prev_t_loss = prev_t_step = prev_t_prune = 0.0
    t_start = time.time()

    for it in range(config.num_iterations):
        batch = ray_sampler(rays_per_batch)

        # θ curriculum — broad gate early, sharper as training progresses.
        # Review feedback: without this, θ collapses to its floor on
        # every primitive and the P_E gate becomes razor-thin, so NSQs
        # that drift outside their PSQs never find their way back.
        theta_min_eff = _theta_min_curriculum(config, it)

        t0 = time.time()
        render = render_rays(
            scene,
            batch.origins, batch.dirs,
            num_samples=config.num_samples_per_ray,
            near=config.near_plane,
            far=config.far_plane,
            mu=config.mu_gate_offset,
            theta_min=theta_min_eff,
            background=config.background_color,
        )
        timings["render"] += time.time() - t0

        t0 = time.time()
        loss, parts = total_loss(
            scene, render,
            rgb_gt=batch.rgb_gt, mask_gt=batch.mask_gt,
            normals_pred=batch.normals_gt,
            lambda_mask=config.lambda_mask,
            lambda_sparse=config.lambda_sparse,
            lambda_entropy=config.lambda_entropy,
            lambda_max=config.lambda_max,
            lambda_norm_reg=config.lambda_norm_reg,
        )
        timings["loss"] += time.time() - t0

        t0 = time.time()
        optimizer.zero_grad()
        # Skip the step if loss or grads are non-finite (NaN protection).
        # sq_implicit can still produce huge finite values near ε=0.05
        # that, combined with the softmin, occasionally overflow. The
        # value-clamp + this rollback gives us a clean recovery.
        if not torch.isfinite(loss):
            timings["step"] += time.time() - t0
            if it % config.log_interval == 0 and log_fn is not None:
                parts["iter"] = it; parts["alive"] = scene.num_alive
                parts["nan_skip"] = 1
                log_fn(it, parts)
            continue
        loss.backward()
        # Clip gradients before the Adam step.
        torch.nn.utils.clip_grad_norm_(opt_params, max_norm=1.0)
        # Double-check gradients after clipping (clip doesn't fix NaN)
        any_nan_grad = any(
            (p.grad is not None and not torch.isfinite(p.grad).all())
            for p in opt_params
        )
        if any_nan_grad:
            optimizer.zero_grad()
            timings["step"] += time.time() - t0
            continue
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        clip_to_ranges(scene, config)
        timings["step"] += time.time() - t0

        if it % config.pruning_interval == 0 and it >= config.warmup_iterations:
            t0 = time.time()
            prune(scene, config, verbose=(log_fn is not None))
            # View-dependent pruning (paper §4.2): runs at a coarser
            # cadence than the fast α/scale prune because it requires
            # a full-scene render pass.
            if it % (config.pruning_interval * 3) == 0:
                prune_view_dependent(
                    scene, ray_sampler, config,
                    num_probe_rays=min(8192, rays_per_batch * 8),
                    verbose=(log_fn is not None),
                )
            timings["prune"] += time.time() - t0

        if it % config.log_interval == 0:
            parts["iter"] = it
            parts["alive"] = scene.num_alive
            parts["theta_min_eff"] = theta_min_eff
            # Cheap diagnostics at every log step; expensive P_E probe
            # only every 4th log step.
            probe_this_step = (it % (config.log_interval * 4) == 0)
            diag = _dp_diagnostics(
                scene,
                ray_sampler=ray_sampler if probe_this_step else None,
                theta_min_eff=theta_min_eff,
                mu=config.mu_gate_offset,
            )
            parts.update(diag)
            # Stage timings — average per-iter for the just-completed window.
            # Catches the cost-cliff failure mode: "GPU at 100% but iter
            # rate falls off a cliff because one stage exploded".
            n_window = max(it - prev_log_it, 1)
            parts["t_render_ms"] = 1000.0 * (timings["render"] - prev_t_render) / n_window
            parts["t_loss_ms"] = 1000.0 * (timings["loss"] - prev_t_loss) / n_window
            parts["t_step_ms"] = 1000.0 * (timings["step"] - prev_t_step) / n_window
            parts["t_prune_ms"] = 1000.0 * (timings["prune"] - prev_t_prune) / n_window
            prev_t_render, prev_t_loss = timings["render"], timings["loss"]
            prev_t_step, prev_t_prune = timings["step"], timings["prune"]
            prev_log_it = it
            loss_history.append(parts)
            if log_fn is not None:
                log_fn(it, parts)

        if checkpoint_path and it > 0 and it % config.checkpoint_interval == 0:
            _save_checkpoint(scene, checkpoint_path, it)

    timings["total"] = time.time() - t_start
    return TrainingState(
        scene=scene, iteration=config.num_iterations,
        loss_history=loss_history, timings=timings,
    )


def train_mesh_fit(
    scene: DualPrimScene,
    query_points: torch.Tensor,      # (P, 3) on device
    target_sdf: torch.Tensor,        # (P,) on device
    config: DualPrimConfig,
    *,
    samples_per_batch: int = 4096,
    device: str = "cuda",
    log_fn: Optional[Callable[[int, dict], None]] = None,
    checkpoint_path: Optional[str] = None,
) -> TrainingState:
    """Debug mesh_fit training — supervises dual-primitives against a
    precomputed TSDF of a reference mesh.

    NOT paper-parity: the paper supervises via the differentiable
    renderer, not against an SDF. Use this ONLY as the cheap sanity
    path before spending pod credits.

    Each batch samples ``samples_per_batch`` random query points from
    ``(query_points, target_sdf)`` and runs loss_tsdf + regularizers.

    Pruning and parameter clipping reuse the same helpers as train().
    """
    opt_params = [scene.params]
    # No lighting MLP in mesh_fit mode (no rendering)
    optimizer = torch.optim.Adam(opt_params, lr=config.learning_rate)

    scheduler = None
    if config.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.num_iterations,
            eta_min=config.learning_rate * 0.1,
        )

    P = query_points.shape[0]
    loss_history: list[dict] = []
    timings = {"total": 0.0, "loss": 0.0, "step": 0.0, "prune": 0.0}
    t_start = time.time()

    for it in range(config.num_iterations):
        idx = torch.randint(0, P, (samples_per_batch,), device=device)
        qp = query_points[idx]
        tg = target_sdf[idx]

        theta_min_eff = _theta_min_curriculum(config, it)

        t0 = time.time()
        loss, parts = total_loss_tsdf(
            scene, qp, tg,
            lambda_sparse=config.lambda_sparse,
            lambda_entropy=config.lambda_entropy,
            lambda_max=config.lambda_max,
            mu=config.mu_gate_offset,
            theta_min=theta_min_eff,
        )
        timings["loss"] += time.time() - t0

        t0 = time.time()
        optimizer.zero_grad()
        if not torch.isfinite(loss):
            timings["step"] += time.time() - t0
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(opt_params, max_norm=1.0)
        any_nan_grad = any(
            (p.grad is not None and not torch.isfinite(p.grad).all())
            for p in opt_params
        )
        if any_nan_grad:
            optimizer.zero_grad()
            timings["step"] += time.time() - t0
            continue
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        clip_to_ranges(scene, config)
        timings["step"] += time.time() - t0

        if it % config.pruning_interval == 0 and it >= config.warmup_iterations:
            t0 = time.time()
            prune(scene, config, verbose=(log_fn is not None))
            timings["prune"] += time.time() - t0

        if it % config.log_interval == 0:
            parts["iter"] = it
            parts["alive"] = scene.num_alive
            parts["theta_min_eff"] = theta_min_eff
            diag = _dp_diagnostics(
                scene, ray_sampler=None,
                theta_min_eff=theta_min_eff,
                mu=config.mu_gate_offset,
            )
            parts.update(diag)
            loss_history.append(parts)
            if log_fn is not None:
                log_fn(it, parts)

        if checkpoint_path and it > 0 and it % config.checkpoint_interval == 0:
            _save_checkpoint(scene, checkpoint_path, it)

    timings["total"] = time.time() - t_start
    return TrainingState(
        scene=scene, iteration=config.num_iterations,
        loss_history=loss_history, timings=timings,
    )


def _save_checkpoint(scene: DualPrimScene, path: str, it: int):
    out = Path(path) / f"ckpt_{it:06d}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "params": scene.params.detach().cpu(),
        "alive": scene.alive.detach().cpu(),
        "iteration": it,
    }
    if scene.lighting_mlp is not None:
        state["lighting_mlp"] = scene.lighting_mlp.state_dict()
    torch.save(state, out)
