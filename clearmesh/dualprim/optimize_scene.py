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
from contextlib import nullcontext
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
from clearmesh.dualprim.losses import (
    loss_mask,
    loss_norm_reg,
    loss_rgb,
    total_loss,
    total_loss_tsdf,
)


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

    s_lo, s_hi = config.scale_range
    if config.init_profile == "paper_random":
        init_s_hi = s_hi
        params[:, IDX_PSQ_SCALE] = _uniform(s_lo, init_s_hi, (K, 3))
        params[:, IDX_PSQ_SHAPE] = _uniform(*config.shape_range, (K, 2))
        params[:, IDX_NSQ_SHAPE] = _uniform(*config.shape_range, (K, 2))
        params[:, IDX_ALPHA] = _uniform(*config.alpha_range, (K,))
        theta_lo = max(config.theta_min, 0.05)
        params[:, IDX_THETA] = _uniform(theta_lo, config.sharpness_range[1], (K,))
    elif config.init_profile == "paper_table":
        # Paper Table "primitive_init": fixed initial scale / shape /
        # opacity / sharpness, with random placement handled below.
        # Keep this separate from "paper_random": the latter samples
        # broadly across Table 1 ranges and is useful as a stress test,
        # but it is not the table-style initialization.
        init_s_hi = s_hi
        params[:, IDX_PSQ_SCALE] = torch.full((K, 3), 0.1, device=device)
        params[:, IDX_PSQ_SHAPE] = torch.ones((K, 2), device=device)
        params[:, IDX_NSQ_SHAPE] = torch.ones((K, 2), device=device)
        params[:, IDX_ALPHA] = torch.ones(K, device=device)
        params[:, IDX_THETA] = torch.full((K,), 0.5, device=device)
    elif config.init_profile == "biased":
        # PSQ / NSQ scale — from config range, biased small initially
        init_s_hi = min(s_hi, 0.3)  # start compact so they don't cover the whole cube
        params[:, IDX_PSQ_SCALE] = _uniform(s_lo, init_s_hi, (K, 3))

        # Shape — friend's tuning: bias init toward boxier shapes (lower ε
        # = more box-like; ε=1 is sphere). Manmade objects like a hole-box
        # tend to want sharp primitives, and the optimizer rarely pushes ε
        # downward from a sphere init.
        params[:, IDX_PSQ_SHAPE] = _uniform(0.2, 0.8, (K, 2))
        params[:, IDX_NSQ_SHAPE] = _uniform(0.2, 0.8, (K, 2))

        # α — start small (sparse) so the sparsity loss has room to work
        params[:, IDX_ALPHA] = _uniform(0.3, 0.5, (K,))

        # θ (render sharpness) — mid-range
        params[:, IDX_THETA] = _uniform(0.3, 0.7, (K,))
    else:
        raise ValueError(f"unknown init_profile: {config.init_profile}")

    # Translation + NSQ scale — depends on init strategy
    t_lo, t_hi = config.init_space
    params[:, IDX_PSQ_TRANSLATION] = _uniform(t_lo, t_hi, (K, 3))

    import math
    params[:, IDX_PSQ_ROTATION] = _uniform(-math.pi, math.pi, (K, 3))

    if config.nsq_init_strategy == "coupled":
        # NSQ starts at PSQ position with smaller scale (legacy default).
        params[:, IDX_NSQ_SCALE] = params[:, IDX_PSQ_SCALE] * 0.7
        params[:, IDX_NSQ_TRANSLATION] = (
            params[:, IDX_PSQ_TRANSLATION]
            + _uniform(-0.05, 0.05, (K, 3))
        )
        params[:, IDX_NSQ_ROTATION] = params[:, IDX_PSQ_ROTATION].clone()
    elif config.nsq_init_strategy == "coupled_axial":
        # Round 10 addition: coupled init BUT each NSQ has one
        # randomly-chosen principal axis elongated 3x. Round 9 at
        # K=100 produced 4 near-axis carvers but all with isotropic
        # scale ~0.25 — the local minimum where NSQs poke surface
        # pits instead of carving through the volume. Axial init
        # gives a subset of primitives the shape to span through a
        # 0.6-long hole from the start.
        nsq_base = params[:, IDX_PSQ_SCALE] * 0.7
        # Pick one of 3 axes per primitive uniformly
        axis = torch.randint(0, 3, (K,), generator=g, device=device)
        elongation = 3.0
        # Multiply the chosen axis by elongation
        for i in range(K):
            nsq_base[i, axis[i]] *= elongation
        # Clip to config range so we don't exceed scale_range[1]
        nsq_base = nsq_base.clamp(s_lo, s_hi)
        params[:, IDX_NSQ_SCALE] = nsq_base
        params[:, IDX_NSQ_TRANSLATION] = (
            params[:, IDX_PSQ_TRANSLATION]
            + _uniform(-0.05, 0.05, (K, 3))
        )
        params[:, IDX_NSQ_ROTATION] = params[:, IDX_PSQ_ROTATION].clone()
    elif config.nsq_init_strategy == "independent":
        # Paper-faithful: NSQ random in [-1,1]^3, scale independent.
        if config.init_profile == "paper_table":
            params[:, IDX_NSQ_SCALE] = torch.full((K, 3), 0.1, device=device)
        else:
            params[:, IDX_NSQ_SCALE] = _uniform(s_lo, init_s_hi, (K, 3))
        params[:, IDX_NSQ_TRANSLATION] = _uniform(t_lo, t_hi, (K, 3))
        params[:, IDX_NSQ_ROTATION] = _uniform(-math.pi, math.pi, (K, 3))
    else:
        raise ValueError(f"unknown nsq_init_strategy: {config.nsq_init_strategy}")

    if config.init_profile == "paper_table":
        params[:, IDX_COLOR] = _uniform(*config.color_range, (K, 3))
    else:
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


def reset_opacity(
    scene: DualPrimScene,
    config: DualPrimConfig,
    verbose: bool = False,
) -> int:
    """3DGS-style opacity reset to re-open alpha competition.

    DualPrim says its adaptive pruning follows a strategy similar to
    3DGS. In the official 3DGS implementation, pruning/densification is
    paired with periodic opacity resets. We do not import 3DGS
    densification here, but we do adopt the alpha reset itself to keep
    the primitive competition from freezing once all survivors saturate
    to α≈1.

    Returns the number of alive primitives whose alpha was reduced.
    """
    with torch.no_grad():
        alive = scene.alive
        alpha = scene.params[:, IDX_ALPHA]
        capped = alpha[alive].clamp(min=0.0, max=config.opacity_reset_value)
        changed = int((alpha[alive] > config.opacity_reset_value).sum().item())
        alpha[alive] = capped
    if verbose and changed > 0:
        print(f"[opacity_reset] capped {changed} alive primitives to α<={config.opacity_reset_value}; "
              f"{scene.num_alive} alive")
    return changed


def _theta_min_curriculum(config: DualPrimConfig, iteration: int) -> float:
    """θ-curriculum schedule.

    Returns the EFFECTIVE theta_min for the current training step,
    linearly annealing from config.theta_curriculum_start down to
    config.theta_min over the first config.theta_curriculum_fraction
    of training, then holding at config.theta_min.
    """
    if getattr(config, "gate_mode", "stabilized") == "paper_literal":
        return config.theta_min

    tot = max(config.num_iterations, 1)
    frac = iteration / tot
    stop = config.theta_curriculum_fraction
    if frac >= stop:
        return config.theta_min
    alpha = frac / max(stop, 1e-9)
    return (1.0 - alpha) * config.theta_curriculum_start + alpha * config.theta_min


def _mu_gate_schedule(config: DualPrimConfig, iteration: int) -> float:
    """Optional late-stage μ ramp for a sharper P_E gate.

    Defaults to the constant paper-style behavior when
    ``mu_gate_offset_final`` is unset or equal to ``mu_gate_offset``.
    """
    mu_start = config.mu_gate_offset
    mu_final = (
        config.mu_gate_offset
        if config.mu_gate_offset_final is None
        else config.mu_gate_offset_final
    )
    if mu_final == mu_start:
        return mu_start

    tot = max(config.num_iterations, 1)
    frac = iteration / tot
    start = config.mu_gate_ramp_start_fraction
    if frac <= start:
        return mu_start
    alpha = min(1.0, (frac - start) / max(1.0 - start, 1e-9))
    return (1.0 - alpha) * mu_start + alpha * mu_final


def _norm_reg_schedule(config: DualPrimConfig, iteration: int) -> float:
    """Optional late-stage ramp for the normal consistency weight."""
    lam_start = config.lambda_norm_reg
    lam_final = (
        config.lambda_norm_reg
        if config.lambda_norm_reg_final is None
        else config.lambda_norm_reg_final
    )
    if lam_final == lam_start:
        return lam_start

    tot = max(config.num_iterations, 1)
    frac = iteration / tot
    start = config.norm_reg_ramp_start_fraction
    end = max(start, config.norm_reg_ramp_end_fraction)
    if frac <= start:
        return lam_start
    if frac >= end:
        return lam_final
    alpha = (frac - start) / max(end - start, 1e-9)
    return (1.0 - alpha) * lam_start + alpha * lam_final


def _overlap_schedule(config: DualPrimConfig, iteration: int) -> float:
    """Optional decay/ramp for the PSQ overlap repulsion.

    The overlap term is most useful early, when it breaks the
    stacked-at-origin basin. Late in training it can fight legitimate
    contact between adjacent body/lens primitives, so callers can ramp
    it down after the layout has spread.
    """
    lam_start = config.lambda_overlap
    lam_final = (
        config.lambda_overlap
        if config.lambda_overlap_final is None
        else config.lambda_overlap_final
    )
    if lam_final == lam_start:
        return lam_start

    tot = max(config.num_iterations, 1)
    frac = iteration / tot
    start = config.overlap_ramp_start_fraction
    end = max(start, config.overlap_ramp_end_fraction)
    if frac <= start:
        return lam_start
    if frac >= end:
        return lam_final
    alpha = (frac - start) / max(end - start, 1e-9)
    return (1.0 - alpha) * lam_start + alpha * lam_final


def _region_ownership_schedule(config: DualPrimConfig, iteration: int) -> float:
    """Assignment-phase support prior, optionally released later.

    SuperFit/Marching-Primitives style fitting keeps primitives local
    while they are assigned, then lets the final assembly settle. This
    schedule makes that release explicit instead of leaving a permanent
    r58-style tether.
    """
    lam_start = float(getattr(config, "lambda_region_ownership", 0.0))
    lam_final_cfg = getattr(config, "lambda_region_ownership_final", None)
    if lam_start <= 0.0 or lam_final_cfg is None:
        return lam_start
    lam_final = float(lam_final_cfg)
    if lam_final == lam_start:
        return lam_start

    tot = max(config.num_iterations, 1)
    frac = iteration / tot
    start = float(getattr(config, "region_ownership_ramp_start_fraction", 0.2))
    end = max(start, float(getattr(config, "region_ownership_ramp_end_fraction", 0.5)))
    if frac <= start:
        return lam_start
    if frac >= end:
        return lam_final
    alpha = (frac - start) / max(end - start, 1e-9)
    return (1.0 - alpha) * lam_start + alpha * lam_final


def _adaptive_prune_target(config: DualPrimConfig, iteration: int) -> int:
    """Late compactness schedule for contribution-ranked primitive selection.

    The DualPrim video shows a very overcomplete candidate soup early, then a
    much smaller useful assembly late. This target makes that attrition explicit:
    do not force compactness while primitives are still discovering support, but
    progressively select the best contributors once the coarse shape exists.
    """
    final = int(getattr(config, "adaptive_prune_target_final", 0))
    if final <= 0:
        return 0

    start_count = int(getattr(config, "visual_hull_active_start", 0))
    if start_count <= 0:
        start_count = int(getattr(config, "num_primitives_init", 0))
    min_keep = max(int(getattr(config, "adaptive_prune_min_keep", 8)), 1)
    final = max(final, min_keep)

    frac = iteration / max(int(config.num_iterations), 1)
    start = float(getattr(config, "adaptive_prune_start_fraction", 0.25))
    end = max(start, float(getattr(config, "adaptive_prune_end_fraction", 0.75)))
    if frac <= start:
        return max(start_count, final)
    if frac >= end:
        return final

    alpha = (frac - start) / max(end - start, 1e-9)
    return int(round((1.0 - alpha) * start_count + alpha * final))


def _dp_diagnostics(
    scene: DualPrimScene,
    ray_sampler: Optional["RaySampler"] = None,
    n_probe: int = 1024,
    theta_min_eff: float = 0.01,
    theta_min_nsq: float = 0.01,
    mu: float = 0.0,
    gate_mode: str = "stabilized",
    paper_literal_theta_eps: float = 1e-6,
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
        psq_vol = psq_s.prod(dim=-1).clamp_min(1e-8)
        carve_ratio = (nsq_s.prod(dim=-1) / psq_vol).clamp(0.0, 10.0)
        nsq_offset = (nsq_t - psq_t).norm(dim=-1) / psq_s.norm(dim=-1).clamp_min(1e-8)
        cr_q = torch.quantile(
            carve_ratio.float(),
            torch.tensor([0.1, 0.5, 0.9], device=carve_ratio.device),
        )
        off_q = torch.quantile(
            nsq_offset.float(),
            torch.tensor([0.1, 0.5, 0.9], device=nsq_offset.device),
        )
        out["carve_ratio_p10"] = float(cr_q[0].item())
        out["carve_ratio_p50"] = float(cr_q[1].item())
        out["carve_ratio_p90"] = float(cr_q[2].item())
        out["nsq_offset_p10"] = float(off_q[0].item())
        out["nsq_offset_p50"] = float(off_q[1].item())
        out["nsq_offset_p90"] = float(off_q[2].item())

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
                    theta_min_nsq=theta_min_nsq,
                    gate_mode=gate_mode,
                    paper_literal_theta_eps=paper_literal_theta_eps,
                )
                out["pe_mean_fg"] = float(p_e.mean().item())
                out["pe_max_fg"] = float(p_e.max().item())
        return out


def _estimate_view_contribution(
    scene: DualPrimScene,
    ray_sampler: "RaySampler",
    config: DualPrimConfig,
    *,
    num_probe_rays: int | None = None,
    weight_threshold: float | None = None,
    foreground_only: bool = True,
    min_foreground_rays: int | None = None,
    min_distinct_views: int | None = None,
    mu: float | None = None,
    theta_min: float | None = None,
    verbose: bool = False,
) -> tuple[torch.Tensor | None, int, int]:
    """Estimate per-primitive compositing contribution on foreground probes."""
    from clearmesh.dualprim.renderer import (
        sample_ray_points,
        _scene_field,
        _delta_p_offsets,
        density_from_field,
        _accumulated_transmittance,
    )

    with torch.no_grad():
        if num_probe_rays is None:
            num_probe_rays = config.view_prune_probe_rays
        if weight_threshold is None:
            weight_threshold = config.view_prune_weight_threshold
        if min_foreground_rays is None:
            min_foreground_rays = config.view_prune_min_foreground_rays
        if min_distinct_views is None:
            min_distinct_views = config.view_prune_min_distinct_views
        if mu is None:
            mu = config.mu_gate_offset
        if theta_min is None:
            theta_min = config.theta_min

        if hasattr(ray_sampler, "sample_view_probe"):
            batch = ray_sampler.sample_view_probe(
                num_probe_rays,
                foreground_only=foreground_only,
            )
        else:
            batch = ray_sampler(num_probe_rays)
        view_idx = getattr(batch, "view_idx", None)

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
                return None, n_fg, 0
            origins = batch.origins[fg_mask]
            dirs = batch.dirs[fg_mask]
            if view_idx is not None:
                view_idx = view_idx[fg_mask]
            R_eff = n_fg
        else:
            origins = batch.origins
            dirs = batch.dirs
            if view_idx is not None:
                view_idx = view_idx
            R_eff = num_probe_rays

        points, t_vals, deltas = sample_ray_points(
            origins, dirs,
            near=config.near_plane, far=config.far_plane,
            N=config.num_samples_per_ray,
            perturb=False, device=scene.params.device,
        )

        dp = _delta_p_offsets(
            dirs, deltas,
            delta_p=config.delta_p_value,
            delta_p_mode=config.delta_p_mode,
            delta_p_scale=config.delta_p_scale,
        )
        f_fwd = _scene_field(
            scene, points + dp, mu, theta_min,
            theta_min_nsq=getattr(config, "theta_min_nsq", config.theta_min),
            gate_mode=config.gate_mode,
            paper_literal_theta_eps=config.paper_literal_theta_eps,
        )
        f_bwd = _scene_field(
            scene, points - dp, mu, theta_min,
            theta_min_nsq=getattr(config, "theta_min_nsq", config.theta_min),
            gate_mode=config.gate_mode,
            paper_literal_theta_eps=config.paper_literal_theta_eps,
        )
        sigma_k = density_from_field(
            f_fwd, f_bwd, scene.theta(), theta_min=theta_min,
            gate_mode=config.gate_mode,
            paper_literal_theta_eps=config.paper_literal_theta_eps,
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
        per_ray_prim_contrib = (
            (sigma_k_weighted / denom) * w_ray.unsqueeze(-1)
        ).sum(dim=1)                                          # (R_eff, K)

        if view_idx is not None:
            uniq_views = torch.unique(view_idx)
            if uniq_views.numel() < min_distinct_views:
                if verbose:
                    print(f"[prune/view] only {uniq_views.numel()} distinct views "
                          f"(< {min_distinct_views}); skipping prune cycle")
                return None, R_eff, int(uniq_views.numel())
            per_view_contrib = []
            for v in uniq_views:
                mask_v = view_idx == v
                per_view_contrib.append(
                    per_ray_prim_contrib[mask_v].sum(dim=0) / max(int(mask_v.sum().item()), 1)
                )
            per_prim_contrib = torch.stack(per_view_contrib, dim=0).mean(dim=0)
            n_views_eff = int(uniq_views.numel())
        else:
            # Fallback: normalize by foreground-hit-ray count, NOT total probe rays.
            per_prim_contrib = per_ray_prim_contrib.sum(dim=0) / max(R_eff, 1)
            n_views_eff = 1

        return per_prim_contrib, R_eff, n_views_eff


def prune_view_dependent(
    scene: DualPrimScene,
    ray_sampler: "RaySampler",
    config: DualPrimConfig,
    *,
    num_probe_rays: int | None = None,
    weight_threshold: float | None = None,
    foreground_only: bool = True,
    min_foreground_rays: int | None = None,
    min_distinct_views: int | None = None,
    mu: float | None = None,
    theta_min: float | None = None,
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

    Runs under ``torch.no_grad``.
    """
    with torch.no_grad():
        per_prim_contrib, R_eff, n_views_eff = _estimate_view_contribution(
            scene, ray_sampler, config,
            num_probe_rays=num_probe_rays,
            weight_threshold=weight_threshold,
            foreground_only=foreground_only,
            min_foreground_rays=min_foreground_rays,
            min_distinct_views=min_distinct_views,
            mu=mu,
            theta_min=theta_min,
            verbose=verbose,
        )
        if per_prim_contrib is None:
            return 0
        if weight_threshold is None:
            weight_threshold = config.view_prune_weight_threshold
        kill = scene.alive & (per_prim_contrib < weight_threshold)
        n_killed = int(kill.sum().item())
        scene.alive &= ~kill
        scene.params[kill, IDX_ALPHA] = 0.0

    if verbose and n_killed > 0:
        print(f"[prune/view] killed {n_killed} primitives "
              f"(contribution<{weight_threshold}, over {R_eff} fg rays / {n_views_eff} views); "
              f"{scene.num_alive} alive")
    return n_killed


def prune_to_active_budget(
    scene: DualPrimScene,
    ray_sampler: "RaySampler",
    config: DualPrimConfig,
    target_alive: int,
    *,
    num_probe_rays: int | None = None,
    foreground_only: bool = True,
    min_foreground_rays: int | None = None,
    min_distinct_views: int | None = None,
    mu: float | None = None,
    theta_min: float | None = None,
    verbose: bool = False,
) -> int:
    """Select the top contributing alive primitives until a budget is met."""
    with torch.no_grad():
        min_keep = max(int(getattr(config, "adaptive_prune_min_keep", 8)), 1)
        target_alive = max(int(target_alive), min_keep)
        alive_idx = scene.alive.nonzero(as_tuple=True)[0]
        n_alive = int(alive_idx.numel())
        if n_alive <= target_alive:
            return 0

        per_prim_contrib, R_eff, n_views_eff = _estimate_view_contribution(
            scene, ray_sampler, config,
            num_probe_rays=num_probe_rays,
            foreground_only=foreground_only,
            min_foreground_rays=min_foreground_rays,
            min_distinct_views=min_distinct_views,
            mu=mu,
            theta_min=theta_min,
            verbose=verbose,
        )
        if per_prim_contrib is None:
            return 0

        n_to_kill = n_alive - target_alive
        contrib_alive = torch.nan_to_num(
            per_prim_contrib[alive_idx],
            nan=-float("inf"),
            posinf=float("inf"),
            neginf=-float("inf"),
        )
        kill_order = torch.argsort(contrib_alive, descending=False)[:n_to_kill]
        kill_idx = alive_idx[kill_order]
        scene.alive[kill_idx] = False
        scene.params[kill_idx, IDX_ALPHA] = 0.0

    if verbose and n_to_kill > 0:
        kept_floor = float(torch.topk(contrib_alive, k=target_alive, largest=True).values.min().item())
        print(
            f"[prune/adaptive] killed {n_to_kill} lowest-contribution primitives "
            f"to target={target_alive} (keep_floor={kept_floor:.4g}, "
            f"over {R_eff} fg rays / {n_views_eff} views); {scene.num_alive} alive",
            flush=True,
        )
    return n_to_kill


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


def _param_col_name(col: int) -> str:
    if IDX_PSQ_SCALE.start <= col < IDX_PSQ_SCALE.stop:
        return f"psq_scale[{col - IDX_PSQ_SCALE.start}]"
    if IDX_NSQ_SCALE.start <= col < IDX_NSQ_SCALE.stop:
        return f"nsq_scale[{col - IDX_NSQ_SCALE.start}]"
    if IDX_PSQ_SHAPE.start <= col < IDX_PSQ_SHAPE.stop:
        return f"psq_shape[{col - IDX_PSQ_SHAPE.start}]"
    if IDX_NSQ_SHAPE.start <= col < IDX_NSQ_SHAPE.stop:
        return f"nsq_shape[{col - IDX_NSQ_SHAPE.start}]"
    if col == IDX_ALPHA:
        return "alpha"
    if col == IDX_THETA:
        return "theta"
    if IDX_PSQ_TRANSLATION.start <= col < IDX_PSQ_TRANSLATION.stop:
        return f"psq_translation[{col - IDX_PSQ_TRANSLATION.start}]"
    if IDX_NSQ_TRANSLATION.start <= col < IDX_NSQ_TRANSLATION.stop:
        return f"nsq_translation[{col - IDX_NSQ_TRANSLATION.start}]"
    if IDX_PSQ_ROTATION.start <= col < IDX_PSQ_ROTATION.stop:
        return f"psq_rotation[{col - IDX_PSQ_ROTATION.start}]"
    if IDX_NSQ_ROTATION.start <= col < IDX_NSQ_ROTATION.stop:
        return f"nsq_rotation[{col - IDX_NSQ_ROTATION.start}]"
    if IDX_COLOR.start <= col < IDX_COLOR.stop:
        return f"color[{col - IDX_COLOR.start}]"
    return f"col[{col}]"


def _nan_grad_summary(
    scene: DualPrimScene,
    opt_named_params: list[tuple[str, torch.Tensor]],
    *,
    max_rows: int = 6,
    max_cols: int = 10,
) -> str:
    """Summarize which parameter blocks first went non-finite."""
    out: list[str] = []

    g = scene.params.grad
    if g is not None and not torch.isfinite(g).all():
        bad = ~torch.isfinite(g)
        rows = bad.any(dim=1).nonzero(as_tuple=True)[0].tolist()
        cols = bad.any(dim=0).nonzero(as_tuple=True)[0].tolist()
        col_names = [_param_col_name(c) for c in cols[:max_cols]]
        out.append(f"scene.params rows={rows[:max_rows]} cols={col_names}")

    for name, p in opt_named_params:
        if p is scene.params:
            continue
        if p.grad is not None and not torch.isfinite(p.grad).all():
            n_bad = int((~torch.isfinite(p.grad)).sum().item())
            out.append(f"{name} bad={n_bad}")

    return "; ".join(out) if out else "nonfinite grad but no source summary"


def _scene_value_summary(scene: DualPrimScene) -> str:
    """Compact scalar summary of scene parameter ranges at failure time."""
    with torch.no_grad():
        return (
            f"psq_scale=[{scene.psq_scale().min().item():.3g},{scene.psq_scale().max().item():.3g}] "
            f"nsq_scale=[{scene.nsq_scale().min().item():.3g},{scene.nsq_scale().max().item():.3g}] "
            f"psq_shape=[{scene.psq_shape().min().item():.3g},{scene.psq_shape().max().item():.3g}] "
            f"nsq_shape=[{scene.nsq_shape().min().item():.3g},{scene.nsq_shape().max().item():.3g}] "
            f"theta=[{scene.theta().min().item():.3g},{scene.theta().max().item():.3g}] "
            f"alpha=[{scene.alpha().min().item():.3g},{scene.alpha().max().item():.3g}]"
        )


def _visual_hull_birth_scores(
    scene: DualPrimScene,
    batch: "RaySampleBatch",
    render,
    config: DualPrimConfig,
) -> torch.Tensor | None:
    """Score queued visual-hull slots by current residual evidence.

    This mirrors the density-control idea used by successful explicit
    primitive methods: do not merely add capacity on a timer; allocate it
    near rays where the current model is under-explaining foreground.
    Depth supervision, when available from synthetic or calibrated depth
    views, gives direct 3D residual points. Pure paper/RGB-mask mode falls
    back to a ray-to-region score using the same silhouette evidence.
    """

    anchors = getattr(scene, "region_anchors", None)
    scales = getattr(scene, "region_scales", None)
    active = getattr(scene, "region_active", None)
    if anchors is None or scales is None or active is None:
        return None

    with torch.no_grad():
        device = scene.params.device
        anchors = anchors.to(device=device, dtype=scene.params.dtype)
        scales = scales.to(device=device, dtype=scene.params.dtype).clamp_min(1e-3)
        valid = active.to(device=device) & ~scene.alive
        if not bool(valid.any().item()):
            return None

        mask_gt = batch.mask_gt.detach().to(device=device, dtype=scene.params.dtype)
        mask_pred = render.mask.detach().to(device=device, dtype=scene.params.dtype)
        residual = (mask_gt - mask_pred).clamp_min(0.0)
        fg = mask_gt > 0.5
        keep = fg & (residual > 1e-4)
        if not bool(keep.any().item()):
            scores = torch.zeros(scene.K, device=device, dtype=scene.params.dtype)
            return scores

        origins = batch.origins.detach().to(device=device, dtype=scene.params.dtype)[keep]
        dirs = batch.dirs.detach().to(device=device, dtype=scene.params.dtype)[keep]
        residual = residual[keep]
        sigma = float(getattr(config, "visual_hull_birth_region_sigma", 1.5))
        sigma = max(sigma, 1e-3)

        depth_gt = getattr(batch, "depth_gt", None)
        if depth_gt is not None:
            depth = depth_gt.detach().to(device=device, dtype=scene.params.dtype)[keep]
            depth_ok = torch.isfinite(depth) & (depth > config.near_plane) & (depth < config.far_plane)
        else:
            depth_ok = torch.zeros_like(residual, dtype=torch.bool)

        queued_anchors = anchors[valid]
        queued_scales = scales[valid]
        if bool(depth_ok.any().item()):
            points = origins[depth_ok] + dirs[depth_ok] * depth[depth_ok].unsqueeze(-1)
            diff = (points[:, None, :] - queued_anchors[None, :, :]) / (queued_scales[None, :, :] * sigma)
            dist2 = diff.pow(2).sum(dim=-1)
            weights = torch.exp(-0.5 * dist2)
            res = residual[depth_ok]
        else:
            # Silhouette-only fallback: a region is responsible for a ray
            # when the ray passes near its support box center.
            offset = queued_anchors[None, :, :] - origins[:, None, :]
            t = (offset * dirs[:, None, :]).sum(dim=-1)
            closest = origins[:, None, :] + t.unsqueeze(-1) * dirs[:, None, :]
            radius = queued_scales.norm(dim=-1).clamp_min(1e-3) * sigma
            dist2 = ((queued_anchors[None, :, :] - closest).norm(dim=-1) / radius[None, :]).pow(2)
            in_segment = (t > config.near_plane) & (t < config.far_plane)
            weights = torch.exp(-0.5 * dist2) * in_segment.to(scene.params.dtype)
            res = residual

        denom = weights.sum(dim=0).clamp_min(1.0).sqrt()
        queued_scores = (weights * res[:, None]).sum(dim=0) / denom
        scores = torch.zeros(scene.K, device=device, dtype=scene.params.dtype)
        scores[valid] = queued_scores
        return scores


class RaySampleBatch:
    """Typed batch of training rays.

    All tensors must live on the same device as the scene.

    hole_ray_gt: (R,) bool tensor flagging rays whose pixel lies INSIDE
    the filled silhouette but OUTSIDE the actual mask — i.e. "this
    ray should pass through empty space because the ref mesh has a
    hole here." Used by the open-ray loss (losses.loss_open_ray) to
    specifically reward NSQ carving on these rays. Default: all False
    (backward-compatible; the loss term is then zero).
    """
    def __init__(
        self,
        origins: torch.Tensor,      # (R, 3)
        dirs: torch.Tensor,         # (R, 3)
        rgb_gt: torch.Tensor,       # (R, 3)
        mask_gt: torch.Tensor,      # (R,)
        normals_gt: torch.Tensor,   # (R, 3)
        depth_gt: Optional[torch.Tensor] = None,  # (R,)
        hole_ray_gt: Optional[torch.Tensor] = None,  # (R,) bool
        edge_weight_gt: Optional[torch.Tensor] = None,  # (R,) float/bool
        view_idx: Optional[torch.Tensor] = None,     # (R,) long
    ):
        self.origins = origins
        self.dirs = dirs
        self.rgb_gt = rgb_gt
        self.mask_gt = mask_gt
        self.normals_gt = normals_gt
        self.depth_gt = depth_gt
        self.hole_ray_gt = hole_ray_gt
        self.edge_weight_gt = edge_weight_gt
        self.view_idx = view_idx


def train(
    scene: DualPrimScene,
    ray_sampler: RaySampler,
    config: DualPrimConfig,
    *,
    rays_per_batch: int = 1024,
    device: str = "cuda",
    log_fn: Optional[Callable[[int, dict], None]] = None,
    checkpoint_path: Optional[str] = None,
    trajectory_dir: Optional[str] = None,
    trajectory_iters: Optional[list[int]] = None,
    detect_anomaly: bool = False,
    abort_on_nan_grad: bool = False,
) -> TrainingState:
    """Run per-scene optimization.

    ray_sampler(n) returns a RaySampleBatch of n rays with RGB/mask/
    normal GT. Caller is responsible for building this — see
    scripts/dualprim/run_canary.py for the mesh-rendered-views
    implementation.

    Trajectory snapshots (for training a warm-start / refinement
    predictor downstream): if ``trajectory_dir`` is given, write the
    scene's live primitives to ``{trajectory_dir}/step_{N:06d}.json``
    at every iter in ``trajectory_iters``. Default snapshot schedule
    is canonical (~log-spaced over the run) — see
    clearmesh.dualprim.io.canonical_trajectory_iters.

    Saving JSON mid-training is cheap: no renderer, no gradients, just
    pack live primitives into a dict and dump. Empirically <50ms per
    snapshot even for K=100. Five snapshots over a 15k-iter run is
    negligible overhead but gives the downstream dataset ~5× more
    training points per mesh than endpoint-only.
    """
    from clearmesh.dualprim.io import save_scene_json, canonical_trajectory_iters

    if trajectory_dir is not None:
        if trajectory_iters is None:
            trajectory_iters = canonical_trajectory_iters(config.num_iterations)
        trajectory_iter_set = set(trajectory_iters)
    else:
        trajectory_iter_set = set()
    opt_named_params: list[tuple[str, torch.Tensor]] = [("scene.params", scene.params)]
    if scene.lighting_mlp is not None:
        opt_named_params += [
            (f"lighting_mlp.{name}", p)
            for name, p in scene.lighting_mlp.named_parameters()
        ]
    opt_params = [p for _, p in opt_named_params]
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

    # Heartbeat observability — prints iter + timings every HEARTBEAT_S
    # seconds REGARDLESS of log_interval. Prevents the "slow vs hung"
    # ambiguity: a quiet run that's actually just slow now shows a
    # heartbeat every 60s so watchdogs and humans know it's alive.
    #
    # Friend's review (critical): if iter time balloons to ~3s at
    # K=100 coupled, 200-iter log_interval → 10 min silence → watchdog
    # labels "slow" as "hung". Heartbeat fixes that categorically.
    HEARTBEAT_S = 60.0
    last_heartbeat = time.time()
    prev_shape_grad_eps_by_idx: dict[int, float] = {}
    birth_interval = int(getattr(config, "visual_hull_birth_interval", 0) or 0)
    birth_count = int(getattr(config, "visual_hull_birth_count", 0) or 0)
    birth_stop_fraction = float(getattr(config, "visual_hull_birth_stop_fraction", 0.5))
    birth_strategy = str(getattr(config, "visual_hull_birth_strategy", "residual"))
    if birth_strategy not in {"residual", "scheduled", "hybrid"}:
        raise ValueError(f"unknown visual_hull_birth_strategy: {birth_strategy}")

    for it in range(config.num_iterations):
        batch = ray_sampler(rays_per_batch)

        # θ curriculum — broad gate early, sharper as training progresses.
        # Review feedback: without this, θ collapses to its floor on
        # every primitive and the P_E gate becomes razor-thin, so NSQs
        # that drift outside their PSQs never find their way back.
        theta_min_eff = _theta_min_curriculum(config, it)
        mu_eff = _mu_gate_schedule(config, it)
        lambda_norm_eff = _norm_reg_schedule(config, it)
        lambda_overlap_eff = _overlap_schedule(config, it)
        lambda_region_eff = _region_ownership_schedule(config, it)

        anomaly_ctx = torch.autograd.detect_anomaly(check_nan=True) if detect_anomaly else nullcontext()
        try:
            with anomaly_ctx:
                t0 = time.time()
                render = render_rays(
                    scene,
                    batch.origins, batch.dirs,
                    num_samples=config.num_samples_per_ray,
                    sampling_mode=getattr(config, "sampling_mode", "uniform"),
                    num_importance_samples=getattr(config, "num_importance_samples_per_ray", 0),
                    near=config.near_plane,
                    far=config.far_plane,
                    delta_p=config.delta_p_value,
                    delta_p_mode=config.delta_p_mode,
                    delta_p_scale=config.delta_p_scale,
                    color_weight_mode=config.color_weight_mode,
                    point_normal_weight_mode=config.point_normal_weight_mode,
                    final_normal_normalize=config.final_normal_normalize,
                    mu=mu_eff,
                    theta_min=theta_min_eff,
                    theta_min_nsq=getattr(config, 'theta_min_nsq', config.theta_min),  # friend's #1
                    gate_mode=config.gate_mode,
                    paper_literal_theta_eps=config.paper_literal_theta_eps,
                    background=config.background_color,
                )
                timings["render"] += time.time() - t0

                t0 = time.time()
                loss, parts = total_loss(
                    scene, render,
                    rgb_gt=batch.rgb_gt, mask_gt=batch.mask_gt,
                    normals_pred=batch.normals_gt,
                    depth_gt=getattr(batch, "depth_gt", None),
                    ray_origins=batch.origins,
                    ray_dirs=batch.dirs,
                    lambda_mask=config.lambda_mask,
                    lambda_sparse=config.lambda_sparse,
                    lambda_entropy=config.lambda_entropy,
                    lambda_max=config.lambda_max,
                    lambda_norm_reg=lambda_norm_eff,
                    lambda_depth=getattr(config, "lambda_depth", 0.0),
                    lambda_open_ray=config.lambda_open_ray,
                    lambda_nsq_carve=getattr(config, "lambda_nsq_carve", 0.0),
                    lambda_overlap=lambda_overlap_eff,
                    lambda_region_ownership=lambda_region_eff,
                    lambda_edge_mask=getattr(config, "lambda_edge_mask", 0.0),
                    lambda_shape_box=getattr(config, "lambda_shape_box", 0.0),
                    shape_box_threshold=getattr(config, "shape_box_threshold", 0.30),
                    region_anchor_margin=getattr(config, "region_anchor_margin", 1.0),
                    region_scale_growth=getattr(config, "region_scale_growth", 1.5),
                    hole_ray_gt=getattr(batch, "hole_ray_gt", None),
                    edge_weight_gt=getattr(batch, "edge_weight_gt", None),
                    mask_loss_type=getattr(config, "mask_loss_type", "bce"),
                    normal_loss_type=getattr(config, "normal_loss_type", "l1"),
                    masked_loss_norm_mode=getattr(config, "masked_loss_norm_mode", "global_mean"),
                    primitive_reg_average_mode=getattr(config, "primitive_reg_average_mode", "alive"),
                    mu=mu_eff,
                    theta_min=theta_min_eff,
                    theta_min_nsq=getattr(config, "theta_min_nsq", config.theta_min),
                    gate_mode=config.gate_mode,
                    paper_literal_theta_eps=config.paper_literal_theta_eps,
                    nsq_carve_samples=getattr(config, "nsq_carve_samples", 5),
                    nsq_carve_depth_band=getattr(config, "nsq_carve_depth_band", 0.08),
                    nsq_carve_residual_threshold=getattr(config, "nsq_carve_residual_threshold", 0.05),
                )
                timings["loss"] += time.time() - t0

                t0 = time.time()
                optimizer.zero_grad()
                audit_shape_grads = bool(getattr(config, "log_shape_grad_stats", False)) and (it % config.log_interval == 0)
                audit_cache = None
                if audit_shape_grads:
                    alive_idx = scene.alive.nonzero(as_tuple=True)[0]
                    if alive_idx.numel() > 0:
                        eps_alive = scene.psq_shape().detach()[alive_idx].mean(dim=-1)
                        theta_alive = scene.theta().detach()[alive_idx]

                        def _component_grad(component_loss: torch.Tensor) -> torch.Tensor:
                            grad = torch.autograd.grad(
                                component_loss,
                                scene.params,
                                retain_graph=True,
                                allow_unused=False,
                            )[0]
                            return grad.detach()[alive_idx, 6:8].mean(dim=-1)

                        audit_cache = {
                            "alive_idx": alive_idx.detach(),
                            "eps_alive": eps_alive.detach(),
                            "theta_alive": theta_alive.detach(),
                            "rgb_signed": _component_grad(
                                loss_rgb(
                                    render,
                                    batch.rgb_gt,
                                    batch.mask_gt,
                                    norm_mode=getattr(config, "masked_loss_norm_mode", "global_mean"),
                                )
                            ),
                            "mask_signed": _component_grad(
                                loss_mask(
                                    render,
                                    batch.mask_gt,
                                    loss_type=getattr(config, "mask_loss_type", "bce"),
                                )
                            ),
                            "norm_signed": _component_grad(
                                loss_norm_reg(
                                    render,
                                    batch.normals_gt,
                                    batch.mask_gt,
                                    norm_mode=getattr(config, "masked_loss_norm_mode", "global_mean"),
                                    loss_type=getattr(config, "normal_loss_type", "l1"),
                                )
                            ),
                        }
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
                if audit_shape_grads and audit_cache is not None and scene.params.grad is not None:
                    alive_idx = audit_cache["alive_idx"]
                    eps_alive = audit_cache["eps_alive"]
                    theta_alive = audit_cache["theta_alive"]
                    total_signed = scene.params.grad.detach()[alive_idx, 6:8].mean(dim=-1)
                    total_abs = scene.params.grad.detach()[alive_idx, 6:8].abs().mean(dim=-1)
                    delta_eps = []
                    for idx, eps_val in zip(alive_idx.detach().cpu().tolist(), eps_alive.detach().cpu().tolist()):
                        prev = prev_shape_grad_eps_by_idx.get(int(idx))
                        delta_eps.append(None if prev is None else float(eps_val - prev))
                        prev_shape_grad_eps_by_idx[int(idx)] = float(eps_val)
                    order = torch.argsort(eps_alive)
                    order_cpu = order.detach().cpu().tolist()
                    parts["shape_grad_alive_idx"] = alive_idx[order].detach().cpu().tolist()
                    parts["shape_grad_alive_eps_mean"] = eps_alive[order].detach().cpu().tolist()
                    parts["shape_grad_alive_theta"] = theta_alive[order].detach().cpu().tolist()
                    parts["shape_grad_alive_abs_mean"] = total_abs[order].detach().cpu().tolist()
                    parts["shape_grad_alive_signed_total"] = total_signed[order].detach().cpu().tolist()
                    parts["shape_grad_alive_signed_rgb"] = audit_cache["rgb_signed"][order].detach().cpu().tolist()
                    parts["shape_grad_alive_signed_mask"] = audit_cache["mask_signed"][order].detach().cpu().tolist()
                    parts["shape_grad_alive_signed_norm"] = audit_cache["norm_signed"][order].detach().cpu().tolist()
                    parts["shape_grad_alive_signed_rgb_weighted"] = audit_cache["rgb_signed"][order].detach().cpu().tolist()
                    parts["shape_grad_alive_signed_mask_weighted"] = (
                        config.lambda_mask * audit_cache["mask_signed"][order]
                    ).detach().cpu().tolist()
                    parts["shape_grad_alive_signed_norm_weighted"] = (
                        lambda_norm_eff * audit_cache["norm_signed"][order]
                    ).detach().cpu().tolist()
                    parts["shape_delta_alive_eps_mean"] = [delta_eps[i] for i in order_cpu]
                    parts["shape_grad_abs_p10"] = float(torch.quantile(total_abs, 0.1).item())
                    parts["shape_grad_abs_p50"] = float(torch.quantile(total_abs, 0.5).item())
                    parts["shape_grad_abs_p90"] = float(torch.quantile(total_abs, 0.9).item())
                    parts["shape_grad_signed_p10"] = float(torch.quantile(total_signed, 0.1).item())
                    parts["shape_grad_signed_p50"] = float(torch.quantile(total_signed, 0.5).item())
                    parts["shape_grad_signed_p90"] = float(torch.quantile(total_signed, 0.9).item())
        except RuntimeError:
            print(
                f"[trace_fail] it={it} theta_min_eff={theta_min_eff:.4f} "
                f"{_scene_value_summary(scene)}",
                flush=True,
            )
            raise
        # Clip gradients before the Adam step.
        torch.nn.utils.clip_grad_norm_(opt_params, max_norm=1.0)
        # Double-check gradients after clipping (clip doesn't fix NaN)
        any_nan_grad = any(
            (p.grad is not None and not torch.isfinite(p.grad).all())
            for p in opt_params
        )
        if any_nan_grad:
            summary = _nan_grad_summary(scene, opt_named_params)
            # OBSERVABILITY FIX (round 7 debug): previously this branch
            # silently `continue`'d, so repeated NaN gradients would
            # skip thousands of updates without any log signal. Round 7
            # got stuck in exactly this state for 3000+ iters. Now we
            # always emit a nan_skip log line on any NaN-grad event;
            # if they persist, we at least see them every iter.
            optimizer.zero_grad()
            timings["step"] += time.time() - t0
            if log_fn is not None:
                parts["iter"] = it
                parts["alive"] = scene.num_alive
                parts["nan_grad_skip"] = 1
                parts["nan_grad_summary"] = summary
                # Only push to log_rows / call log_fn once per log_interval
                # to avoid drowning the log file when NaN is persistent.
                # But always print a brief stderr-visible note so ops
                # can see the count grow.
                if it % config.log_interval == 0:
                    log_fn(it, parts)
                else:
                    print(f"[nan_grad] it={it:6d} skipping step offender={summary}",
                          flush=True)
            if abort_on_nan_grad:
                raise RuntimeError(f"nonfinite gradient at iter {it}: {summary}")
            continue
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        clip_to_ranges(scene, config)
        if audit_shape_grads and audit_cache is not None:
            alive_idx = audit_cache["alive_idx"]
            eps_before = audit_cache["eps_alive"]
            eps_after = scene.psq_shape().detach()[alive_idx].mean(dim=-1)
            order = torch.argsort(eps_before)
            parts["shape_step_delta_alive_eps_mean"] = (
                eps_after[order] - eps_before[order]
            ).detach().cpu().tolist()
        timings["step"] += time.time() - t0

        # Heartbeat — unconditional "I'm alive" signal independent of
        # log_interval. Uses print() straight to stdout so it shows up
        # even between log steps on slow runs. Fires at most every
        # HEARTBEAT_S seconds. No cost if log_interval fires faster.
        #
        # Friend's critical observability fix: previously, silent runs
        # of 10+ min (plausible at K=100 coupled ~3s/iter × log_interval=200)
        # were mis-labeled "hung" by watchdogs. Heartbeat makes "slow
        # but progressing" visibly distinct from "hung".
        _now = time.time()
        if _now - last_heartbeat >= HEARTBEAT_S:
            n_win = max(it - prev_log_it, 1)
            r_ms = 1000.0 * (timings["render"] - prev_t_render) / n_win
            s_ms = 1000.0 * (timings["step"] - prev_t_step) / n_win
            print(f"[heartbeat] it={it:6d}/{config.num_iterations} "
                  f"alive={scene.num_alive} elapsed={_now - t_start:.0f}s "
                  f"avg t[r{r_ms:.0f}/s{s_ms:.0f}]ms since last log",
                  flush=True)
            last_heartbeat = _now

        if it % config.pruning_interval == 0 and it >= config.warmup_iterations:
            t0 = time.time()
            adaptive_killed = 0
            adaptive_target = _adaptive_prune_target(config, it)
            prune(scene, config, verbose=(log_fn is not None))
            # View-dependent pruning (paper §4.2): runs at a coarser
            # cadence than the fast α/scale prune because it requires
            # a full-scene render pass.
            if it % (config.pruning_interval * config.view_prune_every_multiplier) == 0:
                prune_view_dependent(
                    scene, ray_sampler, config,
                    num_probe_rays=min(config.view_prune_probe_rays, rays_per_batch * 8),
                    weight_threshold=config.view_prune_weight_threshold,
                    foreground_only=getattr(config, "view_prune_foreground_only", True),
                    min_foreground_rays=config.view_prune_min_foreground_rays,
                    min_distinct_views=config.view_prune_min_distinct_views,
                    mu=mu_eff,
                    theta_min=theta_min_eff,
                    verbose=(log_fn is not None),
                )
            if adaptive_target > 0 and scene.num_alive > adaptive_target:
                adaptive_killed = prune_to_active_budget(
                    scene, ray_sampler, config,
                    target_alive=adaptive_target,
                    num_probe_rays=min(config.view_prune_probe_rays, rays_per_batch * 8),
                    foreground_only=getattr(config, "view_prune_foreground_only", True),
                    min_foreground_rays=config.view_prune_min_foreground_rays,
                    min_distinct_views=config.view_prune_min_distinct_views,
                    mu=mu_eff,
                    theta_min=theta_min_eff,
                    verbose=(log_fn is not None),
                )
            if adaptive_target > 0:
                parts["adaptive_prune_target"] = adaptive_target
                parts["adaptive_prune_killed"] = adaptive_killed
            timings["prune"] += time.time() - t0

        if (
            config.opacity_reset_interval > 0
            and it % config.opacity_reset_interval == 0
            and it >= config.warmup_iterations
        ):
            reset_opacity(scene, config, verbose=(log_fn is not None))

        if (
            birth_interval > 0
            and birth_count > 0
            and it > 0
            and it % birth_interval == 0
            and (it / max(config.num_iterations, 1)) <= birth_stop_fraction
        ):
            from clearmesh.dualprim.view_init import activate_visual_hull_regions

            score_tensor = None
            min_score = None
            label = birth_strategy
            if birth_strategy in {"residual", "hybrid"}:
                score_tensor = _visual_hull_birth_scores(scene, batch, render, config)
                min_score = float(getattr(config, "visual_hull_birth_min_score", 1e-4))
            n_born, born_idx, born_scores = activate_visual_hull_regions(
                scene, birth_count, scores=score_tensor, min_score=min_score,
                return_details=True,
            )
            if n_born == 0 and birth_strategy == "hybrid":
                n_born, born_idx, born_scores = activate_visual_hull_regions(
                    scene, birth_count, return_details=True,
                )
                label = "hybrid-fallback"
            if n_born > 0:
                parts["birth_count"] = n_born
                parts["birth_strategy"] = label
                parts["birth_indices"] = born_idx
                if born_scores:
                    parts["birth_score_max"] = max(born_scores)
                    parts["birth_score_min"] = min(born_scores)
                print(
                    f"[birth/visual_hull/{label}] activated {n_born} queued primitives "
                    f"at it={it}; alive={scene.num_alive}/{scene.K}"
                    + (f" score=[{min(born_scores):.4g},{max(born_scores):.4g}]"
                       if born_scores else ""),
                    flush=True,
                )

        if it % config.log_interval == 0:
            parts["iter"] = it
            parts["alive"] = scene.num_alive
            parts["theta_min_eff"] = theta_min_eff
            parts["mu_gate_eff"] = mu_eff
            parts["lambda_norm_eff"] = lambda_norm_eff
            parts["lambda_overlap_eff"] = lambda_overlap_eff
            parts["lambda_region_eff"] = lambda_region_eff
            if scene.num_alive > 0:
                psq_eps_alive = scene.psq_shape()[scene.alive].reshape(-1)
                eps_q = torch.quantile(
                    psq_eps_alive,
                    torch.tensor([0.1, 0.5, 0.9], device=psq_eps_alive.device),
                )
                parts["eps_psq_mean"] = float(psq_eps_alive.mean().item())
                parts["eps_psq_p10"] = float(eps_q[0].item())
                parts["eps_psq_p50"] = float(eps_q[1].item())
                parts["eps_psq_p90"] = float(eps_q[2].item())
            # Cheap diagnostics at every log step; expensive P_E probe
            # only every 4th log step.
            probe_this_step = (it % (config.log_interval * 4) == 0)
            diag = _dp_diagnostics(
                scene,
                ray_sampler=ray_sampler if probe_this_step else None,
                theta_min_eff=theta_min_eff,
                theta_min_nsq=getattr(config, "theta_min_nsq", config.theta_min),
                mu=mu_eff,
                gate_mode=config.gate_mode,
                paper_literal_theta_eps=config.paper_literal_theta_eps,
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

        # Trajectory snapshot (JSON, for downstream warm-start training).
        # Use (it + 1) semantics: snapshot AFTER the step has completed.
        # So a schedule of [1000, 3000, ...] captures state post-iter-999,
        # post-iter-2999, etc. — the "after N optimization steps" state.
        if trajectory_dir is not None and (it + 1) in trajectory_iter_set:
            save_scene_json(
                scene,
                Path(trajectory_dir) / f"step_{it + 1:06d}.json",
                iteration=it + 1,
            )

    # Always save a final-step snapshot if we're tracking trajectories.
    # Cheap insurance against off-by-one schedule mistakes and gives a
    # predictable end-of-run filename downstream loaders can key on.
    if trajectory_dir is not None:
        save_scene_json(
            scene,
            Path(trajectory_dir) / f"step_{config.num_iterations:06d}.json",
            iteration=config.num_iterations,
        )

    if checkpoint_path:
        _save_checkpoint(scene, checkpoint_path, config.num_iterations)

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
            theta_min_nsq=getattr(config, "theta_min_nsq", config.theta_min),
            primitive_reg_average_mode=getattr(config, "primitive_reg_average_mode", "alive"),
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
                theta_min_nsq=getattr(config, "theta_min_nsq", config.theta_min),
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
    for name in ("region_anchors", "region_scales", "region_active"):
        value = getattr(scene, name, None)
        if value is not None:
            state[name] = value.detach().cpu()
    torch.save(state, out)
