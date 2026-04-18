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

    NOTE on α: we DO NOT clamp α here. α has a soft cap via L_max
    (Eq 17 = ReLU(α − 1)), which only fires if α is allowed to
    exceed 1 during the forward pass. Hard-clamping α each step
    effectively disables L_max — α never goes above 1, so the
    gradient from L_max is always 0. The paper relies on L_max as
    a learned regularizer, not as a box constraint. We clamp α
    strictly only at export time (see DualPrimConfig.export_alpha_threshold).

    Similarly we don't hard-clamp shape exponents into Table 1 range
    here — ``sq_implicit`` has its own internal clamp so the math
    stays stable, but we leave the raw parameter free so its
    gradient doesn't get killed.
    """
    with torch.no_grad():
        scene.params[:, IDX_PSQ_SCALE].clamp_(*config.scale_range)
        scene.params[:, IDX_NSQ_SCALE].clamp_(*config.scale_range)
        # Shape + α are soft-regularized by L_* losses, not hard-clamped.
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


def prune_view_dependent(
    scene: DualPrimScene,
    ray_sampler: "RaySampler",
    config: DualPrimConfig,
    *,
    num_probe_rays: int = 8192,
    weight_threshold: float = 1e-3,
    verbose: bool = False,
) -> int:
    """Kill primitives with negligible rendering weight across viewpoints.

    Paper §4.2: "We prune primitives with negligible rendering weights
    across all viewpoints." A primitive is redundant if no ray through
    any view ever accumulates significant contribution from it.

    Implementation: run the standard renderer over ``num_probe_rays``
    rays sampled from the ray_sampler (which should cover all N
    viewpoints), accumulate per-primitive total rendering weight, and
    kill any alive primitive whose total is below ``weight_threshold``.

    Per-primitive rendering weight at sample n of ray r is
        w_k(r, n) = σ_k_weighted(r, n) · transmittance(r, n) · δ(r, n)
    which is the contribution of primitive k to the alpha-composite at
    that sample. We sum over samples and rays to get a scalar per
    primitive.

    This runs under ``torch.no_grad`` so it's cheap compared to a
    training step.

    Returns the number of primitives killed.
    """
    from clearmesh.dualprim.renderer import (
        sample_ray_points,
        _scene_field,
        density_from_field,
        _accumulated_transmittance,
    )

    with torch.no_grad():
        batch = ray_sampler(num_probe_rays)
        points, t_vals, deltas = sample_ray_points(
            batch.origins, batch.dirs,
            near=config.near_plane, far=config.far_plane,
            N=config.num_samples_per_ray,
            perturb=False, device=scene.params.device,
        )

        # Density, with α-weighting (matches the main render path)
        dp = batch.dirs.unsqueeze(1) * 0.01  # same default Δp
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
        alpha_k = scene.alpha().clamp(0.0, 1.0)
        sigma_k_weighted = sigma_k * (alpha_k * alive).view(1, 1, -1)

        sigma = sigma_k_weighted.sum(dim=-1)                 # (R, N)
        alpha_ray = 1.0 - torch.exp(-sigma * deltas)         # (R, N)
        trans = _accumulated_transmittance(alpha_ray)        # (R, N)
        # Per-sample ray weight
        w_ray = alpha_ray * trans                            # (R, N)

        # Per-primitive contribution: w_ray * (σ_k / σ).
        # Aggregate over rays and samples.
        denom = sigma.unsqueeze(-1) + 1e-8
        per_prim_contrib = (
            (sigma_k_weighted / denom) * w_ray.unsqueeze(-1)
        ).sum(dim=(0, 1))                                     # (K,)

        # Normalize so threshold is comparable across num_probe_rays
        per_prim_contrib = per_prim_contrib / max(num_probe_rays, 1)

        kill = scene.alive & (per_prim_contrib < weight_threshold)
        n_killed = int(kill.sum().item())
        scene.alive &= ~kill
        scene.params[kill, IDX_ALPHA] = 0.0

    if verbose and n_killed > 0:
        print(f"[prune/view] killed {n_killed} primitives "
              f"(contribution < {weight_threshold}); {scene.num_alive} alive")
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
    t_start = time.time()

    for it in range(config.num_iterations):
        batch = ray_sampler(rays_per_batch)

        t0 = time.time()
        render = render_rays(
            scene,
            batch.origins, batch.dirs,
            num_samples=config.num_samples_per_ray,
            near=config.near_plane,
            far=config.far_plane,
            mu=config.mu_gate_offset,
            theta_min=config.theta_min,
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
        loss.backward()
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

        t0 = time.time()
        loss, parts = total_loss_tsdf(
            scene, qp, tg,
            lambda_sparse=config.lambda_sparse,
            lambda_entropy=config.lambda_entropy,
            lambda_max=config.lambda_max,
            mu=config.mu_gate_offset,
            theta_min=config.theta_min,
        )
        timings["loss"] += time.time() - t0

        t0 = time.time()
        optimizer.zero_grad()
        loss.backward()
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
