"""NeuS-style volumetric renderer over dual-primitive SDF.

Paper §3.2 Renderer. Equations implemented:

  (1)  I_render(o, v) = Σ_{i=1..N} ∏_{j<i} (1 − α_j) · α_i · c_i
         standard alpha-composite. α_i = 1 − exp(−σ_i δ_i).

  (7)  σ_S(p) = max( [Φ(f(p+Δp)/θ_S) − Φ(f(p−Δp)/θ_S)] / Φ(f(p+Δp)/θ_S), 0 )
         NeuS-style density from the implicit field. The numerator is
         a finite-difference approximation of the derivative of the
         sigmoid CDF along the ray direction; dividing by the CDF
         value normalizes.

       (final per-point density is Σ_k α_k · σ_{S_k}(p))

  (8)  c(p) = Σ_k c_basic(p, S_k) · σ_{S_k}(p) / σ(p) + C(p)
         per-sample color = α-weighted basic color plus MLP-lit residual.

  (9)  M_render(o, v) = alpha-composite with color replaced by 1.
  (10) N_render(o, v) = alpha-composite of surface normals.
  (11) n(p) = Σ_k n_p,S_k(p) · σ_{S_k}(p) / σ(p)

Notes on NOT SPECIFIED details (see params.py):
  - Δp ray step size: paper doesn't give; defaults to a fraction of δ.
  - N (samples/ray): not given; default 64.
  - How background is handled: assume uniform from config.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F

from clearmesh.dualprim.types import DualPrimScene
from clearmesh.dualprim.superquadric import (
    sq_implicit,
    sq_implicit_grad,
    effectiveness_probability,
    combined_field,
    combined_normal,
)


# ---------------------------------------------------------------------
# Lighting MLP (used in Eq 8 as C(p))
# ---------------------------------------------------------------------

class LightingMLP(torch.nn.Module):
    """Small MLP that predicts view-dependent lighting residual.

    Paper §5.1: "4 layers with Xavier initialization". Hidden width
    NOT specified; we default to 128.
    """

    def __init__(self, hidden: int = 128, layers: int = 4):
        super().__init__()
        # Input: (point(3) + view_dir(3) + normal(3)) = 9
        input_dim = 9
        mlp_layers: list[torch.nn.Module] = []
        in_f = input_dim
        for i in range(layers - 1):
            lin = torch.nn.Linear(in_f, hidden)
            torch.nn.init.xavier_uniform_(lin.weight)
            torch.nn.init.zeros_(lin.bias)
            mlp_layers.append(lin)
            mlp_layers.append(torch.nn.ReLU())
            in_f = hidden
        head = torch.nn.Linear(in_f, 3)
        torch.nn.init.xavier_uniform_(head.weight)
        torch.nn.init.zeros_(head.bias)
        mlp_layers.append(head)
        self.net = torch.nn.Sequential(*mlp_layers)

    def forward(self, points, view_dirs, normals):
        x = torch.cat([points, view_dirs, normals], dim=-1)
        return self.net(x)


# ---------------------------------------------------------------------
# Ray sampling
# ---------------------------------------------------------------------

def sample_ray_points(
    ray_origins: torch.Tensor,       # (R, 3)
    ray_dirs: torch.Tensor,          # (R, 3) unit
    near: float, far: float, N: int,
    perturb: bool = True,
    device=None,
):
    """Uniform-stratified sampling along each ray.

    Returns:
        points: (R, N, 3)
        dists : (R, N)     — distance to camera per sample
        deltas: (R, N)     — spacing between adjacent samples (for α)
    """
    R = ray_origins.shape[0]
    if device is None:
        device = ray_origins.device
    t_vals = torch.linspace(0.0, 1.0, N, device=device)
    t_vals = near + (far - near) * t_vals         # (N,)
    t_vals = t_vals.unsqueeze(0).expand(R, N)      # (R, N)

    if perturb:
        mids = 0.5 * (t_vals[:, 1:] + t_vals[:, :-1])
        upper = torch.cat([mids, t_vals[:, -1:]], dim=-1)
        lower = torch.cat([t_vals[:, :1], mids], dim=-1)
        rand = torch.rand_like(t_vals)
        t_vals = lower + (upper - lower) * rand

    points = ray_origins.unsqueeze(1) + t_vals.unsqueeze(-1) * ray_dirs.unsqueeze(1)
    # deltas: distance between adjacent samples
    deltas = torch.diff(t_vals, dim=-1)
    # Paper Eq 1 defines δ_i as adjacent sample spacing. The previous
    # NeRF-style 1e10 tail distance forces the final alpha to 1 for any
    # non-zero terminal density, which is especially destructive for
    # hole rays: a tiny stray sigma at the far sample makes the entire
    # ray opaque. Reuse the last finite interval instead.
    if N > 1:
        last = deltas[:, -1:]
    else:
        last = torch.full_like(t_vals[:, :1], far - near)
    deltas = torch.cat([deltas, last], dim=-1)
    return points, t_vals, deltas


# ---------------------------------------------------------------------
# Density (Eq 7)
# ---------------------------------------------------------------------

def _psi(f: torch.Tensor, theta: torch.Tensor, theta_min: float = 0.01) -> torch.Tensor:
    """Sigmoid of f/θ — the CDF used in Eq 7."""
    t = theta.clamp(min=theta_min)
    return torch.sigmoid(f / t)


def density_from_field(
    f_fwd: torch.Tensor,     # f at sample + Δp (..., K)
    f_bwd: torch.Tensor,     # f at sample − Δp (..., K)
    theta: torch.Tensor,     # (K,)
    eps: float = 1e-8,
    theta_min: float = 0.01,
):
    """Paper Eq 7 — NeuS-style SDF → density.

        σ_S(p) = max(
            (Φ(f(p+Δp,S)/θ_S) − Φ(f(p−Δp,S)/θ_S)) / Φ(f(p+Δp,S)/θ_S),
            0
        )

    The denominator is Φ of the FORWARD point — NOT the midpoint.
    (Previous version used Φ(f_mid/θ) which is a different formula and
    was caught in Phase A review.)

    The mid-point field is not used in this function; it was removed
    from the signature to prevent it from being accidentally reintroduced.
    """
    cdf_fwd = torch.nan_to_num(_psi(f_fwd, theta, theta_min), nan=0.0, posinf=1.0, neginf=0.0)
    cdf_bwd = torch.nan_to_num(_psi(f_bwd, theta, theta_min), nan=0.0, posinf=1.0, neginf=0.0)
    num = torch.nan_to_num(cdf_fwd - cdf_bwd, nan=0.0, posinf=1.0, neginf=0.0)
    den = cdf_fwd.clamp(min=eps)
    # Theoretically this ratio lives in [0, 1]. Clamp there explicitly
    # so numeric junk from saturated implicits does not amplify through
    # the transmittance chain.
    sigma = torch.nan_to_num(num / den, nan=0.0, posinf=1.0, neginf=0.0)
    return sigma.clamp(min=0.0, max=1.0)


# ---------------------------------------------------------------------
# Full render (RGB, mask, normals)
# ---------------------------------------------------------------------

class RenderOutput(NamedTuple):
    rgb: torch.Tensor       # (R, 3)
    mask: torch.Tensor      # (R,) in [0, 1]
    normals: torch.Tensor   # (R, 3)
    opacity: torch.Tensor   # (R,) in [0, 1] — same as mask, kept for clarity


def render_rays(
    scene: DualPrimScene,
    ray_origins: torch.Tensor,
    ray_dirs: torch.Tensor,
    *,
    num_samples: int = 64,
    near: float = 0.1,
    far: float = 4.0,
    delta_p: float = 0.01,      # Eq 7 finite-diff step; NOT SPECIFIED IN PAPER
    mu: float = 0.0,
    theta_min: float = 0.01,
    background: tuple = (1.0, 1.0, 1.0),
    perturb: bool = True,
) -> RenderOutput:
    """Volumetric render a batch of rays.

    Implements Eq 1 (alpha-composite), Eq 7 (density from SDF), Eq 8
    (color = α-weighted basic + MLP residual), Eq 9 (mask), Eq 10-11
    (normals).

    All geometry terms use the DualPrim *combined* field from Eq 5.
    """
    R = ray_origins.shape[0]
    points, t_vals, deltas = sample_ray_points(
        ray_origins, ray_dirs, near, far, num_samples, perturb=perturb,
    )
    # points: (R, N, 3)

    # Forward/backward for Eq 7 finite diff along ray.
    # Paper's Eq 7 denominator is the FORWARD point, not the midpoint,
    # so we don't need to evaluate the field at the midpoint here.
    dp = ray_dirs.unsqueeze(1) * delta_p  # (R, 1, 3) → broadcast to (R, N, 3)
    pts_fwd = points + dp
    pts_bwd = points - dp

    f_fwd = _scene_field(scene, pts_fwd, mu, theta_min)   # (R, N, K)
    f_bwd = _scene_field(scene, pts_bwd, mu, theta_min)

    # Per-primitive density from Eq 7
    sigma_k = density_from_field(
        f_fwd, f_bwd, scene.theta(), theta_min=theta_min,
    )  # (R, N, K)
    sigma_k = torch.nan_to_num(sigma_k, nan=0.0, posinf=1.0, neginf=0.0)

    # Apply alpha weighting (pruning via alive mask + learned α)
    alive = scene.alive.to(sigma_k.dtype)                   # (K,)
    alpha_k = scene.alpha().clamp(0.0, 1.0)                 # (K,)
    weight_k = (alpha_k * alive).view(1, 1, -1)             # (1, 1, K)
    sigma_k_weighted = sigma_k * weight_k                    # (R, N, K)

    sigma = torch.nan_to_num(
        sigma_k_weighted.sum(dim=-1), nan=0.0, posinf=scene.K, neginf=0.0,
    )                                                        # (R, N)

    # Alpha compositing — Eq 1
    optical = torch.nan_to_num(sigma * deltas, nan=0.0, posinf=80.0, neginf=0.0)
    alpha = 1.0 - torch.exp(-optical.clamp(min=0.0, max=80.0))  # (R, N)
    alpha = torch.nan_to_num(alpha, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    trans = torch.nan_to_num(_accumulated_transmittance(alpha), nan=0.0, posinf=1.0, neginf=0.0)
    weights = alpha * trans                                  # (R, N)

    # ----- color (Eq 8) -----
    c_basic = scene.color()                                  # (K, 3)
    # per-sample basic color = Σ_k c_basic_k · σ_k / σ
    # (with α weights baked into σ_k already)
    denom = sigma.unsqueeze(-1) + 1e-8
    c_per_sample = torch.einsum(
        "rnk,kc->rnc", sigma_k_weighted, c_basic,
    ) / denom                                                # (R, N, 3)

    # ----- composited normal (Eq 10-11) — compute ONCE -----
    # Used for both the lighting MLP residual and the final normal
    # output. Each _scene_normal call evaluates sq_implicit_grad twice
    # (PSQ + NSQ) which expands to 12 sq_implicit forwards via FD-grad,
    # plus 2 more sq_implicit for the gate — so calling it twice was
    # adding 28 extra sq_implicit evaluations per iter to the autograd
    # graph. Caching this single result cuts the backward pass cost
    # roughly in half on profiling.
    normals_per_sample = _scene_normal(scene, points, sigma_k_weighted, sigma, mu, theta_min)
    normals_per_sample = torch.nan_to_num(normals_per_sample, nan=0.0, posinf=0.0, neginf=0.0)

    # Lighting residual: MLP on (point, view_dir, weighted_normal)
    if scene.lighting_mlp is not None:
        view_dirs_exp = ray_dirs.unsqueeze(1).expand_as(points)
        # The paper specifies an MLP residual C(p), but does not require
        # geometry gradients to flow through the normal feature input.
        # Detaching here removes one large and numerically fragile
        # backward path without changing the forward render.
        lighting = scene.lighting_mlp(points, view_dirs_exp, normals_per_sample.detach())
        c_per_sample = c_per_sample + lighting

    rgb = torch.nan_to_num((weights.unsqueeze(-1) * c_per_sample).sum(dim=1), nan=0.0)   # (R, 3)

    # ----- mask (Eq 9) -----
    mask = torch.nan_to_num(weights.sum(dim=1), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

    normals = (weights.unsqueeze(-1) * normals_per_sample).sum(dim=1)
    normals = torch.nan_to_num(F.normalize(normals, dim=-1, eps=1e-8), nan=0.0, posinf=0.0, neginf=0.0)

    # Background
    bg = torch.tensor(background, device=rgb.device, dtype=rgb.dtype)
    rgb = rgb + (1.0 - mask).unsqueeze(-1) * bg

    return RenderOutput(rgb=rgb, mask=mask, normals=normals, opacity=mask)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _scene_field(scene, points, mu, theta_min):
    f_psq = sq_implicit(
        points, scene.psq_translation(), scene.psq_rotation(),
        scene.psq_scale(), scene.psq_shape(),
    )
    f_nsq = sq_implicit(
        points, scene.nsq_translation(), scene.nsq_rotation(),
        scene.nsq_scale(), scene.nsq_shape(),
    )
    p_e = effectiveness_probability(f_psq, f_nsq, scene.theta(), mu=mu, theta_min=theta_min)
    return combined_field(f_psq, f_nsq, p_e)


def _scene_normal(scene, points, sigma_k_weighted, sigma, mu, theta_min):
    """Per-sample weighted normal from the combined field (Eq 11)."""
    grad_psq = sq_implicit_grad(
        points, scene.psq_translation(), scene.psq_rotation(),
        scene.psq_scale(), scene.psq_shape(),
    )
    grad_nsq = sq_implicit_grad(
        points, scene.nsq_translation(), scene.nsq_rotation(),
        scene.nsq_scale(), scene.nsq_shape(),
    )
    f_psq = sq_implicit(
        points, scene.psq_translation(), scene.psq_rotation(),
        scene.psq_scale(), scene.psq_shape(),
    )
    f_nsq = sq_implicit(
        points, scene.nsq_translation(), scene.nsq_rotation(),
        scene.nsq_scale(), scene.nsq_shape(),
    )
    p_e = effectiveness_probability(f_psq, f_nsq, scene.theta(), mu=mu, theta_min=theta_min)
    n_k = combined_normal(grad_psq, grad_nsq, p_e)        # (..., K, 3)

    # Eq 11: weight by sigma_k / sigma
    denom = sigma.unsqueeze(-1) + 1e-8
    weighted = torch.einsum("rnk,rnkc->rnc", sigma_k_weighted, n_k) / denom
    return weighted


def _accumulated_transmittance(alpha: torch.Tensor) -> torch.Tensor:
    """Cumulative product of (1 − α) along the ray, exclusive."""
    trans = torch.cumprod(
        torch.cat([torch.ones_like(alpha[:, :1]), 1.0 - alpha + 1e-10], dim=-1),
        dim=-1,
    )[:, :-1]
    return trans
