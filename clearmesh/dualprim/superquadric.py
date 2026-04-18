"""Superquadric implicit, dual-primitive combined field, and P_E gate.

Equations implemented (paper §3.1 Formulation, §3.2 Renderer):

  (2)  f(p, Q) = (|p'_x / a_x|^(2/ε2) + |p'_y / a_y|^(2/ε2))^(ε2/ε1)
                 + |p'_z / a_z|^(2/ε1) − 1
          where p' = R_Q^-1 (p − T_Q)  is the point in the SQ's local frame
          The "−1" makes f(p,Q) < 0 inside, 0 on surface, > 0 outside.

  (4)  P_E(p, S) = Φ(−f(p,PSQ)/θ_S(p) − μ) · Φ(−f(p,NSQ)/θ_S(p) − μ)
          Sigmoid-gated "effectiveness probability" that the NSQ
          subtracts at p. Only high when p is *inside both* PSQ and
          NSQ AND the primitive is sharp enough.

  (5)  f(p, S) = f(p, PSQ) · (1 − P_E(p, S)) − f(p, NSQ) · P_E(p, S)

  (6)  n(p, S) = Normalize(f'(p, PSQ)) · (1 − P_E(p, S))
                − Normalize(f'(p, NSQ)) · P_E(p, S)

This is the CORRECT subtraction operator (smooth, gated, fully
differentiable) — NOT hard CSG `max(f_psq, −f_nsq)`. The gradient
flows through P_E, letting the NSQ location and scale move during
optimization based on the combined field's mismatch with supervision.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from clearmesh.dualprim.types import DualPrimScene


# ---------------------------------------------------------------------
# Rotation helpers (XYZ Euler → 3x3, batched)
# ---------------------------------------------------------------------

def _euler_xyz_to_mat(euler: torch.Tensor) -> torch.Tensor:
    """Batched XYZ intrinsic Euler → 3x3 rotation matrix.

    euler: (..., 3) in radians.
    returns: (..., 3, 3).
    """
    rx, ry, rz = euler[..., 0], euler[..., 1], euler[..., 2]
    cx, sx = torch.cos(rx), torch.sin(rx)
    cy, sy = torch.cos(ry), torch.sin(ry)
    cz, sz = torch.cos(rz), torch.sin(rz)
    zero = torch.zeros_like(cx)
    one = torch.ones_like(cx)

    Rx = torch.stack([
        torch.stack([one, zero, zero], -1),
        torch.stack([zero, cx, -sx], -1),
        torch.stack([zero, sx, cx], -1),
    ], -2)
    Ry = torch.stack([
        torch.stack([cy, zero, sy], -1),
        torch.stack([zero, one, zero], -1),
        torch.stack([-sy, zero, cy], -1),
    ], -2)
    Rz = torch.stack([
        torch.stack([cz, -sz, zero], -1),
        torch.stack([sz, cz, zero], -1),
        torch.stack([zero, zero, one], -1),
    ], -2)
    return Rz @ Ry @ Rx


# ---------------------------------------------------------------------
# Superquadric implicit (Eq 2)
# ---------------------------------------------------------------------

def sq_implicit(
    points: torch.Tensor,        # (..., 3) query points
    translation: torch.Tensor,   # (K, 3)
    rotation: torch.Tensor,      # (K, 3) Euler XYZ in radians
    scale: torch.Tensor,         # (K, 3) — (a_x, a_y, a_z)
    shape: torch.Tensor,         # (K, 2) — (ε1, ε2)
    eps: float = 1e-6,
) -> torch.Tensor:
    """Per-primitive superquadric implicit. Paper Eq 2.

    Returns:
        f: (..., K) where negative = inside, zero = on surface, positive
           = outside. This is the "f(p, Q) − 1" style implicit as in
           the paper.

    points: any batch shape ending in 3.
    """
    # Broadcast points to include K dim
    K = translation.shape[0]
    p = points.unsqueeze(-2)                   # (..., 1, 3)
    R = _euler_xyz_to_mat(rotation)            # (K, 3, 3)
    R_inv = R.transpose(-2, -1)                # R is orthogonal
    # p' = R_inv (p − T)
    rel = p - translation                      # (..., K, 3)
    p_local = torch.einsum("...ki,kij->...kj", rel, R_inv)

    # Clamp scale so division is safe
    a = scale.clamp(min=eps)                   # (K, 3)
    # Clamp shape to paper range
    e1 = shape[..., 0].clamp(min=0.05, max=2.0)  # (K,)
    e2 = shape[..., 1].clamp(min=0.05, max=2.0)

    # |p' / a|^(2/ε2) style terms — use abs + pow, add eps to prevent nan
    X = (p_local[..., 0].abs() / a[..., 0] + eps)
    Y = (p_local[..., 1].abs() / a[..., 1] + eps)
    Z = (p_local[..., 2].abs() / a[..., 2] + eps)

    inner = X ** (2.0 / e2) + Y ** (2.0 / e2)
    f = inner ** (e2 / e1) + Z ** (2.0 / e1)
    return f - 1.0   # < 0 inside


def sq_implicit_grad(
    points: torch.Tensor,
    translation, rotation, scale, shape,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Analytic gradient of sq_implicit w.r.t. points.

    Returns (..., K, 3). Used in Eq 6 for the surface normal term.
    This is a numerical wrapper that uses torch.autograd under the
    hood for simplicity — a closed-form version is possible but noisy
    near axis-aligned planes.
    """
    p = points.detach().clone().requires_grad_(True)
    f = sq_implicit(p, translation, rotation, scale, shape, eps=eps)
    # sum over K so grad has shape (..., 3) per-K; we want (..., K, 3)
    # Trick: use grad_outputs to get per-K gradients
    grads = []
    for k in range(f.shape[-1]):
        g = torch.autograd.grad(
            f[..., k].sum(), p, retain_graph=True, create_graph=p.requires_grad,
        )[0]
        grads.append(g)
    return torch.stack(grads, dim=-2)  # (..., K, 3)


# ---------------------------------------------------------------------
# P_E effectiveness gate (Eq 4) + combined field (Eq 5) + normal (Eq 6)
# ---------------------------------------------------------------------

def effectiveness_probability(
    f_psq: torch.Tensor,    # (..., K)
    f_nsq: torch.Tensor,    # (..., K)
    theta: torch.Tensor,    # (K,) — per-primitive sharpness
    mu: float = 0.0,
    theta_min: float = 0.01,
) -> torch.Tensor:
    """Paper Eq 4 — smooth gated activation of the NSQ.

        P_E(p, S) = Φ(−f(p,PSQ)/θ_S − μ) · Φ(−f(p,NSQ)/θ_S − μ)

    Φ = sigmoid. P_E → 1 when BOTH PSQ and NSQ implicit evaluate deep
    inside (large negative f), modulated by θ_S. Modulating by θ_S
    lets each primitive learn its own boundary sharpness.

    Returns (..., K) in [0, 1].
    """
    # theta broadcasts as (K,) → (..., K) automatically
    t = theta.clamp(min=theta_min)
    term_psq = torch.sigmoid(-f_psq / t - mu)
    term_nsq = torch.sigmoid(-f_nsq / t - mu)
    return term_psq * term_nsq


def combined_field(
    f_psq: torch.Tensor,
    f_nsq: torch.Tensor,
    p_e: torch.Tensor,
) -> torch.Tensor:
    """Paper Eq 5 — smooth dual-primitive combined field.

        f(p, S) = f_PSQ · (1 − P_E) − f_NSQ · P_E

    Inside regions where P_E ≈ 1 (NSQ active), the combined field
    FLIPS sign relative to the NSQ — a deeply-inside NSQ has large
    negative f_NSQ, so −f_NSQ is positive, and the combined field
    moves from negative (inside PSQ) toward positive (outside).
    That is the smooth analogue of CSG difference.

    Returns (..., K).
    """
    return f_psq * (1.0 - p_e) - f_nsq * p_e


def combined_normal(
    grad_psq: torch.Tensor,   # (..., K, 3)
    grad_nsq: torch.Tensor,   # (..., K, 3)
    p_e: torch.Tensor,        # (..., K)
    eps: float = 1e-8,
) -> torch.Tensor:
    """Paper Eq 6 — combined surface normal.

        n(p, S) = Normalize(f'_PSQ) (1 − P_E) − Normalize(f'_NSQ) P_E

    Returns (..., K, 3), not further normalized across primitives.
    """
    n_psq = F.normalize(grad_psq, dim=-1, eps=eps)
    n_nsq = F.normalize(grad_nsq, dim=-1, eps=eps)
    return n_psq * (1.0 - p_e).unsqueeze(-1) - n_nsq * p_e.unsqueeze(-1)


# ---------------------------------------------------------------------
# Scene-level wrappers
# ---------------------------------------------------------------------

def scene_combined_field(
    points: torch.Tensor,
    scene: DualPrimScene,
    mu: float = 0.0,
    theta_min: float = 0.01,
    with_normals: bool = False,
):
    """One-shot evaluation of the combined field per primitive.

    Returns either:
        f_combined (..., K)
    or
        (f_combined, n_combined) when with_normals=True — where
        n_combined has shape (..., K, 3).

    Callers typically aggregate across K using α + density (see
    renderer.py).
    """
    f_psq = sq_implicit(
        points,
        scene.psq_translation(), scene.psq_rotation(),
        scene.psq_scale(), scene.psq_shape(),
    )
    f_nsq = sq_implicit(
        points,
        scene.nsq_translation(), scene.nsq_rotation(),
        scene.nsq_scale(), scene.nsq_shape(),
    )
    p_e = effectiveness_probability(
        f_psq, f_nsq, scene.theta(), mu=mu, theta_min=theta_min,
    )
    f_combined = combined_field(f_psq, f_nsq, p_e)

    if not with_normals:
        return f_combined

    grad_psq = sq_implicit_grad(
        points,
        scene.psq_translation(), scene.psq_rotation(),
        scene.psq_scale(), scene.psq_shape(),
    )
    grad_nsq = sq_implicit_grad(
        points,
        scene.nsq_translation(), scene.nsq_rotation(),
        scene.nsq_scale(), scene.nsq_shape(),
    )
    n_combined = combined_normal(grad_psq, grad_nsq, p_e)
    return f_combined, n_combined
