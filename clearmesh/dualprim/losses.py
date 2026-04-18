"""Six training losses from paper §4.1.

  (13) L_rgb      = Σ ||I_render − I_gt|| · M_gt          (masked L1 RGB)
  (14) L_mask     = Σ BCE(M_render, M_gt)
  (15) L_sp       = (1/K) Σ_p α(p)                         (sparsity on α)
  (16) L_e        = −(1/K) Σ_p α log α + (1−α) log(1−α)   (entropy on α)
  (17) L_max      = (1/K) Σ_p ReLU(α(p) − 1)               (soft cap α ≤ 1)
  (18) L_norm_reg = Σ ||N_render − N_pred|| · M_gt

Combined (Eq 12):
     L = L_rgb + λ_mask L_mask + λ_sp L_sp + λ_e L_e
         + λ_max L_max + λ_norm_reg L_norm_reg

Note: α in Eqs 15-17 refers to the PER-PRIMITIVE opacity (the α field
of each dual-primitive, K scalars), NOT the per-sample α_i from
alpha-compositing in Eq 1. Those are unfortunately both called α in
the paper. The primitive-level α is what gets pruned.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from clearmesh.dualprim.renderer import RenderOutput
from clearmesh.dualprim.types import DualPrimScene


def loss_rgb(
    render: RenderOutput,
    rgb_gt: torch.Tensor,          # (R, 3)
    mask_gt: torch.Tensor,         # (R,) in [0, 1]
) -> torch.Tensor:
    """Eq 13 — masked L1 on RGB."""
    diff = (render.rgb - rgb_gt).abs().sum(dim=-1)    # (R,)
    return (diff * mask_gt).mean()


def loss_mask(
    render: RenderOutput,
    mask_gt: torch.Tensor,         # (R,) in [0, 1]
    eps: float = 1e-6,
) -> torch.Tensor:
    """Eq 14 — per-ray BCE on rendered mask vs GT mask."""
    m = render.mask.clamp(eps, 1.0 - eps)
    return F.binary_cross_entropy(m, mask_gt)


def loss_sparsity(scene: DualPrimScene) -> torch.Tensor:
    """Eq 15 — mean per-primitive α. Drives pruning.

    Only counts alive primitives so dead rows don't contribute.
    """
    alpha = scene.alpha().clamp(0.0, 1.0)
    alive = scene.alive.to(alpha.dtype)
    return (alpha * alive).sum() / (alive.sum() + 1e-8)


def loss_entropy(scene: DualPrimScene, eps: float = 1e-6) -> torch.Tensor:
    """Eq 16 — binary entropy on per-primitive α, pushing to {0, 1}."""
    alpha = scene.alpha().clamp(eps, 1.0 - eps)
    alive = scene.alive.to(alpha.dtype)
    h = -(alpha * torch.log(alpha) + (1.0 - alpha) * torch.log(1.0 - alpha))
    return (h * alive).sum() / (alive.sum() + 1e-8)


def loss_max(scene: DualPrimScene) -> torch.Tensor:
    """Eq 17 — soft cap α ≤ 1 via ReLU(α − 1)."""
    alpha = scene.alpha()
    alive = scene.alive.to(alpha.dtype)
    return (F.relu(alpha - 1.0) * alive).sum() / (alive.sum() + 1e-8)


def loss_norm_reg(
    render: RenderOutput,
    normals_pred: torch.Tensor,    # (R, 3) — from StableNormal or analytic
    mask_gt: torch.Tensor,         # (R,)
) -> torch.Tensor:
    """Eq 18 — masked L1 on surface normals."""
    diff = (render.normals - normals_pred).abs().sum(dim=-1)
    return (diff * mask_gt).mean()


# ---------------------------------------------------------------------
# Combined loss
# ---------------------------------------------------------------------

def total_loss(
    scene: DualPrimScene,
    render: RenderOutput,
    rgb_gt: torch.Tensor,
    mask_gt: torch.Tensor,
    normals_pred: torch.Tensor,
    *,
    lambda_mask: float = 1.0,
    lambda_sparse: float = 0.01,
    lambda_entropy: float = 0.01,
    lambda_max: float = 0.1,
    lambda_norm_reg: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Eq 12 — weighted sum of the 6 terms.

    Returns (scalar loss, dict for logging).
    """
    l_rgb = loss_rgb(render, rgb_gt, mask_gt)
    l_mask = loss_mask(render, mask_gt)
    l_sp = loss_sparsity(scene)
    l_e = loss_entropy(scene)
    l_max = loss_max(scene)
    l_norm = loss_norm_reg(render, normals_pred, mask_gt)

    total = (
        l_rgb
        + lambda_mask * l_mask
        + lambda_sparse * l_sp
        + lambda_entropy * l_e
        + lambda_max * l_max
        + lambda_norm_reg * l_norm
    )
    parts = {
        "rgb": l_rgb.item(),
        "mask": l_mask.item(),
        "sparse": l_sp.item(),
        "entropy": l_e.item(),
        "max": l_max.item(),
        "norm": l_norm.item(),
        "total": total.item(),
    }
    return total, parts
