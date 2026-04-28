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


def _masked_reduce(
    values: torch.Tensor,
    mask: torch.Tensor,
    *,
    mode: str = "global_mean",
    eps: float = 1e-8,
) -> torch.Tensor:
    weighted = values * mask
    if mode == "fg_mean":
        return weighted.sum() / (mask.sum() + eps)
    return weighted.mean()


def loss_rgb(
    render: RenderOutput,
    rgb_gt: torch.Tensor,          # (R, 3)
    mask_gt: torch.Tensor,         # (R,) in [0, 1]
    *,
    norm_mode: str = "global_mean",
) -> torch.Tensor:
    """Eq 13 — masked L1 on RGB."""
    diff = (render.rgb - rgb_gt).abs().sum(dim=-1)    # (R,)
    return _masked_reduce(diff, mask_gt, mode=norm_mode)


def loss_mask(
    render: RenderOutput,
    mask_gt: torch.Tensor,         # (R,) in [0, 1]
    eps: float = 1e-6,
    loss_type: str = "bce",
) -> torch.Tensor:
    """Eq 14 — per-ray loss on rendered mask vs GT mask.

    loss_type:
      "bce" — paper's binary cross-entropy (eps-clamped for stability).
              Strong penalty on confident wrong predictions.
              Gradient -1/(1-m) unbounded near m=1; with mask_gt=0,
              this can hit ~1e6 magnitude per ray at eps=1e-6, which
              cascades through the rendering backward chain and
              produces NaN when aggregated over many rays.
      "mse" — mean squared error. Gradient 2*(m - mask_gt) bounded
              by ±2 per element. Much NaN-safer. Empirically round
              11 showed ~90% NaN-skip with BCE drops to ~0% with MSE.
    """
    if loss_type == "mse":
        return F.mse_loss(render.mask, mask_gt)
    m = render.mask.clamp(eps, 1.0 - eps)
    return F.binary_cross_entropy(m, mask_gt)


def loss_edge_mask(
    render: RenderOutput,
    mask_gt: torch.Tensor,
    edge_weight_gt: torch.Tensor | None,
    eps: float = 1e-6,
    loss_type: str = "bce",
) -> torch.Tensor:
    """Extra mask loss on silhouette/boundary rays.

    This is not in the paper. It is a hard-surface diagnostic: ordinary
    RGB/mask supervision can fit a cuboid silhouette with rounded SQs,
    so we add focused pressure exactly where sharp edges are visible.
    """
    if edge_weight_gt is None or edge_weight_gt.sum() <= 0:
        return render.mask.sum() * 0
    w = edge_weight_gt.to(render.mask.dtype)
    if loss_type == "mse":
        per = (render.mask - mask_gt).pow(2)
    else:
        m = render.mask.clamp(eps, 1.0 - eps)
        per = F.binary_cross_entropy(m, mask_gt, reduction="none")
    return (per * w).sum() / (w.sum() + eps)


def loss_sparsity(
    scene: DualPrimScene,
    *,
    average_mode: str = "alive",
) -> torch.Tensor:
    """Eq 15 — mean per-primitive α. Drives pruning.

    Only counts alive primitives so dead rows don't contribute.
    """
    alpha = scene.alpha().clamp(0.0, 1.0)
    alive = scene.alive.to(alpha.dtype)
    num = (alpha * alive).sum()
    den = scene.K if average_mode == "fixed_k" else alive.sum() + 1e-8
    return num / den


def loss_entropy(
    scene: DualPrimScene,
    eps: float = 1e-6,
    *,
    average_mode: str = "alive",
) -> torch.Tensor:
    """Eq 16 — binary entropy on per-primitive α, pushing to {0, 1}."""
    alpha = scene.alpha().clamp(eps, 1.0 - eps)
    alive = scene.alive.to(alpha.dtype)
    h = -(alpha * torch.log(alpha) + (1.0 - alpha) * torch.log(1.0 - alpha))
    num = (h * alive).sum()
    den = scene.K if average_mode == "fixed_k" else alive.sum() + 1e-8
    return num / den


def loss_max(
    scene: DualPrimScene,
    *,
    average_mode: str = "alive",
) -> torch.Tensor:
    """Eq 17 — soft cap α ≤ 1 via ReLU(α − 1)."""
    alpha = scene.alpha()
    alive = scene.alive.to(alpha.dtype)
    num = (F.relu(alpha - 1.0) * alive).sum()
    den = scene.K if average_mode == "fixed_k" else alive.sum() + 1e-8
    return num / den


def loss_shape_box(
    scene: DualPrimScene,
    *,
    threshold: float = 0.30,
    average_mode: str = "alive",
) -> torch.Tensor:
    """One-sided PSQ ε prior for hard-surface diagnostics.

    Penalizes only ε above `threshold`; below it, the term is zero. This
    lets us test whether the optimizer merely needs a small anti-rounding
    bias, without forcing already-boxy primitives lower.
    """
    eps_vals = scene.psq_shape()
    alive = scene.alive.to(eps_vals.dtype).unsqueeze(-1)
    penalty = F.relu(eps_vals - threshold)
    num = (penalty * alive).sum()
    den = (scene.K * eps_vals.shape[-1]) if average_mode == "fixed_k" else alive.sum() * eps_vals.shape[-1] + 1e-8
    return num / den


def loss_norm_reg(
    render: RenderOutput,
    normals_pred: torch.Tensor,    # (R, 3) — from StableNormal or analytic
    mask_gt: torch.Tensor,         # (R,)
    *,
    norm_mode: str = "global_mean",
    loss_type: str = "l1",
) -> torch.Tensor:
    """Eq 18 — masked surface-normal consistency."""
    if loss_type == "angular":
        nr = F.normalize(render.normals, dim=-1, eps=1e-6)
        ng = F.normalize(normals_pred, dim=-1, eps=1e-6)
        diff = 1.0 - (nr * ng).sum(dim=-1).clamp(-1.0, 1.0)
    else:
        diff = (render.normals - normals_pred).abs().sum(dim=-1)
    return _masked_reduce(diff, mask_gt, mode=norm_mode)


def loss_depth(
    render: RenderOutput,
    depth_gt: torch.Tensor,         # (R,) expected foreground depth
    mask_gt: torch.Tensor,          # (R,)
    *,
    norm_mode: str = "fg_mean",
) -> torch.Tensor:
    """Synthetic GT depth diagnostic.

    This is not in the DualPrim paper. It targets the r45 failure mode:
    RGB/mask/normal supervision can prefer a rounded ε≈0.5 solution even
    when a sharp ε≈0.1 solution is used as initialization. Depth supplies
    dense O(N^2) hard-surface signal over visible faces, not just O(N)
    silhouette signal.
    """
    if depth_gt is None:
        return render.mask.sum() * 0
    if render.depth is None:
        return render.mask.sum() * 0
    diff = (render.depth - depth_gt).abs()
    return _masked_reduce(diff, mask_gt, mode=norm_mode)


def loss_overlap(scene: DualPrimScene) -> torch.Tensor:
    """Pairwise PSQ bounding-sphere repulsion (friend's #4 audit fix).

       overlap_ij = ReLU(r_i + r_j - ||t_i - t_j||) ** 2
    where r_i = max(psq_scale[i, :]) is the conservative bounding radius.
    """
    K = scene.K
    if K < 2 or scene.alive.sum().item() < 2:
        return scene.params.new_zeros(())
    scales = scene.psq_scale()              # (K, 3)
    translations = scene.psq_translation()  # (K, 3)
    radii = scales.amax(dim=-1)             # (K,)
    diffs = translations.unsqueeze(0) - translations.unsqueeze(1)
    dists = diffs.norm(dim=-1)              # (K, K)
    radius_sums = radii.unsqueeze(0) + radii.unsqueeze(1)
    overlap = torch.relu(radius_sums - dists)
    alive_f = scene.alive.to(scales.dtype)
    pair_mask = alive_f.unsqueeze(0) * alive_f.unsqueeze(1)
    diag_mask = 1.0 - torch.eye(K, device=scales.device, dtype=scales.dtype)
    pair_mask = pair_mask * diag_mask
    n_pairs = pair_mask.sum() + 1e-8
    return (overlap.pow(2) * pair_mask).sum() / n_pairs


def loss_open_ray(
    render: RenderOutput,
    hole_ray_gt: torch.Tensor,     # (R,) bool — True if this ray should pass through
) -> torch.Tensor:
    """NOT IN THE PAPER — topology-aware augmentation of the mask loss.

    Problem: the standard mask loss (loss_mask above) penalizes
    predicted mask > 0 at hole pixels, which the optimizer can satisfy
    EITHER by:
      (a) shrinking PSQs so no primitive reaches this ray (reduces
          mass globally, fights with silhouette)
      (b) NSQ-carving this ray (what we want)

    Both get equal credit from BCE. Friend's review: "for rays that
    correspond to GT hole pixels, encourage low predicted opacity /
    high transmittance." This function gives that signal DIRECTLY by
    computing the mean predicted mask on exactly those hole rays
    (squared for strong gradient near zero).

    Returns a scalar loss; zero if `hole_ray_gt` is None or has no
    True entries. Safe to always-include in the total loss.

    Round 7 debug: observed silently-NaN gradients that caused the
    optimizer to skip all updates for 3000+ iters. Root cause: the
    renderer occasionally produces non-finite mask values on rays
    where sq_implicit saturates, and backward from NaN = NaN grad =
    silent skip. Guarded now via nan_to_num + clamp.
    """
    if hole_ray_gt is None or hole_ray_gt.sum() == 0:
        # Use .sum() * 0 instead of new_zeros(()) so the loss stays
        # connected to the graph — avoids autograd edge cases when
        # batch-to-batch the output is sometimes detached.
        return render.mask.sum() * 0
    m = render.mask[hole_ray_gt]
    # Guard against any non-finite values from render-side numerics.
    # render.mask should naturally be in [0, 1] but we've seen rare
    # cases where sq_implicit overflow + softmin-weighted-sum produces
    # NaN that propagates. This clip is cheap insurance.
    m = torch.nan_to_num(m, nan=0.0, posinf=1.0, neginf=0.0)
    m = m.clamp(0.0, 1.0)
    # L1 instead of squared. Initial round 7 used squared which had
    # two problems:
    #   (1) gradient = 2m at mask values near 1 is large — with
    #       lambda_open_ray=5 and m~1.0, individual-ray grad contribution
    #       can dominate the backward pass and produce NaN via accumulated
    #       sq_implicit FD-grad overflow in the rendering graph.
    #   (2) near zero (where we want to land), gradient vanishes,
    #       making the loss weak exactly where it should be effective.
    # L1 (absolute value, .mean() of non-negative m is identity) has
    # CONSTANT gradient magnitude and is NaN-resistant.
    return m.mean()


# ---------------------------------------------------------------------
# TSDF loss — NOT in the paper
#
# The paper only supervises via the differentiable renderer. We add
# this separately so `mode="mesh_fit"` (a debug / sanity path that
# skips rendering entirely) has something real to optimize. Do NOT
# use this path for paper-parity runs.
# ---------------------------------------------------------------------

def loss_tsdf(
    scene: DualPrimScene,
    query_points: torch.Tensor,    # (P, 3) query positions
    target_sdf: torch.Tensor,      # (P,) target signed distance
    *,
    mu: float = 0.0,
    theta_min: float = 0.01,
    truncation: float = 0.1,
    beta: float = 8.0,
) -> torch.Tensor:
    """Clamped-L1 between combined dual-primitive field and a target SDF.

    For each query point p, compares the scene's combined field
    f(p, S) (paper Eq 5) against a known target SDF sample (e.g. from
    mesh2sdf on the ground-truth mesh). Both are clamped to
    [-truncation, +truncation] before taking the difference.

    Aggregation across primitives (updated in review round 2):

      scene SDF at p = α-weighted smooth-min of per-primitive SDFs

    In weighted-LogSumExp form:

      softmin_α(f_k) = -(1/β) · logsumexp(log(α_k) − β·f_k)

    This uses α as a CONTINUOUS weight (not just an alive/dead
    boolean, as the previous version did). Primitives with small α
    are suppressed in the min regardless of their SDF value. Dead
    primitives (alive=False) get α set to 0 here so log(α) = −∞
    kills their contribution exactly.

    Note: this aggregation is specific to mesh_fit mode. It is NOT
    paper-parity — the paper supervises via the differentiable
    renderer where α enters as a density scale.
    """
    from clearmesh.dualprim.superquadric import scene_combined_field
    f_comb = scene_combined_field(
        query_points, scene, mu=mu, theta_min=theta_min,
    )   # (P, K)

    alpha = scene.alpha().clamp(0.0, 1.0)
    alive = scene.alive.to(alpha.dtype)
    effective_alpha = alpha * alive                        # (K,)

    # Add a tiny ε to avoid log(0); ε ≪ any live α so live primitives
    # dominate unambiguously.
    alpha_eps = 1e-6
    log_w = torch.log(effective_alpha + alpha_eps)         # (K,) in (-∞, 0]

    # Weighted soft-min: softmin_w(f) = -log(Σ w · exp(-β·f)) / β
    #                  = -logsumexp(log w - β·f) / β
    f_scene = -torch.logsumexp(
        log_w.unsqueeze(0) - beta * f_comb, dim=-1,
    ) / beta

    # Only the TARGET is truncated — NOT the prediction. If we clamp
    # both, a badly-wrong prediction at initialisation (when f_scene
    # is tens-to-thousands because all query points are outside every
    # random-init primitive) saturates against the clamp's max, which
    # has zero gradient. That kills the optimization entirely at the
    # very place we need signal the most.
    #
    # Clamping only the target + using L1 on the difference gives:
    #   - Inside the target's ±truncation band: normal L1 matching.
    #   - Outside: loss increases linearly with |prediction|, which
    #     is the correct thing to penalise (a prediction shouldn't
    #     be ±1000 when the target is a truncated SDF of ~±0.1).
    target_t = target_sdf.clamp(-truncation, truncation)
    return (f_scene - target_t).abs().mean()


def total_loss_tsdf(
    scene: DualPrimScene,
    query_points: torch.Tensor,
    target_sdf: torch.Tensor,
    *,
    lambda_sparse: float = 0.01,
    lambda_entropy: float = 0.01,
    lambda_max: float = 0.1,
    mu: float = 0.0,
    theta_min: float = 0.01,
    truncation: float = 0.1,
    primitive_reg_average_mode: str = "alive",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Total loss for mesh_fit mode.

    Combines:
      - loss_tsdf (the actual geometry-matching term)
      - loss_sparsity / loss_entropy / loss_max (regularizers shared
        with paper mode)
    """
    l_tsdf = loss_tsdf(
        scene, query_points, target_sdf,
        mu=mu, theta_min=theta_min, truncation=truncation,
    )
    l_sp = loss_sparsity(scene, average_mode=primitive_reg_average_mode)
    l_e = loss_entropy(scene, average_mode=primitive_reg_average_mode)
    l_max = loss_max(scene, average_mode=primitive_reg_average_mode)

    total = (
        l_tsdf
        + lambda_sparse * l_sp
        + lambda_entropy * l_e
        + lambda_max * l_max
    )
    parts = {
        "tsdf": l_tsdf.item(),
        "sparse": l_sp.item(),
        "entropy": l_e.item(),
        "max": l_max.item(),
        "total": total.item(),
    }
    return total, parts


# ---------------------------------------------------------------------
# Combined loss
# ---------------------------------------------------------------------

def total_loss(
    scene: DualPrimScene,
    render: RenderOutput,
    rgb_gt: torch.Tensor,
    mask_gt: torch.Tensor,
    normals_pred: torch.Tensor,
    depth_gt: torch.Tensor | None = None,
    *,
    lambda_mask: float = 1.0,
    lambda_sparse: float = 0.01,
    lambda_entropy: float = 0.01,
    lambda_max: float = 0.1,
    lambda_norm_reg: float = 0.1,
    lambda_depth: float = 0.0,
    lambda_open_ray: float = 0.0,
    lambda_overlap: float = 0.0,
    lambda_edge_mask: float = 0.0,
    lambda_shape_box: float = 0.0,
    shape_box_threshold: float = 0.30,
    hole_ray_gt=None,
    edge_weight_gt=None,
    mask_loss_type: str = "bce",
    masked_loss_norm_mode: str = "global_mean",
    primitive_reg_average_mode: str = "alive",
    normal_loss_type: str = "l1",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Eq 12 — weighted sum of the 6 terms + optional open-ray loss.

    Returns (scalar loss, dict for logging).
    """
    l_rgb = loss_rgb(render, rgb_gt, mask_gt, norm_mode=masked_loss_norm_mode)
    l_mask = loss_mask(render, mask_gt, loss_type=mask_loss_type)
    l_sp = loss_sparsity(scene, average_mode=primitive_reg_average_mode)
    l_e = loss_entropy(scene, average_mode=primitive_reg_average_mode)
    l_max = loss_max(scene, average_mode=primitive_reg_average_mode)
    l_edge = (
        loss_edge_mask(render, mask_gt, edge_weight_gt, loss_type=mask_loss_type)
        if lambda_edge_mask > 0 else render.mask.new_zeros(())
    )
    l_shape_box = (
        loss_shape_box(
            scene,
            threshold=shape_box_threshold,
            average_mode=primitive_reg_average_mode,
        )
        if lambda_shape_box > 0 else render.mask.new_zeros(())
    )
    l_norm = loss_norm_reg(
        render,
        normals_pred,
        mask_gt,
        norm_mode=masked_loss_norm_mode,
        loss_type=normal_loss_type,
    )
    l_depth = (
        loss_depth(render, depth_gt, mask_gt, norm_mode="fg_mean")
        if lambda_depth > 0 else render.mask.new_zeros(())
    )
    # New: topology-aware open-ray loss (zero if no hole rays provided)
    l_open = loss_open_ray(render, hole_ray_gt) if lambda_open_ray > 0 else render.mask.new_zeros(())
    l_overlap = loss_overlap(scene) if lambda_overlap > 0 else render.mask.new_zeros(())

    total = (
        l_rgb
        + lambda_mask * l_mask
        + lambda_sparse * l_sp
        + lambda_entropy * l_e
        + lambda_max * l_max
        + lambda_norm_reg * l_norm
        + lambda_depth * l_depth
        + lambda_open_ray * l_open
        + lambda_overlap * l_overlap
        + lambda_edge_mask * l_edge
        + lambda_shape_box * l_shape_box
    )
    parts = {
        "rgb": l_rgb.item(),
        "mask": l_mask.item(),
        "sparse": l_sp.item(),
        "entropy": l_e.item(),
        "max": l_max.item(),
        "norm": l_norm.item(),
        "depth": l_depth.item(),
        "open": l_open.item(),
        "overlap": l_overlap.item(),
        "edge_mask": l_edge.item(),
        "shape_box": l_shape_box.item(),
        "total": total.item(),
    }
    return total, parts
