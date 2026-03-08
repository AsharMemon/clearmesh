"""Multi-objective losses for Stage 2 training.

Loss components:
  1. SDF supervision — truncated, surface-weighted L1 (core loss)
  2. Eikonal regularisation — |∇SDF| ≈ 1 near the surface
  3. Chamfer distance — point-cloud similarity on extracted meshes
  4. Normal consistency — face-normal alignment at nearest neighbours
  5. Edge sharpness — penalise soft edges where GT is sharp
  6. Watertight regularisation — penalise non-manifold / boundary edges

SDF-specific improvements (following UltraShape / DeepSDF best practices):
  - Truncation:  clamp GT & pred SDF to ±τ  (default τ = 0.1)
  - Surface weighting:  up-weight points near the zero-crossing
  - Near-surface sampling:  dataset should sample 60 %+ near surface
  - Eikonal:  finite-difference ∇SDF magnitude regularisation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# SDF losses
# ---------------------------------------------------------------------------

def sdf_supervision_loss(
    pred_sdf: torch.Tensor,
    gt_sdf: torch.Tensor,
    truncation: float = 0.1,
    surface_weight: float = 5.0,
) -> torch.Tensor:
    """Truncated, surface-weighted L1 SDF loss.

    1. Clamp both pred and GT to [-τ, τ]  (ignore far-field)
    2. Weight points near the zero-crossing higher  (surface band matters most)
    3. Return weighted mean L1

    Args:
        pred_sdf: (B, N, 1)
        gt_sdf: (B, N, 1)
        truncation: τ — clamp range
        surface_weight: max weight multiplier at the surface (exponential decay)
    """
    gt_c = gt_sdf.clamp(-truncation, truncation)
    pred_c = pred_sdf.clamp(-truncation, truncation)

    # Exponential surface weighting:  w = 1 + (W-1) · exp(-|gt| / σ)
    sigma = truncation / 3.0  # most weight concentrated within ±σ
    weights = 1.0 + (surface_weight - 1.0) * torch.exp(-gt_c.abs() / sigma)

    return (weights * (pred_c - gt_c).abs()).mean()


def eikonal_loss(
    pred_sdf: torch.Tensor,
    gt_sdf: torch.Tensor,
    truncation: float = 0.1,
    positions: torch.Tensor | None = None,
    k: int = 6,
) -> torch.Tensor:
    """Eikonal regularisation: encourage |∇SDF| ≈ 1 near the surface.

    For sparse voxel inputs (the common case), uses k-nearest-neighbour
    spatial gradients.  For each near-surface point we find its k spatial
    neighbours, compute (ΔSDF / Δpos) for each pair, and penalise
    deviations from unit gradient magnitude.

    Args:
        pred_sdf:  (B, N, 1)  predicted SDF values
        gt_sdf:    (B, N, 1)  ground-truth SDF values (for surface mask)
        truncation: τ — only penalise inside this band
        positions: (B, N, 3)  world-space positions of the tokens.
                   Required for sparse inputs.  If None, falls back to
                   a 1-D finite-difference approximation (grid-like layout).
        k: number of spatial neighbours for gradient estimation (default 6,
           matching the 6-connected neighbourhood on a regular grid)

    For dense grid SDF (B, R, R, R) see ``eikonal_loss_grid``.
    """
    if positions is None:
        # Legacy fallback: 1-D finite differences (only valid for grid-ordered tokens)
        dx = pred_sdf[:, 1:, :] - pred_sdf[:, :-1, :]
        grad_mag = dx.abs()
        near_surface = gt_sdf[:, :-1, :].abs() < truncation
        if near_surface.sum() == 0:
            return torch.tensor(0.0, device=pred_sdf.device)
        return ((grad_mag[near_surface] - 1.0) ** 2).mean()

    B, N, _ = pred_sdf.shape
    if N < k + 1:
        return torch.tensor(0.0, device=pred_sdf.device)

    losses = []
    for b in range(B):
        # Near-surface mask
        near_mask = gt_sdf[b, :, 0].abs() < truncation  # (N,)
        if near_mask.sum() == 0:
            continue

        pos_b = positions[b]        # (N, 3) float
        sdf_b = pred_sdf[b, :, 0]   # (N,)

        # k-NN on positions (using cdist — N is small, typically <8192)
        with torch.no_grad():
            dists = torch.cdist(pos_b.unsqueeze(0), pos_b.unsqueeze(0)).squeeze(0)  # (N, N)
            # Exclude self (set diagonal to inf)
            dists.fill_diagonal_(float('inf'))
            # Get k nearest neighbours
            actual_k = min(k, N - 1)
            _, nn_idx = dists.topk(actual_k, dim=1, largest=False)  # (N, k)

        # Compute spatial gradients via finite differences to neighbours
        nn_sdf = sdf_b[nn_idx]           # (N, k)
        nn_pos = pos_b[nn_idx]           # (N, k, 3)
        delta_sdf = nn_sdf - sdf_b.unsqueeze(1)              # (N, k)
        delta_pos = nn_pos - pos_b.unsqueeze(1)               # (N, k, 3)
        delta_dist = delta_pos.norm(dim=-1).clamp(min=1e-8)   # (N, k)

        # |∇SDF| ≈ |ΔSDF| / |Δpos| for each neighbour pair
        grad_mag = (delta_sdf.abs() / delta_dist)  # (N, k)

        # Average over neighbours, then penalise deviation from 1
        grad_mag_mean = grad_mag.mean(dim=1)  # (N,)

        loss_b = ((grad_mag_mean[near_mask] - 1.0) ** 2).mean()
        losses.append(loss_b)

    if not losses:
        return torch.tensor(0.0, device=pred_sdf.device)
    return torch.stack(losses).mean()


def eikonal_loss_grid(
    sdf_grid: torch.Tensor,
    truncation: float = 0.1,
) -> torch.Tensor:
    """Eikonal regularisation on a dense SDF grid (B, R, R, R).

    Central-difference gradient → penalise |∇SDF| ≠ 1 near the surface.
    """
    # Central differences (interior only)
    dx = (sdf_grid[:, 2:, 1:-1, 1:-1] - sdf_grid[:, :-2, 1:-1, 1:-1]) / 2
    dy = (sdf_grid[:, 1:-1, 2:, 1:-1] - sdf_grid[:, 1:-1, :-2, 1:-1]) / 2
    dz = (sdf_grid[:, 1:-1, 1:-1, 2:] - sdf_grid[:, 1:-1, 1:-1, :-2]) / 2

    grad_mag = (dx ** 2 + dy ** 2 + dz ** 2).sqrt()

    # Interior SDF values (same crop)
    sdf_interior = sdf_grid[:, 1:-1, 1:-1, 1:-1]
    near_surface = sdf_interior.abs() < truncation

    if near_surface.sum() == 0:
        return torch.tensor(0.0, device=sdf_grid.device)

    return ((grad_mag[near_surface] - 1.0) ** 2).mean()


# ---------------------------------------------------------------------------
# Mesh-space losses (used when FlexiCubes extraction succeeds)
# ---------------------------------------------------------------------------

def chamfer_distance(
    pred_points: torch.Tensor,
    gt_points: torch.Tensor,
    num_samples: int = 10_000,
) -> torch.Tensor:
    """Bidirectional Chamfer distance."""
    if pred_points.shape[1] > num_samples:
        idx = torch.randperm(pred_points.shape[1], device=pred_points.device)[:num_samples]
        pred_points = pred_points[:, idx]
    if gt_points.shape[1] > num_samples:
        idx = torch.randperm(gt_points.shape[1], device=gt_points.device)[:num_samples]
        gt_points = gt_points[:, idx]

    diff = pred_points.unsqueeze(2) - gt_points.unsqueeze(1)  # (B, N, M, 3)
    dist_sq = (diff ** 2).sum(-1)
    return dist_sq.min(2).values.mean() + dist_sq.min(1).values.mean()


def normal_consistency_loss(
    pred_normals: torch.Tensor,
    gt_normals: torch.Tensor,
    pred_points: torch.Tensor,
    gt_points: torch.Tensor,
) -> torch.Tensor:
    """Normal alignment at nearest-neighbour correspondences."""
    diff = pred_points.unsqueeze(2) - gt_points.unsqueeze(1)
    nn_idx = (diff ** 2).sum(-1).argmin(2)
    B, N, _ = pred_normals.shape
    matched = torch.gather(gt_normals, 1, nn_idx.unsqueeze(-1).expand(-1, -1, 3))
    cos_sim = F.cosine_similarity(pred_normals, matched, dim=-1)
    return (1 - cos_sim.abs()).mean()


def edge_sharpness_loss(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    dihedral_threshold: float = 0.5,
) -> torch.Tensor:
    """Penalise rounded edges (encourage sharp dihedral angles)."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    normals = F.normalize(torch.cross(v1 - v0, v2 - v0, dim=-1), dim=-1)

    edges: dict[tuple[int, int], list[int]] = {}
    for fi in range(faces.shape[0]):
        for j in range(3):
            e = tuple(sorted([faces[fi, j].item(), faces[fi, (j + 1) % 3].item()]))
            edges.setdefault(e, []).append(fi)

    losses = []
    for face_list in edges.values():
        if len(face_list) == 2:
            cos_a = (normals[face_list[0]] * normals[face_list[1]]).sum()
            losses.append(F.relu(cos_a - dihedral_threshold))

    if not losses:
        return torch.tensor(0.0, device=vertices.device)
    return torch.stack(losses).mean()


def watertight_loss(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Fraction of non-manifold edges (target: 0)."""
    edge_count: dict[tuple[int, int], int] = {}
    for fi in range(faces.shape[0]):
        for j in range(3):
            e = tuple(sorted([faces[fi, j].item(), faces[fi, (j + 1) % 3].item()]))
            edge_count[e] = edge_count.get(e, 0) + 1

    total = len(edge_count)
    if total == 0:
        return torch.tensor(0.0, device=vertices.device)
    bad = sum(1 for c in edge_count.values() if c != 2)
    return torch.tensor(bad / total, device=vertices.device, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Combined loss
# ---------------------------------------------------------------------------

class ClearMeshLoss(nn.Module):
    """Multi-objective loss for Stage 2 training.

    Always computes:
      - Truncated, surface-weighted SDF loss
      - Eikonal regularisation

    Optionally (when FlexiCubes extraction provides meshes):
      - Chamfer distance
      - Normal consistency
      - Edge sharpness
      - Watertight regularisation
    """

    def __init__(
        self,
        sdf_weight: float = 1.0,
        eikonal_weight: float = 0.1,
        chamfer_weight: float = 1.0,
        normal_weight: float = 0.5,
        edge_weight: float = 0.3,
        watertight_weight: float = 0.2,
        sdf_truncation: float = 0.1,
        sdf_surface_weight: float = 5.0,
    ):
        super().__init__()
        self.sdf_weight = sdf_weight
        self.eikonal_weight = eikonal_weight
        self.chamfer_weight = chamfer_weight
        self.normal_weight = normal_weight
        self.edge_weight = edge_weight
        self.watertight_weight = watertight_weight
        self.sdf_truncation = sdf_truncation
        self.sdf_surface_weight = sdf_surface_weight

    def forward(
        self,
        pred_sdf: torch.Tensor,
        gt_sdf: torch.Tensor,
        # Sparse token positions for spatial eikonal
        positions: torch.Tensor | None = None,
        # Optional mesh-space args (from FlexiCubes extraction)
        extracted_vertices: torch.Tensor | None = None,
        extracted_faces: torch.Tensor | None = None,
        gt_vertices: torch.Tensor | None = None,
        gt_faces: torch.Tensor | None = None,
        pred_points: torch.Tensor | None = None,
        gt_points: torch.Tensor | None = None,
        pred_normals: torch.Tensor | None = None,
        gt_normals: torch.Tensor | None = None,
        # Optional grid SDF for Eikonal
        pred_sdf_grid: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        losses: dict[str, torch.Tensor] = {}

        # --- SDF supervision (always) ---
        losses["sdf"] = self.sdf_weight * sdf_supervision_loss(
            pred_sdf, gt_sdf, self.sdf_truncation, self.sdf_surface_weight
        )

        # --- Eikonal (always, either grid or flat) ---
        if self.eikonal_weight > 0:
            if pred_sdf_grid is not None:
                losses["eikonal"] = self.eikonal_weight * eikonal_loss_grid(
                    pred_sdf_grid, self.sdf_truncation
                )
            else:
                losses["eikonal"] = self.eikonal_weight * eikonal_loss(
                    pred_sdf, gt_sdf, self.sdf_truncation,
                    positions=positions,
                )

        # --- Mesh-space losses (when FlexiCubes provides them) ---
        if pred_points is not None and gt_points is not None:
            losses["chamfer"] = self.chamfer_weight * chamfer_distance(pred_points, gt_points)

        if pred_normals is not None and gt_normals is not None:
            losses["normal"] = self.normal_weight * normal_consistency_loss(
                pred_normals, gt_normals, pred_points, gt_points
            )

        if extracted_vertices is not None and extracted_faces is not None:
            losses["edge"] = self.edge_weight * edge_sharpness_loss(
                extracted_vertices, extracted_faces
            )
            losses["watertight"] = self.watertight_weight * watertight_loss(
                extracted_vertices, extracted_faces
            )

        losses["total"] = sum(losses.values())
        return losses
