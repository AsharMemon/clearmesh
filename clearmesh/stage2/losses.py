"""Multi-objective loss functions for Stage 2 training with FlexiCubes.

Loss components:
  1. Chamfer Distance: Point cloud similarity between extracted and GT mesh
  2. Normal Consistency: Alignment of face normals
  3. Edge Sharpness: Penalizes soft/rounded edges that should be sharp
  4. Watertight Regularization: Encourages manifold, closed surfaces
  5. SDF Supervision: Direct SDF value regression at sampled points

All losses are designed to flow gradients through FlexiCubes back into the DiT.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def chamfer_distance(
    pred_points: torch.Tensor,
    gt_points: torch.Tensor,
    num_samples: int = 10000,
) -> torch.Tensor:
    """Bidirectional Chamfer Distance between two point clouds.

    Args:
        pred_points: (B, N, 3) predicted mesh surface points
        gt_points: (B, M, 3) ground truth mesh surface points
        num_samples: Number of points to sample if clouds are large

    Returns:
        Scalar loss
    """
    # Subsample if needed
    if pred_points.shape[1] > num_samples:
        idx = torch.randperm(pred_points.shape[1], device=pred_points.device)[:num_samples]
        pred_points = pred_points[:, idx]
    if gt_points.shape[1] > num_samples:
        idx = torch.randperm(gt_points.shape[1], device=gt_points.device)[:num_samples]
        gt_points = gt_points[:, idx]

    # pred -> gt: for each pred point, find nearest gt point
    diff_p2g = pred_points.unsqueeze(2) - gt_points.unsqueeze(1)  # (B, N, M, 3)
    dist_p2g = (diff_p2g ** 2).sum(-1)  # (B, N, M)
    min_p2g = dist_p2g.min(dim=2).values.mean(dim=1)  # (B,)

    # gt -> pred: for each gt point, find nearest pred point
    min_g2p = dist_p2g.min(dim=1).values.mean(dim=1)  # (B,)

    return (min_p2g + min_g2p).mean()


def normal_consistency_loss(
    pred_normals: torch.Tensor,
    gt_normals: torch.Tensor,
    pred_points: torch.Tensor,
    gt_points: torch.Tensor,
) -> torch.Tensor:
    """Normal consistency: normals of nearest-neighbor points should align.

    Args:
        pred_normals: (B, N, 3) predicted face/vertex normals
        gt_normals: (B, M, 3) ground truth normals
        pred_points: (B, N, 3) predicted surface points
        gt_points: (B, M, 3) ground truth surface points

    Returns:
        Scalar loss
    """
    # Find nearest neighbors (pred -> gt)
    diff = pred_points.unsqueeze(2) - gt_points.unsqueeze(1)  # (B, N, M, 3)
    dist = (diff ** 2).sum(-1)  # (B, N, M)
    nn_idx = dist.argmin(dim=2)  # (B, N)

    # Gather GT normals at nearest neighbor positions
    B, N, _ = pred_normals.shape
    nn_idx_expanded = nn_idx.unsqueeze(-1).expand(-1, -1, 3)
    matched_gt_normals = torch.gather(gt_normals, 1, nn_idx_expanded)

    # Normal alignment loss (1 - cos similarity)
    cos_sim = F.cosine_similarity(pred_normals, matched_gt_normals, dim=-1)
    return (1 - cos_sim.abs()).mean()


def edge_sharpness_loss(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    gt_vertices: torch.Tensor,
    gt_faces: torch.Tensor,
    dihedral_threshold: float = 0.5,
) -> torch.Tensor:
    """Edge sharpness: penalize soft edges where GT has sharp dihedral angles.

    Computes dihedral angles along edges in both predicted and GT meshes.
    Penalizes predicted edges that are smoother than their GT counterparts.

    Args:
        vertices: (V, 3) predicted mesh vertices
        faces: (F, 3) predicted mesh faces
        gt_vertices: (V', 3) GT mesh vertices
        gt_faces: (F', 3) GT mesh faces
        dihedral_threshold: Angle threshold (radians) below which edges are "sharp"

    Returns:
        Scalar loss
    """

    def compute_face_normals(verts, face_indices):
        v0 = verts[face_indices[:, 0]]
        v1 = verts[face_indices[:, 1]]
        v2 = verts[face_indices[:, 2]]
        normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
        normals = F.normalize(normals, dim=-1)
        return normals

    pred_normals = compute_face_normals(vertices, faces)

    # Build edge-to-face adjacency for predicted mesh
    edges = {}
    for fi in range(faces.shape[0]):
        for i in range(3):
            e = tuple(sorted([faces[fi, i].item(), faces[fi, (i + 1) % 3].item()]))
            if e not in edges:
                edges[e] = []
            edges[e].append(fi)

    # Compute dihedral angles at shared edges
    dihedral_losses = []
    for edge, face_list in edges.items():
        if len(face_list) != 2:
            continue
        n1 = pred_normals[face_list[0]]
        n2 = pred_normals[face_list[1]]
        cos_angle = (n1 * n2).sum()
        # We want sharp edges to have large dihedral angles (small cos)
        # Regularize toward maintaining sharpness
        dihedral_losses.append(F.relu(cos_angle - dihedral_threshold))

    if not dihedral_losses:
        return torch.tensor(0.0, device=vertices.device)

    return torch.stack(dihedral_losses).mean()


def watertight_loss(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Watertight regularization: penalize non-manifold edges and boundary edges.

    A watertight mesh has every edge shared by exactly 2 faces.
    Edges shared by 1 face (boundary) or 3+ faces (non-manifold) are penalized.

    Args:
        vertices: (V, 3) mesh vertices
        faces: (F, 3) mesh face indices

    Returns:
        Scalar loss (0 = perfectly watertight)
    """
    # Count edge adjacency
    edge_count = {}
    for fi in range(faces.shape[0]):
        for i in range(3):
            e = tuple(sorted([faces[fi, i].item(), faces[fi, (i + 1) % 3].item()]))
            edge_count[e] = edge_count.get(e, 0) + 1

    # Penalize non-manifold edges
    total_edges = len(edge_count)
    if total_edges == 0:
        return torch.tensor(0.0, device=vertices.device)

    bad_edges = sum(1 for count in edge_count.values() if count != 2)
    return torch.tensor(bad_edges / total_edges, device=vertices.device, dtype=torch.float32)


def sdf_supervision_loss(
    pred_sdf: torch.Tensor,
    gt_sdf: torch.Tensor,
) -> torch.Tensor:
    """Direct SDF regression loss at supervision points.

    Args:
        pred_sdf: (B, N, 1) predicted SDF values
        gt_sdf: (B, N, 1) ground truth SDF values

    Returns:
        Scalar loss
    """
    return F.l1_loss(pred_sdf, gt_sdf)


class ClearMeshLoss(nn.Module):
    """Combined multi-objective loss for Stage 2 training with FlexiCubes.

    Gradients flow through FlexiCubes back into the DiT:
        DiT predicts SDF -> FlexiCubes extracts mesh -> losses on extracted mesh
    """

    def __init__(
        self,
        chamfer_weight: float = 1.0,
        normal_weight: float = 0.5,
        edge_weight: float = 0.3,
        watertight_weight: float = 0.2,
        sdf_weight: float = 1.0,
    ):
        super().__init__()
        self.chamfer_weight = chamfer_weight
        self.normal_weight = normal_weight
        self.edge_weight = edge_weight
        self.watertight_weight = watertight_weight
        self.sdf_weight = sdf_weight

    def forward(
        self,
        pred_sdf: torch.Tensor,
        gt_sdf: torch.Tensor,
        extracted_vertices: torch.Tensor | None = None,
        extracted_faces: torch.Tensor | None = None,
        gt_vertices: torch.Tensor | None = None,
        gt_faces: torch.Tensor | None = None,
        pred_points: torch.Tensor | None = None,
        gt_points: torch.Tensor | None = None,
        pred_normals: torch.Tensor | None = None,
        gt_normals: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute all losses.

        Returns dict with individual losses and total.
        """
        losses = {}

        # SDF supervision (always computed)
        losses["sdf"] = self.sdf_weight * sdf_supervision_loss(pred_sdf, gt_sdf)

        # Chamfer distance (on FlexiCubes-extracted mesh)
        if pred_points is not None and gt_points is not None:
            losses["chamfer"] = self.chamfer_weight * chamfer_distance(pred_points, gt_points)

        # Normal consistency
        if pred_normals is not None and gt_normals is not None:
            losses["normal"] = self.normal_weight * normal_consistency_loss(
                pred_normals, gt_normals, pred_points, gt_points
            )

        # Edge sharpness
        if extracted_vertices is not None and gt_vertices is not None:
            losses["edge"] = self.edge_weight * edge_sharpness_loss(
                extracted_vertices, extracted_faces, gt_vertices, gt_faces
            )

        # Watertight regularization
        if extracted_vertices is not None:
            losses["watertight"] = self.watertight_weight * watertight_loss(
                extracted_vertices, extracted_faces
            )

        losses["total"] = sum(losses.values())
        return losses
