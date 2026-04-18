"""Mesh exportation via Boolean difference per dual-primitive (§3.3).

Per-primitive export:
  1. Tessellate PSQ on a local grid (marching cubes on its implicit).
  2. Tessellate NSQ on the same grid.
  3. mesh_out = PSQ_mesh \\ NSQ_mesh   (boolean difference)

Scene-level export:
  * Discard primitives with α < T_export (paper T_export = 0.5).
  * Discard primitives killed by pruning.
  * Concatenate the per-primitive meshes. Optionally union across
    primitives via manifold3d if a single watertight mesh is desired.

Backends:
  * manifold3d (preferred — exact boolean, vertex-shared at seams)
  * trimesh.boolean (falls back to manifold3d under the hood in modern
    trimesh, but we provide an explicit wrapper)

NOT SPECIFIED IN PAPER:
  * Tessellation resolution for marching cubes. Default 128^3 per
    primitive local grid.
  * Whether to union across primitives or just concatenate.
"""

from __future__ import annotations

import math
import warnings
from typing import List, Optional

import numpy as np
import torch
import trimesh

from clearmesh.dualprim.params import DualPrimConfig
from clearmesh.dualprim.types import DualPrimitive, DualPrimScene
from clearmesh.dualprim.superquadric import sq_implicit, _euler_xyz_to_mat


# ---------------------------------------------------------------------
# Per-primitive tessellation
# ---------------------------------------------------------------------

def tessellate_superquadric(
    translation: torch.Tensor,     # (3,)
    rotation: torch.Tensor,        # (3,) Euler XYZ rad
    scale: torch.Tensor,           # (3,)
    shape: torch.Tensor,           # (2,)
    resolution: int = 128,
    margin: float = 1.4,
) -> trimesh.Trimesh:
    """Marching-cubes tessellate one SQ.

    Grid spans [-margin*max_scale, +margin*max_scale] around the SQ
    translation to guarantee we capture the full zero level set.
    """
    from skimage.measure import marching_cubes

    device = translation.device
    half = float(margin * scale.max().item())
    if half < 1e-3:
        return trimesh.Trimesh()

    lin = torch.linspace(-half, half, resolution, device=device)
    gx, gy, gz = torch.meshgrid(lin, lin, lin, indexing="ij")
    pts = torch.stack([gx, gy, gz], dim=-1) + translation.view(1, 1, 1, 3)
    pts_flat = pts.reshape(-1, 3)

    # sq_implicit expects (K,*) sized params; wrap as K=1
    t = translation.view(1, 3)
    r = rotation.view(1, 3)
    s = scale.view(1, 3)
    sh = shape.view(1, 2)
    f = sq_implicit(pts_flat, t, r, s, sh).reshape(resolution, resolution, resolution)
    f_np = f.detach().cpu().numpy()

    if not (f_np.min() < 0 < f_np.max()):
        return trimesh.Trimesh()

    try:
        verts, faces, _, _ = marching_cubes(f_np, level=0.0)
    except (ValueError, RuntimeError):
        return trimesh.Trimesh()

    # Grid index -> world
    verts = verts / (resolution - 1) * (2 * half) - half
    verts = verts + translation.detach().cpu().numpy()
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


# ---------------------------------------------------------------------
# Boolean difference
# ---------------------------------------------------------------------

def boolean_difference(
    psq_mesh: trimesh.Trimesh,
    nsq_mesh: trimesh.Trimesh,
    backend: str = "manifold3d",
) -> trimesh.Trimesh:
    """Compute psq \\ nsq. Returns psq if nsq is empty."""
    if len(nsq_mesh.faces) == 0:
        return psq_mesh
    if len(psq_mesh.faces) == 0:
        return trimesh.Trimesh()

    if backend == "manifold3d":
        try:
            import manifold3d as m3d
            m_psq = m3d.Manifold(
                m3d.Mesh(
                    vert_properties=np.asarray(psq_mesh.vertices, dtype=np.float32),
                    tri_verts=np.asarray(psq_mesh.faces, dtype=np.uint32),
                )
            )
            m_nsq = m3d.Manifold(
                m3d.Mesh(
                    vert_properties=np.asarray(nsq_mesh.vertices, dtype=np.float32),
                    tri_verts=np.asarray(nsq_mesh.faces, dtype=np.uint32),
                )
            )
            diff = m_psq - m_nsq
            out_mesh = diff.to_mesh()
            return trimesh.Trimesh(
                vertices=np.asarray(out_mesh.vert_properties[:, :3]),
                faces=np.asarray(out_mesh.tri_verts),
                process=False,
            )
        except ImportError:
            warnings.warn("[dualprim/export] manifold3d not installed; falling back to trimesh")
            backend = "trimesh"

    if backend == "trimesh":
        try:
            return trimesh.boolean.difference([psq_mesh, nsq_mesh])
        except Exception as e:
            warnings.warn(f"[dualprim/export] trimesh.boolean.difference failed: {e}; "
                          f"returning PSQ mesh only")
            return psq_mesh

    raise ValueError(f"Unknown backend: {backend}")


# ---------------------------------------------------------------------
# Scene-level export (§3.3)
# ---------------------------------------------------------------------

def export_dual_primitive(
    dp: DualPrimitive,
    resolution: int = 128,
    backend: str = "manifold3d",
) -> trimesh.Trimesh:
    """Per dual-primitive: mesh = PSQ_mesh \\ NSQ_mesh."""
    psq = tessellate_superquadric(
        dp.psq_translation, dp.psq_rotation_rad,
        dp.psq_scale, dp.psq_shape,
        resolution=resolution,
    )
    nsq = tessellate_superquadric(
        dp.nsq_translation, dp.nsq_rotation_rad,
        dp.nsq_scale, dp.nsq_shape,
        resolution=resolution,
    )
    return boolean_difference(psq, nsq, backend=backend)


def export_scene(
    scene: DualPrimScene,
    config: DualPrimConfig,
    *,
    union_all: bool = False,
) -> tuple[trimesh.Trimesh, List[trimesh.Trimesh]]:
    """Export the full scene to a single mesh + per-primitive meshes.

    Args:
        scene: trained DualPrimScene
        config: has ``export_alpha_threshold`` + ``tessellation_resolution``
                + ``boolean_backend``
        union_all: if True, boolean-union all per-primitive meshes into
                   a single watertight scene mesh (slow but clean).
                   If False, just concatenate (fast, not watertight
                   across primitives).

    Returns:
        scene_mesh: the single exported mesh
        per_prim:   per-dual-primitive meshes (same order as input)
    """
    alpha = scene.alpha().detach().cpu().numpy()
    alive = scene.alive.detach().cpu().numpy()
    keep_mask = alive & (alpha >= config.export_alpha_threshold)
    keep_idx = np.where(keep_mask)[0]

    per_prim: List[trimesh.Trimesh] = []
    for i in keep_idx:
        dp = scene.get_primitive(int(i))
        m = export_dual_primitive(
            dp,
            resolution=config.tessellation_resolution,
            backend=config.boolean_backend,
        )
        per_prim.append(m)

    if len(per_prim) == 0:
        return trimesh.Trimesh(), []

    if union_all:
        try:
            scene_mesh = trimesh.boolean.union(per_prim)
        except Exception as e:
            warnings.warn(f"[dualprim/export] union failed ({e}); concatenating")
            scene_mesh = trimesh.util.concatenate(per_prim)
    else:
        scene_mesh = trimesh.util.concatenate(per_prim)

    return scene_mesh, per_prim
