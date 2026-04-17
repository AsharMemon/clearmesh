"""Light-SQ port — mesh -> superquadric decomposition.

Upstream: Light-SQ (arXiv 2509.24986, SIGGRAPH Asia 2025).
The upstream repo (johannwyh/Light-SQ) was a README stub at the time
of writing, so this is a from-paper port of the CORE algorithm only:

  1. Build a 100^3 TSDF of the input mesh.
  2. Greedy loop:
       a. Fit one superquadric to the current TSDF (Adam on L2 loss).
       b. "SDF carve": subtract the fitted primitive's volume from the
          target TSDF so the next iteration sees what's left.
       c. Stop when residual occupied-voxel count drops below threshold.
  3. Return the list of fitted superquadrics, plus a tessellated union
     mesh rebuilt by sampling each primitive's implicit surface with
     marching cubes.

Skipped vs the full paper (acceptable for a first prototype):
  - Structure-aware convex decomposition + block-regrow-fill (§3.3).
    Those reduce fragmentation at convex partition boundaries and
    improve accuracy — without them this is basically Marching-Primitives
    in spirit.
  - Adaptive residual pruning by Main/Connector/Offcut classes (§3.4).
    We prune on a single size threshold.
  - EM-style σ² update (§A). We use plain Adam on L2 loss.

Output quality expectation:
  - Cleaner than EMS / SPAGHETTI on TRELLIS-regime input (where
    Light-SQ's benchmark lives)
  - Worse than full paper (expect 2-3x more residual error)
  - Good enough to demonstrate feasibility and compose with CAD-Recode

Licence: this is a code port of an academic paper. The paper text is
© SIGGRAPH; this implementation is MIT-compatible since the algorithm
is described openly. The original repo has no licence yet.

Usage:
    from clearmesh.refit.light_sq import LightSQRefiner

    refiner = LightSQRefiner(grid_res=100, n_iters=200, max_primitives=60)
    result = refiner.fit(mesh)
    print(f"{len(result.primitives)} superquadrics")
    union_mesh = refiner.compile(result.primitives)
    union_mesh.export("refit.glb")
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import trimesh


__all__ = ["LightSQRefiner", "SuperQuadric", "LightSQResult"]


# =====================================================================
# Superquadric primitive
# =====================================================================

@dataclass
class SuperQuadric:
    """11-parameter superquadric.

    Implicit signed radial distance (Eq. 4 in the paper):
        f(x, y, z) = ((|X|^(2/e2) + |Y|^(2/e2))^(e2/e1) + |Z|^(2/e1))^(e1/2)
                     - 1
        phi(x)     = (1 - f^(-e1/2)) * ||g^-1(x)||_2
    where (X, Y, Z) = g^-1(x) / a, g is the rigid transform (R t).

    f < 1: inside the superquadric
    f = 1: on the surface
    f > 1: outside
    """

    # Shape (roundness) exponents in [0.1, 2.0]
    e1: float = 1.0
    e2: float = 1.0
    # Scale along local axes
    ax: float = 0.3
    ay: float = 0.3
    az: float = 0.3
    # Rotation as 3D Euler (XYZ intrinsic)
    rx: float = 0.0
    ry: float = 0.0
    rz: float = 0.0
    # Translation
    tx: float = 0.0
    ty: float = 0.0
    tz: float = 0.0

    def to_vector(self) -> np.ndarray:
        return np.array(
            [self.e1, self.e2, self.ax, self.ay, self.az,
             self.rx, self.ry, self.rz, self.tx, self.ty, self.tz],
            dtype=np.float64,
        )

    @classmethod
    def from_vector(cls, v: np.ndarray) -> "SuperQuadric":
        return cls(*[float(x) for x in v])


def _rotation_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """XYZ intrinsic Euler to 3x3 rotation."""
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _sq_f(
    points: np.ndarray,
    e1: float, e2: float,
    a: np.ndarray, R: np.ndarray, t: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Canonical superquadric implicit. points: (N, 3)."""
    # To local frame
    local = (points - t) @ R  # (N, 3)
    X = np.abs(local[:, 0]) / max(a[0], eps)
    Y = np.abs(local[:, 1]) / max(a[1], eps)
    Z = np.abs(local[:, 2]) / max(a[2], eps)
    # Standard SQ implicit (Barr 1981)
    inner = X ** (2.0 / e2) + Y ** (2.0 / e2)
    f = inner ** (e2 / e1) + Z ** (2.0 / e1)
    return f


def _sq_phi(
    points: np.ndarray, sq: SuperQuadric, eps: float = 1e-6,
) -> np.ndarray:
    """Signed radial distance approximation (paper Eq. 4).

    phi < 0: inside, phi > 0: outside. Units roughly match world distance.
    """
    a = np.array([sq.ax, sq.ay, sq.az], dtype=np.float64)
    R = _rotation_matrix(sq.rx, sq.ry, sq.rz)
    t = np.array([sq.tx, sq.ty, sq.tz], dtype=np.float64)

    f = _sq_f(points, sq.e1, sq.e2, a, R, t, eps)
    f = np.clip(f, eps, 1e9)
    r = np.linalg.norm(points - t, axis=-1)
    # Sign: inside if f < 1
    inside = f < 1.0
    phi = (1.0 - f ** (-sq.e1 / 2.0)) * r
    phi[inside] = -np.abs(phi[inside])
    phi[~inside] = np.abs(phi[~inside])
    return phi


# =====================================================================
# TSDF volume
# =====================================================================

def build_tsdf(
    mesh: trimesh.Trimesh,
    resolution: int = 100,
    tau: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a signed, truncated distance field on a regular grid over
    [-1, 1]^3.

    Returns:
        phi:     (R, R, R) truncated SDF, negative inside.
        coords:  (R, R, R, 3) voxel-centre world coordinates.

    Prefers ``mesh2sdf.compute`` (OpenMP + BVH, seconds on 100^3 for
    100k-face mesh). Falls back to ``trimesh.contains`` + scipy EDT
    if mesh2sdf isn't installed (warning: that fallback fans out to
    100GB+ memory on >100k-face meshes).
    """
    if tau is None:
        tau = 2.0 / resolution  # Paper: tau = voxel edge

    lin = np.linspace(-1.0 + 1.0 / resolution, 1.0 - 1.0 / resolution, resolution)
    gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
    coords = np.stack([gx, gy, gz], axis=-1).astype(np.float64)

    # --- Preferred: mesh2sdf ---
    try:
        import mesh2sdf  # type: ignore
    except ImportError:
        mesh2sdf = None

    if mesh2sdf is not None:
        # mesh2sdf expects the mesh to live inside [-1+1/N, 1-1/N]^3
        # and returns a (N, N, N) signed distance with POSITIVE inside.
        verts = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        sdf_pos_inside = mesh2sdf.compute(
            verts, faces,
            size=resolution,
            fix=False,
            level=2.0 / resolution,
            return_mesh=False,
        )
        # Flip sign to match paper: negative inside
        sd = -sdf_pos_inside.astype(np.float64)
        phi = np.clip(sd, -tau, +tau)
        return phi, coords

    # --- Fallback: occupancy + EDT ---
    import warnings
    warnings.warn(
        "[light_sq] mesh2sdf not installed; falling back to "
        "trimesh.contains which is slow and memory-heavy on >100k-face "
        "meshes. `pip install mesh2sdf` for a 10-50x speedup."
    )
    flat = coords.reshape(-1, 3)
    chunk = 100_000
    occupied = np.zeros(len(flat), dtype=bool)
    for i in range(0, len(flat), chunk):
        occupied[i:i + chunk] = mesh.contains(flat[i:i + chunk])
    occ_grid = occupied.reshape(resolution, resolution, resolution)

    if occ_grid.sum() == 0 or occ_grid.sum() == occ_grid.size:
        phi = np.full_like(occ_grid, +tau if not occ_grid.any() else -tau, dtype=np.float64)
        return phi, coords

    from scipy.ndimage import distance_transform_edt
    voxel_size = 2.0 / resolution
    d_outside = distance_transform_edt(~occ_grid) * voxel_size
    d_inside = distance_transform_edt(occ_grid) * voxel_size
    sd = np.where(occ_grid, -d_inside, d_outside)
    phi = np.clip(sd, -tau, +tau)
    return phi, coords


# =====================================================================
# Single-primitive optimisation
# =====================================================================

def _init_from_moments(
    coords: np.ndarray,
    mask: np.ndarray,
    scale_frac: float = 0.35,
    seed_region_frac: float = 0.25,
    rng_seed: int = 0,
) -> SuperQuadric:
    """Initialise a SQ inside a SUB-REGION of the occupied set.

    Rationale: seeding with a full-extent PCA gives the fit procedure
    a SQ whose bounding box already covers the whole model. One pass
    of optimisation easily collapses that to a "rounded cuboid of the
    whole thing" and the carving step removes nearly all voxels, so
    the greedy loop terminates after 1 primitive.

    What we do instead:
      1. Pick a random occupied voxel as a seed centre.
      2. Restrict to the occupied voxels within ``seed_region_frac *
         mesh_radius`` of that seed.
      3. Compute PCA on that restricted cluster.
      4. Scale the SQ to ``scale_frac`` times the PCA standard
         deviation (so it covers only the local feature, not the full
         bounding box).

    This produces a sequence of locally-fit SQs that each cover a
    different part of the object, which is what the paper's
    block-regrow-fill strategy achieves more formally.
    """
    occ = coords[mask]
    if len(occ) == 0:
        return SuperQuadric()

    rng = np.random.default_rng(rng_seed)
    # Seed voxel: weighted random pick (biases toward interior by
    # sampling uniformly over the occupied set)
    seed = occ[rng.integers(0, len(occ))]

    # Restrict to local region around seed
    mesh_radius = max(
        np.linalg.norm(occ.max(axis=0) - occ.min(axis=0)) / 2.0,
        1e-3,
    )
    r = seed_region_frac * mesh_radius
    dist = np.linalg.norm(occ - seed, axis=1)
    local = occ[dist < r]
    if len(local) < 50:
        # Too small a region; fall back to the full set but scaled small
        local = occ

    c = local.mean(axis=0)
    rel = local - c
    cov = (rel.T @ rel) / max(len(local), 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    axes = eigvecs[:, order]
    # sqrt(eig) is a standard-deviation along that axis. Multiply by
    # scale_frac to get a SUB-extent SQ.
    scales = np.sqrt(np.maximum(eigvals[order], 1e-6)) * 2.0 * scale_frac
    scales = np.clip(scales, 0.04, 0.6)

    rx = math.atan2(axes[2, 1], axes[2, 2])
    ry = math.atan2(-axes[2, 0], math.hypot(axes[2, 1], axes[2, 2]))
    rz = math.atan2(axes[1, 0], axes[0, 0])
    return SuperQuadric(
        e1=1.0, e2=1.0,
        ax=scales[0], ay=scales[1], az=scales[2],
        rx=rx, ry=ry, rz=rz,
        tx=c[0], ty=c[1], tz=c[2],
    )


def _fit_one(
    phi_target: np.ndarray,
    coords: np.ndarray,
    tau: float,
    n_iters: int = 200,
    lr: float = 0.01,
    n_samples: int = 20000,
    init: Optional[SuperQuadric] = None,
    init_seed: int = 0,
    verbose: bool = False,
) -> SuperQuadric:
    """Fit a single superquadric to the target TSDF by minimising L2 on
    the truncated SDF over a sample of points.

    This is a numpy-based finite-difference Adam — simpler than pytorch
    autograd and adequate since we only have 11 params.
    """
    R = phi_target.shape[0]
    # Focus sampling on voxels that are inside the target surface
    # (phi_target < 0) with some outside neighbourhood for gradient.
    occ_mask = phi_target < 0
    if occ_mask.sum() == 0:
        return SuperQuadric()

    flat_coords = coords.reshape(-1, 3)
    flat_phi = phi_target.reshape(-1)

    # Sample n_samples points, biased ~70% inside / 30% outside
    n_in = int(n_samples * 0.7)
    n_out = n_samples - n_in
    inside_idx = np.where(occ_mask.ravel())[0]
    outside_idx = np.where(~occ_mask.ravel())[0]
    rng = np.random.default_rng(0)
    pick_in = rng.choice(inside_idx, size=min(n_in, len(inside_idx)), replace=False)
    pick_out = rng.choice(outside_idx, size=min(n_out, len(outside_idx)), replace=False)
    pick = np.concatenate([pick_in, pick_out])
    sample_coords = flat_coords[pick]
    sample_phi = flat_phi[pick]

    if init is None:
        init = _init_from_moments(coords, occ_mask, rng_seed=init_seed)
    params = init.to_vector()

    def loss_at(p: np.ndarray) -> float:
        sq = SuperQuadric.from_vector(p)
        # Clip shape exponents and scales to valid ranges
        if min(sq.e1, sq.e2) < 0.1 or max(sq.e1, sq.e2) > 2.0:
            return 1e9
        if min(sq.ax, sq.ay, sq.az) < 1e-3:
            return 1e9
        phi_pred = _sq_phi(sample_coords, sq)
        phi_pred_t = np.clip(phi_pred, -tau, +tau)
        return float(np.mean((phi_pred_t - sample_phi) ** 2))

    # Simple Adam on finite-differenced gradient
    step = np.array(
        [0.02, 0.02, 0.005, 0.005, 0.005, 0.01, 0.01, 0.01, 0.005, 0.005, 0.005]
    )  # finite-diff step per param (matched to param scale)
    m = np.zeros_like(params)
    v = np.zeros_like(params)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    best_params = params.copy()
    best_loss = loss_at(params)

    for it in range(n_iters):
        grad = np.zeros_like(params)
        base = loss_at(params)
        for i in range(len(params)):
            p_plus = params.copy()
            p_plus[i] += step[i]
            l_plus = loss_at(p_plus)
            grad[i] = (l_plus - base) / step[i]

        # Clip gradient
        gnorm = np.linalg.norm(grad)
        if gnorm > 50.0:
            grad = grad * (50.0 / gnorm)

        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad * grad
        mh = m / (1 - beta1 ** (it + 1))
        vh = v / (1 - beta2 ** (it + 1))
        params = params - lr * mh / (np.sqrt(vh) + eps)

        l = loss_at(params)
        if l < best_loss:
            best_loss = l
            best_params = params.copy()

        if verbose and (it % 20 == 0):
            print(f"  [fit] it={it:3d} loss={l:.5f} best={best_loss:.5f}")

    return SuperQuadric.from_vector(best_params)


# =====================================================================
# SDF carving
# =====================================================================

def _carve(
    phi_target: np.ndarray,
    coords: np.ndarray,
    sq: SuperQuadric,
    tau: float,
) -> np.ndarray:
    """Paper Eq. 7. Update the target TSDF to remove the volume covered
    by the fitted primitive.

    Cases per voxel x:
      Both target and primitive INSIDE   -> phi(x) = -phi_theta(x)
      Target inside, primitive outside   -> phi(x) = max(-phi_theta, phi)
      Target outside                     -> phi(x) unchanged

    Then re-clamp to [-tau, +tau].
    """
    flat = coords.reshape(-1, 3)
    phi_theta = _sq_phi(flat, sq).reshape(phi_target.shape)

    target_inside = phi_target < 0
    prim_inside = phi_theta < 0

    new_phi = phi_target.copy()
    # Case 1: both inside
    mask_both = target_inside & prim_inside
    new_phi[mask_both] = -phi_theta[mask_both]  # now effectively "positive" = just-outside
    # Case 2: target inside, primitive outside
    mask_2 = target_inside & ~prim_inside
    new_phi[mask_2] = np.maximum(-phi_theta[mask_2], phi_target[mask_2])
    # Case 3 unchanged

    new_phi = np.clip(new_phi, -tau, +tau)
    return new_phi


# =====================================================================
# Result object + refiner
# =====================================================================

@dataclass
class LightSQResult:
    primitives: List[SuperQuadric]
    final_residual_voxels: int
    n_iterations: int
    timings: dict = field(default_factory=dict)


class LightSQRefiner:
    """Minimal Light-SQ: greedy fit + carve loop.

    See module docstring for which paper features are skipped.
    """

    def __init__(
        self,
        grid_res: int = 100,
        max_primitives: int = 60,
        n_iters: int = 200,
        lr: float = 0.01,
        min_residual_voxels: int = 500,
        prune_size_threshold: float = 0.02,
    ):
        self.grid_res = grid_res
        self.max_primitives = max_primitives
        self.n_iters = n_iters
        self.lr = lr
        self.min_residual_voxels = min_residual_voxels
        self.prune_size_threshold = prune_size_threshold

    def _normalise(
        self, mesh: trimesh.Trimesh,
    ) -> tuple[trimesh.Trimesh, np.ndarray, float]:
        """Fit mesh into [-1, 1]^3 cube."""
        c = mesh.bounding_box.centroid.copy()
        m = mesh.copy()
        m.apply_translation(-c)
        ext = max(m.extents)
        scale = ext if ext > 0 else 1.0
        m.apply_scale(2.0 / scale * 0.98)  # 2% margin to avoid TSDF edge clipping
        return m, np.asarray(c), float(scale / 2.0 / 0.98)

    def fit(self, mesh: trimesh.Trimesh, verbose: bool = True) -> LightSQResult:
        """Greedy superquadric decomposition of mesh."""
        timings = {}
        t0 = time.time()
        norm_mesh, centroid, inv_scale = self._normalise(mesh)

        t1 = time.time()
        phi, coords = build_tsdf(norm_mesh, self.grid_res)
        tau = 2.0 / self.grid_res
        timings["tsdf"] = time.time() - t1
        if verbose:
            occ_init = int((phi < 0).sum())
            print(f"[light_sq] built {self.grid_res}^3 TSDF in "
                  f"{timings['tsdf']:.1f}s ({occ_init:,} occupied voxels)")

        prims: List[SuperQuadric] = []
        for i in range(self.max_primitives):
            n_occ = int((phi < 0).sum())
            if verbose:
                print(f"[light_sq] prim {i+1}/{self.max_primitives} "
                      f"residual_occ={n_occ:,}")
            if n_occ < self.min_residual_voxels:
                if verbose:
                    print(f"[light_sq] residual below threshold — done")
                break

            t_fit = time.time()
            sq = _fit_one(
                phi, coords, tau,
                n_iters=self.n_iters, lr=self.lr,
                init_seed=i * 7919 + 1,   # different seed per iteration
                verbose=False,
            )
            fit_dt = time.time() - t_fit

            # Prune tiny primitives (Sec §3.4 simplified)
            min_axis = min(sq.ax, sq.ay, sq.az)
            if min_axis < self.prune_size_threshold:
                if verbose:
                    print(f"[light_sq]   pruned (min_axis={min_axis:.3f} < "
                          f"{self.prune_size_threshold})")
                break

            prims.append(sq)
            phi = _carve(phi, coords, sq, tau)
            if verbose:
                print(f"[light_sq]   fit in {fit_dt:.1f}s, "
                      f"sq=(a={sq.ax:.3f},{sq.ay:.3f},{sq.az:.3f} "
                      f"t={sq.tx:+.2f},{sq.ty:+.2f},{sq.tz:+.2f} "
                      f"e1={sq.e1:.2f} e2={sq.e2:.2f})")

        timings["total"] = time.time() - t0
        return LightSQResult(
            primitives=prims,
            final_residual_voxels=int((phi < 0).sum()),
            n_iterations=len(prims),
            timings=timings,
        )

    # ---------------- compile to mesh ----------------------------------

    def compile(
        self,
        primitives: List[SuperQuadric],
        resolution: int = 128,
    ) -> trimesh.Trimesh:
        """Build a single trimesh by unioning per-primitive meshes.

        Each superquadric is tessellated via marching cubes on its
        own implicit grid, then concatenated. True boolean-union is
        more expensive — concatenate is fine as a first visualisation
        since the overlapping volumes still render as a single blob.
        """
        if not primitives:
            return trimesh.Trimesh()
        meshes = []
        for sq in primitives:
            try:
                meshes.append(self._marching_cubes_sq(sq, resolution))
            except Exception as e:
                import warnings
                warnings.warn(f"[light_sq] marching cubes failed on a primitive: {e}")
        if not meshes:
            return trimesh.Trimesh()
        return trimesh.util.concatenate(meshes)

    @staticmethod
    def _marching_cubes_sq(sq: SuperQuadric, resolution: int) -> trimesh.Trimesh:
        """Grid-sample the SQ implicit and run marching cubes."""
        from skimage.measure import marching_cubes

        # Bounding cube around the SQ
        c = np.array([sq.tx, sq.ty, sq.tz])
        half = 1.5 * max(sq.ax, sq.ay, sq.az)
        lin = np.linspace(-half, +half, resolution)
        gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
        pts = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3) + c
        phi = _sq_phi(pts, sq).reshape(resolution, resolution, resolution)
        if not (phi.min() < 0 < phi.max()):
            return trimesh.Trimesh()
        verts, faces, _, _ = marching_cubes(phi, level=0.0)
        # Map grid indices back to world
        verts = verts / (resolution - 1) * (2 * half) - half + c
        return trimesh.Trimesh(vertices=verts, faces=faces, process=False)
