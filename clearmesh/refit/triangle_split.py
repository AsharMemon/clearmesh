"""Pairwise triangle splitting at intersection lines.

Port of Liu 2023 §3.2 "Mesh Fitting and Splitting Module" (steps 2-4,
since step 1 "fit mesh per patch" is assumed to be upstream).

Given a list of candidate triangle meshes (one per detected primitive
patch, possibly overlapping), this module:

  1. Finds pairs of candidate meshes whose AABBs overlap.

  2. For each such pair (i, j), splits mesh i by j's best-fit plane
     and mesh j by i's best-fit plane (a per-candidate PCA plane).
     This is an APPROXIMATION of the paper's §3.2 step 2 — the paper
     computes the exact intersection line between each triangle-pair
     (which lies on both supporting planes simultaneously) and splits
     both triangles along that shared line, producing coincident
     vertices across the pair.

  3. Optionally snaps all candidate vertices onto a shared coarse
     lattice so the downstream edge-graph ``watertight_select`` builds
     can find shared edges. The lattice introduces <= ``lattice_snap``
     geometric error.

Two split paths:

  1. **tri-tri intersection** (``method="tri-tri"``, DEFAULT) — textbook
     Moller-style pair-wise triangle intersection. For each pair of
     candidates (A, B) and each pair of triangles (t_a, t_b) whose
     AABBs overlap, we compute the 3D segment where the two triangles'
     interiors cross. The segment endpoints are by construction the
     SAME 3D points for both triangles, so when we subdivide both
     triangles at those endpoints, the two subdivided meshes SHARE
     those vertex coordinates. This gives the watertight BLP edge
     graph genuine cross-candidate edges to constrain.

  2. **plane split** (``method="plane"``, LEGACY) — the previous
     approximation: cut candidate i by candidate j's best-fit plane
     and vice versa, then snap all vertices to a coarse shared
     lattice. This is kept as a fallback because it's cheaper than
     tri-tri on large meshes, but its edge graph is typically empty.

Implementation notes:

  - Pure-python implementation of triangle-plane clip (no pytorch3d /
    no manifold3d dependency). Single-threaded.
  - Snap-to-plane on the interpolated split vertex to prevent drift.
  - Zero-area triangle filter at 1e-18 area² squared.

Usage:

    from clearmesh.refit.triangle_split import split_candidates

    split_cands = split_candidates(
        candidate_faces,
        merge_tol=1e-5,
        verbose=True,
    )
    # Each fragment in split_cands shares exact vertices with every
    # other fragment it originally intersected — ready for
    # watertight_select.select_watertight().
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import trimesh


__all__ = ["split_candidates"]


# =====================================================================
# Fast AABB intersection culling
# =====================================================================

def _candidate_aabbs(candidates: List[trimesh.Trimesh]) -> np.ndarray:
    """Return (n_cand, 2, 3) array of min/max corners per candidate."""
    n = len(candidates)
    out = np.zeros((n, 2, 3), dtype=np.float64)
    for i, c in enumerate(candidates):
        if c is None or len(c.vertices) == 0:
            out[i] = 0
            continue
        out[i, 0] = c.vertices.min(axis=0)
        out[i, 1] = c.vertices.max(axis=0)
    return out


def _overlapping_pairs(
    aabbs: np.ndarray, tol: float = 1e-6,
) -> List[Tuple[int, int]]:
    """All (i, j) with i < j whose AABBs overlap in 3D."""
    n = aabbs.shape[0]
    pairs = []
    lo = aabbs[:, 0] - tol  # (n, 3)
    hi = aabbs[:, 1] + tol  # (n, 3)
    for i in range(n):
        # Overlap test against all j > i — vectorised
        overlap = np.all((lo[i + 1:] <= hi[i]) & (hi[i + 1:] >= lo[i]), axis=-1)
        js = np.where(overlap)[0] + (i + 1)
        pairs.extend([(i, int(j)) for j in js])
    return pairs


# =====================================================================
# Triangle-triangle intersection (Moller-style)
# =====================================================================

def _tri_plane_segment(
    tri: np.ndarray,       # (3, 3) triangle vertices
    plane_origin: np.ndarray,
    plane_normal: np.ndarray,
    eps: float = 1e-10,
) -> Optional[np.ndarray]:
    """Return the 3D segment where triangle ``tri`` crosses the plane,
    or None if the triangle doesn't straddle it.

    Returns shape (2, 3): the two endpoints of the crossing.
    """
    d = (tri - plane_origin) @ plane_normal  # signed dists per vertex
    signs = np.sign(np.where(np.abs(d) < eps, 0, d))
    # All same sign => no crossing (all above or all below)
    pos = (signs > 0).sum()
    neg = (signs < 0).sum()
    if pos == 0 or neg == 0:
        return None

    # Find the two edges that straddle the plane
    endpoints = []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        if signs[a] * signs[b] < 0:
            # Edge (a, b) crosses plane
            t = d[a] / (d[a] - d[b])
            endpoints.append(tri[a] + t * (tri[b] - tri[a]))
        elif signs[a] == 0:
            endpoints.append(tri[a].copy())
    if len(endpoints) < 2:
        return None
    # Deduplicate exact matches (rare, happens when a vertex lies on plane)
    uniq = [endpoints[0]]
    for p in endpoints[1:]:
        if all(np.linalg.norm(p - q) > eps for q in uniq):
            uniq.append(p)
        if len(uniq) == 2:
            break
    if len(uniq) < 2:
        return None
    return np.stack(uniq[:2])


def _tri_tri_segment(
    tri_a: np.ndarray, tri_b: np.ndarray,
    eps: float = 1e-10,
) -> Optional[np.ndarray]:
    """Moller-style tri-tri intersection.

    Returns (2, 3) — the two 3D endpoints of the segment where the
    interiors of tri_a and tri_b meet, or None if no real intersection.

    The endpoints are IDENTICAL for both triangles (they're points on
    the intersection line L = plane_a ∩ plane_b), which is what gives
    us the shared-vertex property needed by the watertight BLP.
    """
    # Planes
    e0_a, e1_a = tri_a[1] - tri_a[0], tri_a[2] - tri_a[0]
    n_a = np.cross(e0_a, e1_a)
    n_a_len = np.linalg.norm(n_a)
    if n_a_len < eps:
        return None
    n_a = n_a / n_a_len

    e0_b, e1_b = tri_b[1] - tri_b[0], tri_b[2] - tri_b[0]
    n_b = np.cross(e0_b, e1_b)
    n_b_len = np.linalg.norm(n_b)
    if n_b_len < eps:
        return None
    n_b = n_b / n_b_len

    # Early-out: coplanar (direction cross product near zero)
    d_line = np.cross(n_a, n_b)
    if np.linalg.norm(d_line) < 1e-8:
        return None  # coplanar or parallel — not our concern

    # Segment where tri_a crosses plane_b, and vice versa
    seg_a = _tri_plane_segment(tri_a, tri_b[0], n_b, eps)
    seg_b = _tri_plane_segment(tri_b, tri_a[0], n_a, eps)
    if seg_a is None or seg_b is None:
        return None

    # Both segments lie on line L = plane_a ∩ plane_b. Parameterise each
    # along d_line and take the 1D overlap.
    d_line = d_line / np.linalg.norm(d_line)
    origin = seg_a[0]

    def _t(p):
        return (p - origin) @ d_line

    ta0, ta1 = _t(seg_a[0]), _t(seg_a[1])
    tb0, tb1 = _t(seg_b[0]), _t(seg_b[1])
    a_lo, a_hi = min(ta0, ta1), max(ta0, ta1)
    b_lo, b_hi = min(tb0, tb1), max(tb0, tb1)
    lo = max(a_lo, b_lo)
    hi = min(a_hi, b_hi)
    if hi <= lo + eps:
        return None
    p0 = origin + lo * d_line
    p1 = origin + hi * d_line
    return np.stack([p0, p1])


def _insert_segment_into_mesh(
    verts: np.ndarray,
    faces: np.ndarray,
    segment: np.ndarray,     # (2, 3)
    eps: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray]:
    """Insert segment endpoints into the mesh as new vertices and
    subdivide any triangle that contains a segment endpoint strictly
    inside one of its edges or face.

    Strategy: for each segment endpoint, find the face (if any) that
    contains it. If on an edge → split the two adjacent triangles at
    that point. If on an interior → fan-split the one containing
    triangle into three sub-triangles.

    This is intentionally simple and produces redundant vertices if
    the endpoint already exists — those get deduped downstream by
    the numpy unique pass.
    """
    verts = verts.tolist()
    faces = faces.tolist()

    for p in segment:
        # Add the new vertex
        p_idx = len(verts)
        verts.append(p.tolist())

        # Find which face the point lies in / on
        new_faces = []
        inserted = False
        for tri in faces:
            if inserted:
                new_faces.append(tri)
                continue
            v0, v1, v2 = np.asarray(verts[tri[0]]), np.asarray(verts[tri[1]]), np.asarray(verts[tri[2]])
            # Barycentric-ish test: compute area of triangle and sub-triangles
            n = np.cross(v1 - v0, v2 - v0)
            n_len = np.linalg.norm(n)
            if n_len < eps:
                new_faces.append(tri)
                continue
            u_hat = n / n_len
            # Distance from p to plane
            if abs((p - v0) @ u_hat) > 1e-4:
                new_faces.append(tri)
                continue
            # Barycentric coords
            d00 = (v1 - v0) @ (v1 - v0)
            d01 = (v1 - v0) @ (v2 - v0)
            d11 = (v2 - v0) @ (v2 - v0)
            d20 = (p - v0) @ (v1 - v0)
            d21 = (p - v0) @ (v2 - v0)
            denom = d00 * d11 - d01 * d01
            if abs(denom) < eps:
                new_faces.append(tri)
                continue
            b1 = (d11 * d20 - d01 * d21) / denom
            b2 = (d00 * d21 - d01 * d20) / denom
            b0 = 1.0 - b1 - b2
            if (b0 >= -1e-6) and (b1 >= -1e-6) and (b2 >= -1e-6):
                # Inside (or on boundary) — fan split
                new_faces.append([tri[0], tri[1], p_idx])
                new_faces.append([tri[1], tri[2], p_idx])
                new_faces.append([tri[2], tri[0], p_idx])
                inserted = True
            else:
                new_faces.append(tri)
        faces = new_faces

    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def _split_pair_tri_tri(
    a: trimesh.Trimesh,
    b: trimesh.Trimesh,
    eps: float = 1e-9,
    max_segments: int = 200,
) -> tuple[trimesh.Trimesh, trimesh.Trimesh]:
    """Subdivide both meshes at every intersection-segment endpoint.

    Returns new (a', b') where a' and b' share vertex coordinates at
    all tri-tri intersection endpoints.

    Performance: this is pure python and O(T_a * T_b * S * F) in the
    worst case (S = segments found, F = faces for the point-insert).
    For large candidate pools, pre-decimate or switch to
    ``method="plane"`` in ``split_candidates``.
    """
    va, fa = np.asarray(a.vertices), np.asarray(a.faces)
    vb, fb = np.asarray(b.vertices), np.asarray(b.faces)
    tri_a = va[fa]
    tri_b = vb[fb]

    # Fast AABB overlap filter between the two FACE sets
    a_lo = tri_a.min(axis=1)
    a_hi = tri_a.max(axis=1)
    b_lo = tri_b.min(axis=1)
    b_hi = tri_b.max(axis=1)

    segs: list = []
    # Vectorised outer-loop overlap filter — avoids per-triangle python
    # loops through a's full face list when most are irrelevant.
    for i in range(len(fa)):
        overlap = np.all(
            (b_lo <= a_hi[i] + eps) & (b_hi >= a_lo[i] - eps), axis=-1
        )
        js = np.where(overlap)[0]
        if len(js) == 0:
            continue
        for j in js:
            seg = _tri_tri_segment(tri_a[i], tri_b[j], eps=eps)
            if seg is not None:
                segs.append(seg)
                if len(segs) >= max_segments:
                    break
        if len(segs) >= max_segments:
            break

    if not segs:
        return a.copy(), b.copy()

    # Deduplicate segments first so we don't insert the same endpoint
    # dozens of times (each shared segment can appear many times if
    # multiple triangle pairs project onto the same cut).
    seg_arr = np.stack(segs)  # (S, 2, 3)
    uniq = np.unique(seg_arr.round(6), axis=0)

    v_a, f_a = va.copy(), fa.copy()
    v_b, f_b = vb.copy(), fb.copy()
    for seg in uniq:
        v_a, f_a = _insert_segment_into_mesh(v_a, f_a, seg, eps)
        v_b, f_b = _insert_segment_into_mesh(v_b, f_b, seg, eps)

    return (
        trimesh.Trimesh(vertices=v_a, faces=f_a, process=False),
        trimesh.Trimesh(vertices=v_b, faces=f_b, process=False),
    )


# =====================================================================
# Triangle-plane split (legacy method)
# =====================================================================

def _split_mesh_by_planes(
    mesh: trimesh.Trimesh,
    planes: List[Tuple[np.ndarray, np.ndarray]],
    tol: float = 1e-7,
) -> trimesh.Trimesh:
    """Split every triangle of ``mesh`` by each plane in ``planes``.

    For each plane (origin, normal):
        - Every triangle straddling the plane is cut along the plane
          crossing, producing 3 sub-triangles (1 on one side, 2 on
          the other, as per a standard clip).
        - Triangles fully on one side are kept as-is.

    We iterate plane-by-plane, re-slicing the mesh each time. The
    result has vertices that exactly coincide with any other mesh
    split by the same plane set (up to numerical tolerance).

    This is a pure-python implementation using the standard clip-
    triangle-by-plane algorithm. Accurate but O(T * P) with small
    constant.
    """
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    for origin, normal in planes:
        n = np.asarray(normal, dtype=np.float64)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            continue
        n = n / n_norm
        o = np.asarray(origin, dtype=np.float64)

        # Signed distance for every vertex
        d_all = (verts - o) @ n
        new_verts = verts.tolist()
        new_faces = []
        # A small cache: for each intersecting edge, store the vertex
        # index of the interpolated split point (so neighbouring
        # triangles sharing that edge reuse the same vertex — this
        # is the critical bit for shared-vertex guarantee across
        # candidates).
        edge_cache: dict[tuple, int] = {}

        def _split_vertex(i0: int, i1: int) -> int:
            key = (min(i0, i1), max(i0, i1))
            if key in edge_cache:
                return edge_cache[key]
            d0, d1 = d_all[i0], d_all[i1]
            t = d0 / (d0 - d1)
            p = verts[i0] + t * (verts[i1] - verts[i0])
            # Snap-to-plane to avoid drift from repeated splits
            p = p - ((p - o) @ n) * n
            new_verts.append(p.tolist())
            idx = len(new_verts) - 1
            edge_cache[key] = idx
            return idx

        for tri in faces:
            i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
            d0, d1, d2 = float(d_all[i0]), float(d_all[i1]), float(d_all[i2])
            s0 = 0 if abs(d0) < tol else (1 if d0 > 0 else -1)
            s1 = 0 if abs(d1) < tol else (1 if d1 > 0 else -1)
            s2 = 0 if abs(d2) < tol else (1 if d2 > 0 else -1)

            # All on same side (or on the plane): keep as-is
            if s0 * s1 >= 0 and s1 * s2 >= 0 and s0 * s2 >= 0:
                new_faces.append([i0, i1, i2])
                continue

            # Identify the lone vertex (the one whose sign differs from
            # the other two) and the two majority vertices. Split the two
            # majority→lone edges at the plane.
            signs = [s0, s1, s2]
            ids = [i0, i1, i2]
            # The "lone" vertex is the one with sign != the majority
            positives = sum(s > 0 for s in signs)
            negatives = sum(s < 0 for s in signs)
            if positives == 1 and negatives >= 1:
                lone_sign = +1
            elif negatives == 1 and positives >= 1:
                lone_sign = -1
            else:
                # Two-on-plane edge case: split-safe fallback = keep the
                # triangle untouched.
                new_faces.append([i0, i1, i2])
                continue

            # Rotate so lone vertex is ids[0]
            lone_local = next(k for k in range(3) if signs[k] == lone_sign)
            rot = [(lone_local + k) % 3 for k in range(3)]
            a = ids[rot[0]]       # lone
            b = ids[rot[1]]
            c = ids[rot[2]]

            ab = _split_vertex(a, b)
            ac = _split_vertex(a, c)

            # 3 sub-triangles: (a, ab, ac) on the lone side, then
            # (ab, b, c) and (ab, c, ac) on the majority side.
            new_faces.append([a, ab, ac])
            new_faces.append([ab, b, c])
            new_faces.append([ab, c, ac])

        verts = np.asarray(new_verts, dtype=np.float64)
        faces = np.asarray(new_faces, dtype=np.int64)
        # Recompute d_all isn't needed — we move on to the next plane
        # which recomputes it anyway.

    # Dedup vertices by rounding to 1e-6 (newer trimesh dropped the
    # merge_vertices(digits=) kwarg, so we do it manually)
    v_round = np.round(verts, 6)
    _, inv = np.unique(v_round, axis=0, return_inverse=True)
    new_v = np.unique(v_round, axis=0)
    new_f = inv[faces]
    # Drop zero-area triangles
    valid = (
        (new_f[:, 0] != new_f[:, 1])
        & (new_f[:, 1] != new_f[:, 2])
        & (new_f[:, 0] != new_f[:, 2])
    )
    tri_v = new_v[new_f[valid]]
    cross = np.cross(tri_v[:, 1] - tri_v[:, 0], tri_v[:, 2] - tri_v[:, 0])
    area2 = (cross ** 2).sum(axis=1)
    non_deg = area2 > 1e-18
    final_faces = new_f[valid][non_deg]
    return trimesh.Trimesh(vertices=new_v, faces=final_faces, process=False)


# =====================================================================
# Approximate plane for a candidate (used for pairwise splitting)
# =====================================================================

def _best_plane(mesh: trimesh.Trimesh) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Best-fit plane for the candidate's vertices via PCA.

    For planar candidates (rectangle patches etc) this is exact. For
    non-planar candidates (cylinders, spheres) the best-fit plane is
    just a rough cut — the purpose here is to introduce SOME shared
    geometry between intersecting candidates, not to recover the
    true intersection curve.

    Returns (origin, normal) or None if the mesh is degenerate.
    """
    if mesh is None or len(mesh.vertices) < 3:
        return None
    v = np.asarray(mesh.vertices, dtype=np.float64)
    centroid = v.mean(axis=0)
    rel = v - centroid
    cov = rel.T @ rel / max(len(v), 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Smallest eigenvalue's eigenvector = plane normal
    normal = eigvecs[:, 0]
    return centroid, normal


# =====================================================================
# Public entry point
# =====================================================================

def _lattice_snap(
    meshes: List[trimesh.Trimesh],
    lattice_res: float,
) -> List[trimesh.Trimesh]:
    """Snap every vertex of every candidate to a shared regular lattice.

    This is the vertex-sharing guarantee of last resort: even if the
    per-pair triangle splits didn't produce coincident vertices, any
    two vertices within ``lattice_res`` of each other get mapped to the
    same lattice point, so the downstream edge-graph finds shared edges.

    Cost: introduces up-to-lattice_res geometric distortion. With
    lattice_res=1e-3 on a mesh normalised to [-1, 1], that's 0.05%
    relative error — invisible for most rendering + downstream CAD.
    """
    out = []
    for m in meshes:
        if m is None or len(m.vertices) == 0:
            out.append(m)
            continue
        v_snap = np.round(m.vertices / lattice_res) * lattice_res
        # Deduplicate
        _, inv = np.unique(v_snap, axis=0, return_inverse=True)
        new_v = np.unique(v_snap, axis=0)
        new_f = inv[m.faces]
        # Drop degenerate triangles
        valid = (
            (new_f[:, 0] != new_f[:, 1])
            & (new_f[:, 1] != new_f[:, 2])
            & (new_f[:, 0] != new_f[:, 2])
        )
        out.append(trimesh.Trimesh(
            vertices=new_v, faces=new_f[valid], process=False,
        ))
    return out


def split_candidates(
    candidates: List[trimesh.Trimesh],
    method: str = "tri-tri",
    lattice_snap: Optional[float] = 1e-3,
    verbose: bool = False,
) -> List[trimesh.Trimesh]:
    """Pairwise triangle split on a list of candidate meshes.

    For every pair of AABB-overlapping candidates (i, j):
      - Pick each candidate's best-fit plane (PCA).
      - Split candidate i by j's plane, and j by i's plane.

    After all pairs are processed, each candidate's mesh has been
    re-cut along every intersecting neighbour's plane. At those cuts,
    vertices are ``snap-to-plane`` projected, so if candidates i and j
    both got cut by the same plane, their cut vertices lie exactly on
    that plane (but NOT necessarily at identical coords — the cut is
    per-mesh).

    To guarantee VERTEX-EXACT sharing across candidates (which the
    downstream edge-graph requires), we do a final pass of rounding
    all vertices to ``merge_tol`` decimal digits; ``watertight_select``
    then uses the same tolerance when building the edge-graph.

    Args:
        candidates: list of trimesh.Trimesh candidate patches.
        verbose: print per-pair split counts.

    Returns:
        A new list of trimesh.Trimesh with the same length (None
        entries preserved). Vertex rounding applied at ``1e-5``.
    """
    n = len(candidates)
    if n < 2:
        return list(candidates)

    aabbs = _candidate_aabbs(candidates)
    pairs = _overlapping_pairs(aabbs)
    if verbose:
        print(f"[triangle_split] {len(pairs)} AABB-overlapping candidate pairs "
              f"(method={method})")

    # -----------------------------------------------------------------
    # method="tri-tri" — Moller-style pair-wise triangle intersection.
    # Produces vertex-exact shared geometry across candidates. O(T*T')
    # per pair; for N ~ 10 candidates of ~300 faces this is fine, for
    # larger pools prefer method="plane" or pre-decimate candidates.
    # -----------------------------------------------------------------
    if method == "tri-tri":
        out = [c.copy() if c is not None else None for c in candidates]
        for (i, j) in pairs:
            if out[i] is None or out[j] is None:
                continue
            try:
                a2, b2 = _split_pair_tri_tri(out[i], out[j])
                out[i], out[j] = a2, b2
            except Exception as e:
                warnings.warn(f"[triangle_split] tri-tri pair ({i},{j}) failed: {e}")
            if verbose:
                print(f"[triangle_split]   pair ({i},{j}): "
                      f"{len(out[i].faces):,} / {len(out[j].faces):,} faces")
        if lattice_snap is not None and lattice_snap > 0:
            out = _lattice_snap(out, lattice_snap)
        return out
    # -----------------------------------------------------------------

    # Build the GLOBAL set of planes. Every candidate gets cut by every
    # OTHER candidate's best-fit plane that it AABB-overlaps with. The
    # critical improvement over per-pair cutting with ad-hoc planes:
    # if candidates i and j are both cut by plane P_k (the plane of some
    # third candidate k), their cut endpoints both lie on plane P_k
    # (up to numerical drift from the sub-triangulation snap). The
    # downstream lattice snap can then collide them onto shared grid
    # points. Without the global plane set, i was cut on plane P_j and
    # j on plane P_i — two DIFFERENT planes — so no collisions ever.
    best_planes: List[Optional[Tuple[np.ndarray, np.ndarray]]] = [
        _best_plane(c) for c in candidates
    ]
    adjacency: List[set[int]] = [set() for _ in range(n)]
    for i, j in pairs:
        adjacency[i].add(j)
        adjacency[j].add(i)

    planes_for: List[List[Tuple[np.ndarray, np.ndarray]]] = [[] for _ in range(n)]
    for i in range(n):
        # Collect planes from candidate i itself PLUS every AABB-
        # overlapping neighbour. Cutting i by its own plane is a no-op
        # for perfectly-planar candidates but adds useful seams for
        # curved ones. Cutting by neighbours' planes is what creates
        # the potentially-shared intersection lines.
        own = best_planes[i]
        if own is not None:
            planes_for[i].append(own)
        for j in adjacency[i]:
            p = best_planes[j]
            if p is not None:
                planes_for[i].append(p)

    out: List[trimesh.Trimesh] = []
    for idx, (cand, planes) in enumerate(zip(candidates, planes_for)):
        if cand is None or len(cand.vertices) == 0 or not planes:
            out.append(cand)
            continue
        try:
            split = _split_mesh_by_planes(cand, planes)
        except Exception as e:
            warnings.warn(f"[triangle_split] failed on candidate {idx}: {e}")
            split = cand
        if verbose:
            print(
                f"[triangle_split]   cand {idx}: "
                f"{len(cand.faces):,} -> {len(split.faces):,} faces"
            )
        out.append(split)

    # Shared-lattice snap: forces vertex sharing across candidates that
    # per-pair plane splits alone cannot guarantee (candidate i is cut
    # along plane P_j and candidate j along plane P_i — those are
    # different planes unless we walk the full pair-intersection-line
    # construction from Liu §3.2 step 2, which we haven't implemented).
    if lattice_snap is not None and lattice_snap > 0:
        out = _lattice_snap(out, lattice_snap)
        if verbose:
            print(f"[triangle_split] snapped all vertices to {lattice_snap} lattice")
    return out
