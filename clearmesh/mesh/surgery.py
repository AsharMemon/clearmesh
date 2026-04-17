"""Mesh surgery: surgical artifact removal via raycasting + hole filling.

Philosophy
----------
The diffusion-based edit pipeline (Easy3E + UltraShape + polish) is the
right tool for semantic changes — adding wings, changing material,
re-interpreting shape. It is the WRONG tool for removing small
artifacts, because the image-edit model (InstructPix2Pix) has no
concept of "artifact" — it just reinterprets the whole image, which
typically blurs out fine detail along the way.

For surgical corrections (remove this bump, delete this spur, clean
this fused region), the right primitive is pure geometry:

  1. Identify faces in the mesh that comprise the artifact
  2. Delete those faces
  3. Fill the resulting hole smoothly

No diffusion. No blurring. Preserves all surrounding detail bit-exact.

Typical runtime: 3-10 seconds (dominated by ray/BVH construction).

API
---
  remove_by_mask(mesh, mask_img, camera=...) : 2D-mask-based removal
  remove_by_spike_detection(mesh, ...)       : statistical outlier removal
  fill_hole_smooth(mesh)                     : smooth hole fill utility
"""

from __future__ import annotations

import numpy as np
import trimesh


# ---------------------------------------------------------------------------
# 2D-mask-based removal (raycasting)
# ---------------------------------------------------------------------------

def remove_by_mask(
    mesh: trimesh.Trimesh,
    mask_image,  # PIL.Image.Image | np.ndarray
    camera=None,
    mask_threshold: float = 0.5,
    ray_samples: int = 512,
    max_depth_faces: int = 4,
    fill_holes: bool = True,
    verbose: bool = False,
) -> trimesh.Trimesh:
    """Delete faces whose front-most hits lie inside a 2D mask, then fill
    the hole.

    This is the equivalent of a "2D click-to-heal" tool: the user paints
    a region in image space (via the canonical camera) and we remove the
    FIRST LAYER of mesh faces visible through each pixel of that region.
    Deeper faces (occluded, behind the first hit) are left untouched, so
    we only cut away the SURFACE layer of the artifact without punching
    through the whole object.

    Args:
        mesh: Input trimesh.Trimesh.
        mask_image: 2D mask. PIL.Image (mode L or RGB) or numpy (H, W)
            with values in [0, 255] or [0, 1]. White = delete, black = keep.
        camera: CanonicalCamera instance. Defaults to TRELLIS.2's canonical
            front view (eye=(0,0,2), yfov=40 deg).
        mask_threshold: Pixel value (after normalization) above which we
            treat the pixel as "delete here".
        ray_samples: Approx. number of rays to cast along each axis of the
            mask image. 512 is plenty for artifact-scale masks.
        max_depth_faces: How many layers deep to remove per ray. 1 is
            standard (only the first visible face). Higher values remove
            layered artifacts (e.g., two thin shells sticking out).
        fill_holes: Whether to fill the resulting holes. If False, returns
            the raw mesh with a hole (useful for debugging the cut).
        verbose: Print stats per step.

    Returns:
        Repaired trimesh.Trimesh with the artifact region excised.
    """
    import time
    from PIL import Image

    from clearmesh.editing.camera import CanonicalCamera

    # --- Normalize mask ---
    if hasattr(mask_image, "convert"):
        mask_img = mask_image.convert("L")
        mask_arr = np.asarray(mask_img, dtype=np.float32) / 255.0
    elif isinstance(mask_image, np.ndarray):
        mask_arr = mask_image.astype(np.float32)
        if mask_arr.max() > 1.5:
            mask_arr = mask_arr / 255.0
        if mask_arr.ndim == 3:
            mask_arr = mask_arr.mean(axis=-1)
    else:
        raise TypeError(f"mask_image must be PIL or ndarray, got {type(mask_image)}")

    H, W = mask_arr.shape
    cam = camera or CanonicalCamera.trellis2_default(image_size=max(H, W))

    # --- Find pixel locations where mask > threshold ---
    ys, xs = np.where(mask_arr > mask_threshold)
    if len(xs) == 0:
        if verbose:
            print("[surgery] mask is empty; returning input")
        return mesh

    # Subsample if too many pixels (not all needed for artifact removal)
    if len(xs) > ray_samples ** 2:
        step = int(np.ceil(len(xs) / (ray_samples ** 2)))
        xs = xs[::step]
        ys = ys[::step]
    if verbose:
        print(f"[surgery] casting {len(xs)} rays through mask pixels")

    # --- Build rays from the camera through each masked pixel ---
    t0 = time.time()
    mesh_centered = mesh.copy()
    mesh_centered.vertices = mesh.vertices - mesh.centroid
    extent = mesh.extents.max()
    if extent > 0:
        mesh_centered.vertices /= extent
    # mesh is now in a unit-centered frame matching the canonical camera

    ray_origins, ray_directions = _pixels_to_rays(xs, ys, cam, H, W)
    if verbose:
        print(f"[surgery] built rays in {time.time()-t0:.2f}s")

    # --- Raycast and collect front face indices ---
    t0 = time.time()
    locations, ray_ids, face_ids = mesh_centered.ray.intersects_location(
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        multiple_hits=(max_depth_faces > 1),
    )
    if verbose:
        print(f"[surgery] raycast {len(locations)} intersections in {time.time()-t0:.2f}s")

    if len(face_ids) == 0:
        if verbose:
            print("[surgery] no face intersections; returning input")
        return mesh

    # If multiple hits allowed, keep only the N nearest per ray
    if max_depth_faces > 1 and len(ray_ids) > 0:
        # Sort by ray_id, then by depth (distance from origin along ray)
        depths = np.linalg.norm(
            locations - ray_origins[ray_ids], axis=1
        )
        order = np.lexsort((depths, ray_ids))
        ray_ids = ray_ids[order]
        face_ids = face_ids[order]

        # For each ray, take first max_depth_faces
        _, first_idx = np.unique(ray_ids, return_index=True)
        keep_mask = np.zeros(len(ray_ids), dtype=bool)
        for start in first_idx:
            end = min(start + max_depth_faces, len(ray_ids))
            # Must still be same ray
            same_ray_end = start
            while same_ray_end < end and ray_ids[same_ray_end] == ray_ids[start]:
                same_ray_end += 1
            keep_mask[start:same_ray_end] = True
        face_ids = face_ids[keep_mask]

    # --- Remove those faces ---
    unique_face_ids = np.unique(face_ids)
    if verbose:
        print(f"[surgery] removing {len(unique_face_ids):,} faces "
              f"({len(unique_face_ids) / len(mesh.faces) * 100:.2f}% of mesh)")

    keep_face_mask = np.ones(len(mesh.faces), dtype=bool)
    keep_face_mask[unique_face_ids] = False
    cut_mesh = _safe_submesh(mesh, np.where(keep_face_mask)[0])
    if cut_mesh is None:
        if verbose:
            print("[surgery] no faces would remain after cut; returning input")
        return mesh

    # --- Fill hole ---
    if fill_holes:
        t0 = time.time()
        cut_mesh = fill_hole_smooth(cut_mesh, verbose=verbose)
        if verbose:
            print(f"[surgery] hole-fill in {time.time()-t0:.2f}s")

    return cut_mesh


# ---------------------------------------------------------------------------
# Statistical spike / outlier detection
# ---------------------------------------------------------------------------

def remove_by_spike_detection(
    mesh: trimesh.Trimesh,
    k_neighbors: int = 8,
    spike_std_threshold: float = 3.0,
    fill_holes: bool = True,
    verbose: bool = False,
) -> trimesh.Trimesh:
    """Detect and remove 'spike' vertices — those far from their local
    neighborhood average relative to the global vertex displacement std.

    Complementary to remove_by_mask. Use this when you can't easily paint
    a 2D mask, but the artifact is a clear geometric anomaly (e.g., a
    thin protrusion sticking out of a smooth surface).

    Caveat: can over-detect on legitimately spiky features like gear teeth.
    Use a conservative threshold and inspect results.
    """
    import time
    t0 = time.time()

    # Find k nearest neighbors for every vertex
    from scipy.spatial import cKDTree
    kd = cKDTree(mesh.vertices)
    dists, idx = kd.query(mesh.vertices, k=k_neighbors + 1)  # +1 for self

    # For each vertex: how far is it from its neighbor cluster mean?
    neighbor_means = mesh.vertices[idx[:, 1:]].mean(axis=1)
    offsets = np.linalg.norm(mesh.vertices - neighbor_means, axis=1)
    global_std = offsets.std()
    threshold = spike_std_threshold * global_std

    spike_vertex_mask = offsets > threshold
    if verbose:
        print(
            f"[surgery] detected {spike_vertex_mask.sum():,} spike verts "
            f"({spike_vertex_mask.mean()*100:.2f}%), threshold={threshold:.4f}"
        )

    if spike_vertex_mask.sum() == 0:
        return mesh

    # Remove faces that reference any spike vertex
    face_has_spike = spike_vertex_mask[mesh.faces].any(axis=1)
    keep_faces = np.where(~face_has_spike)[0]
    cut_mesh = _safe_submesh(mesh, keep_faces)
    if cut_mesh is None:
        if verbose:
            print("[surgery] no non-spike faces remaining; returning input")
        return mesh

    if verbose:
        print(f"[surgery] removed {face_has_spike.sum():,} faces in {time.time()-t0:.2f}s")

    if fill_holes:
        cut_mesh = fill_hole_smooth(cut_mesh, verbose=verbose)

    return cut_mesh


# ---------------------------------------------------------------------------
# Hole filling
# ---------------------------------------------------------------------------

def remove_by_bounding_box(
    mesh: trimesh.Trimesh,
    bbox_min,
    bbox_max,
    coords: str = "normalized",
    fill_holes: bool = True,
    smooth_iterations: int = 3,
    verbose: bool = False,
) -> trimesh.Trimesh:
    """Delete all faces whose centroid falls inside a world-space AABB.

    This is the "surgical cube" — the cleanest and most predictable
    artifact-removal primitive. You specify a 3D box in world space,
    every face fully inside gets deleted, and the resulting hole gets
    filled. No raycasting, no projection ambiguity, no grazing-angle
    artifacts on the surrounding mesh.

    Args:
        mesh: Input trimesh.Trimesh.
        bbox_min, bbox_max: (x, y, z) tuples or length-3 arrays defining
            the axis-aligned box.
        coords: Coordinate frame for the bbox:
          - "normalized" (default): bbox is in the unit-centered frame
            used by all our canonical rendering / Easy3E logic. Mesh
            is temporarily re-centered and rescaled for filtering.
          - "world": bbox is in the input mesh's raw world coordinates.
        fill_holes: Whether to fill the resulting hole.
        smooth_iterations: Patch-only Taubin iterations for hole fill.
        verbose: Print stats.

    Returns:
        Repaired trimesh.Trimesh with faces inside the bbox removed.
    """
    import time

    bbox_min = np.asarray(bbox_min, dtype=np.float32)
    bbox_max = np.asarray(bbox_max, dtype=np.float32)

    # Compute face centroids in the chosen coordinate frame
    if coords == "normalized":
        # Replicate the normalization used in our canonical renders:
        #   vertices -> vertices - centroid, then divide by max extent.
        centered = mesh.vertices - mesh.centroid
        ext = mesh.extents.max()
        if ext > 0:
            centered = centered / ext
        face_centroids = centered[mesh.faces].mean(axis=1)
    elif coords == "world":
        face_centroids = mesh.triangles.mean(axis=1)
    else:
        raise ValueError(f"bad coords mode: {coords}")

    # Faces to delete
    in_box = np.all(
        (face_centroids >= bbox_min) & (face_centroids <= bbox_max),
        axis=1,
    )
    if verbose:
        print(
            f"[bbox_surgery] bbox({coords})={bbox_min.tolist()} -> "
            f"{bbox_max.tolist()}; {in_box.sum():,} / {len(in_box):,} faces inside "
            f"({in_box.mean()*100:.2f}%)"
        )

    if in_box.sum() == 0:
        if verbose:
            print("[bbox_surgery] bbox contains no faces; returning input")
        return mesh

    keep_face_idx = np.where(~in_box)[0]
    cut_mesh = _safe_submesh(mesh, keep_face_idx)
    if cut_mesh is None:
        if verbose:
            print("[bbox_surgery] bbox would delete all faces; returning input")
        return mesh

    if fill_holes:
        t0 = time.time()
        cut_mesh = fill_hole_smooth(
            cut_mesh,
            smooth_iterations=smooth_iterations,
            verbose=verbose,
        )
        if verbose:
            print(f"[bbox_surgery] hole-fill in {time.time()-t0:.2f}s")

    return cut_mesh


def _patch_laplacian_relax(
    mesh: trimesh.Trimesh,
    patch_vertex_idx: np.ndarray,
    iterations: int = 20,
    lamb: float = 0.7,
    verbose: bool = False,
) -> trimesh.Trimesh:
    """Dirichlet-boundary Laplacian smoothing: move ONLY the listed
    vertices toward the average of their neighbors while all other
    vertices stay fixed.

    This is the canonical way to relax a newly-added patch into the
    surrounding mesh smoothly without perturbing the original geometry.

    Args:
        mesh: Trimesh with the patch already added.
        patch_vertex_idx: Indices of vertices that are FREE to move
            (everything else is a hard constraint).
        iterations: Smoothing passes. 10-30 is a good range.
        lamb: Step size per iteration (0 < lamb <= 1). 0.7 converges
            fast without oscillating on well-shaped patches.
        verbose: Print stats.
    """
    if len(patch_vertex_idx) == 0:
        return mesh

    import time
    t0 = time.time()

    # Build a PATCH-LOCAL adjacency: only the 1-ring around patch verts
    # matters. Full-mesh scatter_add per iteration was O(E) for all E
    # edges of a 2M-face mesh — way too slow. Here we pre-filter edges
    # to only those touching a patch vert (which is a tiny fraction).
    V = len(mesh.vertices)
    patch_mask = np.zeros(V, dtype=bool)
    patch_mask[patch_vertex_idx] = True

    edges = mesh.edges_unique  # (E, 2)
    touches_patch = patch_mask[edges].any(axis=1)
    local_edges = edges[touches_patch]
    num_local_patch_verts = len(patch_vertex_idx)
    if verbose:
        print(
            f"[surgery] relaxation scope: {len(local_edges):,} / {len(edges):,} edges "
            f"touching {num_local_patch_verts:,} patch verts"
        )

    verts = mesh.vertices.astype(np.float64).copy()

    # CSR-style accumulators sized to full mesh but only updated via
    # local_edges — O(len(local_edges)) per iter
    for i in range(iterations):
        neighbor_sums = np.zeros((V, 3), dtype=np.float64)
        neighbor_counts = np.zeros(V, dtype=np.int64)
        np.add.at(neighbor_sums, local_edges[:, 0], verts[local_edges[:, 1]])
        np.add.at(neighbor_sums, local_edges[:, 1], verts[local_edges[:, 0]])
        np.add.at(neighbor_counts, local_edges[:, 0], 1)
        np.add.at(neighbor_counts, local_edges[:, 1], 1)

        # Only touch patch verts' positions
        counts = np.maximum(neighbor_counts[patch_vertex_idx], 1)[:, None]
        averages = neighbor_sums[patch_vertex_idx] / counts
        verts[patch_vertex_idx] = (
            (1 - lamb) * verts[patch_vertex_idx] + lamb * averages
        )

    out = trimesh.Trimesh(
        vertices=verts.astype(np.float32),
        faces=mesh.faces,
        process=False,
    )
    if verbose:
        print(f"[surgery] patch relaxation: {iterations} iters "
              f"on {num_local_patch_verts:,} verts in {time.time()-t0:.2f}s")
    return out


def _flatten_patch_onto_rim_plane(
    mesh: trimesh.Trimesh,
    rim_idx: np.ndarray,
    interior_idx: np.ndarray,
    rim_pull: float = 0.5,
    verbose: bool = False,
) -> trimesh.Trimesh:
    """Fit a best-fit plane through rim verts, project interior patch
    verts onto that plane, and pull rim verts partway toward it.

    Uses SVD-based least-squares plane fitting (standard technique:
    centroid = mean of points, normal = smallest eigenvector of the
    covariance matrix).

    Args:
        mesh: Trimesh with patch already filled + relaxed.
        rim_idx: Indices of rim verts (both patch + non-patch refs).
            Plane is fit to these.
        interior_idx: Indices of interior patch verts (patch-only).
            Snapped fully onto the plane.
        rim_pull: 0.0 keeps rim verts put (may still show spikes),
                  1.0 snaps rim fully coplanar (cleanest cap but
                  slightly distorts surrounding mesh where rim
                  verts are shared).
    """
    if len(rim_idx) < 3:
        return mesh

    verts = mesh.vertices.copy()
    rim_points = verts[rim_idx].astype(np.float64)

    # Fit plane via PCA: centroid + smallest eigenvector as normal
    centroid = rim_points.mean(axis=0)
    centered = rim_points - centroid
    # Use SVD for numerical stability
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    # Last row of vt is the axis of smallest variance = plane normal
    normal = vt[-1] / np.linalg.norm(vt[-1])

    # Project a vertex onto the plane: p' = p - ((p - centroid) . normal) * normal
    def project_to_plane(pts):
        disp = (pts - centroid) @ normal
        return pts - disp[:, None] * normal

    # Snap interior verts fully (rim_pull=1 for interior always)
    if len(interior_idx) > 0:
        verts[interior_idx] = project_to_plane(
            verts[interior_idx].astype(np.float64)
        ).astype(verts.dtype)

    # Pull rim verts partway (lerp with rim_pull)
    if rim_pull > 0.0:
        rim_projected = project_to_plane(verts[rim_idx].astype(np.float64))
        verts[rim_idx] = (
            (1.0 - rim_pull) * verts[rim_idx]
            + rim_pull * rim_projected.astype(verts.dtype)
        )

    if verbose:
        # Residual = how far rim verts were from the plane (RMS)
        residuals = np.abs((rim_points - centroid) @ normal)
        print(
            f"[surgery] flat-cap plane fit: normal={normal.round(3).tolist()}, "
            f"rim residual RMS={np.sqrt((residuals**2).mean()):.4f}, "
            f"max={residuals.max():.4f}; snapped {len(interior_idx)} interior "
            f"+ pulled {len(rim_idx)} rim verts (rim_pull={rim_pull})"
        )

    return trimesh.Trimesh(vertices=verts, faces=mesh.faces, process=False)


def _subdivide_faces(
    mesh: trimesh.Trimesh,
    face_indices: np.ndarray,
) -> tuple[trimesh.Trimesh, np.ndarray]:
    """1-level midpoint subdivide of specific faces.

    Each triangle becomes 4 smaller triangles. Returns the new mesh
    and the indices of the NEW vertices (which will be the patch
    interior we can relax).
    """
    if len(face_indices) == 0:
        return mesh, np.array([], dtype=np.int64)

    V = mesh.vertices
    F = mesh.faces
    # Identify edges that need splitting
    tri_faces = F[face_indices]  # (Nsub, 3)
    edge_pairs = np.vstack([
        tri_faces[:, [0, 1]],
        tri_faces[:, [1, 2]],
        tri_faces[:, [2, 0]],
    ])
    edge_pairs_sorted = np.sort(edge_pairs, axis=1)
    unique_edges, inverse = np.unique(edge_pairs_sorted, axis=0, return_inverse=True)

    # New midpoint vertices
    midpoints = (V[unique_edges[:, 0]] + V[unique_edges[:, 1]]) / 2.0
    new_verts = np.vstack([V, midpoints])
    mid_offset = len(V)

    # For each subdivided face, create 4 new triangles
    num_sub_faces = len(tri_faces)
    mid01 = mid_offset + inverse[:num_sub_faces]
    mid12 = mid_offset + inverse[num_sub_faces:2*num_sub_faces]
    mid20 = mid_offset + inverse[2*num_sub_faces:]

    new_sub_faces = np.stack([
        np.stack([tri_faces[:, 0], mid01, mid20], axis=1),
        np.stack([tri_faces[:, 1], mid12, mid01], axis=1),
        np.stack([tri_faces[:, 2], mid20, mid12], axis=1),
        np.stack([mid01, mid12, mid20], axis=1),
    ], axis=0).reshape(-1, 3)

    # Keep non-subdivided faces as-is
    keep_mask = np.ones(len(F), dtype=bool)
    keep_mask[face_indices] = False
    final_faces = np.vstack([F[keep_mask], new_sub_faces])

    new_mesh = trimesh.Trimesh(
        vertices=new_verts,
        faces=final_faces,
        process=False,
    )
    # Indices of newly-added midpoint vertices
    new_vert_idx = np.arange(mid_offset, len(new_verts))
    return new_mesh, new_vert_idx


def fill_hole_smooth(
    mesh: trimesh.Trimesh,
    smooth_iterations: int = 3,
    cumesh_max_perimeter: float = 2.0,
    # subdivide_patch uses trimesh.remesh.subdivide which correctly
    # handles T-junctions by splitting shared edges on both sides.
    # Our earlier naive midpoint subdivision only touched patch faces,
    # creating half-split edges and spike artifacts. The built-in
    # remesher subdivides both the patch and adjacent non-patch faces
    # along shared edges, maintaining manifoldness.
    subdivide_patch: bool = True,
    patch_relax_iterations: int = 100,
    patch_relax_lamb: float = 0.5,
    # Flat-cap mode: after relaxation, snap interior patch verts to the
    # best-fit plane through the rim verts and pull rim verts partway
    # toward that plane. Eliminates "slight spikes" from non-coplanar
    # rim vertices that Laplacian relaxation alone can't resolve.
    flat_cap: bool = True,
    flat_cap_rim_pull: float = 0.5,  # 0 = rim stays put, 1 = fully coplanar
    verbose: bool = False,
) -> trimesh.Trimesh:
    """Fill boundary holes with a cascade and lightly smooth the patches.

    Cascade (each stage runs only if the previous left boundary edges):
      1. trimesh.fill_holes()  - fast, works on simple small holes.
      2. cumesh fill_holes     - CUDA-accelerated, handles complex
                                 multi-loop boundaries and large holes.
                                 Uses max_hole_perimeter = cumesh_max_perimeter
                                 (default 2.0 = HUGE, closes almost anything).
      3. Light Taubin smoothing - blends patch seams, only if we added
                                  geometry (preserves original surface).
    """
    import time

    def boundary_edge_count(m: trimesh.Trimesh) -> int:
        try:
            from trimesh.grouping import group_rows
            return int(len(group_rows(np.sort(m.edges, axis=1), require_count=1)))
        except Exception:
            return -1

    t_total = time.time()
    mesh = mesh.copy()
    before_faces = len(mesh.faces)
    before_verts = len(mesh.vertices)

    # --- Stage 1: trimesh.fill_holes (fast path) ---
    try:
        mesh.fill_holes()
    except Exception as e:
        if verbose:
            print(f"[surgery] trimesh.fill_holes raised {e}; continuing")
    stage1_faces = len(mesh.faces) - before_faces
    be1 = boundary_edge_count(mesh)
    if verbose:
        print(f"[surgery] stage 1 trimesh.fill_holes: +{stage1_faces} faces, {be1} residual boundary edges")

    # --- Stage 2: cumesh aggressive fill (CUDA) for any remaining loops ---
    before_stage2_faces = len(mesh.faces)
    if be1 > 0:
        try:
            import cumesh  # type: ignore
            import torch  # type: ignore
            if torch.cuda.is_available():
                t0 = time.time()
                v = torch.tensor(np.asarray(mesh.vertices), dtype=torch.float32, device="cuda")
                f = torch.tensor(np.asarray(mesh.faces), dtype=torch.int32, device="cuda")
                cm = cumesh.CuMesh()
                cm.init(v, f)
                cm.get_edges()
                cm.get_boundary_info()
                if cm.num_boundaries > 0:
                    cm.get_vertex_edge_adjacency()
                    cm.get_vertex_boundary_adjacency()
                    cm.get_manifold_boundary_adjacency()
                    cm.read_manifold_boundary_adjacency()
                    cm.get_boundary_connected_components()
                    cm.get_boundary_loops()
                    if cm.num_boundary_loops > 0:
                        cm.fill_holes(max_hole_perimeter=cumesh_max_perimeter)
                        nv, nf = cm.read()
                        mesh = trimesh.Trimesh(
                            vertices=nv.cpu().numpy(),
                            faces=nf.cpu().numpy(),
                            process=False,
                        )
                        if verbose:
                            print(
                                f"[surgery] stage 2 cumesh.fill_holes("
                                f"max_perimeter={cumesh_max_perimeter}): "
                                f"+{len(mesh.faces) - before_stage2_faces} faces "
                                f"in {time.time() - t0:.2f}s"
                            )
        except ImportError:
            if verbose:
                print("[surgery] cumesh unavailable; skipping stage 2")
        except Exception as e:
            import warnings
            warnings.warn(f"[surgery] cumesh fill failed ({e}); stage 2 skipped")

    total_new_faces = len(mesh.faces) - before_faces
    total_new_verts = len(mesh.vertices) - before_verts
    final_be = boundary_edge_count(mesh)
    if verbose:
        print(
            f"[surgery] total added {total_new_faces} patch faces "
            f"+ {total_new_verts} patch verts, "
            f"final boundary edges = {final_be}"
        )

    # --- Stage 3: identify the patch faces and relax them ---
    # Key insight: cumesh's fill_holes REUSES existing boundary vertices
    # to close the hole, so checking for new verts misses most of the
    # patch. Instead we identify patch FACES (added during fill) and
    # compute patch VERTS as those referenced ONLY by patch faces (new
    # internal verts, tiny handful) plus the RIM verts referenced by
    # both patch and non-patch faces.
    #
    # Under Dirichlet constraints we pin verts whose most neighbors are
    # non-patch so the surrounding mesh stays bit-exact.
    patch_face_idx = np.arange(before_faces, len(mesh.faces))

    if len(patch_face_idx) > 0:
        # Optional: properly subdivide the patch using trimesh's built-in
        # remesher (handles T-junctions with adjacent non-patch faces
        # automatically). This gives the Laplacian room to smooth.
        if subdivide_patch:
            # trimesh.remesh.subdivide processes the given face subset
            # but properly adds midpoint verts to adjacent faces too,
            # avoiding the T-junction-spike failure mode.
            try:
                from trimesh import remesh as _remesh
                pre_verts = len(mesh.vertices)
                pre_faces = len(mesh.faces)
                new_v, new_f = _remesh.subdivide(
                    mesh.vertices, mesh.faces, face_index=patch_face_idx
                )
                mesh = trimesh.Trimesh(vertices=new_v, faces=new_f, process=False)
                if verbose:
                    print(
                        f"[surgery] trimesh.remesh.subdivide: "
                        f"{pre_faces:,} -> {len(mesh.faces):,} faces, "
                        f"{pre_verts:,} -> {len(mesh.vertices):,} verts"
                    )
                # Recompute patch faces: they're now the tail of the faces array
                patch_face_idx = np.arange(
                    pre_faces if pre_faces < len(mesh.faces) else 0,
                    len(mesh.faces),
                )
            except Exception as e:
                import warnings
                warnings.warn(f"[surgery] subdivide failed ({e}); skipping")

        # Find patch-only verts (referenced ONLY by patch faces) and rim
        # verts (referenced by both patch and non-patch faces). Rim verts
        # get light relaxation; non-patch-only verts are hard-fixed.
        all_face_mask = np.zeros(len(mesh.faces), dtype=bool)
        all_face_mask[patch_face_idx] = True
        patch_vert_refs = np.zeros(len(mesh.vertices), dtype=np.int32)
        non_patch_vert_refs = np.zeros(len(mesh.vertices), dtype=np.int32)
        for fi, is_patch in enumerate(all_face_mask):
            for v in mesh.faces[fi]:
                if is_patch:
                    patch_vert_refs[v] += 1
                else:
                    non_patch_vert_refs[v] += 1
        # A vert is "free to move" if it is patch-only (no non-patch ref)
        # OR on the rim (both refs; allow movement but will be pulled by
        # its non-patch neighbors back toward surrounding surface).
        free_vert_mask = patch_vert_refs > 0
        rim_vert_mask = free_vert_mask & (non_patch_vert_refs > 0)
        interior_patch_mask = free_vert_mask & (non_patch_vert_refs == 0)

        relax_idx = np.where(free_vert_mask)[0]
        if verbose:
            print(
                f"[surgery] patch ownership: {interior_patch_mask.sum():,} interior, "
                f"{rim_vert_mask.sum():,} rim, {free_vert_mask.sum():,} total relaxable"
            )

        if patch_relax_iterations > 0 and len(relax_idx) > 0:
            mesh = _patch_laplacian_relax(
                mesh,
                patch_vertex_idx=relax_idx,
                iterations=patch_relax_iterations,
                lamb=patch_relax_lamb,
                verbose=verbose,
            )

        # --- Stage 4: flat-plane projection of interior patch ---
        # After relaxation, the cap can still have "slight spikes" because
        # rim verts are at slightly different heights (the cut contour
        # wasn't planar). Fit a best-fit plane through the rim, project
        # every INTERIOR patch vert onto that plane exactly, and pull
        # rim verts partway toward it. This produces a flat cap that
        # smoothly continues from the (still-preserved) surrounding
        # geometry while eliminating the crown-spike pattern.
        if flat_cap and len(relax_idx) > 0:
            rim_idx = np.where(rim_vert_mask)[0]
            interior_idx = np.where(interior_patch_mask)[0]

            if len(rim_idx) >= 3 and len(interior_idx) > 0:
                mesh = _flatten_patch_onto_rim_plane(
                    mesh,
                    rim_idx=rim_idx,
                    interior_idx=interior_idx,
                    rim_pull=flat_cap_rim_pull,
                    verbose=verbose,
                )

    # Legacy: very light whole-mesh Taubin (kept for backcompat/tuning).
    # Skip by default; the patch relaxation above is strictly better.
    if smooth_iterations > 0 and total_new_faces > 0 and len(patch_face_idx) == 0:
        # Only runs if we couldn't track patch faces (defensive path)
        try:
            trimesh.smoothing.filter_taubin(
                mesh,
                lamb=0.3,
                nu=-0.32,
                iterations=smooth_iterations,
            )
        except Exception as e:
            import warnings
            warnings.warn(f"[surgery] Taubin fallback failed ({e})")

    if verbose:
        print(f"[surgery] fill_hole_smooth done in {time.time() - t_total:.2f}s")
    return mesh


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pixels_to_rays(xs, ys, camera, H, W):
    """Convert pixel coords (xs, ys) + canonical camera → ray origins
    and directions for trimesh.ray casting.

    Rays originate at the camera eye and shoot through each pixel's
    world position at unit depth.
    """
    # Camera params
    eye = _camera_eye(camera)
    view = camera.view_matrix.numpy() if hasattr(camera.view_matrix, "numpy") else np.asarray(camera.view_matrix)
    proj = camera.projection_matrix().numpy() if hasattr(camera.projection_matrix(), "numpy") else np.asarray(camera.projection_matrix())

    # Pixel → NDC (inverse of (u,v) = ((ndc_x+1)*0.5*W, (1-ndc_y)*0.5*H))
    ndc_x = (xs.astype(np.float32) / W) * 2.0 - 1.0
    ndc_y = 1.0 - (ys.astype(np.float32) / H) * 2.0

    # NDC → camera-space direction. Use near plane at z=-1 for unit depth.
    inv_proj = np.linalg.inv(proj)
    near_clip = np.stack(
        [ndc_x, ndc_y, -np.ones_like(ndc_x), np.ones_like(ndc_x)], axis=1
    )  # (N, 4) — z=-1 (near plane), w=1
    near_cam = near_clip @ inv_proj.T
    near_cam = near_cam[:, :3] / near_cam[:, 3:4]  # perspective divide

    # Camera → world
    inv_view = np.linalg.inv(view)
    near_world_hom = np.concatenate(
        [near_cam, np.ones((near_cam.shape[0], 1), dtype=np.float32)], axis=1
    )
    near_world = near_world_hom @ inv_view.T
    near_world = near_world[:, :3] / near_world[:, 3:4]

    origins = np.tile(eye[np.newaxis, :], (near_world.shape[0], 1))
    directions = near_world - origins
    directions /= np.linalg.norm(directions, axis=1, keepdims=True) + 1e-12
    return origins.astype(np.float32), directions.astype(np.float32)


def _safe_submesh(mesh: trimesh.Trimesh, face_indices: np.ndarray):
    """Build a submesh from face indices, robust to edge cases.

    trimesh's submesh(append=True) sometimes returns a numpy array of
    face indices when the result is empty or has trivial geometry. Build
    the submesh manually to guarantee a trimesh.Trimesh return.
    """
    if len(face_indices) == 0:
        return None

    kept_faces = np.asarray(mesh.faces, dtype=np.int64)[face_indices]
    # Remap vertex indices so unused verts are dropped
    referenced = np.zeros(len(mesh.vertices), dtype=bool)
    referenced[kept_faces.ravel()] = True
    if not referenced.any():
        return None

    new_vert_idx = np.cumsum(referenced) - 1
    new_faces = new_vert_idx[kept_faces]
    new_verts = mesh.vertices[referenced]
    return trimesh.Trimesh(vertices=new_verts, faces=new_faces, process=False)


def _camera_eye(camera):
    """Extract camera world position from a CanonicalCamera."""
    view = camera.view_matrix.numpy() if hasattr(camera.view_matrix, "numpy") else np.asarray(camera.view_matrix)
    # view = world->camera, so camera position in world = inverse translation
    inv_view = np.linalg.inv(view)
    return inv_view[:3, 3].astype(np.float32)
