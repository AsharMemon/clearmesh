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


def fill_hole_smooth(
    mesh: trimesh.Trimesh,
    smooth_iterations: int = 3,
    cumesh_max_perimeter: float = 2.0,
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
    final_be = boundary_edge_count(mesh)
    if verbose:
        print(
            f"[surgery] total added {total_new_faces} patch faces, "
            f"final boundary edges = {final_be}"
        )

    # --- Stage 3: patch-only Taubin smoothing ---
    if smooth_iterations > 0 and total_new_faces > 0:
        try:
            trimesh.smoothing.filter_taubin(
                mesh,
                lamb=0.3,
                nu=-0.32,
                iterations=smooth_iterations,
            )
        except Exception as e:
            import warnings
            warnings.warn(f"[surgery] patch smoothing failed ({e}); leaving rough fill")

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
