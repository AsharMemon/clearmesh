"""Mesh repair and print-readiness: watertight manifold geometry for 3D printing.

Pipeline:
  1. PyMeshFix: Hole filling, manifold repair, self-intersection removal
  2. Trimesh validation: Watertight check, degenerate face removal
  3. Optional: Open3D-based cleaning for edge cases
  4. Print-readiness: orientation, hollowing with drain holes, full validation

Print-readiness checklist (automated):
  (1) Watertight/manifold validation
  (2) Scale to target base size (25mm/32mm/40mm)
  (3) Orient for minimal supports (Z-up, feet on build plate)
  (4) Optional auto-hollowing with drain holes for resin savings
  (5) Export as binary STL (smaller file size)
"""

import numpy as np
import trimesh


def repair_mesh(
    mesh: trimesh.Trimesh,
    fix_normals: bool = True,
    remove_degenerate: bool = True,
    fill_holes: bool = True,
    verbose: bool = False,
) -> trimesh.Trimesh:
    """Full mesh repair pipeline.

    Args:
        mesh: Input mesh (possibly non-manifold, open, etc.)
        fix_normals: Ensure consistent face winding
        remove_degenerate: Remove zero-area faces
        fill_holes: Fill holes to make watertight
        verbose: Print repair statistics

    Returns:
        Repaired trimesh.Trimesh
    """
    vertices = np.array(mesh.vertices, dtype=np.float64)
    faces = np.array(mesh.faces, dtype=np.int32)

    if verbose:
        print(f"Input: {vertices.shape[0]} verts, {faces.shape[0]} faces")
        print(f"  Watertight: {mesh.is_watertight}")
        print(f"  Volume: {mesh.is_volume}")

    # Step 1: PyMeshFix repair
    try:
        import pymeshfix

        fixer = pymeshfix.MeshFix(vertices, faces)
        fixer.repair(verbose=verbose)
        vertices = np.array(fixer.v, dtype=np.float64)
        faces = np.array(fixer.f, dtype=np.int32)
    except ImportError:
        if verbose:
            print("pymeshfix not available, using trimesh-only repair")
    except Exception as e:
        if verbose:
            print(f"PyMeshFix failed: {e}, continuing with trimesh repair")

    # Step 2: Rebuild mesh and clean
    repaired = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)

    if remove_degenerate:
        # Remove zero-area faces
        repaired.remove_degenerate_faces()
        # Remove unreferenced vertices
        repaired.remove_unreferenced_vertices()
        # Merge duplicate vertices
        repaired.merge_vertices()

    if fix_normals:
        repaired.fix_normals()

    if fill_holes and not repaired.is_watertight:
        repaired.fill_holes()

    if verbose:
        print(f"Output: {repaired.vertices.shape[0]} verts, {repaired.faces.shape[0]} faces")
        print(f"  Watertight: {repaired.is_watertight}")
        print(f"  Volume: {repaired.is_volume}")

    return repaired


def repair_with_open3d(mesh: trimesh.Trimesh, verbose: bool = False) -> trimesh.Trimesh:
    """Alternative repair using Open3D for edge cases.

    Open3D's Poisson reconstruction can recover topology when
    PyMeshFix fails on severely damaged meshes.

    Args:
        mesh: Input mesh
        verbose: Print statistics

    Returns:
        Repaired mesh
    """
    try:
        import open3d as o3d
    except ImportError:
        if verbose:
            print("Open3D not available, falling back to basic repair")
        return repair_mesh(mesh, verbose=verbose)

    # Convert to Open3D
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
    o3d_mesh.compute_vertex_normals()

    # Remove non-manifold edges/vertices
    o3d_mesh.remove_non_manifold_edges()
    o3d_mesh.remove_degenerate_triangles()
    o3d_mesh.remove_duplicated_triangles()
    o3d_mesh.remove_duplicated_vertices()
    o3d_mesh.remove_unreferenced_vertices()

    # Convert back to trimesh
    vertices = np.asarray(o3d_mesh.vertices)
    faces = np.asarray(o3d_mesh.triangles)

    repaired = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)

    if not repaired.is_watertight:
        # Try Poisson reconstruction as last resort
        if verbose:
            print("Attempting Poisson reconstruction...")

        pcd = o3d_mesh.sample_points_poisson_disk(number_of_points=50000)
        pcd.estimate_normals()

        poisson_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=9
        )

        # Trim low-density vertices
        density_threshold = np.quantile(np.asarray(densities), 0.05)
        vertices_to_remove = np.asarray(densities) < density_threshold
        poisson_mesh.remove_vertices_by_mask(vertices_to_remove)

        vertices = np.asarray(poisson_mesh.vertices)
        faces = np.asarray(poisson_mesh.triangles)
        repaired = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)

    if verbose:
        print(f"Open3D repair result: {repaired.vertices.shape[0]} verts, watertight={repaired.is_watertight}")

    return repaired


def validate_for_printing(mesh: trimesh.Trimesh) -> dict:
    """Validate mesh is ready for 3D printing.

    Returns a report of potential issues.
    """
    report = {
        "watertight": mesh.is_watertight,
        "volume": mesh.is_volume,
        "vertex_count": mesh.vertices.shape[0],
        "face_count": mesh.faces.shape[0],
        "bounding_box_mm": None,
        "issues": [],
    }

    if not mesh.is_watertight:
        report["issues"].append("Mesh is not watertight — slicer may fail")

    if not mesh.is_volume:
        report["issues"].append("Mesh does not enclose a volume")

    # Check for inverted normals
    if mesh.is_watertight and mesh.volume < 0:
        report["issues"].append("Inverted normals detected (negative volume)")

    # Check for thin walls (faces very close together)
    extents = mesh.extents
    min_extent = extents.min()
    if min_extent < 0.001:
        report["issues"].append(f"Very thin dimension: {min_extent:.4f} units")

    # Check face count (slicers struggle with >5M faces)
    if mesh.faces.shape[0] > 5_000_000:
        report["issues"].append(f"High face count ({mesh.faces.shape[0]}) — consider decimation")

    report["printable"] = len(report["issues"]) == 0

    return report


def orient_for_printing(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Orient mesh for minimal supports during 3D printing.

    Places the flattest surface down (Z-up) to minimize overhangs.
    For character models, typically means feet on the build plate.

    Args:
        mesh: Input mesh

    Returns:
        Reoriented mesh
    """
    mesh = mesh.copy()

    # Find the face with the largest area that faces downward
    face_normals = mesh.face_normals
    face_areas = mesh.area_faces

    # Score each face by how "flat and down-facing" it is
    # Prefer large faces with normals pointing in -Z
    down_score = -face_normals[:, 2] * face_areas
    best_face = np.argmax(down_score)
    best_normal = face_normals[best_face]

    # Rotate so best_normal aligns with -Z (pointing down)
    target = np.array([0.0, 0.0, -1.0])
    rotation = _rotation_between_vectors(best_normal, target)
    mesh.apply_transform(rotation)

    # Translate so lowest point is at Z=0 (build plate)
    z_min = mesh.vertices[:, 2].min()
    mesh.vertices[:, 2] -= z_min

    return mesh


def _rotation_between_vectors(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Compute 4x4 rotation matrix to rotate v1 to v2."""
    v1 = v1 / (np.linalg.norm(v1) + 1e-10)
    v2 = v2 / (np.linalg.norm(v2) + 1e-10)

    cross = np.cross(v1, v2)
    dot = np.dot(v1, v2)

    if np.linalg.norm(cross) < 1e-8:
        # Vectors are parallel
        if dot > 0:
            return np.eye(4)
        else:
            # 180-degree rotation around any perpendicular axis
            perp = np.array([1, 0, 0]) if abs(v1[0]) < 0.9 else np.array([0, 1, 0])
            axis = np.cross(v1, perp)
            axis /= np.linalg.norm(axis)
            return trimesh.transformations.rotation_matrix(np.pi, axis)

    skew = np.array([
        [0, -cross[2], cross[1]],
        [cross[2], 0, -cross[0]],
        [-cross[1], cross[0], 0],
    ])
    rotation_3x3 = np.eye(3) + skew + skew @ skew / (1 + dot)

    mat = np.eye(4)
    mat[:3, :3] = rotation_3x3
    return mat


def add_drain_holes(
    mesh: trimesh.Trimesh,
    hole_radius_mm: float = 1.5,
    num_holes: int = 2,
) -> trimesh.Trimesh:
    """Add drain holes to a hollowed mesh for resin printing.

    Drain holes allow uncured resin to escape from the interior,
    reducing weight and preventing pressure buildup during post-curing.

    Holes are placed on the bottom surface (build plate side).

    Args:
        mesh: Hollowed watertight mesh (in mm)
        hole_radius_mm: Radius of each drain hole
        num_holes: Number of holes to add

    Returns:
        Mesh with drain holes (boolean subtraction)
    """
    mesh = mesh.copy()

    # Find the lowest Z positions for hole placement
    z_min = mesh.vertices[:, 2].min()
    bottom_verts = mesh.vertices[mesh.vertices[:, 2] < z_min + 0.5]

    if bottom_verts.shape[0] == 0:
        return mesh

    # Find spread-out positions on the bottom
    centroid = bottom_verts.mean(axis=0)
    hole_positions = []

    if num_holes == 1:
        hole_positions.append(centroid)
    else:
        # Spread holes around the bottom
        for i in range(num_holes):
            angle = 2 * np.pi * i / num_holes
            offset = np.array([
                np.cos(angle) * hole_radius_mm * 3,
                np.sin(angle) * hole_radius_mm * 3,
                0,
            ])
            pos = centroid + offset
            pos[2] = z_min  # On the bottom
            hole_positions.append(pos)

    # Create cylinders for boolean subtraction
    for pos in hole_positions:
        hole = trimesh.creation.cylinder(
            radius=hole_radius_mm,
            height=hole_radius_mm * 4,  # Tall enough to penetrate
            sections=32,
        )
        hole.apply_translation(pos)

        try:
            mesh = mesh.difference(hole)
        except Exception:
            # Boolean operations can fail — skip this hole
            pass

    return mesh


def full_print_preparation(
    mesh: trimesh.Trimesh,
    target_scale_mm: float | None = None,
    orient: bool = True,
    hollow: bool = False,
    wall_thickness_mm: float = 1.5,
    drain_holes: bool = False,
    drain_hole_radius_mm: float = 1.5,
    verbose: bool = False,
) -> trimesh.Trimesh:
    """Full automated print-readiness pipeline.

    Applies all preparation steps in order:
      1. Repair (watertight, manifold)
      2. Orient for minimal supports
      3. Scale to target size
      4. Hollow (optional, for resin savings)
      5. Add drain holes (optional, if hollowed)

    Args:
        mesh: Input mesh
        target_scale_mm: Target height in mm (None = keep current scale)
        orient: Orient for minimal supports
        hollow: Hollow the mesh for resin printing
        wall_thickness_mm: Wall thickness when hollowed
        drain_holes: Add drain holes (only if hollow=True)
        drain_hole_radius_mm: Drain hole radius
        verbose: Print progress

    Returns:
        Print-ready mesh
    """
    if verbose:
        print("=== Print Preparation Pipeline ===")

    # Step 1: Repair
    if verbose:
        print("Step 1/5: Repairing mesh...")
    mesh = repair_mesh(mesh, verbose=verbose)

    # Step 2: Orient
    if orient:
        if verbose:
            print("Step 2/5: Orienting for minimal supports...")
        mesh = orient_for_printing(mesh)

    # Step 3: Scale
    if target_scale_mm is not None:
        if verbose:
            print(f"Step 3/5: Scaling to {target_scale_mm}mm...")
        current_height = mesh.extents[2]
        if current_height > 0:
            scale_factor = target_scale_mm / current_height
            mesh.apply_scale(scale_factor)

    # Step 4: Hollow
    if hollow:
        if verbose:
            print(f"Step 4/5: Hollowing (wall={wall_thickness_mm}mm)...")
        from clearmesh.utils.scale import hollow_mesh
        mesh = hollow_mesh(mesh, wall_thickness_mm=wall_thickness_mm)

    # Step 5: Drain holes
    if hollow and drain_holes:
        if verbose:
            print(f"Step 5/5: Adding drain holes (r={drain_hole_radius_mm}mm)...")
        mesh = add_drain_holes(mesh, hole_radius_mm=drain_hole_radius_mm)

    # Final validation
    report = validate_for_printing(mesh)
    if verbose:
        print(f"\nPrint validation: {'PASS' if report['printable'] else 'ISSUES FOUND'}")
        for issue in report["issues"]:
            print(f"  - {issue}")

    return mesh
