"""Scale normalization for 3D printing miniatures.

Common tabletop miniature scales:
  - 28mm: Standard D&D/Warhammer (heroic 28mm = ~32mm actual)
  - 32mm: Modern standard for many wargames
  - 54mm: Collectible/display scale
  - 75mm: Large display figures

The scale refers to the height of a "standard human" figure from
feet to eye level. Non-humanoid models are scaled proportionally.
"""

import numpy as np
import trimesh


# Standard miniature scales (height of a ~1.75m human, feet to eyes)
SCALES = {
    "28mm": 28.0,
    "32mm": 32.0,
    "54mm": 54.0,
    "75mm": 75.0,
}

# Assumed real-world height of the model (in model units = meters)
DEFAULT_HUMAN_HEIGHT_M = 1.75


def scale_to_mm(
    mesh: trimesh.Trimesh,
    target_height_mm: float,
    reference_height: float | None = None,
) -> trimesh.Trimesh:
    """Scale a mesh to a specific height in millimeters.

    Args:
        mesh: Input mesh (units assumed to be normalized or meters)
        target_height_mm: Target height in millimeters
        reference_height: Known reference height of the model.
            If None, uses the mesh's Z-extent as height.

    Returns:
        Scaled mesh (new copy)
    """
    mesh = mesh.copy()

    if reference_height is not None:
        current_height = reference_height
    else:
        # Use Z-axis extent as height (feet to top of head)
        current_height = mesh.extents[2]  # Z is typically up

    if current_height <= 0:
        raise ValueError("Mesh has zero height — cannot scale")

    scale_factor = target_height_mm / current_height
    mesh.apply_scale(scale_factor)

    return mesh


def scale_to_preset(mesh: trimesh.Trimesh, preset: str) -> trimesh.Trimesh:
    """Scale to a standard miniature preset.

    Args:
        mesh: Input mesh
        preset: One of '28mm', '32mm', '54mm', '75mm'

    Returns:
        Scaled mesh
    """
    if preset not in SCALES:
        raise ValueError(f"Unknown preset: {preset}. Supported: {list(SCALES.keys())}")

    return scale_to_mm(mesh, SCALES[preset])


def add_base(
    mesh: trimesh.Trimesh,
    base_diameter_mm: float = 25.0,
    base_height_mm: float = 3.0,
) -> trimesh.Trimesh:
    """Add a circular base to a miniature for printing stability.

    Args:
        mesh: Input mesh (already scaled to mm)
        base_diameter_mm: Diameter of circular base
        base_height_mm: Height/thickness of the base

    Returns:
        Combined mesh with base
    """
    # Create cylinder base
    base = trimesh.creation.cylinder(
        radius=base_diameter_mm / 2.0,
        height=base_height_mm,
        sections=64,
    )

    # Position base below the mesh
    mesh_bottom = mesh.bounds[0][2]  # Lowest Z point
    base_center_z = mesh_bottom - base_height_mm / 2.0
    base.apply_translation([0, 0, base_center_z])

    # Combine
    combined = trimesh.util.concatenate([mesh, base])
    return combined


def hollow_mesh(
    mesh: trimesh.Trimesh,
    wall_thickness_mm: float = 1.5,
) -> trimesh.Trimesh:
    """Hollow out a mesh for resin printing (saves resin, reduces suction).

    Creates an inner shell offset from the outer surface.

    Args:
        mesh: Input watertight mesh (in mm)
        wall_thickness_mm: Wall thickness

    Returns:
        Hollowed mesh
    """
    if not mesh.is_watertight:
        raise ValueError("Mesh must be watertight before hollowing")

    # Create inner shell by offsetting vertices inward along normals
    inner = mesh.copy()
    inner.invert()  # Flip normals for inner surface

    # Offset inward
    normals = mesh.vertex_normals
    inner.vertices = mesh.vertices - normals * wall_thickness_mm

    # Combine outer and inner shells
    combined = trimesh.util.concatenate([mesh, inner])

    return combined
