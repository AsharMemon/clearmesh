"""Export meshes to various formats: STL (printing), GLB (textured), FBX (rigged).

STL: For 3D printing (Chitubox, Lychee Slicer)
GLB: For web viewers and general 3D (textured, compact)
FBX: For game engines (Unity, Unreal) with skeleton and animation
OBJ: Universal interchange format
"""

import os
from pathlib import Path

import numpy as np
import trimesh


def export_stl(mesh: trimesh.Trimesh, output_path: str) -> str:
    """Export as binary STL for 3D printing.

    STL is the standard format for FDM and resin 3D printers.
    Binary STL is preferred over ASCII for smaller file size.

    Args:
        mesh: Mesh to export (should be watertight)
        output_path: Output file path (.stl)

    Returns:
        Absolute path to exported file
    """
    output_path = str(Path(output_path).with_suffix(".stl"))
    mesh.export(output_path, file_type="stl")
    return os.path.abspath(output_path)


def export_glb(mesh: trimesh.Trimesh, output_path: str, texture: np.ndarray | None = None) -> str:
    """Export as GLB (binary glTF) with optional texture.

    GLB is a compact binary format widely supported by web viewers,
    game engines, and 3D tools.

    Args:
        mesh: Mesh to export
        output_path: Output file path (.glb)
        texture: Optional (H, W, 3) uint8 texture image

    Returns:
        Absolute path to exported file
    """
    output_path = str(Path(output_path).with_suffix(".glb"))

    if texture is not None:
        from PIL import Image

        # Create texture material
        tex_image = Image.fromarray(texture)
        material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=tex_image,
        )
        mesh.visual = trimesh.visual.TextureVisuals(material=material)

    mesh.export(output_path, file_type="glb")
    return os.path.abspath(output_path)


def export_obj(mesh: trimesh.Trimesh, output_path: str) -> str:
    """Export as Wavefront OBJ.

    Universal interchange format. Exports .obj + .mtl files.

    Args:
        mesh: Mesh to export
        output_path: Output file path (.obj)

    Returns:
        Absolute path to exported file
    """
    output_path = str(Path(output_path).with_suffix(".obj"))
    mesh.export(output_path, file_type="obj")
    return os.path.abspath(output_path)


def export_fbx(
    mesh: trimesh.Trimesh,
    output_path: str,
    skeleton: dict | None = None,
    skin_weights: np.ndarray | None = None,
) -> str:
    """Export as FBX with optional skeleton and skinning weights.

    FBX is required for Unity/Unreal Engine import with animation support.
    Uses pygltflib for conversion or direct FBX SDK if available.

    Args:
        mesh: Mesh to export
        output_path: Output file path (.fbx)
        skeleton: Joint hierarchy dict from auto-rigging
        skin_weights: (V, num_joints) skinning weight matrix

    Returns:
        Absolute path to exported file
    """
    output_path = str(Path(output_path).with_suffix(".fbx"))

    if skeleton is not None and skin_weights is not None:
        # Export rigged mesh — delegate to Puppeteer's FBX exporter
        # which handles skeleton + skinning natively
        try:
            _export_rigged_fbx(mesh, skeleton, skin_weights, output_path)
        except ImportError:
            # Fallback: export as GLB (most engines accept GLB too)
            print("FBX export requires Puppeteer. Exporting as GLB instead.")
            glb_path = output_path.replace(".fbx", ".glb")
            export_glb(mesh, glb_path)
            return os.path.abspath(glb_path)
    else:
        # Static mesh — export as GLB (FBX for static is unnecessary)
        # But respect the user's format choice
        try:
            import pyfbx

            _export_static_fbx(mesh, output_path)
        except ImportError:
            # No FBX SDK, export as OBJ
            print("FBX SDK not available. Exporting as OBJ instead.")
            obj_path = output_path.replace(".fbx", ".obj")
            export_obj(mesh, obj_path)
            return os.path.abspath(obj_path)

    return os.path.abspath(output_path)


def _export_rigged_fbx(
    mesh: trimesh.Trimesh,
    skeleton: dict,
    skin_weights: np.ndarray,
    output_path: str,
):
    """Export rigged mesh using Puppeteer's native FBX pipeline."""
    import sys

    puppeteer_dir = "/mnt/data/Puppeteer"
    if puppeteer_dir not in sys.path:
        sys.path.insert(0, puppeteer_dir)

    # Puppeteer provides FBX export utilities
    from utils.export import export_to_fbx

    export_to_fbx(
        vertices=mesh.vertices,
        faces=mesh.faces,
        joints=skeleton,
        weights=skin_weights,
        output_path=output_path,
    )


def _export_static_fbx(mesh: trimesh.Trimesh, output_path: str):
    """Export static mesh as FBX using pyfbx."""
    import pyfbx

    writer = pyfbx.FBXWriter()
    writer.add_mesh(
        vertices=mesh.vertices.tolist(),
        faces=mesh.faces.tolist(),
        name="clearmesh_output",
    )
    writer.write(output_path)


def export_mesh(
    mesh: trimesh.Trimesh,
    output_path: str,
    format: str = "glb",
    texture: np.ndarray | None = None,
    skeleton: dict | None = None,
    skin_weights: np.ndarray | None = None,
) -> str:
    """Export mesh in the specified format.

    Convenience function that dispatches to format-specific exporters.

    Args:
        mesh: Mesh to export
        output_path: Output file path
        format: One of 'stl', 'glb', 'obj', 'fbx'
        texture: Optional texture for GLB
        skeleton: Optional skeleton for FBX
        skin_weights: Optional skin weights for FBX

    Returns:
        Absolute path to exported file
    """
    exporters = {
        "stl": lambda: export_stl(mesh, output_path),
        "glb": lambda: export_glb(mesh, output_path, texture),
        "obj": lambda: export_obj(mesh, output_path),
        "fbx": lambda: export_fbx(mesh, output_path, skeleton, skin_weights),
    }

    if format not in exporters:
        raise ValueError(f"Unknown format: {format}. Supported: {list(exporters.keys())}")

    return exporters[format]()
