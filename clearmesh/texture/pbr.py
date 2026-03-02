"""PBR texture handling for digital variants.

For 3D printing: textures are irrelevant (resin prints are monochrome).
For digital preview and VTT marketplace: PBR textures add visual quality.

Sources:
  - Hunyuan3D 2.5 / TRELLIS.2 built-in PBR (albedo, normal, roughness, metallic)
  - Manual enhancement via InstaMAT or Material Maker

This module extracts, stores, and re-applies PBR maps from the generation
pipeline for export in textured formats (GLB, FBX).

Usage:
    pbr = PBRTextures.from_pipeline_output(trellis2_output)
    pbr.save('/path/to/textures/')
    textured_mesh = pbr.apply(mesh)
"""

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class PBRTextures:
    """PBR material maps for a mesh.

    Standard PBR workflow: albedo (base color), normal map,
    roughness, metallic. Optional: ambient occlusion, emissive.
    """

    albedo: np.ndarray | None = None  # (H, W, 3) uint8
    normal: np.ndarray | None = None  # (H, W, 3) uint8
    roughness: np.ndarray | None = None  # (H, W, 1) uint8
    metallic: np.ndarray | None = None  # (H, W, 1) uint8
    ao: np.ndarray | None = None  # (H, W, 1) uint8, ambient occlusion
    emissive: np.ndarray | None = None  # (H, W, 3) uint8

    @classmethod
    def from_pipeline_output(cls, pipeline_output) -> "PBRTextures":
        """Extract PBR textures from TRELLIS.2 / Hunyuan3D output.

        Pipeline outputs may include texture maps as part of the
        generated 3D asset.
        """
        pbr = cls()

        # Try extracting from TRELLIS.2's O-Voxel output
        if hasattr(pipeline_output, "texture"):
            tex = pipeline_output.texture
            if hasattr(tex, "cpu"):
                tex = tex.cpu().numpy()
            if tex.ndim == 3 and tex.shape[2] >= 3:
                pbr.albedo = (tex[:, :, :3] * 255).clip(0, 255).astype(np.uint8)

        # Try extracting from GLB materials
        if hasattr(pipeline_output, "visual") and hasattr(pipeline_output.visual, "material"):
            mat = pipeline_output.visual.material
            if hasattr(mat, "baseColorTexture") and mat.baseColorTexture is not None:
                pbr.albedo = np.array(mat.baseColorTexture)[:, :, :3]
            if hasattr(mat, "normalTexture") and mat.normalTexture is not None:
                pbr.normal = np.array(mat.normalTexture)[:, :, :3]

        return pbr

    @classmethod
    def from_directory(cls, path: str) -> "PBRTextures":
        """Load PBR textures from a directory of image files."""
        pbr = cls()
        p = Path(path)

        map_files = {
            "albedo": ["albedo.png", "basecolor.png", "diffuse.png", "color.png"],
            "normal": ["normal.png", "normal_map.png"],
            "roughness": ["roughness.png", "rough.png"],
            "metallic": ["metallic.png", "metalness.png", "metal.png"],
            "ao": ["ao.png", "ambient_occlusion.png", "occlusion.png"],
            "emissive": ["emissive.png", "emission.png"],
        }

        for attr, filenames in map_files.items():
            for fname in filenames:
                fpath = p / fname
                if fpath.exists():
                    img = np.array(Image.open(fpath))
                    setattr(pbr, attr, img)
                    break

        return pbr

    def save(self, output_dir: str):
        """Save PBR textures as PNG files."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        for name, data in [
            ("albedo", self.albedo),
            ("normal", self.normal),
            ("roughness", self.roughness),
            ("metallic", self.metallic),
            ("ao", self.ao),
            ("emissive", self.emissive),
        ]:
            if data is not None:
                Image.fromarray(data).save(out / f"{name}.png")

    def apply(self, mesh) -> "trimesh.Trimesh":
        """Apply PBR textures to a trimesh mesh.

        Creates a PBR material and assigns it to the mesh's visual.
        """
        import trimesh

        if self.albedo is None:
            return mesh

        material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.fromarray(self.albedo),
            normalTexture=Image.fromarray(self.normal) if self.normal is not None else None,
            roughnessFactor=0.5,
            metallicFactor=0.0,
        )

        if hasattr(mesh, "visual"):
            mesh.visual = trimesh.visual.TextureVisuals(material=material)

        return mesh

    @property
    def has_textures(self) -> bool:
        """Check if any texture maps are available."""
        return any([
            self.albedo is not None,
            self.normal is not None,
            self.roughness is not None,
            self.metallic is not None,
        ])
