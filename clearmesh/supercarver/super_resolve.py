"""Geometry super-resolution: coarse mesh → fine surface detail.

SuperCarver (2025) adds sculpted micro-detail (scale mail, fabric weave,
skin pores, weapon engravings) as actual geometric displacement — critical
for 3D printing where texture maps are irrelevant.

Two-stage framework:
  1. Normal map prediction: renders coarse mesh from multiple views, uses
     prior-guided normal diffusion to predict high-detail normal maps
  2. Inverse rendering: carves detail back into mesh geometry via noise-
     resistant inverse rendering using a deformable distance field

Fallback: CraftsMan3D (CVPR 2025) provides similar coarse-to-fine refinement
with an automatic global geometry refiner. Less powerful but available now.

Usage:
    resolver = GeometrySuperResolver()
    detailed_mesh = resolver.super_resolve(coarse_mesh)
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import trimesh


class GeometrySuperResolver:
    """Geometry super-resolution via SuperCarver or CraftsMan3D fallback.

    Adds fine geometric detail to coarse meshes. Applied per-part after
    NDC/FlexiCubes extraction. Best on organic parts (faces, bodies, cloaks).
    Can be skipped on simple geometric parts (flat bases, simple shields).

    Args:
        method: 'supercarver' or 'craftsman3d'
        model_dir: Base directory for model installations
        device: CUDA device
    """

    def __init__(
        self,
        method: str = "supercarver",
        model_dir: str = "/mnt/data",
        device: str = "cuda",
    ):
        self.method = method
        self.model_dir = model_dir
        self.device = device

        self.paths = {
            "supercarver": os.path.join(model_dir, "SuperCarver"),
            "craftsman3d": os.path.join(model_dir, "CraftsMan3D"),
        }

    def is_available(self) -> bool:
        """Check if the selected method is installed."""
        primary = os.path.isdir(self.paths.get(self.method, ""))
        fallback = os.path.isdir(self.paths.get("craftsman3d", ""))
        return primary or fallback

    def super_resolve(
        self,
        mesh: trimesh.Trimesh,
        detail_level: str = "medium",
        num_views: int = 8,
    ) -> trimesh.Trimesh:
        """Add fine geometric detail to a coarse mesh.

        Args:
            mesh: Coarse input mesh
            detail_level: 'low' | 'medium' | 'high' — controls subdivision depth
            num_views: Number of views for normal map prediction

        Returns:
            High-detail mesh with geometric micro-detail
        """
        if self.method == "supercarver" and os.path.isdir(self.paths["supercarver"]):
            return self._supercarver_resolve(mesh, detail_level, num_views)
        elif os.path.isdir(self.paths.get("craftsman3d", "")):
            return self._craftsman_resolve(mesh, detail_level)
        else:
            print("No super-resolution model available. Using subdivision fallback.")
            return self._subdivision_fallback(mesh, detail_level)

    def _supercarver_resolve(
        self,
        mesh: trimesh.Trimesh,
        detail_level: str,
        num_views: int,
    ) -> trimesh.Trimesh:
        """SuperCarver: normal prediction → inverse rendering → geometry carving."""
        sc_dir = self.paths["supercarver"]

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.obj")
            output_path = os.path.join(tmpdir, "output.obj")
            mesh.export(input_path)

            detail_map = {"low": "1", "medium": "2", "high": "3"}

            subprocess.run(
                [
                    sys.executable,
                    "run.py",
                    "--input", input_path,
                    "--output", output_path,
                    "--detail_level", detail_map.get(detail_level, "2"),
                    "--num_views", str(num_views),
                ],
                cwd=sc_dir,
                check=True,
                capture_output=True,
            )

            return trimesh.load(output_path, force="mesh")

    def _craftsman_resolve(
        self,
        mesh: trimesh.Trimesh,
        detail_level: str,
    ) -> trimesh.Trimesh:
        """CraftsMan3D fallback: automatic global geometry refiner."""
        cm_dir = self.paths["craftsman3d"]

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.obj")
            output_path = os.path.join(tmpdir, "output.obj")
            mesh.export(input_path)

            subprocess.run(
                [
                    sys.executable,
                    "refine.py",
                    "--input", input_path,
                    "--output", output_path,
                ],
                cwd=cm_dir,
                check=True,
                capture_output=True,
            )

            return trimesh.load(output_path, force="mesh")

    @staticmethod
    def _subdivision_fallback(
        mesh: trimesh.Trimesh,
        detail_level: str,
    ) -> trimesh.Trimesh:
        """Simple subdivision as fallback when no super-resolution model is available.

        Loop subdivision increases face count and smooths geometry. Not as good
        as learned approaches but better than nothing.
        """
        iterations = {"low": 1, "medium": 2, "high": 3}.get(detail_level, 2)

        result = mesh.copy()
        for _ in range(iterations):
            result = result.subdivide()

        return result

    def should_apply(self, part_label: str) -> bool:
        """Decide whether to apply super-resolution to a part.

        Apply to organic parts where detail matters (faces, bodies, cloaks).
        Skip on simple geometric parts (flat bases, simple shields).
        """
        skip_labels = {"base", "pedestal", "stand", "platform", "ground"}
        return part_label.lower() not in skip_labels
