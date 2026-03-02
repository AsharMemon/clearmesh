"""Retopology via BPT or TreeMeshGPT for clean artist-quality topology.

BPT (Blocked and Patchified Tokenization, Tencent, CVPR 2025):
  - Compresses mesh token sequences by ~75% via block-wise indexing
  - Generates meshes exceeding 8,000 faces (vs 800 for MeshAnything)
  - Handles complex miniature geometry

TreeMeshGPT (CVPR 2025):
  - Autoregressive tree sequencing with 9-bit discretization
  - Up to 11,000 faces for models needing higher face counts

For 3D printing: retopology is optional (slicers handle high poly fine).
For digital/game-ready: retopology produces efficient quad topology,
reducing file size while maintaining visual detail.

Usage:
    retopo = Retopologizer(method='bpt')
    clean_mesh = retopo.retopologize(high_poly_mesh, target_faces=8000)
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import trimesh


class Retopologizer:
    """Clean topology generation via BPT or TreeMeshGPT.

    Args:
        method: 'bpt' or 'treemeshgpt'
        model_dir: Base directory for model installations
        device: CUDA device
    """

    def __init__(
        self,
        method: str = "bpt",
        model_dir: str = "/mnt/data",
        device: str = "cuda",
    ):
        self.method = method
        self.model_dir = model_dir
        self.device = device

        self.paths = {
            "bpt": os.path.join(model_dir, "bpt"),
            "treemeshgpt": os.path.join(model_dir, "TreeMeshGPT"),
        }

    def is_available(self) -> bool:
        """Check if retopology model is installed."""
        return os.path.isdir(self.paths.get(self.method, ""))

    def retopologize(
        self,
        mesh: trimesh.Trimesh,
        target_faces: int = 8000,
    ) -> trimesh.Trimesh:
        """Generate clean topology from high-poly mesh.

        Args:
            mesh: High-poly input mesh
            target_faces: Target face count (BPT max ~8K, TreeMeshGPT ~11K)

        Returns:
            Clean mesh with efficient topology
        """
        if self.method == "bpt" and os.path.isdir(self.paths["bpt"]):
            target_faces = min(target_faces, 8000)
            return self._bpt_retopo(mesh, target_faces)
        elif self.method == "treemeshgpt" and os.path.isdir(self.paths["treemeshgpt"]):
            target_faces = min(target_faces, 11000)
            return self._treemeshgpt_retopo(mesh, target_faces)
        else:
            print("No neural retopology available. Using decimation fallback.")
            return self._decimation_fallback(mesh, target_faces)

    def _bpt_retopo(self, mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
        """BPT: blocked and patchified tokenization retopology."""
        bpt_dir = self.paths["bpt"]

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.obj")
            output_path = os.path.join(tmpdir, "output.obj")
            mesh.export(input_path)

            subprocess.run(
                [
                    sys.executable,
                    "run.py",
                    "--input", input_path,
                    "--output", output_path,
                    "--target_faces", str(target_faces),
                ],
                cwd=bpt_dir,
                check=True,
                capture_output=True,
            )

            return trimesh.load(output_path, force="mesh")

    def _treemeshgpt_retopo(self, mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
        """TreeMeshGPT: autoregressive tree sequencing retopology."""
        tmg_dir = self.paths["treemeshgpt"]

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.obj")
            output_path = os.path.join(tmpdir, "output.obj")
            mesh.export(input_path)

            subprocess.run(
                [
                    sys.executable,
                    "run.py",
                    "--input", input_path,
                    "--output", output_path,
                    "--max_faces", str(target_faces),
                ],
                cwd=tmg_dir,
                check=True,
                capture_output=True,
            )

            return trimesh.load(output_path, force="mesh")

    @staticmethod
    def _decimation_fallback(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
        """Simple quadric decimation when neural retopology is unavailable.

        Uses trimesh's built-in simplification. Produces acceptable results
        for non-critical use cases.
        """
        if mesh.faces.shape[0] <= target_faces:
            return mesh

        # Simplify using quadric decimation
        ratio = target_faces / mesh.faces.shape[0]
        simplified = mesh.simplify_quadric_decimation(target_faces)

        return simplified
