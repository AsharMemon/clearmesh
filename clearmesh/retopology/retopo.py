"""Retopology via BPT, TreeMeshGPT, or QuadGPT.

Three neural backends are supported (pick via `method=`):

BPT (Blocked and Patchified Tokenization, Tencent, CVPR 2025):
  - Compresses mesh token sequences by ~75% via block-wise indexing
  - Generates triangle meshes up to ~8,000 faces
  - Input: mesh, Output: mesh (triangle retopology)

TreeMeshGPT (CVPR 2025):
  - Autoregressive tree sequencing with 9-bit discretization
  - Triangle meshes up to ~11,000 faces
  - Input: mesh, Output: mesh (triangle retopology)

QuadGPT (arxiv:2509.21420, ICLR 2026 submission):
  - First autoregressive **quad-mesh** generator, end-to-end native quads
  - Conditions on a point cloud with normals (40,960 samples) — *not* on the
    input mesh directly. The input mesh is used only as a surface to sample
    points from, so the output is a fresh retopology rather than an edit.
  - Generates mixed quad/triangle topology; trained on 500–20,000 face models.
  - Reference hyperparameters from the paper: top-k=10, top-p=0.95, T=0.5.
  - Context window: 36,864 tokens; speed ~230 tok/s on a single A100.

  As of this writing QuadGPT's code has not been released. The CLI shape
  used here (``run.py --input_pc <ply> --output <obj> --top_k 10 --top_p
  0.95 --temperature 0.5``) matches the paper's described API and is what
  the scaffold will invoke once the repo lands. `is_available()` checks
  for the install, so this backend stays gated until then.

For 3D printing: retopology is optional (slicers handle high poly fine).
For digital/game-ready: retopology produces efficient topology, reducing
file size while maintaining visual detail. QuadGPT is preferred for
subdivision-surface / sculpting workflows that need quads.

Usage:
    retopo = Retopologizer(method='quadgpt')  # or 'bpt', 'treemeshgpt'
    clean_mesh = retopo.retopologize(high_poly_mesh, target_faces=8000)
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import trimesh


# QuadGPT training/context bounds from the paper (arxiv:2509.21420).
QUADGPT_MAX_FACES = 20_000
QUADGPT_NUM_POINTS = 40_960  # Dense surface sampling used at inference.


class Retopologizer:
    """Clean topology generation via BPT, TreeMeshGPT, or QuadGPT.

    Args:
        method: 'bpt' | 'treemeshgpt' | 'quadgpt'
        model_dir: Base directory for model installations
        device: CUDA device
        quadgpt_top_k: Top-k sampling cutoff (paper default: 10).
        quadgpt_top_p: Nucleus sampling threshold (paper default: 0.95).
        quadgpt_temperature: Sampling temperature (paper default: 0.5).
        quadgpt_num_points: Points sampled from the input surface
            (paper default: 40,960).
    """

    VALID_METHODS = ("bpt", "treemeshgpt", "quadgpt")

    def __init__(
        self,
        method: str = "bpt",
        model_dir: str = "/mnt/data",
        device: str = "cuda",
        quadgpt_top_k: int = 10,
        quadgpt_top_p: float = 0.95,
        quadgpt_temperature: float = 0.5,
        quadgpt_num_points: int = QUADGPT_NUM_POINTS,
    ):
        if method not in self.VALID_METHODS:
            raise ValueError(
                f"Unknown retopology method {method!r}. "
                f"Valid: {self.VALID_METHODS}"
            )
        self.method = method
        self.model_dir = model_dir
        self.device = device

        self.paths = {
            "bpt": os.path.join(model_dir, "bpt"),
            "treemeshgpt": os.path.join(model_dir, "TreeMeshGPT"),
            "quadgpt": os.path.join(model_dir, "QuadGPT"),
        }

        self.quadgpt_top_k = quadgpt_top_k
        self.quadgpt_top_p = quadgpt_top_p
        self.quadgpt_temperature = quadgpt_temperature
        self.quadgpt_num_points = quadgpt_num_points

    def is_available(self) -> bool:
        """Check if the configured retopology model is installed."""
        return os.path.isdir(self.paths.get(self.method, ""))

    def retopologize(
        self,
        mesh: trimesh.Trimesh,
        target_faces: int = 8000,
    ) -> trimesh.Trimesh:
        """Generate clean topology from a high-poly mesh.

        Args:
            mesh: High-poly input mesh.
            target_faces: Target face count. Clamped per-backend:
                BPT ≤ 8K, TreeMeshGPT ≤ 11K, QuadGPT ≤ 20K.

        Returns:
            Clean mesh with efficient topology.
        """
        if self.method == "bpt" and os.path.isdir(self.paths["bpt"]):
            target_faces = min(target_faces, 8000)
            return self._bpt_retopo(mesh, target_faces)
        if self.method == "treemeshgpt" and os.path.isdir(self.paths["treemeshgpt"]):
            target_faces = min(target_faces, 11000)
            return self._treemeshgpt_retopo(mesh, target_faces)
        if self.method == "quadgpt" and os.path.isdir(self.paths["quadgpt"]):
            target_faces = min(target_faces, QUADGPT_MAX_FACES)
            return self._quadgpt_retopo(mesh, target_faces)

        print(
            f"No neural retopology available for method={self.method!r}. "
            "Using decimation fallback."
        )
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

    def _quadgpt_retopo(self, mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
        """QuadGPT: native autoregressive quad-mesh generation.

        QuadGPT is point-cloud-conditioned, not mesh-conditioned. We sample
        a dense point cloud with normals from the input mesh surface (the
        paper uses 40,960 points, 6D: xyz + normal), write it to a PLY, and
        invoke QuadGPT's CLI. The returned mesh replaces the input entirely.
        """
        qg_dir = self.paths["quadgpt"]

        # Sample surface points + normals. trimesh.sample.sample_surface
        # returns (N, 3) point positions and face_indices, from which we
        # look up the per-point face normal.
        n_pts = self.quadgpt_num_points
        points, face_idx = trimesh.sample.sample_surface(mesh, n_pts)
        normals = mesh.face_normals[face_idx]
        pc6 = np.concatenate(
            [np.asarray(points, dtype=np.float32), np.asarray(normals, dtype=np.float32)],
            axis=1,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            pc_path = os.path.join(tmpdir, "input_pc.ply")
            output_path = os.path.join(tmpdir, "output.obj")

            _write_point_cloud_ply(pc_path, pc6)

            subprocess.run(
                [
                    sys.executable,
                    "run.py",
                    "--input_pc", pc_path,
                    "--output", output_path,
                    "--target_faces", str(target_faces),
                    "--top_k", str(self.quadgpt_top_k),
                    "--top_p", str(self.quadgpt_top_p),
                    "--temperature", str(self.quadgpt_temperature),
                ],
                cwd=qg_dir,
                check=True,
                capture_output=True,
            )

            return trimesh.load(output_path, force="mesh")

    @staticmethod
    def _decimation_fallback(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
        """Simple quadric decimation when neural retopology is unavailable.

        Uses trimesh's built-in simplification (backed by
        ``fast_simplification``). Produces acceptable results for
        non-critical use cases. If the decimation backend is not
        installed, returns the mesh untouched rather than crashing — the
        rest of the pipeline keeps running.
        """
        if mesh.faces.shape[0] <= target_faces:
            return mesh

        try:
            return mesh.simplify_quadric_decimation(target_faces)
        except (ModuleNotFoundError, ImportError) as e:
            print(
                f"Decimation fallback unavailable ({e}). "
                "Install `fast_simplification` or a neural retopology "
                "backend. Returning original mesh."
            )
            return mesh


def _write_point_cloud_ply(path: str, pc6: np.ndarray) -> None:
    """Write a 6D (xyz + normal) point cloud as ASCII PLY.

    QuadGPT's paper specifies a point cloud with normals as conditioning.
    Using PLY here because it's the most universally supported point-cloud
    format and trimesh/open3d/plyfile can all read it back. If the
    published QuadGPT repo expects a different format (.xyz, .npy, etc.),
    swap the writer — the sampling logic stays the same.
    """
    if pc6.ndim != 2 or pc6.shape[1] != 6:
        raise ValueError(f"Expected (N, 6) point cloud, got {pc6.shape}")

    n = pc6.shape[0]
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float nx\n"
        "property float ny\n"
        "property float nz\n"
        "end_header\n"
    )
    with open(path, "w") as f:
        f.write(header)
        np.savetxt(f, pc6, fmt="%.6f")
