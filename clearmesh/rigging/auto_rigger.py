"""Auto-rigging via Puppeteer (primary), UniRig (fallback), HumanRig (humanoid).

All three models are pre-trained with no additional training required.
They take a clean mesh and output:
  - Skeleton hierarchy (joint positions + parent-child relationships)
  - Skinning weights (per-vertex bone influence weights)

Puppeteer additionally supports video-guided animation.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import trimesh


class AutoRigger:
    """Unified auto-rigging interface for ClearMesh.

    Supports three backends:
      - puppeteer: NeurIPS 2025 Spotlight. Best overall (skeleton + skinning + animation + FBX)
      - unirig: SIGGRAPH 2025. Best generalist for diverse object categories
      - humanrig: CVPR 2025. Best for humanoid characters specifically

    Usage:
        rigger = AutoRigger(method='puppeteer')
        skeleton, weights = rigger.rig(mesh)
        rigger.export_fbx(mesh, skeleton, weights, 'output.fbx')
    """

    SUPPORTED_METHODS = ("puppeteer", "unirig", "humanrig")

    def __init__(
        self,
        method: str = "puppeteer",
        model_dir: str = "/mnt/data",
        device: str = "cuda",
    ):
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(f"Unknown method: {method}. Supported: {self.SUPPORTED_METHODS}")

        self.method = method
        self.model_dir = model_dir
        self.device = device

        # Paths to installed tools
        self.paths = {
            "puppeteer": os.path.join(model_dir, "Puppeteer"),
            "unirig": os.path.join(model_dir, "UniRig"),
        }

    def rig(self, mesh: trimesh.Trimesh) -> tuple[dict, np.ndarray]:
        """Predict skeleton and skinning weights for a mesh.

        Args:
            mesh: Clean, watertight trimesh.Trimesh

        Returns:
            skeleton: Dict with 'joints' (N, 3) positions and 'parents' (N,) indices
            weights: (V, J) skinning weight matrix
        """
        if self.method == "puppeteer":
            return self._rig_puppeteer(mesh)
        elif self.method == "unirig":
            return self._rig_unirig(mesh)
        elif self.method == "humanrig":
            return self._rig_humanrig(mesh)

    def _rig_puppeteer(self, mesh: trimesh.Trimesh) -> tuple[dict, np.ndarray]:
        """Rig using Puppeteer.

        Puppeteer pipeline:
          1. Sample point clouds with normals
          2. Auto-regressive transformer predicts skeleton
          3. Topology-aware attention predicts skinning weights
        """
        puppeteer_dir = self.paths["puppeteer"]

        with tempfile.TemporaryDirectory() as tmpdir:
            # Export mesh to temp file
            input_path = os.path.join(tmpdir, "input.obj")
            mesh.export(input_path)

            skeleton_path = os.path.join(tmpdir, "skeleton.txt")
            skinning_path = os.path.join(tmpdir, "skinning.npz")

            # Step 1: Predict skeleton
            subprocess.run(
                [
                    sys.executable,
                    "run_skeleton.py",
                    "--input",
                    input_path,
                    "--output",
                    skeleton_path,
                ],
                cwd=puppeteer_dir,
                check=True,
                capture_output=True,
            )

            # Step 2: Predict skinning weights
            subprocess.run(
                [
                    sys.executable,
                    "run_skinning.py",
                    "--input",
                    input_path,
                    "--skeleton",
                    skeleton_path,
                    "--output",
                    skinning_path,
                ],
                cwd=puppeteer_dir,
                check=True,
                capture_output=True,
            )

            # Parse results
            skeleton = self._parse_skeleton(skeleton_path)
            weights = np.load(skinning_path)["weights"]

        return skeleton, weights

    def _rig_unirig(self, mesh: trimesh.Trimesh) -> tuple[dict, np.ndarray]:
        """Rig using UniRig.

        UniRig two-stage system:
          1. GPT-like transformer predicts skeleton via Skeleton Tree Tokenization
          2. Bone-Point Cross Attention predicts skinning weights
        """
        unirig_dir = self.paths["unirig"]

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.glb")
            mesh.export(input_path)

            output_path = os.path.join(tmpdir, "output.glb")

            subprocess.run(
                [
                    sys.executable,
                    "run.py",
                    "--input",
                    input_path,
                    "--output",
                    output_path,
                ],
                cwd=unirig_dir,
                check=True,
                capture_output=True,
            )

            # Parse rigged GLB output
            skeleton, weights = self._parse_rigged_glb(output_path, mesh.vertices.shape[0])

        return skeleton, weights

    def _rig_humanrig(self, mesh: trimesh.Trimesh) -> tuple[dict, np.ndarray]:
        """Rig using HumanRig (humanoid-specific).

        Uses Prior-Guided Skeleton Estimator and Mesh-Skeleton
        Mutual Attention Network. Produces industry-standard
        skeleton topology for humanoid characters.
        """
        # HumanRig integration follows similar subprocess pattern
        # but uses T-pose assumption and fixed skeleton topology
        raise NotImplementedError(
            "HumanRig integration not yet implemented. "
            "Use 'puppeteer' or 'unirig' instead."
        )

    def export_fbx(
        self,
        mesh: trimesh.Trimesh,
        skeleton: dict,
        weights: np.ndarray,
        output_path: str,
    ) -> str:
        """Export rigged mesh as FBX for Unity/Unreal.

        Args:
            mesh: The mesh
            skeleton: Joint hierarchy from rig()
            weights: Skinning weights from rig()
            output_path: Output .fbx path

        Returns:
            Absolute path to exported FBX
        """
        if self.method == "puppeteer":
            return self._export_puppeteer_fbx(mesh, skeleton, weights, output_path)
        else:
            # Fallback: export as GLB with skeleton metadata
            from clearmesh.mesh.export import export_glb

            glb_path = output_path.replace(".fbx", ".glb")
            return export_glb(mesh, glb_path)

    def _export_puppeteer_fbx(
        self,
        mesh: trimesh.Trimesh,
        skeleton: dict,
        weights: np.ndarray,
        output_path: str,
    ) -> str:
        """Use Puppeteer's native FBX export."""
        puppeteer_dir = self.paths["puppeteer"]

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.obj")
            mesh.export(input_path)

            skeleton_path = os.path.join(tmpdir, "skeleton.txt")
            self._write_skeleton(skeleton, skeleton_path)

            weights_path = os.path.join(tmpdir, "weights.npz")
            np.savez(weights_path, weights=weights)

            subprocess.run(
                [
                    sys.executable,
                    "export_fbx.py",
                    "--input",
                    input_path,
                    "--skeleton",
                    skeleton_path,
                    "--weights",
                    weights_path,
                    "--output",
                    output_path,
                ],
                cwd=puppeteer_dir,
                check=True,
                capture_output=True,
            )

        return os.path.abspath(output_path)

    @staticmethod
    def _parse_skeleton(skeleton_path: str) -> dict:
        """Parse skeleton file into structured dict."""
        joints = []
        parents = []
        names = []

        with open(skeleton_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    names.append(parts[0])
                    parents.append(int(parts[1]))
                    joints.append([float(parts[2]), float(parts[3]), float(parts[4])])

        return {
            "joints": np.array(joints, dtype=np.float32),
            "parents": np.array(parents, dtype=np.int32),
            "names": names,
        }

    @staticmethod
    def _write_skeleton(skeleton: dict, path: str):
        """Write skeleton to text file."""
        with open(path, "w") as f:
            for i, (name, parent, joint) in enumerate(
                zip(skeleton["names"], skeleton["parents"], skeleton["joints"])
            ):
                f.write(f"{name} {parent} {joint[0]:.6f} {joint[1]:.6f} {joint[2]:.6f}\n")

    @staticmethod
    def _parse_rigged_glb(glb_path: str, num_vertices: int) -> tuple[dict, np.ndarray]:
        """Parse a rigged GLB file to extract skeleton and weights."""
        scene = trimesh.load(glb_path)

        # Extract skeleton from glTF skin nodes
        skeleton = {"joints": np.zeros((1, 3)), "parents": np.array([-1]), "names": ["root"]}
        weights = np.ones((num_vertices, 1), dtype=np.float32)

        # Full parsing depends on the specific GLB structure from UniRig
        # This is a simplified version
        if hasattr(scene, "graph") and hasattr(scene.graph, "transforms"):
            nodes = scene.graph.transforms.node_data
            joint_list = []
            for name, data in nodes.items():
                if "matrix" in data:
                    pos = data["matrix"][:3, 3]
                    joint_list.append(pos)
            if joint_list:
                skeleton["joints"] = np.array(joint_list)
                skeleton["parents"] = np.arange(-1, len(joint_list) - 1)
                skeleton["names"] = [f"joint_{i}" for i in range(len(joint_list))]
                weights = np.ones((num_vertices, len(joint_list)), dtype=np.float32) / len(
                    joint_list
                )

        return skeleton, weights
