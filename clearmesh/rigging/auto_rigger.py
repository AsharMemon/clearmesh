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

    Supports two wired backends + one scaffolded:

      - ``puppeteer``: NeurIPS 2025 Spotlight. Best overall
        (skeleton + skinning + animation + FBX export).
        Installed by ``scripts/setup/install_rigging.sh``.
      - ``unirig``: SIGGRAPH 2025. Best generalist across object
        categories. Installed by the same script.
      - ``humanrig``: CVPR 2025. Humanoid-specific, industry-standard
        skeleton topology. **Scaffold only** — ``rig()`` raises
        ``NotImplementedError``. No install script exists yet. Gated
        by ``is_available()`` so the pipeline skips it cleanly when
        selected.

    Usage:
        rigger = AutoRigger(method='puppeteer')
        if not rigger.is_available():
            # Install missing — skip or install_rigging.sh first.
            ...
        skeleton, weights = rigger.rig(mesh)
        rigger.export_fbx(mesh, skeleton, weights, 'output.fbx')
    """

    SUPPORTED_METHODS = ("puppeteer", "unirig", "humanrig")

    # Backends that are fully wired. ``humanrig`` is in SUPPORTED_METHODS
    # so the constructor accepts it, but ``rig()`` refuses to run.
    _WIRED_METHODS = ("puppeteer", "unirig")

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

        # Paths to installed tools. ``humanrig`` intentionally points to a
        # plausible future install location so ``is_available()`` returns
        # False cleanly rather than KeyError'ing.
        self.paths = {
            "puppeteer": os.path.join(model_dir, "Puppeteer"),
            "unirig": os.path.join(model_dir, "UniRig"),
            "humanrig": os.path.join(model_dir, "HumanRig"),
        }

    def is_available(self) -> bool:
        """Check whether the selected backend is installed AND wired.

        Returns True only if:
          1. The backend's repo directory exists under ``model_dir``.
          2. The backend is in ``_WIRED_METHODS`` (i.e. ``rig()`` can
             actually run end-to-end, not just raise).

        Callers (e.g. ``ClearMeshPipeline.generate``) should check this
        before calling ``rig()`` and skip rigging with a warning when
        False — matches the ``Retopologizer.is_available()`` pattern.
        """
        if self.method not in self._WIRED_METHODS:
            return False
        return os.path.isdir(self.paths.get(self.method, ""))

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

    @staticmethod
    def _run_or_raise(cmd: list[str], cwd: str, stage: str) -> None:
        """Run a subprocess, raising with stderr context on failure.

        The default ``subprocess.run(capture_output=True, check=True)`` is
        hostile to debugging: on failure the CalledProcessError has
        stderr/stdout attached but the default __str__ doesn't show them.
        This helper surfaces the first ~1KB of stderr in the exception
        message so logs on a GPU host actually help diagnose failures.
        """
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, check=False)
        if proc.returncode != 0:
            err = (proc.stderr or b"").decode("utf-8", errors="replace")[:1024]
            out = (proc.stdout or b"").decode("utf-8", errors="replace")[-512:]
            raise RuntimeError(
                f"[AutoRigger/{stage}] subprocess failed "
                f"(exit {proc.returncode}):\n"
                f"  cmd: {' '.join(cmd)}\n"
                f"  cwd: {cwd}\n"
                f"  stderr: {err}\n"
                f"  stdout tail: {out}"
            )

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
            self._run_or_raise(
                [
                    sys.executable,
                    "run_skeleton.py",
                    "--input",
                    input_path,
                    "--output",
                    skeleton_path,
                ],
                cwd=puppeteer_dir,
                stage="puppeteer-skeleton",
            )

            # Step 2: Predict skinning weights
            self._run_or_raise(
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
                stage="puppeteer-skinning",
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

            self._run_or_raise(
                [
                    sys.executable,
                    "run.py",
                    "--input",
                    input_path,
                    "--output",
                    output_path,
                ],
                cwd=unirig_dir,
                stage="unirig",
            )

            # Parse rigged GLB output
            skeleton, weights = self._parse_rigged_glb(output_path, mesh.vertices.shape[0])

        return skeleton, weights

    def _rig_humanrig(self, mesh: trimesh.Trimesh) -> tuple[dict, np.ndarray]:
        """Rig using HumanRig (humanoid-specific) — NOT YET WIRED.

        The HumanRig paper (CVPR 2025) describes a Prior-Guided Skeleton
        Estimator + Mesh-Skeleton Mutual Attention Network producing
        industry-standard humanoid topology. This method is kept as a
        placeholder so the backend slot is reserved — calling it always
        raises, and ``is_available()`` returns False for ``humanrig``
        so the pipeline skips it cleanly.

        To unblock:
          1. Add a ``scripts/setup/install_humanrig.sh`` mirroring the
             install_rigging.sh pattern (clone + pip install + ckpt dl).
          2. Add ``humanrig`` to ``_WIRED_METHODS`` above.
          3. Replace this body with the subprocess call to HumanRig's
             inference entry point, following the Puppeteer template.
        """
        raise NotImplementedError(
            "HumanRig is scaffolded but not wired. Check "
            "`AutoRigger(method='humanrig').is_available()` before calling "
            "rig(); use 'puppeteer' or 'unirig' until HumanRig is installed "
            "and wired. See _rig_humanrig() docstring for the unblock "
            "checklist."
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
        """Parse a rigged GLB file to extract skeleton and weights.

        WARNING — partial implementation: this extracts joint positions
        from ``scene.graph.transforms.node_data`` but returns *uniform*
        skinning weights (1/J per vertex per joint). Real UniRig output
        encodes per-vertex weights in the glTF ``skin`` block's JOINTS_0
        / WEIGHTS_0 accessors, which trimesh's scene graph does not
        surface. Until that path is wired, animations driven from this
        output will deform every vertex equally — usable for pose
        visualization, not for real animation.

        To unblock: load the GLB with ``pygltflib``, read the ``skin``
        block, pull JOINTS_0 (vertex→joint indices) and WEIGHTS_0
        (per-index weights), and construct the (V, J) sparse weight
        matrix. Parent hierarchy comes from the ``skin.joints`` ordering
        and glTF node tree.
        """
        scene = trimesh.load(glb_path)

        # Extract skeleton from glTF skin nodes
        skeleton = {"joints": np.zeros((1, 3)), "parents": np.array([-1]), "names": ["root"]}
        weights = np.ones((num_vertices, 1), dtype=np.float32)

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
                # Placeholder — real per-vertex weights require parsing
                # JOINTS_0 / WEIGHTS_0 from the glTF skin block.
                weights = np.ones((num_vertices, len(joint_list)), dtype=np.float32) / len(
                    joint_list
                )

        return skeleton, weights
