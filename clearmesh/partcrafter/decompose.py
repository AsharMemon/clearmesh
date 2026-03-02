"""PartCrafter part decomposition: single image → multiple semantic 3D parts.

PartCrafter (NeurIPS 2025) is the first structured 3D generative model that
jointly synthesizes multiple semantically meaningful, geometrically distinct
meshes from a single RGB image.

Architecture:
  - Built on TripoSG backbone
  - Compositional latent space with disentangled per-part tokens
  - Learnable part identity embeddings
  - Hierarchical attention: local per-part detail + global structural consistency

Output: Separate mesh files per part (sword, armor, head, body, base, etc.)
Each part is independently processed through subsequent pipeline stages,
enabling selective NDC on hard surfaces and organic smoothing on faces.

VRAM: ~48GB+ (A100 80GB recommended)
Inference: 3-4 minutes per model

Usage:
    decomposer = PartDecomposer()
    parts = decomposer.decompose(image, num_parts=6)
    for part in parts:
        print(f"{part.label}: {part.mesh.vertices.shape[0]} verts")
"""

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image


@dataclass
class MeshPart:
    """A single decomposed mesh part with semantic label."""

    mesh: trimesh.Trimesh
    label: str  # e.g. "sword", "armor", "head", "body", "base"
    part_id: int
    is_hard_surface: bool = False  # Hint for downstream processing

    @property
    def category(self) -> str:
        """Classify part as 'hard' or 'organic' for selective processing."""
        hard_keywords = {"sword", "shield", "weapon", "armor", "helmet", "base", "pedestal", "gem", "staff"}
        if self.is_hard_surface:
            return "hard"
        if any(kw in self.label.lower() for kw in hard_keywords):
            return "hard"
        return "organic"


class PartDecomposer:
    """Part decomposition via PartCrafter.

    Generates separable parts in one shot without pre-segmentation.
    Can reconstruct occluded/invisible parts using learned priors.

    Args:
        partcrafter_dir: Path to PartCrafter installation
        device: CUDA device
    """

    def __init__(
        self,
        partcrafter_dir: str = "/mnt/data/PartCrafter",
        device: str = "cuda",
    ):
        self.partcrafter_dir = partcrafter_dir
        self.device = device
        self._model = None

    def is_available(self) -> bool:
        """Check if PartCrafter is installed."""
        return os.path.isdir(self.partcrafter_dir)

    def decompose(
        self,
        image: Image.Image | str,
        num_parts: int | None = None,
        min_parts: int = 2,
        max_parts: int = 16,
    ) -> list[MeshPart]:
        """Decompose an image into semantic 3D parts.

        Args:
            image: Input PIL Image or path
            num_parts: Hint for expected part count (None = auto-detect)
            min_parts: Minimum parts to generate
            max_parts: Maximum parts to generate

        Returns:
            List of MeshPart objects, one per semantic part
        """
        if isinstance(image, str):
            image = Image.open(image)

        if not self.is_available():
            raise RuntimeError(
                f"PartCrafter not found at {self.partcrafter_dir}. "
                "Install with: scripts/setup/install_partcrafter.sh"
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save input image
            input_path = os.path.join(tmpdir, "input.png")
            image.save(input_path)
            output_dir = os.path.join(tmpdir, "parts")
            os.makedirs(output_dir)

            # Build command
            cmd = [
                sys.executable,
                "run.py",
                "--input", input_path,
                "--output_dir", output_dir,
            ]
            if num_parts is not None:
                cmd.extend(["--num_parts", str(num_parts)])

            # Run PartCrafter
            subprocess.run(
                cmd,
                cwd=self.partcrafter_dir,
                check=True,
                capture_output=True,
            )

            # Collect output parts
            parts = self._load_parts(output_dir)

        return parts

    def _load_parts(self, output_dir: str) -> list[MeshPart]:
        """Load decomposed parts from PartCrafter output directory."""
        parts = []
        output_path = Path(output_dir)

        # PartCrafter outputs numbered OBJ files with optional label metadata
        for i, mesh_file in enumerate(sorted(output_path.glob("*.obj"))):
            try:
                mesh = trimesh.load(str(mesh_file), force="mesh")
            except Exception:
                continue

            # Try to extract label from filename or metadata
            label = mesh_file.stem  # e.g. "part_0_sword" or "0"
            # Clean up label
            label = label.replace("part_", "").strip("_0123456789").strip("_") or f"part_{i}"

            parts.append(MeshPart(
                mesh=mesh,
                label=label,
                part_id=i,
            ))

        # Auto-classify hard surfaces
        for part in parts:
            part.is_hard_surface = self._classify_surface(part)

        return parts

    @staticmethod
    def _classify_surface(part: MeshPart) -> bool:
        """Heuristic classification of hard vs organic surface.

        Hard surfaces have flatter faces and sharper dihedral angles.
        """
        mesh = part.mesh
        if mesh.faces.shape[0] < 10:
            return False

        # Check face normal variance — hard surfaces have more uniform normals
        face_normals = mesh.face_normals
        normal_std = np.std(face_normals, axis=0).mean()

        # Hard surfaces: lower normal variance (flatter patches)
        return normal_std < 0.4

    def decompose_or_passthrough(
        self,
        image: Image.Image | str,
        mesh: trimesh.Trimesh,
        **kwargs,
    ) -> list[MeshPart]:
        """Decompose if PartCrafter is available, otherwise wrap mesh as single part.

        This allows the rest of the pipeline to always work with a list of parts.
        """
        if self.is_available():
            try:
                return self.decompose(image, **kwargs)
            except Exception as e:
                print(f"PartCrafter failed: {e}. Using single-part fallback.")

        # Fallback: single part
        return [MeshPart(mesh=mesh, label="whole", part_id=0)]
