"""TripoSF (SparseFlex) watertight refiner wrapper.

TripoSF is a sparse deformable FlexiCubes VAE that takes an input mesh and
produces a watertight 1024^3 reconstruction with arbitrary topology. We
use it as a final post-processing stage after UltraShape to convert the
open/hole-y output into a clean watertight mesh.

Upstream: https://github.com/VAST-AI-Research/TripoSF (MIT license)
Weights:  https://huggingface.co/VAST-AI/TripoSF (not gated)
Paper:    SparseFlex, arXiv:2503.21732 (ICCV 2025)

Runs via subprocess for the same reason UltraShape does: it has its own
CUDA extensions (torch-scatter, spconv) that can clash with TRELLIS.2's
compiled ops when both are loaded in the same process. Subprocess
isolation keeps each CUDA extension set to itself.

Usage:

    refiner = TripoSFRefiner(
        triposf_dir="/workspace/TripoSF",
        config_path="/workspace/TripoSF/configs/TripoSFVAE_1024.yaml",
    )
    watertight_mesh = refiner.refine(coarse_mesh)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import trimesh


@dataclass
class TripoSFConfig:
    """TripoSF runs at a fixed internal resolution determined by the
    checkpoint; the config YAML controls it. The only tunable we expose
    is an I/O format choice for the serialized intermediate mesh."""

    intermediate_format: str = "obj"  # "obj" or "glb"


class TripoSFRefiner:
    """Run TripoSF as a subprocess-isolated post-process.

    Instance state is minimal — we just remember paths. The actual VAE
    loads inside the subprocess each call (~3-5s overhead). For pipelines
    that call refine() many times in a row, pre-warming could be added
    via a persistent subprocess, but that's not needed for our once-
    per-edit demo pattern.
    """

    def __init__(
        self,
        triposf_dir: str = "/workspace/TripoSF",
        config_path: str | None = None,
    ):
        self.triposf_dir = Path(triposf_dir)
        self.config_path = config_path or str(
            self.triposf_dir / "configs" / "TripoSFVAE_1024.yaml"
        )

    def refine(
        self,
        coarse_mesh: trimesh.Trimesh | str | Path,
        config: TripoSFConfig | None = None,
    ) -> trimesh.Trimesh:
        """Convert an open / non-watertight mesh into a watertight 1024^3
        reconstruction via TripoSF VAE.

        Args:
            coarse_mesh: Input mesh (trimesh.Trimesh or path). Anything
                TripoSF's normalize_mesh can read (.obj, .glb, .ply).
            config: Optional TripoSFConfig; defaults are reasonable.

        Returns:
            trimesh.Trimesh with watertight topology (typically millions
            of verts at 1024^3).

        Raises:
            RuntimeError: if the subprocess fails (stderr tail in message).
            FileNotFoundError: if the TripoSF repo or checkpoint is missing.
        """
        cfg = config or TripoSFConfig()

        # Locate the subprocess script
        import clearmesh
        pkg_root = Path(clearmesh.__file__).resolve().parent.parent
        script_path = pkg_root / "scripts" / "run_triposf_subprocess.py"
        if not script_path.exists():
            raise FileNotFoundError(
                f"TripoSF subprocess script not found at {script_path}"
            )
        if not Path(self.config_path).exists():
            raise FileNotFoundError(
                f"TripoSF config not found at {self.config_path}"
            )

        tmp_root = Path(tempfile.mkdtemp(prefix="triposf_refine_"))
        try:
            # Serialize input
            ext = cfg.intermediate_format.lower().lstrip(".")
            assert ext in ("obj", "glb"), f"bad intermediate format: {ext}"
            in_path = tmp_root / f"input.{ext}"
            out_path = tmp_root / "watertight.obj"

            if isinstance(coarse_mesh, (str, Path)):
                shutil.copy(coarse_mesh, in_path)
            else:
                coarse_mesh.export(in_path)

            # Run subprocess
            cmd = [
                sys.executable,
                str(script_path),
                "--mesh-path", str(in_path),
                "--output", str(out_path),
                "--config", str(self.config_path),
                "--triposf-dir", str(self.triposf_dir),
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"TripoSF subprocess failed (exit {result.returncode}). "
                    f"stderr tail:\n{result.stderr[-1500:]}"
                )
            if not out_path.exists():
                raise RuntimeError(
                    f"TripoSF subprocess returned 0 but no output at {out_path}. "
                    f"stdout tail: {result.stdout[-500:]}"
                )
            return trimesh.load(str(out_path), force="mesh")

        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)
