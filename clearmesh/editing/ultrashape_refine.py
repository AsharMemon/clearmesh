"""UltraShape-1.0 refinement wrapper.

UltraShape (Peng Cheng Lab / PKU-YuanGroup) is a coarse-to-fine mesh
refiner that takes a reference image + a coarse mesh and produces a
higher-fidelity refined mesh via voxel-conditioned diffusion.

Upstream: https://github.com/PKU-YuanGroup/UltraShape-1.0
Weights:  https://huggingface.co/infinith/UltraShape

License note: UltraShape inherits the **Tencent Hunyuan 3D 2.1 Community
License** (non-commercial, territorial restrictions exclude EU/UK/SK).
Any use of this module must comply with that license. In particular:
  - Outputs cannot be used to train competing AI models.
  - Over 1M monthly active users requires separate Tencent approval.
  - Not usable at all in EU, UK, or South Korea.
If you ship a commercial product, do NOT enable UltraShape refinement.

Usage (after install_ultrashape.sh has run on the pod):

    from clearmesh.editing.ultrashape_refine import UltraShapeRefiner

    refiner = UltraShapeRefiner(
        ultrashape_dir="/workspace/UltraShape-1.0",
        ckpt_path="/workspace/UltraShape-1.0/checkpoints/ultrashape_v1.pt",
        config_path="/workspace/UltraShape-1.0/configs/infer_dit2.yaml",
    )
    refined_mesh = refiner.refine(
        coarse_mesh=trellis2_output_mesh,
        reference_image=source_pil_image,
        num_steps=50,
    )

Integration with Easy3EEditor:
    set ``EditOptions.enable_ultrashape=True`` and optionally
    ``ultrashape_ckpt`` / ``ultrashape_config``; the editor will call
    ``refiner.refine`` automatically after the main decode.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import trimesh
from PIL import Image


@dataclass
class UltraShapeConfig:
    """Knobs that map 1:1 to UltraShape's run.sh arguments."""

    num_inference_steps: int = 50
    num_latents: int = 32768
    chunk_size: int = 8000
    octree_res: int = 1024  # marching cubes resolution for final mesh
    scale: float = 0.99
    seed: int = 42
    remove_bg: bool = False
    low_vram: bool = False


class UltraShapeRefiner:
    """Wraps UltraShapePipeline and manages its lifecycle.

    Loads lazily (UltraShape adds ~2 GB of VRAM) so you can create the
    refiner at editor startup without paying the memory cost unless it's
    actually used.
    """

    def __init__(
        self,
        ultrashape_dir: str = "/workspace/UltraShape-1.0",
        ckpt_path: str | None = None,
        config_path: str | None = None,
        device: str = "cuda",
    ):
        self.ultrashape_dir = Path(ultrashape_dir)
        self.ckpt_path = ckpt_path or str(self.ultrashape_dir / "checkpoints" / "ultrashape_v1.pt")
        self.config_path = config_path or str(self.ultrashape_dir / "configs" / "infer_dit_refine.yaml")
        self.device = device

        self._pipeline = None
        self._loader = None
        self._config = None
        self._voxel_res = None
        self._rembg = None

    @property
    def pipeline(self):
        """Lazy-load the UltraShape pipeline + its components."""
        if self._pipeline is not None:
            return self._pipeline

        # Make sure ultrashape is importable
        if str(self.ultrashape_dir) not in sys.path:
            sys.path.insert(0, str(self.ultrashape_dir))

        from omegaconf import OmegaConf
        from ultrashape.utils.misc import instantiate_from_config
        from ultrashape.surface_loaders import SharpEdgeSurfaceLoader
        from ultrashape.pipelines import UltraShapePipeline

        print(f"[ultrashape] Loading config from {self.config_path}...")
        config = OmegaConf.load(self.config_path)
        self._config = config

        print("[ultrashape] Instantiating VAE, DiT, Conditioner, Scheduler...")
        vae = instantiate_from_config(config.model.params.vae_config)
        dit = instantiate_from_config(config.model.params.dit_cfg)
        conditioner = instantiate_from_config(config.model.params.conditioner_config)
        scheduler = instantiate_from_config(config.model.params.scheduler_cfg)
        image_processor = instantiate_from_config(config.model.params.image_processor_cfg)

        print(f"[ultrashape] Loading weights from {self.ckpt_path}...")
        weights = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        vae.load_state_dict(weights["vae"], strict=True)
        dit.load_state_dict(weights["dit"], strict=True)
        conditioner.load_state_dict(weights["conditioner"], strict=True)

        vae.eval().to(self.device)
        dit.eval().to(self.device)
        conditioner.eval().to(self.device)
        if hasattr(vae, "enable_flashvdm_decoder"):
            vae.enable_flashvdm_decoder()

        self._pipeline = UltraShapePipeline(
            vae=vae, model=dit, scheduler=scheduler,
            conditioner=conditioner, image_processor=image_processor,
        )

        self._voxel_res = config.model.params.vae_config.params.voxel_query_res
        self._loader = SharpEdgeSurfaceLoader(
            num_sharp_points=204800,
            num_uniform_points=204800,
        )
        print(f"[ultrashape] Loaded. voxel_query_res={self._voxel_res}")
        return self._pipeline

    def refine(
        self,
        coarse_mesh: trimesh.Trimesh | str | Path,
        reference_image: Image.Image | str | Path,
        config: UltraShapeConfig | None = None,
    ) -> trimesh.Trimesh:
        """Refine a coarse mesh using the reference image.

        Runs UltraShape in a **subprocess** to avoid the cuBVH double-
        registration conflict: TRELLIS.2's ``cumesh`` wheel bundles a
        pybind11 class named ``cuBVH``, and UltraShape's standalone
        ``cubvh`` registers the same symbol, causing the second import
        in the parent process to fail. Running in a fresh subprocess
        means only one ``cuBVH`` gets registered per process.

        Subprocess overhead: ~5s for imports, plus the actual refinement
        time (~20-40s for 50 steps at octree_res=1024).

        Args:
            coarse_mesh: Coarse mesh from TRELLIS.2 (or any source). Can be
                a Trimesh object or a path to a GLB/OBJ.
            reference_image: Reference image used for conditioning. Can be
                PIL.Image or a path.
            config: Optional UltraShapeConfig; defaults to UltraShape's
                published inference settings.

        Returns:
            Refined trimesh.Trimesh.

        Raises:
            RuntimeError: if the subprocess exits non-zero (error message
                from subprocess stderr included).
            FileNotFoundError: if checkpoint, config, or subprocess script
                is missing.
        """
        cfg = config or UltraShapeConfig()

        # Locate the subprocess script (scripts/run_ultrashape_subprocess.py)
        # relative to the clearmesh package root.
        import clearmesh
        pkg_root = Path(clearmesh.__file__).resolve().parent.parent
        subprocess_script = pkg_root / "scripts" / "run_ultrashape_subprocess.py"
        if not subprocess_script.exists():
            raise FileNotFoundError(
                f"UltraShape subprocess script not found at {subprocess_script}. "
                "Ensure the full clearmesh repo is on the pod, not just the package."
            )

        # --- Prepare workspace in a temp dir (inputs + outputs on disk) ---
        import tempfile
        import subprocess as _subprocess

        tmp_root = Path(tempfile.mkdtemp(prefix="ultrashape_"))
        try:
            coarse_mesh_path = tmp_root / "coarse.glb"
            image_path = tmp_root / "reference.png"
            output_path = tmp_root / "refined.glb"

            # Serialize inputs
            if isinstance(coarse_mesh, (str, Path)):
                import shutil
                shutil.copy(coarse_mesh, coarse_mesh_path)
            else:
                coarse_mesh.export(coarse_mesh_path)

            if isinstance(reference_image, (str, Path)):
                Image.open(str(reference_image)).save(image_path)
            else:
                # Coerce to RGB before saving so PIL doesn't drop alpha info
                reference_image.save(image_path)

            # --- Run subprocess ---
            cmd = [
                sys.executable,
                str(subprocess_script),
                "--coarse-mesh", str(coarse_mesh_path),
                "--image", str(image_path),
                "--output", str(output_path),
                "--ckpt", str(self.ckpt_path),
                "--config", str(self.config_path),
                "--ultrashape-dir", str(self.ultrashape_dir),
                "--steps", str(cfg.num_inference_steps),
                "--octree-res", str(cfg.octree_res),
                "--seed", str(cfg.seed),
                "--num-latents", str(cfg.num_latents),
                "--chunk-size", str(cfg.chunk_size),
                "--scale", str(cfg.scale),
            ]
            if cfg.remove_bg:
                cmd.append("--remove-bg")

            print(f"[ultrashape] Launching subprocess: {' '.join(cmd[:3])} ...")
            result = _subprocess.run(
                cmd, capture_output=True, text=True, check=False,
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"UltraShape subprocess failed (exit {result.returncode}). "
                    f"stderr:\n{result.stderr[-2000:]}"
                )

            # --- Load refined mesh ---
            if not output_path.exists():
                raise RuntimeError(
                    f"UltraShape subprocess returned 0 but no output at {output_path}. "
                    f"stdout: {result.stdout[-500:]}"
                )

            return trimesh.load(str(output_path), force="mesh")

        finally:
            import shutil
            shutil.rmtree(tmp_root, ignore_errors=True)

    def unload(self):
        """Free VRAM by dropping the pipeline. Useful for sprint-after-demo."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
            self._loader = None
            self._config = None
            torch.cuda.empty_cache()
