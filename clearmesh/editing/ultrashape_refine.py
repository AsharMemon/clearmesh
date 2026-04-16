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

        Args:
            coarse_mesh: Coarse mesh from TRELLIS.2 (or any source). Can be
                a Trimesh object or a path to a GLB/OBJ.
            reference_image: Reference image used for conditioning. Can be
                PIL.Image or a path.
            config: Optional UltraShapeConfig; defaults to UltraShape's
                published inference settings.

        Returns:
            Refined trimesh.Trimesh.
        """
        cfg = config or UltraShapeConfig()

        # Lazy load
        pipeline = self.pipeline

        # --- Prepare image ---
        if isinstance(reference_image, (str, Path)):
            reference_image = Image.open(str(reference_image))
        if cfg.remove_bg or reference_image.mode != "RGBA":
            from ultrashape.rembg import BackgroundRemover  # type: ignore
            if self._rembg is None:
                self._rembg = BackgroundRemover()
            reference_image = self._rembg(reference_image)

        # --- Prepare mesh ---
        # SharpEdgeSurfaceLoader accepts a path — if caller passed a trimesh,
        # write to a temp file first.
        if isinstance(coarse_mesh, (str, Path)):
            mesh_path = str(coarse_mesh)
            cleanup_path = None
        else:
            import tempfile
            tf = tempfile.NamedTemporaryFile(suffix=".glb", delete=False)
            coarse_mesh.export(tf.name)
            mesh_path = tf.name
            cleanup_path = tf.name

        try:
            surface = self._loader(mesh_path, normalize_scale=cfg.scale).to(
                self.device, dtype=torch.float16
            )
            pc = surface[:, :, :3]  # (B, N, 3)

            from ultrashape.utils import voxelize_from_point
            _, voxel_idx = voxelize_from_point(
                pc, cfg.num_latents, resolution=self._voxel_res
            )

            generator = torch.Generator(self.device).manual_seed(cfg.seed)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                mesh_list, _ = pipeline(
                    image=reference_image,
                    voxel_cond=voxel_idx,
                    generator=generator,
                    box_v=1.0,
                    mc_level=0.0,
                    octree_resolution=cfg.octree_res,
                    num_inference_steps=cfg.num_inference_steps,
                    num_chunks=cfg.chunk_size,
                )

            mesh_out = mesh_list[0]
            # Ensure it's a trimesh.Trimesh (UltraShape returns a compatible type)
            if not isinstance(mesh_out, trimesh.Trimesh):
                import numpy as np
                v = np.asarray(mesh_out.vertices)
                f = np.asarray(mesh_out.faces)
                mesh_out = trimesh.Trimesh(vertices=v, faces=f)
            return mesh_out

        finally:
            if cleanup_path and os.path.exists(cleanup_path):
                try:
                    os.unlink(cleanup_path)
                except OSError:
                    pass

    def unload(self):
        """Free VRAM by dropping the pipeline. Useful for sprint-after-demo."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
            self._loader = None
            self._config = None
            torch.cuda.empty_cache()
