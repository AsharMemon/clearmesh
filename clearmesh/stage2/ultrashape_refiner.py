#!/usr/bin/env python3
"""UltraShape-based Stage 2 refinement for ClearMesh.

Wraps UltraShape 1.0 (PKU-YuanGroup, arxiv:2512.21185) as a drop-in
replacement for our custom-trained RefinementDiT. UltraShape takes a
coarse mesh + reference image and produces a refined high-detail mesh
using voxel-conditioned DiT refinement.

Validated April 2026 on TRELLIS.2 coarse outputs: adds ~14% detail,
~5 min on an L40, no custom training required.

Requirements:
  - UltraShape-1.0 repo cloned to `ultrashape_dir` (default: /workspace/UltraShape-1.0)
  - Checkpoint `ultrashape_v1.pt` downloaded from `infinith/UltraShape` HF repo
  - Dependencies: cubvh, pytorch_lightning, transformers, timm, pymeshlab, flash-attn

Usage:
    from clearmesh.stage2.ultrashape_refiner import UltraShapeRefiner

    refiner = UltraShapeRefiner(
        ultrashape_dir="/workspace/UltraShape-1.0",
        checkpoint="/workspace/checkpoints/ultrashape_v1.pt",
    )
    refined_mesh = refiner.refine(coarse_mesh, reference_image)
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

import torch
import trimesh
from PIL import Image


class UltraShapeRefiner:
    """Stage 2 refiner using UltraShape 1.0.

    This class is lazy: models are loaded on first call to `refine()`.
    After loading, the VAE, DiT, conditioner, scheduler, and image processor
    are held on device for repeated refinements.
    """

    def __init__(
        self,
        ultrashape_dir: str = "/workspace/UltraShape-1.0",
        checkpoint: str = "/workspace/checkpoints/ultrashape_v1.pt",
        config_path: str | None = None,
        device: str | None = None,
        low_vram: bool = False,
    ):
        self.ultrashape_dir = ultrashape_dir
        self.checkpoint = checkpoint
        self.config_path = config_path or os.path.join(
            ultrashape_dir, "configs", "infer_dit_refine.yaml"
        )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.low_vram = low_vram

        # Lazy-loaded components
        self._pipeline = None
        self._config = None
        self._surface_loader = None

    def _ensure_on_path(self) -> None:
        """Add UltraShape repo to sys.path if not already there."""
        if not os.path.isdir(self.ultrashape_dir):
            raise FileNotFoundError(
                f"UltraShape repo not found at {self.ultrashape_dir}. "
                "Clone from https://github.com/PKU-YuanGroup/UltraShape-1.0"
            )
        if self.ultrashape_dir not in sys.path:
            sys.path.insert(0, self.ultrashape_dir)

    def _load(self) -> None:
        """Load UltraShape pipeline components (one-time)."""
        if self._pipeline is not None:
            return

        self._ensure_on_path()

        from omegaconf import OmegaConf
        from ultrashape.pipelines import UltraShapePipeline
        from ultrashape.surface_loaders import SharpEdgeSurfaceLoader
        from ultrashape.utils.misc import instantiate_from_config

        if not os.path.exists(self.checkpoint):
            raise FileNotFoundError(
                f"UltraShape checkpoint not found at {self.checkpoint}. "
                "Download with: hf download infinith/UltraShape ultrashape_v1.pt"
            )

        print(f"Loading UltraShape from {self.config_path}...")
        t0 = time.time()
        config = OmegaConf.load(self.config_path)

        vae = instantiate_from_config(config.model.params.vae_config)
        dit = instantiate_from_config(config.model.params.dit_cfg)
        conditioner = instantiate_from_config(config.model.params.conditioner_config)
        scheduler = instantiate_from_config(config.model.params.scheduler_cfg)
        image_processor = instantiate_from_config(config.model.params.image_processor_cfg)

        # Load weights
        weights = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        vae.load_state_dict(weights["vae"], strict=True)
        dit.load_state_dict(weights["dit"], strict=True)
        conditioner.load_state_dict(weights["conditioner"], strict=True)

        # Move to device (unless low_vram mode)
        if not self.low_vram:
            vae = vae.eval().to(self.device)
            dit = dit.eval().to(self.device)
            conditioner = conditioner.eval().to(self.device)
        else:
            vae.eval()
            dit.eval()
            conditioner.eval()

        if hasattr(vae, "enable_flashvdm_decoder"):
            vae.enable_flashvdm_decoder()

        pipeline = UltraShapePipeline(
            vae=vae,
            model=dit,
            scheduler=scheduler,
            conditioner=conditioner,
            image_processor=image_processor,
        )
        if self.low_vram:
            pipeline.enable_model_cpu_offload()

        self._pipeline = pipeline
        self._config = config
        self._surface_loader = SharpEdgeSurfaceLoader(
            num_sharp_points=204800,
            num_uniform_points=204800,
        )

        print(f"UltraShape loaded in {time.time() - t0:.1f}s")

    @torch.no_grad()
    def refine(
        self,
        coarse_mesh: trimesh.Trimesh | str | Path,
        reference_image: Image.Image | str | Path,
        num_steps: int = 50,
        octree_resolution: int = 512,
        num_latents: int = 32768,
        chunk_size: int = 8000,
        scale: float = 0.99,
        seed: int = 42,
    ) -> trimesh.Trimesh:
        """Refine a coarse mesh using the reference image.

        Args:
            coarse_mesh: Coarse mesh from TRELLIS.2 (trimesh object or GLB/OBJ path).
            reference_image: Original reference image (PIL Image or path).
                Should be RGBA with background removed.
            num_steps: Diffusion steps. 50 is quality, 25 is fast.
            octree_resolution: Marching cubes resolution (512 or 1024).
            num_latents: Number of latent tokens (32768 standard).
            chunk_size: VAE decode chunk size.
            scale: Mesh normalization scale.
            seed: Random seed.

        Returns:
            Refined trimesh.Trimesh.
        """
        self._load()

        # Handle mesh input — UltraShape's SurfaceLoader needs a file path
        if isinstance(coarse_mesh, trimesh.Trimesh):
            with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as f:
                coarse_mesh.export(f.name)
                mesh_path = f.name
            cleanup_mesh = True
        else:
            mesh_path = str(coarse_mesh)
            cleanup_mesh = False

        try:
            # Handle image input
            if isinstance(reference_image, (str, Path)):
                image = Image.open(str(reference_image))
            else:
                image = reference_image
            if image.mode != "RGBA":
                image = image.convert("RGBA")

            # Import voxelization at call time (UltraShape path must be set)
            from ultrashape.surface_loaders import SharpEdgeSurfaceLoader  # noqa: F401
            from ultrashape.utils import voxelize_from_point

            # Voxelize the coarse mesh for conditioning
            voxel_res = self._config.model.params.vae_config.params.voxel_query_res
            surface = self._surface_loader(mesh_path, normalize_scale=scale)
            surface = surface.to(self.device, dtype=torch.float16)
            pc = surface[:, :, :3]
            _, voxel_idx = voxelize_from_point(pc, num_latents, resolution=voxel_res)

            # Run the pipeline
            generator = torch.Generator(self.device).manual_seed(seed)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                mesh_list, _ = self._pipeline(
                    image=image,
                    voxel_cond=voxel_idx,
                    generator=generator,
                    box_v=1.0,
                    mc_level=0.0,
                    octree_resolution=octree_resolution,
                    num_inference_steps=num_steps,
                    num_chunks=chunk_size,
                )

            refined = mesh_list[0]
            # UltraShape may return its own mesh wrapper; normalize to trimesh
            if not isinstance(refined, trimesh.Trimesh):
                refined = trimesh.Trimesh(
                    vertices=refined.vertices,
                    faces=refined.faces,
                )
            return refined

        finally:
            if cleanup_mesh:
                try:
                    os.unlink(mesh_path)
                except OSError:
                    pass

    def unload(self) -> None:
        """Free VRAM by deleting the pipeline."""
        self._pipeline = None
        self._config = None
        self._surface_loader = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
