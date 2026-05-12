#!/usr/bin/env python3
"""UltraShape-based Stage 2 refinement for ClearMesh.

Wraps UltraShape 1.0 (PKU-YuanGroup, arxiv:2512.21185) as a drop-in
replacement for our custom-trained RefinementDiT. UltraShape takes a
coarse mesh + reference image and produces a refined high-detail mesh
using voxel-conditioned DiT refinement. ClearMesh intentionally replaces
UltraShape's Hunyuan3D-2.1 coarse-mesh stage with TRELLIS.2, while keeping the
released UltraShape refinement settings aligned with the paper/repo defaults.

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
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch
import trimesh
from PIL import Image


ULTRASHAPE_PAPER_NUM_STEPS = 50
ULTRASHAPE_PAPER_OCTREE_RESOLUTION = 1024
ULTRASHAPE_PAPER_NUM_LATENTS = 32768
ULTRASHAPE_PAPER_CHUNK_SIZE = 8000
ULTRASHAPE_PAPER_NORMALIZE_SCALE = 0.99
ULTRASHAPE_PAPER_SEED = 42
ULTRASHAPE_SURFACE_UNIFORM_POINTS = 204800
ULTRASHAPE_SURFACE_SHARP_POINTS = 204800


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
        isolated_process: bool = True,
        subprocess_timeout_seconds: int = 7200,
        remove_background: bool = False,
    ):
        self.ultrashape_dir = ultrashape_dir
        self.checkpoint = checkpoint
        self.config_path = config_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.low_vram = low_vram
        self.isolated_process = isolated_process
        self.subprocess_timeout_seconds = subprocess_timeout_seconds
        self.remove_background = remove_background

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

        try:
            from ultrashape.utils.misc import instantiate_from_config
        except ImportError:
            from ultrashape.utils import instantiate_from_config

        if not os.path.exists(self.checkpoint):
            raise FileNotFoundError(
                f"UltraShape checkpoint not found at {self.checkpoint}. "
                "Download with: hf download infinith/UltraShape ultrashape_v1.pt"
            )

        config_path = self._resolve_config_path()
        print(f"Loading UltraShape from {config_path}...")
        t0 = time.time()
        config = OmegaConf.load(config_path)

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
            num_sharp_points=ULTRASHAPE_SURFACE_SHARP_POINTS,
            num_uniform_points=ULTRASHAPE_SURFACE_UNIFORM_POINTS,
        )

        print(f"UltraShape loaded in {time.time() - t0:.1f}s")

    @torch.no_grad()
    def refine(
        self,
        coarse_mesh: trimesh.Trimesh | str | Path,
        reference_image: Image.Image | str | Path,
        num_steps: int = ULTRASHAPE_PAPER_NUM_STEPS,
        octree_resolution: int = ULTRASHAPE_PAPER_OCTREE_RESOLUTION,
        num_latents: int = ULTRASHAPE_PAPER_NUM_LATENTS,
        chunk_size: int = ULTRASHAPE_PAPER_CHUNK_SIZE,
        scale: float = ULTRASHAPE_PAPER_NORMALIZE_SCALE,
        seed: int = ULTRASHAPE_PAPER_SEED,
    ) -> trimesh.Trimesh:
        """Refine a coarse mesh using the reference image.

        Args:
            coarse_mesh: Coarse mesh from TRELLIS.2 (trimesh object or GLB/OBJ path).
            reference_image: Original reference image (PIL Image or path).
                Should be RGBA with background removed.
            num_steps: Diffusion steps. 50 is quality, 25 is fast.
            octree_resolution: Marching cubes resolution; UltraShape's
                released inference script defaults to 1024.
            num_latents: Number of latent tokens; paper/repo inference uses
                32768.
            chunk_size: VAE decode chunk size; released inference default is
                8000.
            scale: Mesh normalization scale; released inference default is
                0.99.
            seed: Random seed.

        Returns:
            Refined trimesh.Trimesh.
        """
        if self.isolated_process:
            return self._refine_in_subprocess(
                coarse_mesh=coarse_mesh,
                reference_image=reference_image,
                num_steps=num_steps,
                octree_resolution=octree_resolution,
                num_latents=num_latents,
                chunk_size=chunk_size,
                scale=scale,
                seed=seed,
            )

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
            # Match UltraShape's official inference script: non-RGBA images go
            # through rembg instead of receiving an all-opaque alpha channel.
            if self.remove_background or image.mode != "RGBA":
                from ultrashape.rembg import BackgroundRemover

                image = BackgroundRemover()(image)
            else:
                image = image.convert("RGBA")

            # Import voxelization at call time (UltraShape path must be set)
            from ultrashape.surface_loaders import SharpEdgeSurfaceLoader  # noqa: F401
            try:
                from ultrashape.utils import voxelize_from_point
            except ImportError:
                from ultrashape.utils.voxelize import voxelize_from_point

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

    def _resolve_config_path(self) -> str:
        if self.config_path and os.path.exists(self.config_path):
            return self.config_path
        candidates = [
            os.path.join(self.ultrashape_dir, "configs", "infer_dit_refine.yaml"),
            os.path.join(self.ultrashape_dir, "configs", "infer_dit2.yaml"),
            os.path.join(self.ultrashape_dir, "configs", "infer_dit.yaml"),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                self.config_path = candidate
                return candidate
        raise FileNotFoundError(
            "No UltraShape inference config found. Tried: "
            + ", ".join(candidates)
        )

    def _refine_in_subprocess(
        self,
        coarse_mesh: trimesh.Trimesh | str | Path,
        reference_image: Image.Image | str | Path,
        *,
        num_steps: int,
        octree_resolution: int,
        num_latents: int,
        chunk_size: int,
        scale: float,
        seed: int,
    ) -> trimesh.Trimesh:
        """Run UltraShape in a fresh Python process.

        TRELLIS.2 and UltraShape can both register a pybind11 ``cuBVH`` type.
        Keeping UltraShape in a separate process avoids the double-registration
        failure when the Python API runs TRELLIS first and refinement second.
        """
        import clearmesh

        repo_root = Path(clearmesh.__file__).resolve().parent.parent
        runner = repo_root / "scripts" / "product" / "run_ultrashape_refinement.py"
        if not runner.exists():
            raise FileNotFoundError(f"UltraShape runner not found: {runner}")

        tmp_root = Path(tempfile.mkdtemp(prefix="clearmesh_ultrashape_"))
        try:
            coarse_path = tmp_root / "coarse.glb"
            image_path = tmp_root / "reference.png"
            output_dir = tmp_root / "out"
            output_path = output_dir / "refined.glb"

            if isinstance(coarse_mesh, (str, Path)):
                shutil.copyfile(str(coarse_mesh), coarse_path)
            else:
                coarse_mesh.export(coarse_path)

            if isinstance(reference_image, (str, Path)):
                Image.open(str(reference_image)).save(image_path)
            else:
                reference_image.save(image_path)

            command = [
                sys.executable,
                str(runner),
                "--mesh",
                str(coarse_path),
                "--image",
                str(image_path),
                "--output-dir",
                str(output_dir),
                "--output-name",
                output_path.name,
                "--ultrashape-dir",
                str(self.ultrashape_dir),
                "--checkpoint",
                str(self.checkpoint),
                "--num-steps",
                str(num_steps),
                "--octree-resolution",
                str(octree_resolution),
                "--num-latents",
                str(num_latents),
                "--chunk-size",
                str(chunk_size),
                "--scale",
                str(scale),
                "--seed",
                str(seed),
                "--direct",
            ]
            if self.config_path:
                command.extend(["--config-path", str(self.config_path)])
            if self.low_vram:
                command.append("--low-vram")
            if self.remove_background:
                command.append("--remove-bg")

            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=self.subprocess_timeout_seconds,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"UltraShape subprocess failed with exit {result.returncode}.\n"
                    f"stdout tail:\n{(result.stdout or '')[-2000:]}\n"
                    f"stderr tail:\n{(result.stderr or '')[-2000:]}"
                )
            if not output_path.exists():
                raise RuntimeError(
                    "UltraShape subprocess completed but did not write "
                    f"{output_path}. stdout tail:\n{(result.stdout or '')[-1000:]}"
                )
            mesh = trimesh.load(str(output_path), force="mesh")
            if isinstance(mesh, trimesh.Scene):
                mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
            return mesh
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)
