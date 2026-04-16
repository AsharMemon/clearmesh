#!/usr/bin/env python
"""Subprocess-isolated UltraShape refiner.

Run in a separate process from TRELLIS.2 to avoid the cuBVH double-
registration conflict: both trellis2.cumesh and ultrashape's cubvh
register a pybind11 class named ``cuBVH``, and the second registration
fails with "generic_type: type 'cuBVH' is already registered!".

Inputs/outputs are files on disk so no shared memory:
    --coarse-mesh  <path.glb>       input mesh
    --image        <path.png>       reference image
    --output       <path.glb>       refined mesh (written)
    --ckpt         <path.pt>
    --config       <path.yaml>
    --steps        <int>
    --octree-res   <int>
    --seed         <int>
    --ultrashape-dir <path>

Exit code 0 on success; non-zero on failure with reason on stderr.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coarse-mesh", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ultrashape-dir", required=True)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--octree-res", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-latents", type=int, default=32768)
    ap.add_argument("--chunk-size", type=int, default=8000)
    ap.add_argument("--scale", type=float, default=0.99)
    ap.add_argument("--remove-bg", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, args.ultrashape_dir)

    try:
        import torch
        from PIL import Image
        from omegaconf import OmegaConf
        from ultrashape.pipelines import UltraShapePipeline
        from ultrashape.surface_loaders import SharpEdgeSurfaceLoader
        from ultrashape.utils.misc import instantiate_from_config
        from ultrashape.utils import voxelize_from_point

        device = "cuda"

        print(f"[us-subproc] Loading config from {args.config}", flush=True)
        config = OmegaConf.load(args.config)
        voxel_res = config.model.params.vae_config.params.voxel_query_res

        print("[us-subproc] Instantiating VAE/DiT/Conditioner...", flush=True)
        vae = instantiate_from_config(config.model.params.vae_config)
        dit = instantiate_from_config(config.model.params.dit_cfg)
        conditioner = instantiate_from_config(config.model.params.conditioner_config)
        scheduler = instantiate_from_config(config.model.params.scheduler_cfg)
        image_processor = instantiate_from_config(config.model.params.image_processor_cfg)

        print(f"[us-subproc] Loading weights from {args.ckpt}", flush=True)
        weights = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        vae.load_state_dict(weights["vae"], strict=True)
        dit.load_state_dict(weights["dit"], strict=True)
        conditioner.load_state_dict(weights["conditioner"], strict=True)

        vae.eval().to(device)
        dit.eval().to(device)
        conditioner.eval().to(device)
        if hasattr(vae, "enable_flashvdm_decoder"):
            vae.enable_flashvdm_decoder()

        pipeline = UltraShapePipeline(
            vae=vae, model=dit, scheduler=scheduler,
            conditioner=conditioner, image_processor=image_processor,
        )

        loader = SharpEdgeSurfaceLoader(
            num_sharp_points=204800,
            num_uniform_points=204800,
        )

        # --- Prepare image ---
        print(f"[us-subproc] Loading image {args.image}", flush=True)
        image = Image.open(args.image)
        if args.remove_bg or image.mode != "RGBA":
            from ultrashape.rembg import BackgroundRemover
            image = BackgroundRemover()(image)

        # --- Sample surface points from coarse mesh, then voxelize ---
        print(f"[us-subproc] Loading coarse mesh {args.coarse_mesh}", flush=True)
        surface = loader(args.coarse_mesh, normalize_scale=args.scale).to(
            device, dtype=torch.float16
        )
        pc = surface[:, :, :3]
        _, voxel_idx = voxelize_from_point(pc, args.num_latents, resolution=voxel_res)

        # --- Run refinement ---
        print(f"[us-subproc] Refining with {args.steps} steps, octree={args.octree_res}", flush=True)
        generator = torch.Generator(device).manual_seed(args.seed)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            mesh_list, _ = pipeline(
                image=image,
                voxel_cond=voxel_idx,
                generator=generator,
                box_v=1.0,
                mc_level=0.0,
                octree_resolution=args.octree_res,
                num_inference_steps=args.steps,
                num_chunks=args.chunk_size,
            )

        mesh_out = mesh_list[0]

        # --- Save ---
        print(f"[us-subproc] Saving to {args.output}", flush=True)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        if hasattr(mesh_out, "export"):
            mesh_out.export(args.output)
        else:
            import numpy as np
            import trimesh
            v = np.asarray(mesh_out.vertices)
            f = np.asarray(mesh_out.faces)
            trimesh.Trimesh(vertices=v, faces=f).export(args.output)

        print(f"[us-subproc] DONE verts={len(mesh_out.vertices)}", flush=True)
        return 0

    except Exception as e:
        print(f"[us-subproc] FATAL: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
