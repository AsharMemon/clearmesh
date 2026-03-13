#!/usr/bin/env python3
"""
UltraShape refinement experiment for ClearMesh.

Takes a TRELLIS.2 coarse GLB + reference image, runs UltraShape's
voxel-conditioned DiT refinement, outputs a refined GLB.

Usage:
    python experiments/ultrashape/refine.py \
        --image path/to/image.png \
        --mesh path/to/coarse.glb \
        --output path/to/refined.glb

    # Lower VRAM mode (~16GB, octree_res=384):
    python experiments/ultrashape/refine.py \
        --image path/to/image.png \
        --mesh path/to/coarse.glb \
        --octree_res 384 --low_vram
"""

import argparse
import os
import sys
import time

import torch
import numpy as np


def setup_ultrashape_path():
    """Add UltraShape repo to Python path."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ultrashape_dir = os.path.join(script_dir, "UltraShape-1.0")
    if not os.path.isdir(ultrashape_dir):
        print(f"ERROR: UltraShape not found at {ultrashape_dir}")
        print("Run setup.sh first: bash experiments/ultrashape/setup.sh")
        sys.exit(1)
    if ultrashape_dir not in sys.path:
        sys.path.insert(0, ultrashape_dir)
    return ultrashape_dir


def load_config(ultrashape_dir, args):
    """Load and patch the UltraShape inference config."""
    from omegaconf import OmegaConf

    config_path = os.path.join(ultrashape_dir, "configs", "infer_dit_refine.yaml")
    if not os.path.exists(config_path):
        # Try alternative config names
        for name in ["infer_dit2.yaml", "infer_dit.yaml"]:
            alt = os.path.join(ultrashape_dir, "configs", name)
            if os.path.exists(alt):
                config_path = alt
                break
        else:
            print(f"ERROR: No inference config found in {ultrashape_dir}/configs/")
            print("Available configs:")
            for f in os.listdir(os.path.join(ultrashape_dir, "configs")):
                print(f"  {f}")
            sys.exit(1)

    config = OmegaConf.load(config_path)
    return config


def load_models(config, checkpoint_path, device, low_vram=False):
    """Load UltraShape models from checkpoint."""
    from omegaconf import OmegaConf

    # Dynamically import UltraShape's instantiation utility
    # UltraShape uses a common pattern: instantiate_from_config
    try:
        from ultrashape.utils import instantiate_from_config
    except ImportError:
        # Fallback: try the common diffusion model pattern
        from ultrashape.util import instantiate_from_config

    print("Loading models from checkpoint...")
    t0 = time.time()

    weights = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Instantiate models from config
    vae = instantiate_from_config(config.vae)
    dit = instantiate_from_config(config.model)
    conditioner = instantiate_from_config(config.conditioner)

    # Load state dicts
    # The checkpoint may store weights under different keys
    if "vae" in weights:
        vae.load_state_dict(weights["vae"], strict=False)
    if "dit" in weights:
        dit.load_state_dict(weights["dit"], strict=False)
    elif "model" in weights:
        dit.load_state_dict(weights["model"], strict=False)
    if "conditioner" in weights:
        conditioner.load_state_dict(weights["conditioner"], strict=False)

    # Set eval mode
    vae.eval()
    dit.eval()
    conditioner.eval()

    if not low_vram:
        vae = vae.to(device)
        dit = dit.to(device)
        conditioner = conditioner.to(device)

    # Build scheduler
    scheduler = instantiate_from_config(config.scheduler)

    # Build image processor
    image_processor = instantiate_from_config(config.image_processor)

    print(f"Models loaded in {time.time() - t0:.1f}s")
    return vae, dit, conditioner, scheduler, image_processor


def load_and_voxelize_mesh(mesh_path, ultrashape_dir, num_latents=32768, scale=0.99):
    """Load coarse mesh and convert to voxel conditioning."""
    try:
        from ultrashape.surface_loaders import SharpEdgeSurfaceLoader
        from ultrashape.utils.voxelize import voxelize_from_point
    except ImportError:
        # Try alternative import paths
        from surface_loaders import SharpEdgeSurfaceLoader
        from utils.voxelize import voxelize_from_point

    print(f"Loading coarse mesh: {mesh_path}")
    loader = SharpEdgeSurfaceLoader(
        num_sharp_points=204800,
        num_uniform_points=204800,
    )
    surface = loader(mesh_path, normalize_scale=scale)  # [1, 409600, 7]

    pc = surface[:, :, :3]  # [1, 409600, 3] - just xyz
    print(f"  Point cloud shape: {pc.shape}")
    print(f"  Coordinate range: [{pc.min():.3f}, {pc.max():.3f}]")

    # Voxelize
    _, voxel_idx = voxelize_from_point(pc, token_num=num_latents, resolution=128)
    print(f"  Voxel indices shape: {voxel_idx.shape}")
    print(f"  Unique voxels: {len(torch.unique(voxel_idx.reshape(-1, 3), dim=0))}")

    return voxel_idx


def process_image(image_path, image_processor, conditioner, device):
    """Process reference image through DINOv2 conditioner."""
    from PIL import Image

    print(f"Processing image: {image_path}")
    img = Image.open(image_path).convert("RGBA")
    print(f"  Image size: {img.size}")

    # Use UltraShape's image processor
    img_tensor = image_processor(img)
    if not isinstance(img_tensor, torch.Tensor):
        img_tensor = torch.tensor(img_tensor)
    img_tensor = img_tensor.to(device)

    # Get DINOv2 conditioning
    with torch.no_grad():
        cond = conditioner(img_tensor)

    print(f"  Conditioning shape: {cond.shape if isinstance(cond, torch.Tensor) else type(cond)}")
    return cond


@torch.no_grad()
def run_refinement(
    vae, dit, conditioner, scheduler, image_processor,
    image_path, mesh_path, output_path,
    num_steps=50, guidance_scale=5.0, num_latents=32768,
    octree_res=512, chunk_size=8000, scale=0.99,
    seed=42, device="cuda", low_vram=False,
):
    """Run the full UltraShape refinement pipeline."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Try to use UltraShape's pipeline class directly
    try:
        from ultrashape.pipelines import UltraShapePipeline

        pipeline = UltraShapePipeline(
            vae=vae, model=dit, scheduler=scheduler,
            conditioner=conditioner, image_processor=image_processor,
        )
        if low_vram:
            pipeline.enable_model_cpu_offload()
        else:
            pipeline = pipeline.to(device)

        print(f"\nRunning UltraShape pipeline...")
        print(f"  Steps: {num_steps}, Guidance: {guidance_scale}")
        print(f"  Latents: {num_latents}, Octree res: {octree_res}")

        t0 = time.time()
        result = pipeline(
            image=image_path,
            mesh=mesh_path,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            num_latents=num_latents,
            octree_resolution=octree_res,
            chunk_size=chunk_size,
            seed=seed,
            normalize_scale=scale,
        )
        elapsed = time.time() - t0
        print(f"  Refinement completed in {elapsed:.1f}s")

        # Export mesh
        if hasattr(result, "export"):
            result.export(output_path)
        elif isinstance(result, dict) and "mesh" in result:
            result["mesh"].export(output_path)
        else:
            # Try trimesh export
            import trimesh
            if isinstance(result, trimesh.Trimesh):
                result.export(output_path)
            else:
                print(f"  WARNING: Unexpected result type: {type(result)}")
                print(f"  Attempting to save anyway...")
                result.export(output_path)

        print(f"  Saved refined mesh to: {output_path}")
        return result, elapsed

    except (ImportError, TypeError, AttributeError) as e:
        print(f"Pipeline class not available or incompatible ({e})")
        print("Falling back to manual pipeline...")
        return run_refinement_manual(
            vae, dit, conditioner, scheduler, image_processor,
            image_path, mesh_path, output_path,
            num_steps, guidance_scale, num_latents,
            octree_res, chunk_size, scale, seed, device,
        )


@torch.no_grad()
def run_refinement_manual(
    vae, dit, conditioner, scheduler, image_processor,
    image_path, mesh_path, output_path,
    num_steps, guidance_scale, num_latents,
    octree_res, chunk_size, scale, seed, device,
):
    """Manual pipeline fallback if UltraShapePipeline import fails."""
    from PIL import Image

    try:
        from ultrashape.surface_loaders import SharpEdgeSurfaceLoader
        from ultrashape.utils.voxelize import voxelize_from_point
    except ImportError:
        from surface_loaders import SharpEdgeSurfaceLoader
        from utils.voxelize import voxelize_from_point

    torch.manual_seed(seed)

    # 1. Load and voxelize mesh
    loader = SharpEdgeSurfaceLoader(
        num_sharp_points=204800,
        num_uniform_points=204800,
    )
    surface = loader(mesh_path, normalize_scale=scale)
    pc = surface[:, :, :3]
    _, voxel_idx = voxelize_from_point(pc, token_num=num_latents, resolution=128)
    voxel_idx = voxel_idx.to(device)

    # 2. Process image
    img = Image.open(image_path).convert("RGBA")
    img_tensor = image_processor(img)
    if not isinstance(img_tensor, torch.Tensor):
        img_tensor = torch.tensor(img_tensor)
    img_tensor = img_tensor.to(device)
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)

    # Get DINOv2 conditioning
    cond = conditioner(img_tensor)
    if isinstance(cond, (tuple, list)):
        cond = cond[0]
    # Uncond = zeros for CFG
    uncond = torch.zeros_like(cond)

    # 3. Set up diffusion
    scheduler.set_timesteps(num_steps, device=device)
    timesteps = scheduler.timesteps

    # Init noise
    latents = torch.randn(1, num_latents, 64, device=device, dtype=torch.float32)

    # Duplicate voxel cond for CFG
    voxel_cond_cfg = torch.cat([voxel_idx, voxel_idx], dim=0)
    cond_cfg = torch.cat([cond, uncond], dim=0)

    print(f"\nRunning manual diffusion loop ({num_steps} steps)...")
    t0 = time.time()

    for i, t in enumerate(timesteps):
        latent_input = torch.cat([latents, latents], dim=0)
        timestep = t / scheduler.config.num_train_timesteps

        noise_pred = dit(
            latent_input,
            timestep.unsqueeze(0).expand(2),
            context=cond_cfg,
            voxel_cond=voxel_cond_cfg,
        )

        noise_cond, noise_uncond = noise_pred.chunk(2)
        noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

        latents = scheduler.step(noise_pred, t, latents).prev_sample

        if (i + 1) % 10 == 0:
            print(f"  Step {i+1}/{num_steps}")

    elapsed_diffusion = time.time() - t0
    print(f"  Diffusion done in {elapsed_diffusion:.1f}s")

    # 4. VAE decode
    print("Decoding latents to mesh...")
    t0 = time.time()

    latents = latents / vae.scale_factor
    decoded = vae.decode(latents, chunk_size=chunk_size)

    # Extract mesh via marching cubes
    mesh = vae.latents2mesh(
        decoded,
        octree_resolution=octree_res,
        mc_level=0.0,
        box_v=1.0,
    )

    elapsed_decode = time.time() - t0
    print(f"  Decode done in {elapsed_decode:.1f}s")

    # 5. Export
    import trimesh

    if hasattr(mesh, "vertices") and hasattr(mesh, "faces"):
        # Flip face winding (UltraShape convention)
        tmesh = trimesh.Trimesh(
            vertices=mesh.vertices,
            faces=mesh.faces[:, [0, 2, 1]],
        )
    elif isinstance(mesh, trimesh.Trimesh):
        tmesh = mesh
    else:
        # Try to extract from whatever format
        tmesh = trimesh.Trimesh(
            vertices=np.array(mesh.vertices),
            faces=np.array(mesh.faces),
        )

    tmesh.export(output_path)
    total_elapsed = elapsed_diffusion + elapsed_decode
    print(f"\nSaved refined mesh to: {output_path}")
    print(f"  Vertices: {len(tmesh.vertices)}, Faces: {len(tmesh.faces)}")
    print(f"  Total time: {total_elapsed:.1f}s")

    return tmesh, total_elapsed


def main():
    parser = argparse.ArgumentParser(
        description="Refine a TRELLIS.2 coarse mesh using UltraShape"
    )
    parser.add_argument("--image", required=True, help="Reference image (PNG/JPG)")
    parser.add_argument("--mesh", required=True, help="Coarse mesh from TRELLIS.2 (GLB/OBJ)")
    parser.add_argument("--output", default=None, help="Output path (default: outputs/<name>_refined.glb)")
    parser.add_argument("--checkpoint", default=None, help="Path to ultrashape_v1.pt")
    parser.add_argument("--steps", type=int, default=50, help="Diffusion steps (default: 50)")
    parser.add_argument("--guidance", type=float, default=5.0, help="CFG guidance scale (default: 5.0)")
    parser.add_argument("--num_latents", type=int, default=32768, help="Number of latent tokens (default: 32768)")
    parser.add_argument("--octree_res", type=int, default=512, help="Marching cubes resolution (default: 512)")
    parser.add_argument("--chunk_size", type=int, default=8000, help="VAE decode chunk size (default: 8000)")
    parser.add_argument("--scale", type=float, default=0.99, help="Mesh normalization scale (default: 0.99)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--low_vram", action="store_true", help="Enable CPU offloading for low VRAM")
    parser.add_argument("--device", default="cuda", help="Device (default: cuda)")
    args = parser.parse_args()

    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ultrashape_dir = setup_ultrashape_path()

    # Find checkpoint
    if args.checkpoint is None:
        args.checkpoint = os.path.join(script_dir, "checkpoints", "ultrashape_v1.pt")
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found at {args.checkpoint}")
        print("Run setup.sh first: bash experiments/ultrashape/setup.sh")
        sys.exit(1)

    # Set output path
    if args.output is None:
        os.makedirs(os.path.join(script_dir, "outputs"), exist_ok=True)
        mesh_name = os.path.splitext(os.path.basename(args.mesh))[0]
        args.output = os.path.join(script_dir, "outputs", f"{mesh_name}_refined.glb")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print("=" * 60)
    print("UltraShape Refinement Experiment")
    print("=" * 60)
    print(f"  Image:      {args.image}")
    print(f"  Coarse mesh: {args.mesh}")
    print(f"  Output:     {args.output}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Device:     {args.device}")
    print(f"  Low VRAM:   {args.low_vram}")
    print(f"  Steps:      {args.steps}")
    print(f"  Guidance:   {args.guidance}")
    print(f"  Latents:    {args.num_latents}")
    print(f"  Octree res: {args.octree_res}")
    print()

    # Load config
    config = load_config(ultrashape_dir, args)

    # Load models
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    vae, dit, conditioner, scheduler, image_processor = load_models(
        config, args.checkpoint, device, low_vram=args.low_vram
    )

    # Run refinement
    result, elapsed = run_refinement(
        vae, dit, conditioner, scheduler, image_processor,
        args.image, args.mesh, args.output,
        num_steps=args.steps, guidance_scale=args.guidance,
        num_latents=args.num_latents, octree_res=args.octree_res,
        chunk_size=args.chunk_size, scale=args.scale,
        seed=args.seed, device=device, low_vram=args.low_vram,
    )

    print(f"\nDone! Total pipeline time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
