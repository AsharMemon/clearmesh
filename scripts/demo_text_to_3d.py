"""Text-to-3D demo using the full polish stack.

Flow:
  prompt  --(FLUX.1-schnell)-->  image
  image   --(TRELLIS.2-4B)-->    raw mesh (512->1024 cascade, decoded)
  mesh    --(UltraShape)-->      refined mesh  [optional, non-commercial license]
  mesh    --(polish_mesh)-->     Taubin + vertex merge
  mesh    --(full_print_prep)--> cumesh CUDA repair + orient + decimate
  mesh    --(export)-->          GLB

This script intentionally does NOT go through clearmesh.text_to_3d.TextTo3D,
which routes through the older ClearMeshPipeline that expects a trained
Stage 2 DiT checkpoint. This demo uses TRELLIS.2 directly and then applies
the newer polish chain that was validated in the editing sessions.

Run on pod:
    cd /workspace/clearmesh
    python scripts/demo_text_to_3d.py \
        --prompt "a steampunk gearbox with brass fittings" \
        --out /workspace/demo_text_to_3d
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Assume pod layout
for p in (
    "/workspace/clearmesh",
    "/workspace/UltraShape-1.0",
    "/workspace/TRELLIS.2",
):
    if p not in sys.path and os.path.isdir(p):
        sys.path.insert(0, p)

import torch
import trimesh
from PIL import Image


def _t2i_load(
    model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
    dtype: torch.dtype = torch.bfloat16,
    offload: bool = True,
):
    """Load a text-to-image pipeline.

    Supported families (auto-detected from model_id):
      - FLUX*                    gated, Apache 2.0 w/ HF access
      - Qwen/Qwen-Image*         2025, Apache 2.0, strong single-object prior
      - stable-diffusion-3*      SAI community licence, best composition
      - PixArt-Sigma*            2024 Apache 2.0 DiT
      - SDXL                     2023 default fallback (worst single-object bias)

    SDXL's training corpus is heavy on 2D art / collages, which is why
    prompts like "steampunk gearbox" produce floating-gear sprites
    instead of a single 3D object. Qwen-Image and SD3.5 handle isolated-
    object framing far better.
    """
    print(f"[t2i] loading {model_id}...")
    mid = model_id.lower()
    is_flux = "flux" in mid
    is_qwen = "qwen" in mid
    is_sd3 = "stable-diffusion-3" in mid

    if is_flux:
        from diffusers import FluxPipeline
        pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=dtype)
    elif is_qwen:
        # Qwen-Image uses QwenImagePipeline which ships in diffusers>=0.34
        try:
            from diffusers import QwenImagePipeline
            pipe = QwenImagePipeline.from_pretrained(model_id, torch_dtype=dtype)
        except ImportError:
            from diffusers import AutoPipelineForText2Image
            pipe = AutoPipelineForText2Image.from_pretrained(
                model_id, torch_dtype=dtype, use_safetensors=True,
            )
    else:
        from diffusers import AutoPipelineForText2Image
        pipe = AutoPipelineForText2Image.from_pretrained(
            model_id,
            torch_dtype=dtype,
            variant="fp16" if dtype == torch.float16 else None,
            use_safetensors=True,
        )

    if offload:
        # CPU offload keeps ~10 GB free for TRELLIS.2 + UltraShape
        try:
            pipe.enable_model_cpu_offload()
        except Exception:
            pipe.to("cuda")
    else:
        pipe.to("cuda")
    pipe._is_flux = is_flux
    pipe._is_qwen = is_qwen
    pipe._is_sd3 = is_sd3
    return pipe


def _t2i_image(pipe, prompt: str, seed: int, w: int = 1024, h: int = 1024) -> Image.Image:
    """Generate a single 3D-friendly image from text.

    The modifier string matters a LOT. Key constraints: one single
    object, full body, studio product-shot framing, no 2D collage
    layout, no multiple views in one image. Without these, SDXL in
    particular tends to produce flat grids of objects for mechanical
    prompts.
    """
    modifiers = [
        "single object", "full body", "centered composition",
        "studio product photography", "plain white background",
        "sharp focus", "3/4 view", "one isolated subject only",
    ]
    negative = (
        "collage, grid, multiple objects, duplicates, floating parts, "
        "montage, side by side, diptych, triptych, text, watermark, blur"
    )
    enhanced = f"{prompt}, {', '.join(modifiers)}"
    g = torch.Generator(device="cuda").manual_seed(seed)
    print(f"[t2i] prompt:   {enhanced!r}")
    print(f"[t2i] negative: {negative!r}")
    t0 = time.time()
    is_flux = getattr(pipe, "_is_flux", False)
    is_qwen = getattr(pipe, "_is_qwen", False)
    is_sd3 = getattr(pipe, "_is_sd3", False)
    if is_flux:
        kwargs = dict(num_inference_steps=4, guidance_scale=0.0)
        # FLUX doesn't use negative prompts
        neg_kwargs = {}
    elif is_qwen:
        # Qwen-Image defaults: 50 steps, true_cfg_scale=4.0
        kwargs = dict(num_inference_steps=50, true_cfg_scale=4.0)
        neg_kwargs = dict(negative_prompt=negative)
    elif is_sd3:
        kwargs = dict(num_inference_steps=28, guidance_scale=7.0)
        neg_kwargs = dict(negative_prompt=negative)
    else:
        # SDXL / PixArt default
        kwargs = dict(num_inference_steps=25, guidance_scale=7.0)
        neg_kwargs = dict(negative_prompt=negative)
    try:
        out = pipe(
            prompt=enhanced,
            width=w,
            height=h,
            generator=g,
            **neg_kwargs,
            **kwargs,
        )
    except TypeError:
        # Some pipelines don't accept negative_prompt or true_cfg_scale
        out = pipe(
            prompt=enhanced, width=w, height=h, generator=g, **kwargs,
        )
    print(f"[t2i] done in {time.time() - t0:.1f}s")
    return out.images[0]


def _trellis_mesh(image: Image.Image, out_dir: str):
    print("[trellis2] loading pipeline...")
    t0 = time.time()
    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    pipe = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipe.cuda()
    print(f"[trellis2] loaded in {time.time() - t0:.1f}s")

    t0 = time.time()
    outputs = pipe.run(image, seed=42, sparse_structure_sampler_params={"steps": 12})
    dt = time.time() - t0
    print(f"[trellis2] run() done in {dt:.1f}s")

    # outputs is a dict per TRELLIS.2 API; we want the decoded mesh
    # pipe.run returns {"mesh": [...], ...} with one mesh per input image
    if isinstance(outputs, dict) and "mesh" in outputs:
        mesh = outputs["mesh"][0]
    elif isinstance(outputs, list):
        mesh = outputs[0]
    else:
        mesh = outputs

    # TRELLIS.2 sometimes returns an ovoxel wrapper; fall through to trimesh
    if not isinstance(mesh, trimesh.Trimesh):
        from clearmesh.mesh.extraction import extract_from_ovoxel

        mesh = extract_from_ovoxel(mesh)

    raw_path = os.path.join(out_dir, "01_trellis2_raw.glb")
    mesh.export(raw_path)
    print(f"[trellis2] raw mesh: {len(mesh.vertices):,}v/{len(mesh.faces):,}f -> {raw_path}")
    return mesh, pipe


def _ultrashape_refine(
    mesh,
    image: Image.Image,
    out_dir: str,
    ultrashape_dir: str = "/workspace/UltraShape-1.0",
    ckpt: str | None = None,
    config: str | None = None,
):
    if not os.path.isdir(ultrashape_dir):
        print(f"[ultrashape] skipped (dir not found: {ultrashape_dir})")
        return mesh
    print("[ultrashape] refining...")
    t0 = time.time()
    from clearmesh.editing.ultrashape_refine import UltraShapeRefiner

    refiner = UltraShapeRefiner(
        ultrashape_dir=ultrashape_dir,
        ckpt_path=ckpt or f"{ultrashape_dir}/checkpoints/ultrashape_v1.pt",
        config_path=config or f"{ultrashape_dir}/configs/infer_dit_refine.yaml",
    )
    refined = refiner.refine(
        coarse_mesh=mesh,
        reference_image=image,
        num_steps=50,
        octree_res=1024,
    )
    dt = time.time() - t0
    print(f"[ultrashape] done in {dt:.1f}s")
    refined_path = os.path.join(out_dir, "02_ultrashape.glb")
    refined.export(refined_path)
    print(f"[ultrashape] refined: {len(refined.vertices):,}v/{len(refined.faces):,}f -> {refined_path}")
    return refined


def _r2_polish(mesh, out_dir: str, target_faces: int = 2_000_000):
    """R2 polish order:

      1. polish_mesh (Taubin + vertex merge)
      2. quadric_decimate FIRST to get down to ~2M faces — this makes
         the CUDA repair step tractable (repair on 6.8M is painful even
         on CUDA, and the full_print_preparation path falls back to a
         trimesh-only CPU repair that is essentially infinite on this
         size).
      3. repair_mesh_cuda for watertight cleanup.
    """
    from clearmesh.mesh.repair import polish_mesh, quadric_decimate, repair_mesh_cuda

    print("[r2] polish_mesh (Taubin + vertex merge)...")
    t0 = time.time()
    polished = polish_mesh(
        mesh,
        merge_digits=5,
        taubin_iterations=3,
        taubin_lamb=0.5,
        taubin_nu=-0.53,
        verbose=True,
    )
    print(f"[r2] polish in {time.time() - t0:.1f}s")
    polished.export(os.path.join(out_dir, "03_polished.glb"))

    # Decimate first (before repair) — much faster repair on 2M faces
    if len(polished.faces) > target_faces * 1.2:
        print(f"[r2] decimating {len(polished.faces):,} -> {target_faces:,} faces")
        t0 = time.time()
        polished = quadric_decimate(polished, target_faces=target_faces)
        print(f"[r2] decimate in {time.time() - t0:.1f}s")
        polished.export(os.path.join(out_dir, "03b_decimated.glb"))

    # Fast CUDA repair via cumesh
    print("[r2] repair_mesh_cuda (cumesh)...")
    t0 = time.time()
    try:
        final = repair_mesh_cuda(polished, fill_holes=True, verbose=True)
    except Exception as e:
        print(f"[r2] CUDA repair failed ({e}); returning pre-repair mesh")
        final = polished
    print(f"[r2] repair in {time.time() - t0:.1f}s")

    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", default="/workspace/demo_text_to_3d")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-ultrashape", action="store_true")
    ap.add_argument("--no-offload", action="store_true")
    ap.add_argument(
        "--model-id",
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="HF model id. Default SDXL (ungated). Use 'black-forest-labs/FLUX.1-schnell' if you've accepted the gated license and HF_TOKEN is set.",
    )
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Step 1: Text-to-image -> image
    t2i = _t2i_load(model_id=args.model_id, offload=not args.no_offload)
    image = _t2i_image(t2i, args.prompt, seed=args.seed)
    image_path = os.path.join(args.out, "00_t2i_image.png")
    image.save(image_path)
    print(f"[t2i] image saved: {image_path}")

    # Free text-to-image to reclaim VRAM for TRELLIS.2
    del t2i
    torch.cuda.empty_cache()

    # Step 2: TRELLIS.2 -> raw mesh
    raw, _trellis_pipe = _trellis_mesh(image, args.out)

    # Step 3: UltraShape refinement (optional, non-commercial license)
    refined = raw if args.skip_ultrashape else _ultrashape_refine(raw, image, args.out)

    # Step 4: R2 polish chain
    final = _r2_polish(refined, args.out)

    final_path = os.path.join(args.out, "04_final.glb")
    final.export(final_path)

    print()
    print("=" * 60)
    print(f"DONE")
    print(f"  prompt:   {args.prompt!r}")
    print(f"  verts:    {len(final.vertices):,}")
    print(f"  faces:    {len(final.faces):,}")
    print(f"  water:    {final.is_watertight}")
    print(f"  output:   {final_path}")
    print(f"  folder:   {args.out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
