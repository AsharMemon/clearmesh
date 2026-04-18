"""End-to-end ClearMesh demo: text -> image -> mesh -> polish -> refit.

Chains the whole pipeline in one script so the gains at each stage are
visible side-by-side:

    Stage 1. Text prompt
    Stage 2. Qwen-Image (text-to-image)
    Stage 3. TRELLIS.2-4B (image-to-3D, 512 -> 1024 cascade)
    Stage 4. Polish chain (Taubin + quadric decimate + cumesh CUDA
             repair)
    Stage 5. Light-SQ block-regrow-fill (superquadric refit for
             interpretable / CAD-like output)

Each stage writes a GLB and an orthographic render so you can open
them side-by-side in any viewer. UltraShape is NOT wired in here
because it's non-commercial; add ``--ultrashape`` to include it if
you've accepted the licence.

Run on pod:
    cd /workspace/clearmesh
    python scripts/demo_end_to_end.py \\
        --prompt "a single steampunk compass, whole object" \\
        --out /workspace/demo_e2e_compass
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

for p in ("/workspace/clearmesh",):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch
import trimesh
from PIL import Image


# ---------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------

def render_mesh(
    glb_path: str, out_path: str, angle_deg: float = 45.0, size: int = 720,
) -> None:
    """Fast offscreen render of a GLB from a single angle."""
    import math
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    import pyrender

    mesh = trimesh.load(glb_path, force="mesh")
    if len(mesh.faces) == 0:
        return
    mesh.vertices -= mesh.centroid
    s = mesh.extents.max()
    if s > 0:
        mesh.vertices /= s

    theta = math.radians(angle_deg)
    eye = np.array([2 * math.sin(theta), 0.3, 2 * math.cos(theta)], dtype=np.float32)
    fwd = -eye / np.linalg.norm(eye)
    up_world = np.array([0, 1, 0], dtype=np.float32)
    r_vec = np.cross(fwd, up_world)
    r_vec /= np.linalg.norm(r_vec)
    u_vec = np.cross(r_vec, fwd)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 0] = r_vec
    pose[:3, 1] = u_vec
    pose[:3, 2] = -fwd
    pose[:3, 3] = eye

    scene = pyrender.Scene(ambient_light=(0.3, 0.3, 0.3), bg_color=(255, 255, 255, 255))
    scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))
    cam = pyrender.PerspectiveCamera(yfov=math.radians(40), aspectRatio=1.0)
    scene.add(cam, pose=pose)
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=4.0), pose=pose)
    r = pyrender.OffscreenRenderer(size, size)
    color, _ = r.render(scene)
    r.delete()
    Image.fromarray(color[..., :3]).save(out_path)


# ---------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------

def stage_text_to_image(prompt: str, out_dir: str, seed: int, model_id: str) -> Image.Image:
    from clearmesh.text_to_3d import TextTo3D

    gen = TextTo3D(model_id=model_id)
    img = gen.text_to_image(prompt, seed=seed, width=1024, height=1024)
    img.save(os.path.join(out_dir, "01_image.png"))
    return img


def stage_image_to_3d(image: Image.Image, out_dir: str) -> trimesh.Trimesh:
    import sys
    sys.path.insert(0, "/workspace/TRELLIS.2")
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    from clearmesh.mesh.extraction import extract_from_ovoxel

    print("[stage 3] loading TRELLIS.2-4B")
    pipe = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipe.cuda()

    print("[stage 3] running TRELLIS.2 cascade")
    outputs = pipe.run(image, seed=42, sparse_structure_sampler_params={"steps": 12})
    if isinstance(outputs, dict) and "mesh" in outputs:
        mesh = outputs["mesh"][0]
    elif isinstance(outputs, list):
        mesh = outputs[0]
    else:
        mesh = outputs
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = extract_from_ovoxel(mesh)

    mesh.export(os.path.join(out_dir, "02_trellis2_raw.glb"))
    del pipe
    torch.cuda.empty_cache()
    return mesh


def stage_ultrashape(
    coarse_mesh: trimesh.Trimesh, image: Image.Image, out_dir: str,
    ultrashape_dir: str = "/workspace/UltraShape-1.0",
) -> trimesh.Trimesh:
    """Optional refinement between TRELLIS.2 and polish.

    Closes the open-surface holes that the raw 6.8M-face decode leaves
    behind (cumesh logs ``N boundary loops`` it can't fill — those are
    holes too big for trimesh's hole-filler too). UltraShape rebuilds
    the mesh from voxel diffusion so the output is essentially
    watertight.

    Licence note: UltraShape weights are non-commercial.
    """
    if not os.path.isdir(ultrashape_dir):
        print(f"[stage 3.5] UltraShape dir not found at {ultrashape_dir}; skipping")
        return coarse_mesh
    print("[stage 3.5] UltraShape refinement")
    from clearmesh.editing.ultrashape_refine import UltraShapeRefiner, UltraShapeConfig

    refiner = UltraShapeRefiner(
        ultrashape_dir=ultrashape_dir,
        ckpt_path=f"{ultrashape_dir}/checkpoints/ultrashape_v1.pt",
        config_path=f"{ultrashape_dir}/configs/infer_dit_refine.yaml",
    )
    config = UltraShapeConfig(num_inference_steps=50, octree_res=1024)
    refined = refiner.refine(
        coarse_mesh=coarse_mesh,
        reference_image=image,
        config=config,
    )
    refined.export(os.path.join(out_dir, "02b_ultrashape.glb"))
    print(f"[stage 3.5] UltraShape: "
          f"{len(refined.vertices):,}v / {len(refined.faces):,}f")
    return refined


def _keep_largest_component(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Drop disconnected floating bits; keep only the largest connected
    component. The compass demo had a small "orb" floating off to the
    side that survived cumesh.remove_small_connected_components — its
    threshold is geometric, not relative-volume, so a medium-size noise
    blob can pass through. Splitting + sorting by face count is a much
    sharper filter.
    """
    parts = mesh.split(only_watertight=False)
    if len(parts) <= 1:
        return mesh
    parts_sorted = sorted(parts, key=lambda p: len(p.faces), reverse=True)
    largest = parts_sorted[0]
    print(f"[polish/components] kept largest of {len(parts)} components: "
          f"{len(largest.faces):,}/{len(mesh.faces):,} faces "
          f"(dropped {sum(len(p.faces) for p in parts_sorted[1:]):,})")
    return largest


def _aggressive_fill_holes(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """trimesh.repair.fill_holes is good at the small-hole tail that
    cumesh's CUDA filler skips (cumesh has a hard max_perimeter that
    rejects irregular boundary loops). Run after cumesh to mop up.
    """
    try:
        import trimesh.repair as repair
        before = len(mesh.faces)
        mesh = mesh.copy()
        repair.fill_holes(mesh)
        after = len(mesh.faces)
        if after > before:
            print(f"[polish/fill_holes] added {after - before:,} faces "
                  f"({before:,} -> {after:,})")
    except Exception as e:
        print(f"[polish/fill_holes] skipped: {e}")
    return mesh


def stage_polish(mesh: trimesh.Trimesh, out_dir: str, target_faces: int = 1_500_000) -> trimesh.Trimesh:
    from clearmesh.mesh.repair import polish_mesh, quadric_decimate, repair_mesh_cuda

    print("[stage 4a] polish_mesh")
    m = polish_mesh(mesh, merge_digits=5, taubin_iterations=3, verbose=True)
    if len(m.faces) > target_faces * 1.2:
        print(f"[stage 4b] decimate {len(m.faces):,} -> {target_faces:,}")
        m = quadric_decimate(m, target_faces=target_faces)
    print("[stage 4c] repair_mesh_cuda")
    try:
        m = repair_mesh_cuda(m, fill_holes=True, verbose=True)
    except Exception as e:
        print(f"[stage 4c] CUDA repair failed: {e}")
    print("[stage 4d] keep largest connected component")
    m = _keep_largest_component(m)
    print("[stage 4e] aggressive trimesh fill_holes")
    m = _aggressive_fill_holes(m)
    m.export(os.path.join(out_dir, "03_polished.glb"))
    return m


def stage_refit(
    mesh: trimesh.Trimesh, out_dir: str,
    max_primitives: int = 15, grid_res: int = 100,
) -> tuple:
    from clearmesh.refit import LightSQRefiner

    print(f"[stage 5] Light-SQ block-regrow-fill "
          f"(grid={grid_res}, max_primitives={max_primitives})")
    refiner = LightSQRefiner(
        grid_res=grid_res,
        max_primitives=max_primitives,
        n_iters=60,
        lr=0.015,
    )
    result = refiner.fit_with_regrow(
        mesh, regrow_rounds=1, partition="kmeans", verbose=True,
    )
    compiled = refiner.compile(result.primitives, resolution=96)
    compiled.export(os.path.join(out_dir, "04_refit.glb"))
    return result, compiled


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", default="/workspace/demo_e2e")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--model-id", default="Qwen/Qwen-Image")
    ap.add_argument("--max-primitives", type=int, default=15)
    ap.add_argument("--grid-res", type=int, default=100,
                    help="Light-SQ TSDF grid resolution (100=fast, 200=fine detail)")
    ap.add_argument("--ultrashape", action="store_true",
                    help="Run UltraShape refinement between TRELLIS.2 and polish "
                         "(closes large holes; non-commercial licence)")
    ap.add_argument("--decimate-target", type=int, default=2_000_000,
                    help="Target face count for polish-stage decimation (higher "
                         "preserves more fine detail like compass bezels)")
    ap.add_argument("--skip-refit", action="store_true")
    ap.add_argument("--skip-render", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    total_t0 = time.time()

    # Stage 2: text -> image
    print(f"[e2e] prompt: {args.prompt!r}")
    t0 = time.time()
    image = stage_text_to_image(args.prompt, args.out, args.seed, args.model_id)
    t_t2i = time.time() - t0

    # Stage 3: image -> 3D
    t0 = time.time()
    raw_mesh = stage_image_to_3d(image, args.out)
    t_trellis = time.time() - t0

    # Stage 3.5: UltraShape (optional)
    t_ultra = 0.0
    if args.ultrashape:
        t0 = time.time()
        raw_mesh = stage_ultrashape(raw_mesh, image, args.out)
        t_ultra = time.time() - t0

    # Stage 4: polish
    t0 = time.time()
    polished = stage_polish(raw_mesh, args.out, target_faces=args.decimate_target)
    t_polish = time.time() - t0

    # Stage 5: refit (optional)
    t_refit = 0.0
    refit_info = None
    if not args.skip_refit:
        t0 = time.time()
        result, compiled = stage_refit(
            polished, args.out,
            max_primitives=args.max_primitives,
            grid_res=args.grid_res,
        )
        t_refit = time.time() - t0
        refit_info = (
            f"{len(result.primitives)} primitives, "
            f"{len(compiled.vertices):,}v / {len(compiled.faces):,}f"
        )

    # Renders (all from the same angle for side-by-side comparison)
    if not args.skip_render:
        for name in ("02_trellis2_raw.glb", "02b_ultrashape.glb",
                     "03_polished.glb", "04_refit.glb"):
            p = os.path.join(args.out, name)
            if os.path.exists(p):
                try:
                    render_mesh(p, p.replace(".glb", "_render.png"))
                except Exception as e:
                    print(f"[render] {name} failed: {e}")

    total_t = time.time() - total_t0

    print()
    print("=" * 64)
    print(f"END-TO-END DONE in {total_t:.1f}s")
    print(f"  prompt:     {args.prompt!r}")
    print(f"  out dir:    {args.out}")
    print(f"  t2i:        {t_t2i:.1f}s")
    print(f"  trellis2:   {t_trellis:.1f}s")
    if t_ultra > 0:
        print(f"  ultrashape: {t_ultra:.1f}s")
    print(f"  polish:     {t_polish:.1f}s")
    if refit_info:
        print(f"  refit:      {t_refit:.1f}s  ({refit_info})")
    print("  artefacts:")
    for f in sorted(os.listdir(args.out)):
        p = os.path.join(args.out, f)
        sz = os.path.getsize(p)
        print(f"    {f:35s} {sz/1024:8.1f} KB")
    print("=" * 64)


if __name__ == "__main__":
    main()
