"""Pipeline B: Easy3E edit + UltraShape + TripoSF (full stack)."""

import os, sys, time, numpy as np
sys.path.insert(0, "/workspace/UltraShape-1.0")
sys.path.insert(0, "/workspace/TripoSF")
sys.path.insert(0, "/workspace/clearmesh")

import torch
from PIL import Image
from clearmesh.editing import Easy3EEditor, EditOptions
from trellis2.pipelines import Trellis2ImageTo3DPipeline

OUT_DIR = "/workspace/demo_B_withEdit"
os.makedirs(OUT_DIR, exist_ok=True)

def make_wings_mask(size=512):
    mask = np.zeros((size, size), dtype=np.uint8)
    cy = size // 2
    for y in range(size):
        for x in range(size):
            dx_l = (x - size * 0.22) / (size * 0.25)
            dy_l = (y - cy) / (size * 0.40)
            if dx_l ** 2 + dy_l ** 2 < 1.0: mask[y, x] = 255
            dx_r = (x - size * 0.78) / (size * 0.25)
            dy_r = (y - cy) / (size * 0.40)
            if dx_r ** 2 + dy_r ** 2 < 1.0: mask[y, x] = 255
    return Image.fromarray(mask, mode="L")

mask = make_wings_mask(512)
mask.save(os.path.join(OUT_DIR, "region_mask.png"))

print("[setup] Loading TRELLIS.2 pipeline...")
pipe = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
pipe.cuda()

editor = Easy3EEditor(
    trellis2_dir="/workspace/TRELLIS.2",
    model_dir="/workspace/models/trellis2-4b",
    device="cuda", pipeline=pipe,
)

source_img = Image.open("/workspace/TRELLIS.2/assets/example_image/T.png").convert("RGB")
source_img.save(os.path.join(OUT_DIR, "source.png"))

options = EditOptions(
    num_flow_steps=12, num_repaint_steps=12, guidance_scale=7.5,
    text_num_steps=20, text_image_guidance=2.5, text_guidance_scale=7.5,
    region_mask=mask, mask_dilation=2, mask_blur_radius=12.0, mask_drop_threshold=0.1,
    enable_ultrashape=True,
    ultrashape_dir="/workspace/UltraShape-1.0",
    ultrashape_ckpt="/workspace/UltraShape-1.0/checkpoints/ultrashape_v1.pt",
    ultrashape_config="/workspace/UltraShape-1.0/configs/infer_dit_refine.yaml",
    ultrashape_steps=50, ultrashape_octree_res=1024,
    enable_triposf=True,
    triposf_dir="/workspace/TripoSF",
    triposf_config="/workspace/TripoSF/configs/TripoSFVAE_1024.yaml",
    enable_repair=True, skip_repair_above_verts=500_000,
    export_format="glb",
)

print("[run] Full pipeline: Easy3E + UltraShape + TripoSF...")
t0 = time.time()
result = editor.edit_from_source_image(
    source_image=source_img,
    instruction="add large feathered wings to the sides",
    output_path=os.path.join(OUT_DIR, "final_full_stack.glb"),
    options=options,
)
wall = time.time() - t0

print()
print("=" * 60)
print(f"DONE in {wall:.1f}s")
print(f"Mesh: {result.mesh.vertices.shape[0]:,} verts, {result.mesh.faces.shape[0]:,} faces")
print(f"Watertight: {result.mesh.is_watertight}")
print(f"Timings: {result.timings}")
print(f"Outputs: {OUT_DIR}")
print("=" * 60)
