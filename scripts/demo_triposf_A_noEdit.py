"""Pipeline A: raw TRELLIS.2 + UltraShape + TripoSF (NO Easy3E edit).

Baseline to measure whether TripoSF actually improves mesh quality
over UltraShape alone. No editing, no mask, just pure pipeline.run()
followed by UltraShape refine then TripoSF watertight pass.
"""

import os, sys, time
sys.path.insert(0, "/workspace/UltraShape-1.0")
sys.path.insert(0, "/workspace/TripoSF")
sys.path.insert(0, "/workspace/clearmesh")

import torch
from PIL import Image
from trellis2.pipelines import Trellis2ImageTo3DPipeline

OUT_DIR = "/workspace/demo_A_noEdit"
os.makedirs(OUT_DIR, exist_ok=True)

print("[setup] Loading TRELLIS.2 pipeline...")
pipe = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
pipe.cuda()

source_img = Image.open("/workspace/TRELLIS.2/assets/example_image/T.png").convert("RGB")
source_img.save(os.path.join(OUT_DIR, "source.png"))

print("[run] pipeline.run (full cascade + texture)...")
t0 = time.time()
mesh = pipe.run(source_img, seed=42)[0]
t_trellis = time.time() - t0
print(f"  TRELLIS.2: {t_trellis:.1f}s, verts={mesh.vertices.shape[0]:,}")

# Save raw TRELLIS.2 output
import trimesh, numpy as np
def to_trimesh(m):
    if isinstance(m, trimesh.Trimesh): return m
    v = m.vertices.detach().cpu().numpy() if hasattr(m.vertices, "detach") else np.asarray(m.vertices)
    f = m.faces.detach().cpu().numpy() if hasattr(m.faces, "detach") else np.asarray(m.faces)
    return trimesh.Trimesh(vertices=v, faces=f)

tm = to_trimesh(mesh)
tm.export(os.path.join(OUT_DIR, "01_raw_trellis.glb"))

# --- UltraShape refine ---
print("[run] UltraShape refine...")
from clearmesh.editing.ultrashape_refine import UltraShapeRefiner, UltraShapeConfig
us = UltraShapeRefiner(device="cuda")
t0 = time.time()
us_mesh = us.refine(coarse_mesh=tm, reference_image=source_img, config=UltraShapeConfig(num_inference_steps=50, octree_res=1024))
t_us = time.time() - t0
print(f"  UltraShape: {t_us:.1f}s, verts={us_mesh.vertices.shape[0]:,}, watertight={us_mesh.is_watertight}")
us_mesh.export(os.path.join(OUT_DIR, "02_after_ultrashape.glb"))

# --- TripoSF watertight pass ---
print("[run] TripoSF watertight pass...")
from clearmesh.editing.triposf_refine import TripoSFRefiner, TripoSFConfig
ts = TripoSFRefiner()
t0 = time.time()
ts_mesh = ts.refine(coarse_mesh=us_mesh, config=TripoSFConfig())
t_ts = time.time() - t0
print(f"  TripoSF: {t_ts:.1f}s, verts={ts_mesh.vertices.shape[0]:,}, watertight={ts_mesh.is_watertight}")
ts_mesh.export(os.path.join(OUT_DIR, "03_after_triposf.glb"))

print()
print("=" * 60)
print(f"Total: {t_trellis + t_us + t_ts:.1f}s")
print(f"  TRELLIS.2:  {t_trellis:.1f}s  ({tm.vertices.shape[0]:,} verts)")
print(f"  UltraShape: {t_us:.1f}s  ({us_mesh.vertices.shape[0]:,} verts, wt={us_mesh.is_watertight})")
print(f"  TripoSF:    {t_ts:.1f}s  ({ts_mesh.vertices.shape[0]:,} verts, wt={ts_mesh.is_watertight})")
print(f"Outputs: {OUT_DIR}")
print("=" * 60)
