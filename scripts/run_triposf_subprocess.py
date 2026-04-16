#!/usr/bin/env python
"""Subprocess-isolated TripoSF (SparseFlex) watertight refiner.

Takes an input mesh and produces a watertight 1024^3 reconstruction via
TripoSF VAE. Runs in a subprocess to isolate any CUDA symbol clashes with
the parent pipeline (same pattern as run_ultrashape_subprocess.py).

Arguments:
  --mesh-path  input mesh (.obj/.glb)
  --output     output path (.obj or .glb)
  --config     TripoSF config yaml (default: TripoSF/configs/TripoSFVAE_1024.yaml)
  --triposf-dir TripoSF repo root (default: /workspace/TripoSF)

Exit 0 on success; non-zero with stderr diagnostic on failure.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh-path", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--triposf-dir", required=True)
    args = ap.parse_args()

    sys.path.insert(0, args.triposf_dir)
    os.chdir(args.triposf_dir)

    try:
        import torch
        import trimesh
        from omegaconf import OmegaConf

        # TripoSF defines TripoSFVAEInference INLINE inside inference.py
        # (not in a submodule). Import by running the module as a script-less
        # import via importlib.
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "triposf_inference", Path(args.triposf_dir) / "inference.py"
        )
        tsf_inf = importlib.util.module_from_spec(spec)
        sys.modules["triposf_inference"] = tsf_inf
        # Prevent the module from running main() when loaded
        saved_name = __name__
        try:
            spec.loader.exec_module(tsf_inf)
        except SystemExit:
            pass  # in case the module's __main__ block calls sys.exit

        normalize_mesh = tsf_inf.normalize_mesh
        load_quantized_mesh_original = tsf_inf.load_quantized_mesh_original
        TripoSFVAEInference = tsf_inf.TripoSFVAEInference

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[triposf-subproc] device={device}", flush=True)

        t_load = time.time()
        model = TripoSFVAEInference.from_config(args.config).to(device)
        print(f"[triposf-subproc] Model loaded in {time.time() - t_load:.1f}s", flush=True)

        # Normalize input mesh to TripoSF's expected scale/center
        print(f"[triposf-subproc] Normalizing {args.mesh_path}", flush=True)
        mesh_gt = normalize_mesh(args.mesh_path)

        # TripoSF expects OBJ at a specific path because load_quantized_mesh_original
        # reads from disk. Write the normalized mesh to a tmp file.
        import tempfile
        tmp_root = Path(tempfile.mkdtemp(prefix="triposf_"))
        save_gt = tmp_root / "normalized.obj"
        trimesh.Trimesh(vertices=mesh_gt.vertices.tolist(), faces=mesh_gt.faces.tolist()).export(save_gt)

        print("[triposf-subproc] Loading quantized mesh (sampling points + voxelizing)...", flush=True)
        t0 = time.time()
        sparse_voxels, points_sample = load_quantized_mesh_original(
            str(save_gt),
            volume_resolution=model.cfg.resolution,
            use_normals=model.cfg.use_normals,
            pc_sample_number=model.cfg.sample_points_num,
        )
        print(f"[triposf-subproc] Load time: {time.time() - t0:.2f}s", flush=True)

        # Move to GPU and prepend batch-id column for sparse tensor
        sparse_voxels = sparse_voxels.to(device)
        points_sample = points_sample.to(device)
        sparse_voxels_sp = torch.cat(
            [torch.zeros_like(sparse_voxels[..., :1]), sparse_voxels],
            dim=-1,
        ).int()

        # Run reconstruction
        print("[triposf-subproc] Reconstructing...", flush=True)
        t0 = time.time()
        with torch.cuda.amp.autocast(dtype=torch.float16):
            mesh_recon = model(points_sample[None], sparse_voxels_sp)[0]
        print(f"[triposf-subproc] Reconstruction time: {time.time() - t0:.2f}s", flush=True)

        # Save output
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        out_mesh = trimesh.Trimesh(
            vertices=mesh_recon.vertices.tolist(),
            faces=mesh_recon.faces.tolist(),
        )
        out_mesh.export(args.output)
        print(
            f"[triposf-subproc] DONE verts={len(out_mesh.vertices):,} "
            f"faces={len(out_mesh.faces):,} -> {args.output}",
            flush=True,
        )

        # Cleanup temp dir
        import shutil
        shutil.rmtree(tmp_root, ignore_errors=True)
        return 0

    except Exception as e:
        print(f"[triposf-subproc] FATAL: {type(e).__name__}: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
