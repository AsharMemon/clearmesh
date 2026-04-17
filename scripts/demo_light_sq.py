"""Demo: Light-SQ superquadric refit on a TRELLIS.2-generated mesh.

Runs the minimal Light-SQ port (fit-and-carve greedy loop, no block-
regrow-fill, no adaptive pruning by class). Outputs:

  - ``primitives.json``   the list of fitted SuperQuadric params
  - ``refit.glb``          tessellated union of all primitives
  - ``tsdf_slices.png``    mid-slices of the initial TSDF for sanity

Usage on pod:

    pip install scikit-image rtree
    cd /workspace/clearmesh
    python scripts/demo_light_sq.py \\
        --input /workspace/demo_R2_easy3e/02_after_R2.glb \\
        --out /workspace/demo_light_sq \\
        --grid-res 100 \\
        --max-primitives 40
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

for p in ("/workspace/clearmesh",):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import trimesh
from dataclasses import asdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="/workspace/demo_light_sq")
    ap.add_argument("--grid-res", type=int, default=100)
    ap.add_argument("--max-primitives", type=int, default=40)
    ap.add_argument("--n-iters", type=int, default=150)
    ap.add_argument("--lr", type=float, default=0.015)
    ap.add_argument("--compile-res", type=int, default=96)
    ap.add_argument("--decimate-input", type=int, default=200_000,
                    help="Decimate input mesh to this face count before TSDF (faster)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    from clearmesh.refit.light_sq import LightSQRefiner

    print(f"[light_sq] loading {args.input}")
    mesh = trimesh.load(args.input, force="mesh")
    print(f"[light_sq] input: {len(mesh.vertices):,}v / {len(mesh.faces):,}f")

    # Decimate for speed — TSDF signed_distance on 2M faces is slow
    if len(mesh.faces) > args.decimate_input:
        print(f"[light_sq] decimating to {args.decimate_input:,} faces for speed")
        try:
            from clearmesh.mesh.repair import quadric_decimate
            mesh = quadric_decimate(mesh, target_faces=args.decimate_input)
            print(f"[light_sq] decimated: {len(mesh.vertices):,}v / {len(mesh.faces):,}f")
        except Exception as e:
            print(f"[light_sq] decimate failed ({e}); continuing with full mesh")

    refiner = LightSQRefiner(
        grid_res=args.grid_res,
        max_primitives=args.max_primitives,
        n_iters=args.n_iters,
        lr=args.lr,
    )

    t0 = time.time()
    result = refiner.fit(mesh, verbose=True)
    dt = time.time() - t0
    print(f"\n[light_sq] fit in {dt:.1f}s, "
          f"{len(result.primitives)} primitives, "
          f"residual {result.final_residual_voxels:,} voxels")

    # Save primitives
    prims_json = [asdict(sq) for sq in result.primitives]
    with open(os.path.join(args.out, "primitives.json"), "w") as f:
        json.dump({"primitives": prims_json, "timings": result.timings}, f, indent=2)

    # Compile to mesh
    print(f"[light_sq] compiling tessellated union (res={args.compile_res})...")
    t0 = time.time()
    union = refiner.compile(result.primitives, resolution=args.compile_res)
    dt = time.time() - t0
    print(f"[light_sq] compiled in {dt:.1f}s: "
          f"{len(union.vertices):,}v / {len(union.faces):,}f")

    if len(union.vertices) > 0:
        union.export(os.path.join(args.out, "refit.glb"))

    print("=" * 60)
    print(f"DONE")
    print(f"  input:       {args.input}")
    print(f"  primitives:  {len(result.primitives)}")
    print(f"  refit:       {os.path.join(args.out, 'refit.glb')}")
    print(f"  params:      {os.path.join(args.out, 'primitives.json')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
