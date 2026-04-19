"""Re-export a trained DualPrim scene at different tessellation levels.

Why this exists: the paper's Table 2 reports ~1.5k verts / 0.79k faces
per object, but our default exports come out at ~500k verts / 1M faces
because we tessellate each SQ at marching-cubes resolution 128. That's
a tessellation choice, not a quality difference.

This script reloads a saved primitives.json and re-tessellates at a
chosen lower resolution + optionally takes a true Boolean UNION across
all alive primitives (not just concatenate). The Boolean union
eliminates interior interface geometry between overlapping PSQs and
gives a single watertight scene mesh.

Usage:
    python scripts/dualprim/reexport.py \\
        --in /workspace/dualprim_quality \\
        --out /workspace/dualprim_quality/reexport \\
        --tess-res 32 --union
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

# Make clearmesh importable
for _r in ("/workspace/clearmesh", "/root/clearmesh",
           str(Path(__file__).resolve().parents[2])):
    if os.path.isdir(_r) and _r not in sys.path:
        sys.path.insert(0, _r)

import torch
import trimesh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", required=True,
                    help="Trained run dir (must contain primitives.json)")
    ap.add_argument("--out", dest="out_dir", required=True)
    ap.add_argument("--tess-res", type=int, default=64,
                    help="Marching-cubes resolution per primitive (default 64; 32-48 is paper-ish)")
    ap.add_argument("--union", action="store_true",
                    help="Boolean-union all per-primitive meshes into one (slow, watertight)")
    ap.add_argument("--alpha-threshold", type=float, default=0.5)
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prims_data = json.load(open(in_dir / "primitives.json"))["primitives"]
    print(f"loaded {len(prims_data)} primitives from {in_dir}")

    from clearmesh.dualprim.types import DualPrimitive
    from clearmesh.dualprim.export import export_dual_primitive

    pieces = []
    for i, p in enumerate(prims_data):
        if p["alpha"] < args.alpha_threshold:
            continue
        dp = DualPrimitive(
            psq_scale=torch.tensor(p["psq_scale"]),
            nsq_scale=torch.tensor(p["nsq_scale"]),
            psq_shape=torch.tensor(p["psq_shape"]),
            nsq_shape=torch.tensor(p["nsq_shape"]),
            alpha=torch.tensor(p["alpha"]),
            theta=torch.tensor(p["theta"]),
            psq_translation=torch.tensor(p["psq_translation"]),
            nsq_translation=torch.tensor(p["nsq_translation"]),
            psq_rotation_rad=torch.tensor(p["psq_rotation_rad"]),
            nsq_rotation_rad=torch.tensor(p["nsq_rotation_rad"]),
            color=torch.tensor(p["color"]),
        )
        try:
            m = export_dual_primitive(dp, resolution=args.tess_res, backend="manifold3d")
        except Exception as e:
            print(f"  prim {i}: tess failed: {e}")
            continue
        if len(m.faces) > 0:
            pieces.append(m)
        if len(pieces) % 5 == 0:
            print(f"  tessellated {len(pieces)} primitives so far")

    print(f"alive primitives tessellated: {len(pieces)}")

    if not pieces:
        print("nothing to export")
        return

    # Per-primitive export
    for i, m in enumerate(pieces):
        m.export(out_dir / f"per_prim_{i:03d}.glb")

    if args.union:
        print("Boolean-unioning all primitives...")
        try:
            scene_mesh = trimesh.boolean.union(pieces)
            print(f"  union: {len(scene_mesh.vertices):,}v / {len(scene_mesh.faces):,}f")
        except Exception as e:
            print(f"  union failed ({e}); falling back to concatenate")
            scene_mesh = trimesh.util.concatenate(pieces)
    else:
        scene_mesh = trimesh.util.concatenate(pieces)

    scene_mesh.export(out_dir / "refit.glb")
    print(f"exported {len(scene_mesh.vertices):,}v / {len(scene_mesh.faces):,}f to {out_dir/'refit.glb'}")


if __name__ == "__main__":
    main()
