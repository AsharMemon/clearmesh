"""Export a mesh from a primitives.json snapshot WITHOUT training.

Used for fast evaluation of mid-training states when the training
itself is hung or when we want to measure a trajectory snapshot's
quality directly (does iter 1000 already carve the hole, or do we
need to get to iter 15000?).

Usage:
    python scripts/dualprim/export_from_snapshot.py \\
        --primitives /workspace/dualprim_round3/hole/trajectory/step_001000.json \\
        --out /workspace/snap_export_round3_iter1000 \\
        --union-export
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


_CANDIDATE_ROOTS = ["/workspace/clearmesh",
                    str(Path(__file__).resolve().parents[2])]
for _r in _CANDIDATE_ROOTS:
    if os.path.isdir(_r) and _r not in sys.path:
        sys.path.insert(0, _r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--primitives", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--union-export", action="store_true")
    ap.add_argument("--tessellation-resolution", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--k", type=int, default=100,
                    help="num_primitives_init for the DualPrimConfig.")
    args = ap.parse_args()

    import torch
    from clearmesh.dualprim import DualPrimConfig, export_scene
    from clearmesh.dualprim.io import load_scene_from_json

    device = args.device if torch.cuda.is_available() else "cpu"
    cfg = DualPrimConfig(
        num_primitives_init=args.k,
        tessellation_resolution=args.tessellation_resolution,
    )
    print(f"[export] loading snapshot {args.primitives}")
    scene = load_scene_from_json(
        args.primitives, cfg, device=device, pad_to_K=args.k,
    )
    print(f"[export] loaded {scene.num_alive}/{scene.K} primitives")

    print(f"[export] exporting (union={args.union_export}, "
          f"tess_res={args.tessellation_resolution})")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    scene_mesh, per_prim = export_scene(scene, cfg, union_all=args.union_export)
    scene_mesh.export(out_dir / "refit.glb")
    for i, m in enumerate(per_prim):
        m.export(out_dir / f"per_prim_{i:03d}.glb")
    print(f"[export] wrote {len(per_prim)+1} meshes to {out_dir}")
    print(f"[export] scene: {len(scene_mesh.vertices):,}v / {len(scene_mesh.faces):,}f")


if __name__ == "__main__":
    main()
