"""Demo: CAD-Recode refit on a TRELLIS.2-generated mesh.

Takes an input GLB (e.g. the steampunk engine from last session's R2
polish), downsamples + FPS to 256 points, feeds those to CAD-Recode,
execs the returned CadQuery script, and writes out:

  - ``cadquery_script.py``       the Python script the model generated
  - ``refit_top1.glb``           the tessellated CadQuery solid for the
                                 best-scoring candidate (best Chamfer)
  - ``refit_candN.glb``          each of the other candidates (when
                                 ``--n-candidates > 1``)
  - ``chamfer.json``             per-candidate chamfer scores

Run on pod:
    cd /workspace/clearmesh
    pip install pytorch3d cadquery
    python scripts/demo_cad_recode.py \\
        --input /workspace/demo_R2_easy3e/02_after_R2.glb \\
        --out /workspace/demo_cad_recode \\
        --n-candidates 5

Licence note: CAD-Recode weights are CC-BY-NC-4.0.  RESEARCH ONLY.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Pod layout
for p in ("/workspace/clearmesh",):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

import trimesh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to input GLB/OBJ mesh")
    ap.add_argument("--out", default="/workspace/demo_cad_recode", help="Output dir")
    ap.add_argument("--n-candidates", type=int, default=1, help="Test-time sampling count (10 = paper quality)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument(
        "--attn",
        default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager"],
        help="Attention backend (flash-attn is fastest but can be fragile)",
    )
    ap.add_argument("--model-id", default="filapro/cad-recode-v1.5")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    from clearmesh.refit import CadRecodeRefiner

    print(f"[cad_recode] loading {args.model_id} (attn={args.attn})...")
    t0 = time.time()
    refiner = CadRecodeRefiner(
        model_id=args.model_id,
        attn_implementation=args.attn,
    )
    _ = refiner.model  # force lazy load so load time is logged separately
    print(f"[cad_recode] loaded in {time.time() - t0:.1f}s")

    print(f"[cad_recode] loading input mesh: {args.input}")
    mesh = trimesh.load(args.input, force="mesh")
    print(
        f"[cad_recode] input: {len(mesh.vertices):,}v / {len(mesh.faces):,}f, "
        f"extents={mesh.extents.round(3).tolist()}"
    )

    t0 = time.time()
    result = refiner.refit(
        mesh,
        n_candidates=args.n_candidates,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    dt = time.time() - t0
    print(f"[cad_recode] refit in {dt:.1f}s  (chamfer={result.chamfer})")
    print(f"[cad_recode] generated {len(result.cadquery_code):,} chars of CadQuery")

    # Save the script
    with open(os.path.join(args.out, "cadquery_script.py"), "w") as f:
        f.write(result.cadquery_code)

    # Save best refit mesh
    if result.refit_mesh is not None:
        result.refit_mesh.export(os.path.join(args.out, "refit_top1.glb"))
        print(
            f"[cad_recode] top-1 refit: "
            f"{len(result.refit_mesh.vertices):,}v / "
            f"{len(result.refit_mesh.faces):,}f"
        )
    else:
        print("[cad_recode] top-1 exec/tessellate failed — script saved for inspection")

    # Save all candidates
    per_cand = []
    for idx, (code, m, ch) in enumerate(result.candidates):
        cand_entry = {
            "index": idx,
            "chamfer": ch if ch != float("inf") else None,
            "code_len": len(code),
            "exec_succeeded": m is not None,
            "verts": int(len(m.vertices)) if m is not None else 0,
            "faces": int(len(m.faces)) if m is not None else 0,
        }
        per_cand.append(cand_entry)
        if m is not None and idx > 0:
            m.export(os.path.join(args.out, f"refit_cand{idx}.glb"))

    with open(os.path.join(args.out, "chamfer.json"), "w") as f:
        json.dump({"candidates": per_cand, "total_s": dt}, f, indent=2)

    # Print the generated CadQuery script
    print("\n--- generated CadQuery ---")
    print(result.cadquery_code)
    print("--- end ---\n")

    # Save the point cloud the model actually saw
    pc_glb = trimesh.points.PointCloud(result.point_cloud)
    pc_glb.export(os.path.join(args.out, "input_points_256.ply"))

    print("=" * 60)
    print(f"DONE in {dt:.1f}s")
    print(f"  output:   {args.out}")
    print(f"  chamfer:  {result.chamfer}")
    print(f"  script:   {os.path.join(args.out, 'cadquery_script.py')}")
    print(f"  refit:    {os.path.join(args.out, 'refit_top1.glb')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
