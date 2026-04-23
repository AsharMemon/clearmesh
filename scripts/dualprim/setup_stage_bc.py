"""Prepare Stage B export cleanup followed by Stage C local-refinement setup.

This is the post-checkpoint path we want for the current DualPrim branch:
  1. start from the best coarse DualPrim mesh (for example r6)
  2. apply budgeted Stage B mesh cleanup only
  3. emit global narrow-band and primitive-local Stage C artifacts

It does not retrain DualPrim. It preserves the chosen coarse basin and turns
it into a clean handoff for detail refinement.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import trimesh

_CANDIDATE_ROOTS = [
    "/workspace/clearmesh",
    str(Path(__file__).resolve().parents[2]),
]
for _root in _CANDIDATE_ROOTS:
    if os.path.isdir(_root) and _root not in sys.path:
        sys.path.insert(0, _root)

from clearmesh.dualprim import (
    DualPrimConfig,
    DetailRefineConfig,
    PrimitiveLocalRefineConfig,
    prepare_detail_refine_artifacts,
    prepare_primitive_local_refine_artifacts,
)
from clearmesh.dualprim.export import cleanup_scene_mesh


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coarse-mesh", required=True,
                    help="Exported DualPrim mesh to polish (for example r6 refit.glb).")
    ap.add_argument("--target-mesh", required=True,
                    help="Reference mesh used to measure missing detail.")
    ap.add_argument("--primitives-json", required=True,
                    help="Live primitive JSON for the same checkpoint.")
    ap.add_argument("--out", required=True,
                    help="Output directory for Stage B/C artifacts.")
    ap.add_argument("--cleanup-min-faces", type=int, default=96)
    ap.add_argument("--cleanup-min-area-ratio", type=float, default=0.01)
    ap.add_argument("--smoothing-iters", type=int, default=4)
    ap.add_argument("--smoothing-lambda", type=float, default=0.5)
    ap.add_argument("--smoothing-nu", type=float, default=-0.53)
    ap.add_argument("--local-min-alpha", type=float, default=0.1)
    ap.add_argument("--local-band-radius-frac", type=float, default=0.05)
    ap.add_argument("--local-assignment-margin-frac", type=float, default=0.35)
    ap.add_argument("--local-min-samples", type=int, default=256)
    ap.add_argument("--local-max-samples", type=int, default=4096)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    coarse_mesh = trimesh.load(args.coarse_mesh, force="mesh")
    if isinstance(coarse_mesh, trimesh.Scene):
        if not coarse_mesh.geometry:
            coarse_mesh = trimesh.Trimesh()
        else:
            coarse_mesh = trimesh.util.concatenate(tuple(coarse_mesh.geometry.values()))

    config = DualPrimConfig()
    config.export_cleanup_min_component_faces = args.cleanup_min_faces
    config.export_cleanup_min_component_area_ratio = args.cleanup_min_area_ratio
    config.export_smoothing_iterations = args.smoothing_iters
    config.export_smoothing_lambda = args.smoothing_lambda
    config.export_smoothing_nu = args.smoothing_nu

    cleaned = cleanup_scene_mesh(coarse_mesh, config)
    cleaned_path = out_dir / "stage_b_refined.glb"
    cleaned.export(cleaned_path)

    detail_manifest = prepare_detail_refine_artifacts(
        cleaned_path,
        args.target_mesh,
        out_dir / "detail_refine",
        DetailRefineConfig(),
    )
    primitive_manifest = prepare_primitive_local_refine_artifacts(
        cleaned_path,
        args.target_mesh,
        args.primitives_json,
        out_dir / "primitive_local",
        PrimitiveLocalRefineConfig(
            band_radius_frac=args.local_band_radius_frac,
            assignment_margin_frac=args.local_assignment_margin_frac,
            min_alpha=args.local_min_alpha,
            min_samples_per_primitive=args.local_min_samples,
            max_samples_per_primitive=args.local_max_samples,
        ),
    )

    summary = {
        "stage_b_mesh": str(cleaned_path),
        "detail_refine_manifest": str(out_dir / "detail_refine" / "detail_refine_manifest.json"),
        "primitive_local_manifest": str(out_dir / "primitive_local" / "primitive_local_manifest.json"),
        "stage_b_faces": int(len(cleaned.faces)),
        "stage_b_vertices": int(len(cleaned.vertices)),
        "detail_band_ratio": detail_manifest["detail_signal"]["detail_band_ratio"],
        "primitive_local_artifacts": primitive_manifest.get("summary", {}).get("num_local_artifacts", 0),
    }
    with open(out_dir / "stage_bc_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[stage-bc] stage_b_mesh={cleaned_path}")
    print(f"[stage-bc] detail_refine={summary['detail_refine_manifest']}")
    print(f"[stage-bc] primitive_local={summary['primitive_local_manifest']}")
    print(f"[stage-bc] stage_b_v={summary['stage_b_vertices']:,} stage_b_f={summary['stage_b_faces']:,}")
    print(f"[stage-bc] primitive_local_artifacts={summary['primitive_local_artifacts']}")


if __name__ == "__main__":
    main()
