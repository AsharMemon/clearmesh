"""Run bounded primitive-local Stage C refinement from a manifest."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

_CANDIDATE_ROOTS = [
    "/workspace/clearmesh",
    str(Path(__file__).resolve().parents[2]),
]
for _root in _CANDIDATE_ROOTS:
    if os.path.isdir(_root) and _root not in sys.path:
        sys.path.insert(0, _root)

from clearmesh.dualprim import LocalRefineConfig, train_primitive_local_refiners


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True,
                    help="primitive_local_manifest.json produced by Stage B/C setup.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--smoothness-weight", type=float, default=0.05)
    ap.add_argument("--regularize-weight", type=float, default=0.01)
    ap.add_argument("--jitter-sigma-frac", type=float, default=0.02)
    ap.add_argument("--min-sample-count", type=int, default=256)
    ap.add_argument("--max-primitives", type=int, default=None)
    ap.add_argument("--primitive-ids", default=None,
                    help="Comma-separated live primitive ids to refine. "
                         "If omitted, use the highest-error artifacts.")
    ap.add_argument("--normal-focus-power", type=float, default=1.0)
    ap.add_argument("--distance-focus-power", type=float, default=1.0)
    ap.add_argument("--boundary-focus-weight", type=float, default=0.0)
    ap.add_argument("--boundary-focus-power", type=float, default=2.0)
    ap.add_argument("--weight-cap", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    primitive_ids = None
    if args.primitive_ids:
        primitive_ids = tuple(
            int(tok.strip()) for tok in args.primitive_ids.split(",") if tok.strip()
        )

    cfg = LocalRefineConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        lr=args.lr,
        smoothness_weight=args.smoothness_weight,
        regularize_weight=args.regularize_weight,
        jitter_sigma_frac=args.jitter_sigma_frac,
        min_sample_count=args.min_sample_count,
        max_primitives=args.max_primitives,
        target_primitive_ids=primitive_ids,
        normal_focus_power=args.normal_focus_power,
        distance_focus_power=args.distance_focus_power,
        boundary_focus_weight=args.boundary_focus_weight,
        boundary_focus_power=args.boundary_focus_power,
        weight_cap=args.weight_cap,
        seed=args.seed,
    )
    summary = train_primitive_local_refiners(
        args.manifest,
        args.out,
        cfg=cfg,
        device=args.device,
    )
    print(f"[local-refine] selected={summary['num_selected_primitives']}")
    print(f"[local-refine] mean_baseline={summary['mean_baseline_error']:.6f}")
    print(f"[local-refine] mean_refined={summary['mean_refined_error']:.6f}")
    print(f"[local-refine] mean_improvement={summary['mean_improvement']:.6f}")
    print(f"[local-refine] out={args.out}")


if __name__ == "__main__":
    main()
