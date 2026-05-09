#!/usr/bin/env python3
"""Check whether FACE-token shards are safe artist-mesh training targets."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clearmesh.mesh_heads.face_dataset_gate import (
    evaluate_face_dataset_manifest,
    thresholds_for_profile,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=["strict", "tolerant", "proxy"], default="strict")
    parser.add_argument("--token-family", choices=["coordinate", "paper", "indexed"], default="coordinate")
    parser.add_argument("--max-boundary-edges", type=int, default=None)
    parser.add_argument("--max-nonmanifold-edges", type=int, default=None)
    parser.add_argument("--min-edge-pairing-ratio", type=float, default=None)
    parser.add_argument("--fail-on-violations", action="store_true")
    args = parser.parse_args()

    thresholds = thresholds_for_profile(args.profile)
    if args.max_boundary_edges is not None:
        thresholds = replace(thresholds, max_boundary_edges=args.max_boundary_edges)
    if args.max_nonmanifold_edges is not None:
        thresholds = replace(thresholds, max_nonmanifold_edges=args.max_nonmanifold_edges)
    if args.min_edge_pairing_ratio is not None:
        thresholds = replace(thresholds, min_edge_pairing_ratio=args.min_edge_pairing_ratio)

    report = evaluate_face_dataset_manifest(args.manifest, thresholds, token_family=args.token_family)
    report["manifest"] = str(args.manifest)
    report["profile"] = args.profile
    report["token_family"] = args.token_family
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ["profile", "sample_count", "passing", "failing", "pass_rate", "passes"]}, indent=2))
    print(f"Wrote {args.output}")
    return 1 if args.fail_on_violations and not report["passes"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
