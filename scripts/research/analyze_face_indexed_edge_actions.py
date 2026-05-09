#!/usr/bin/env python3
"""Analyze boundary-edge completion targets for FACE-indexed datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research.train_face_indexed_conditioned_tiny import _edge_action_targets, _load_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--neighbors", default="4,8,16,24,32,48,64")
    args = parser.parse_args()

    samples, num_bins = _load_dataset(args.dataset_dir, limit=args.limit)
    neighbor_counts = [int(value) for value in args.neighbors.split(",") if value.strip()]
    coverage = {count: 0 for count in neighbor_counts}
    total_actions = 0
    local_ranks: list[int] = []
    edge_lengths: list[float] = []
    aspect_ratios: list[float] = []
    per_sample: list[dict[str, object]] = []

    for sample in samples:
        edge_actions, thirds = _edge_action_targets(sample.faces)
        ranks = []
        actions = 0
        vertices = np.asarray(sample.vertices, dtype=np.float64)
        diag = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))) or 1.0
        for face, edge, third in zip(sample.faces, edge_actions, thirds):
            if int(third) < 0 or np.any(edge < 0):
                continue
            actions += 1
            total_actions += 1
            a, b = int(edge[0]), int(edge[1])
            target = int(third)
            midpoint = (vertices[a] + vertices[b]) * 0.5
            order = [int(index) for index in np.argsort(np.linalg.norm(vertices - midpoint, axis=1)) if int(index) not in {a, b}]
            rank = order.index(target) + 1 if target in order else len(order) + 1
            ranks.append(rank)
            local_ranks.append(rank)
            for count in neighbor_counts:
                if rank <= count:
                    coverage[count] += 1
            points = vertices[np.asarray(face, dtype=np.int64)]
            lengths = np.asarray(
                [
                    np.linalg.norm(points[1] - points[0]),
                    np.linalg.norm(points[2] - points[1]),
                    np.linalg.norm(points[0] - points[2]),
                ],
                dtype=np.float64,
            )
            edge_lengths.append(float(np.max(lengths) / diag))
            aspect_ratios.append(float(np.max(lengths) / max(float(np.min(lengths)), 1e-12)))
        per_sample.append(
            {
                "path": str(sample.path),
                "faces": int(len(sample.faces)),
                "edge_actions": int(actions),
                "median_local_rank": _median(ranks),
                "p95_local_rank": _percentile(ranks, 95),
            }
        )

    payload = {
        "dataset_dir": str(args.dataset_dir),
        "samples": int(len(samples)),
        "num_bins": int(num_bins),
        "total_edge_actions": int(total_actions),
        "coverage": {
            str(count): (float(coverage[count] / total_actions) if total_actions else None)
            for count in neighbor_counts
        },
        "local_rank": {
            "median": _median(local_ranks),
            "p95": _percentile(local_ranks, 95),
            "max": int(max(local_ranks)) if local_ranks else None,
        },
        "target_geometry": {
            "edge_length_median": _median(edge_lengths),
            "edge_length_p95": _percentile(edge_lengths, 95),
            "aspect_median": _median(aspect_ratios),
            "aspect_p95": _percentile(aspect_ratios, 95),
        },
        "per_sample": per_sample,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ["samples", "total_edge_actions", "coverage", "local_rank"]}, indent=2, sort_keys=True))
    return 0


def _median(values: list[float] | list[int]) -> float | None:
    if not values:
        return None
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _percentile(values: list[float] | list[int], percentile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


if __name__ == "__main__":
    raise SystemExit(main())
