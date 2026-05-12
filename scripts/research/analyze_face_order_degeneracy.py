#!/usr/bin/env python3
"""Audit FACE paper-order first-face degeneracy.

FACE sorts faces by the lexicographic ZYX coordinate of each face's minimum
vertex. If many faces share or nearly share that first anchor, the first target
face is effectively multimodal: several plausible triangles are equally local
starts for the same shape. This script quantifies that ambiguity from prepared
``paper_tokens`` shards.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_tokens(path: Path) -> np.ndarray:
    data = np.load(path)
    if "paper_tokens" not in data.files:
        raise ValueError(f"{path} missing paper_tokens")
    tokens = np.asarray(data["paper_tokens"], dtype=np.int64)
    if tokens.ndim != 2 or tokens.shape[1] != 9:
        raise ValueError(f"{path} has invalid paper_tokens shape {tokens.shape}")
    return tokens


def _face_min_vertices(tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    faces = np.asarray(tokens, dtype=np.int64).reshape(-1, 3, 3)
    offsets = np.asarray([np.lexsort((face[:, 2], face[:, 1], face[:, 0]))[0] for face in faces], dtype=np.int64)
    mins = faces[np.arange(len(faces)), offsets]
    return mins, offsets


def analyze_tokens(tokens: np.ndarray, *, near_bins: tuple[int, ...] = (1, 2, 4)) -> dict[str, Any]:
    tokens = np.asarray(tokens, dtype=np.int64).reshape(-1, 9)
    if len(tokens) == 0:
        return {"face_count": 0}
    mins, offsets = _face_min_vertices(tokens)
    first_min = mins[0]
    deltas = np.abs(mins - first_min.reshape(1, 3))
    linf = np.max(deltas, axis=1)
    l1 = np.sum(deltas, axis=1)
    same_min = linf == 0
    face0 = tokens[0]
    face0_vertices = face0.reshape(3, 3)
    first_anchor = face0_vertices[0]
    same_slot0_anchor = np.all(tokens.reshape(-1, 3, 3)[:, 0, :] == first_anchor.reshape(1, 3), axis=1)
    ordered = bool(np.array_equal(np.lexsort((mins[:, 2], mins[:, 1], mins[:, 0])), np.arange(len(mins))))
    return {
        "face_count": int(len(tokens)),
        "ordered_by_min_zyx": ordered,
        "first_min_vertex": [int(v) for v in first_min.tolist()],
        "first_face_tokens": [int(v) for v in face0.tolist()],
        "first_face_min_offset": int(offsets[0]),
        "same_min_count": int(np.sum(same_min)),
        "same_slot0_anchor_count": int(np.sum(same_slot0_anchor)),
        "near_min_counts": {str(int(threshold)): int(np.sum(linf <= threshold)) for threshold in near_bins},
        "second_min_linf_gap": None if len(tokens) < 2 else int(np.partition(linf, 1)[1]),
        "second_min_l1_gap": None if len(tokens) < 2 else int(np.partition(l1, 1)[1]),
        "min_linf_p10": float(np.percentile(linf, 10)) if len(tokens) else None,
        "min_linf_p50": float(np.percentile(linf, 50)) if len(tokens) else None,
        "min_linf_p90": float(np.percentile(linf, 90)) if len(tokens) else None,
    }


def _aggregate(rows: list[dict[str, Any]], *, near_bins: tuple[int, ...]) -> dict[str, Any]:
    if not rows:
        return {"sample_count": 0}
    def mean(key: str) -> float:
        return float(np.mean([float(row[key]) for row in rows if row.get(key) is not None]))
    return {
        "sample_count": int(len(rows)),
        "ordered_by_min_zyx_rate": float(np.mean([bool(row.get("ordered_by_min_zyx")) for row in rows])),
        "same_min_gt1_rate": float(np.mean([int(row.get("same_min_count") or 0) > 1 for row in rows])),
        "same_slot0_anchor_gt1_rate": float(np.mean([int(row.get("same_slot0_anchor_count") or 0) > 1 for row in rows])),
        "same_min_count_mean": mean("same_min_count"),
        "same_slot0_anchor_count_mean": mean("same_slot0_anchor_count"),
        "second_min_linf_gap_mean": mean("second_min_linf_gap"),
        "second_min_l1_gap_mean": mean("second_min_l1_gap"),
        "near_min_gt1_rates": {
            str(int(threshold)): float(
                np.mean([int((row.get("near_min_counts") or {}).get(str(int(threshold)), 0)) > 1 for row in rows])
            )
            for threshold in near_bins
        },
        "near_min_count_means": {
            str(int(threshold)): float(
                np.mean([int((row.get("near_min_counts") or {}).get(str(int(threshold)), 0)) for row in rows])
            )
            for threshold in near_bins
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--near-bins", default="1,2,4")
    parser.add_argument("--examples", type=int, default=12)
    args = parser.parse_args()
    near_bins = tuple(int(item) for item in args.near_bins.split(",") if item.strip())
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for dataset_dir in args.dataset_dir:
        paths = sorted(dataset_dir.glob("*.npz"))
        if args.limit:
            paths = paths[: args.limit]
        for path in paths:
            try:
                row = analyze_tokens(_load_tokens(path), near_bins=near_bins)
            except Exception as exc:  # noqa: BLE001 - report corrupt shards without aborting the audit.
                errors.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
                continue
            row["path"] = str(path)
            row["split"] = dataset_dir.name
            rows.append(row)
    rows_sorted = sorted(
        rows,
        key=lambda row: (
            -int(row.get("same_min_count") or 0),
            -int(row.get("same_slot0_anchor_count") or 0),
            int(row.get("second_min_linf_gap") if row.get("second_min_linf_gap") is not None else 10**9),
        ),
    )
    report = {
        "dataset_dirs": [str(path) for path in args.dataset_dir],
        "near_bins": list(near_bins),
        "summary": _aggregate(rows, near_bins=near_bins),
        "examples_most_degenerate": rows_sorted[: max(0, int(args.examples))],
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    if errors:
        print(json.dumps({"errors": len(errors)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
