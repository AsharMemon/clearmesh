#!/usr/bin/env python3
"""Check whether predicted FACE first faces match canonical tie groups.

Exact face-0 divergence can be misleading when many faces share the same
minimum ZYX anchor. This script reads a first-face probe JSON and asks whether
candidate first faces are at least members of the same-anchor equivalence group.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_tokens(path: Path) -> np.ndarray:
    data = np.load(path)
    return np.asarray(data["paper_tokens"], dtype=np.int64).reshape(-1, 9)


def _min_vertex(face_tokens: np.ndarray) -> tuple[int, int, int]:
    face = np.asarray(face_tokens, dtype=np.int64).reshape(3, 3)
    idx = int(np.lexsort((face[:, 2], face[:, 1], face[:, 0]))[0])
    return tuple(int(v) for v in face[idx].tolist())


def _group(tokens: np.ndarray) -> tuple[set[tuple[int, ...]], tuple[int, int, int], int]:
    first_min = _min_vertex(tokens[0])
    members = {tuple(int(v) for v in row.tolist()) for row in tokens if _min_vertex(row) == first_min}
    return members, first_min, len(members)


def _candidate_tokens(item: dict[str, Any], selector: str) -> tuple[int, ...] | None:
    first = item.get("first_face") or {}
    payload = first.get(selector)
    if not isinstance(payload, dict):
        return None
    values = payload.get("tokens")
    if not isinstance(values, list) or len(values) != 9:
        return None
    return tuple(int(v) for v in values)


def analyze(probe: dict[str, Any], *, selectors: tuple[str, ...]) -> dict[str, Any]:
    rows = []
    for item in probe.get("results", []):
        path = Path(str(item.get("path")))
        try:
            tokens = _load_tokens(path)
            members, first_min, group_size = _group(tokens)
        except Exception as exc:  # noqa: BLE001
            rows.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
            continue
        row: dict[str, Any] = {
            "path": str(path),
            "first_min_vertex": list(first_min),
            "same_anchor_group_size": int(group_size),
        }
        for selector in selectors:
            candidate = _candidate_tokens(item, selector)
            if candidate is None:
                row[f"{selector}_available"] = False
                continue
            row[f"{selector}_available"] = True
            row[f"{selector}_exact_row0"] = bool(candidate == tuple(int(v) for v in tokens[0].tolist()))
            row[f"{selector}_in_same_anchor_group"] = bool(candidate in members)
            row[f"{selector}_same_min_vertex"] = bool(_min_vertex(np.asarray(candidate, dtype=np.int64)) == first_min)
        rows.append(row)
    summary: dict[str, Any] = {"attempted": len(rows), "selectors": list(selectors)}
    good_rows = [row for row in rows if "error" not in row]
    summary["valid"] = len(good_rows)
    if good_rows:
        summary["same_anchor_group_size_mean"] = float(np.mean([row["same_anchor_group_size"] for row in good_rows]))
        for selector in selectors:
            available = [row for row in good_rows if row.get(f"{selector}_available")]
            if not available:
                continue
            summary[f"{selector}_exact_row0_rate"] = float(np.mean([bool(row.get(f"{selector}_exact_row0")) for row in available]))
            summary[f"{selector}_in_same_anchor_group_rate"] = float(
                np.mean([bool(row.get(f"{selector}_in_same_anchor_group")) for row in available])
            )
            summary[f"{selector}_same_min_vertex_rate"] = float(
                np.mean([bool(row.get(f"{selector}_same_min_vertex")) for row in available])
            )
    return {"summary": summary, "results": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selectors", default="best_logprob,best_hybrid,best_surface,best_oracle")
    args = parser.parse_args()
    probe = json.loads(args.probe.read_text(encoding="utf-8"))
    selectors = tuple(item.strip() for item in args.selectors.split(",") if item.strip())
    report = analyze(probe, selectors=selectors)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
