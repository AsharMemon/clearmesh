#!/usr/bin/env python3
"""Select topology-safe rows from a FACE token-oracle run.

The oracle tells us which representation/bin/repair combinations preserve a
mesh as a watertight edge graph. This script turns that report into manifests
that can feed the next bounded experiment, without manually copy-pasting paths.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
    return rows


def _csv_values(value: str | None, *, cast=str) -> set[Any] | None:
    if value is None or value == "":
        return None
    return {cast(item.strip()) for item in value.split(",") if item.strip()}


def _matches_filters(
    row: dict[str, Any],
    *,
    families: set[str] | None,
    bins: set[int] | None,
    orders: set[str] | None,
) -> bool:
    if row.get("error"):
        return False
    if families is not None and str(row.get("family")) not in families:
        return False
    if bins is not None and int(row.get("num_bins", -1)) not in bins:
        return False
    if orders is not None and str(row.get("within_face_order")) not in orders:
        return False
    return True


def _topology_view(row: dict[str, Any], repair_mode: str) -> dict[str, Any]:
    if repair_mode == "token":
        return row
    repair = row.get("repair", {}).get(repair_mode)
    if not isinstance(repair, dict):
        return {"watertight_edge_graph": False, "missing_repair_mode": repair_mode}
    return repair


def _row_passes(
    row: dict[str, Any],
    *,
    repair_mode: str,
    max_quantization_face_loss: int | None,
    max_boundary_edges: int | None,
    max_nonmanifold_edges: int | None,
    min_edge_pairing_ratio: float | None,
    require_source_watertight: bool,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    topo = _topology_view(row, repair_mode)
    if require_source_watertight and row.get("source_watertight") is False:
        reasons.append("source_not_watertight")
    if not bool(topo.get("watertight_edge_graph")):
        reasons.append("not_watertight")
    if max_quantization_face_loss is not None and int(row.get("quantization_face_loss", 0)) > max_quantization_face_loss:
        reasons.append("quantization_face_loss")
    if max_boundary_edges is not None and int(topo.get("boundary_edge_count", 0)) > max_boundary_edges:
        reasons.append("boundary_edges")
    if max_nonmanifold_edges is not None and int(topo.get("nonmanifold_edge_count", 0)) > max_nonmanifold_edges:
        reasons.append("nonmanifold_edges")
    if min_edge_pairing_ratio is not None and float(topo.get("edge_pairing_ratio", 0.0)) < min_edge_pairing_ratio:
        reasons.append("edge_pairing_ratio")
    return not reasons, reasons


def _manifest_row(row: dict[str, Any], *, copied_path: Path | None, passes: bool, reasons: list[str]) -> dict[str, Any]:
    output = {
        "path": str(copied_path or row.get("path")),
        "source_path": str(row.get("path")),
        "source_name": row.get("source_name"),
        "family": row.get("family"),
        "num_bins": row.get("num_bins"),
        "within_face_order": row.get("within_face_order"),
        "source_faces": row.get("source_faces"),
        "token_faces": row.get("token_faces"),
        "quantization_face_loss": row.get("quantization_face_loss"),
        "passes": bool(passes),
        "reasons": reasons,
    }
    return output


def _copy_or_link(path: Path, output_dir: Path, mode: str) -> Path | None:
    if mode == "none":
        return None
    if not path.exists():
        return None
    asset_dir = output_dir / "assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    destination = asset_dir / path.name
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    if mode == "copy":
        shutil.copy2(path, destination)
    elif mode == "symlink":
        destination.symlink_to(path.resolve())
    else:
        raise ValueError(f"unknown copy mode: {mode}")
    return destination


def select_rows(
    rows: list[dict[str, Any]],
    *,
    families: set[str] | None,
    bins: set[int] | None,
    orders: set[str] | None,
    repair_mode: str,
    max_quantization_face_loss: int | None,
    max_boundary_edges: int | None,
    max_nonmanifold_edges: int | None,
    min_edge_pairing_ratio: float | None,
    require_source_watertight: bool,
    output_dir: Path,
    copy_mode: str,
) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if _matches_filters(row, families=families, bins=bins, orders=orders)
    ]
    passing: list[dict[str, Any]] = []
    failing: list[dict[str, Any]] = []
    for row in selected:
        passes, reasons = _row_passes(
            row,
            repair_mode=repair_mode,
            max_quantization_face_loss=max_quantization_face_loss,
            max_boundary_edges=max_boundary_edges,
            max_nonmanifold_edges=max_nonmanifold_edges,
            min_edge_pairing_ratio=min_edge_pairing_ratio,
            require_source_watertight=require_source_watertight,
        )
        copied_path = _copy_or_link(Path(str(row.get("path"))), output_dir, copy_mode) if passes else None
        manifest = _manifest_row(row, copied_path=copied_path, passes=passes, reasons=reasons)
        if passes:
            passing.append(manifest)
        else:
            failing.append(manifest)
    pass_rate = float(len(passing) / len(selected)) if selected else 0.0
    return {
        "selected": selected,
        "passing": passing,
        "failing": failing,
        "summary": {
            "selected_count": len(selected),
            "passing_count": len(passing),
            "failing_count": len(failing),
            "pass_rate": pass_rate,
            "repair_mode": repair_mode,
            "families": sorted(families) if families is not None else None,
            "bins": sorted(bins) if bins is not None else None,
            "orders": sorted(orders) if orders is not None else None,
            "max_quantization_face_loss": max_quantization_face_loss,
            "max_boundary_edges": max_boundary_edges,
            "max_nonmanifold_edges": max_nonmanifold_edges,
            "min_edge_pairing_ratio": min_edge_pairing_ratio,
            "require_source_watertight": require_source_watertight,
        },
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-rows", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--families", default=None, help="Comma list, e.g. indexed or paper,indexed.")
    parser.add_argument("--bins", default=None, help="Comma list, e.g. 512,1024.")
    parser.add_argument("--orders", default=None, help="Comma list, e.g. indexed,preserve.")
    parser.add_argument("--repair-mode", choices=["token", "none", "dedupe", "manifold"], default="token")
    parser.add_argument("--max-quantization-face-loss", type=int, default=None)
    parser.add_argument("--max-boundary-edges", type=int, default=None)
    parser.add_argument("--max-nonmanifold-edges", type=int, default=None)
    parser.add_argument("--min-edge-pairing-ratio", type=float, default=None)
    parser.add_argument("--require-source-watertight", action="store_true")
    parser.add_argument("--copy-mode", choices=["none", "copy", "symlink"], default="none")
    parser.add_argument("--min-pass-rate", type=float, default=None)
    parser.add_argument("--fail-if-empty", action="store_true")
    args = parser.parse_args()

    rows = _read_jsonl(args.oracle_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = select_rows(
        rows,
        families=_csv_values(args.families, cast=str),
        bins=_csv_values(args.bins, cast=int),
        orders=_csv_values(args.orders, cast=str),
        repair_mode=args.repair_mode,
        max_quantization_face_loss=args.max_quantization_face_loss,
        max_boundary_edges=args.max_boundary_edges,
        max_nonmanifold_edges=args.max_nonmanifold_edges,
        min_edge_pairing_ratio=args.min_edge_pairing_ratio,
        require_source_watertight=args.require_source_watertight,
        output_dir=args.output_dir,
        copy_mode=args.copy_mode,
    )
    _write_jsonl(args.output_dir / "pass_manifest.jsonl", result["passing"])
    _write_jsonl(args.output_dir / "fail_manifest.jsonl", result["failing"])
    (args.output_dir / "selection_summary.json").write_text(
        json.dumps(result["summary"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    if args.fail_if_empty and not result["passing"]:
        return 2
    if args.min_pass_rate is not None and result["summary"]["pass_rate"] < args.min_pass_rate:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
