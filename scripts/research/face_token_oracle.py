#!/usr/bin/env python3
"""Rapid watertightness oracle for FACE-style mesh tokens.

This is the CPU "wind tunnel" for the FACE path. It answers, before a GPU run:

- Does a target survive tokenization as a watertight edge graph?
- Which failure mode dominates: quantization face loss, boundary edges,
  non-manifold edges, duplicate faces, or degenerate faces?
- Can conservative deterministic repair promote the token sequence to a
  watertight target?
- Do paper coordinate tokens or explicit indexed tokens preserve topology more
  reliably for the same source mesh?

The oracle supports two fast modes:

- existing token shards: ``--dataset-dir path/to/tokens`` or ``--manifest``
- source meshes: ``--mesh-dir path/to/meshes --bins 128,256,512``
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clearmesh.mesh_heads.face_indexed import (  # noqa: E402
    encode_mesh_to_indexed_face_tokens,
    indexed_to_coordinate_tokens,
)
from clearmesh.mesh_heads.face_tokens import (  # noqa: E402
    encode_mesh_to_face_tokens,
    encode_mesh_to_paper_face_tokens,
)
from clearmesh.mesh_heads.face_topology import (  # noqa: E402
    face_token_topology_report,
    repair_face_tokens,
)


MESH_SUFFIXES = {".obj", ".ply", ".stl", ".glb", ".gltf"}


def _parse_csv(value: str, *, cast=str) -> list[Any]:
    return [cast(item.strip()) for item in str(value).split(",") if item.strip()]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
    return rows


def _resolve_record_path(record: dict[str, Any], base_dir: Path) -> Path | None:
    for key in ("path", "target_path", "local_path", "source_path"):
        value = record.get(key)
        if not value:
            continue
        path = Path(str(value))
        candidates = [
            path,
            base_dir / path,
            base_dir / path.name,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
    return None


def _records_from_dataset(dataset_dir: Path, manifest: Path | None) -> list[dict[str, Any]]:
    if manifest is not None:
        records = _load_jsonl(manifest)
        base_dir = manifest.parent
    else:
        manifest_path = dataset_dir / "manifest.jsonl"
        if manifest_path.exists():
            records = _load_jsonl(manifest_path)
            base_dir = manifest_path.parent
        else:
            records = [{"path": str(path), "source_name": path.stem} for path in sorted(dataset_dir.glob("*.npz"))]
            base_dir = dataset_dir

    resolved: list[dict[str, Any]] = []
    for record in records:
        path = _resolve_record_path(record, base_dir)
        if path is None:
            path = _resolve_record_path(record, dataset_dir)
        if path is None:
            unresolved = dict(record)
            unresolved["_oracle_error"] = "missing_npz"
            resolved.append(unresolved)
            continue
        row = dict(record)
        row["_resolved_path"] = str(path)
        resolved.append(row)
    return resolved


def _topology_dict(tokens: np.ndarray) -> dict[str, Any]:
    return face_token_topology_report(np.asarray(tokens, dtype=np.int64)).to_dict()


def _failure_causes(row: dict[str, Any]) -> list[str]:
    causes: list[str] = []
    if row.get("error"):
        return [str(row["error"])]
    if int(row.get("quantization_face_loss", 0)) > 0:
        causes.append("quantization_face_loss")
    if int(row.get("degenerate_face_count", 0)) > 0:
        causes.append("degenerate_faces")
    if int(row.get("duplicate_face_count", 0)) > 0:
        causes.append("duplicate_faces")
    if int(row.get("boundary_edge_count", 0)) > 0:
        causes.append("boundary_edges")
    if int(row.get("nonmanifold_edge_count", 0)) > 0:
        causes.append("nonmanifold_edges")
    if not bool(row.get("watertight_edge_graph", False)):
        causes.append("not_watertight")
    return causes or ["pass"]


def _repair_summaries(tokens: np.ndarray, repair_modes: list[str]) -> dict[str, dict[str, Any]]:
    repaired: dict[str, dict[str, Any]] = {}
    for mode in repair_modes:
        repair_tokens, report = repair_face_tokens(tokens, mode=mode)
        output = report.output_topology
        repaired[mode] = {
            "output_faces": int(report.output_faces),
            "dropped_degenerate_faces": int(report.dropped_degenerate_faces),
            "dropped_duplicate_faces": int(report.dropped_duplicate_faces),
            "dropped_nonmanifold_faces": int(report.dropped_nonmanifold_faces),
            "filled_triangle_holes": int(report.filled_triangle_holes),
            "watertight_edge_graph": bool(output.get("watertight_edge_graph")),
            "boundary_edge_count": int(output.get("boundary_edge_count") or 0),
            "nonmanifold_edge_count": int(output.get("nonmanifold_edge_count") or 0),
            "edge_pairing_ratio": float(output.get("edge_pairing_ratio") or 0.0),
            "face_delta": int(len(repair_tokens) - len(tokens)),
        }
    return repaired


def _row_from_tokens(
    *,
    source_name: str,
    path: str,
    family: str,
    tokens: np.ndarray,
    num_bins: int,
    source_faces: int | None,
    within_face_order: str,
    repair_modes: list[str],
    source_watertight: bool | None = None,
) -> dict[str, Any]:
    topology = _topology_dict(tokens)
    token_faces = int(len(tokens))
    source_faces_value = int(source_faces) if source_faces is not None else None
    face_loss = max(0, source_faces_value - token_faces) if source_faces_value is not None else 0
    row: dict[str, Any] = {
        "source_name": source_name,
        "path": path,
        "family": family,
        "num_bins": int(num_bins),
        "within_face_order": within_face_order,
        "source_faces": source_faces_value,
        "source_watertight": source_watertight,
        "token_faces": token_faces,
        "quantization_face_loss": int(face_loss),
        **topology,
    }
    row["repair"] = _repair_summaries(tokens, repair_modes)
    row["repair_promoted_watertight"] = any(item["watertight_edge_graph"] for item in row["repair"].values())
    row["failure_causes"] = _failure_causes(row)
    return row


def _analyze_npz_record(payload: tuple[dict[str, Any], list[str], list[str]]) -> list[dict[str, Any]]:
    record, families, repair_modes = payload
    if record.get("_oracle_error"):
        return [
            {
                "source_name": record.get("source_name") or record.get("path") or "unknown",
                "path": record.get("path"),
                "family": "unknown",
                "error": record["_oracle_error"],
                "failure_causes": [record["_oracle_error"]],
            }
        ]

    path = Path(str(record["_resolved_path"]))
    source_name = str(record.get("source_name") or path.stem)
    source_faces = record.get("source_faces")
    rows: list[dict[str, Any]] = []
    try:
        with np.load(path, allow_pickle=False) as data:
            num_bins = int(np.asarray(data["num_bins"]).reshape(-1)[0]) if "num_bins" in data else int(record.get("num_bins", 128))
            order = str(np.asarray(data["paper_within_face_order"]).reshape(-1)[0]) if "paper_within_face_order" in data else "unknown"
            if "coordinate" in families and "tokens" in data:
                rows.append(
                    _row_from_tokens(
                        source_name=source_name,
                        path=str(path),
                        family="coordinate",
                        tokens=np.asarray(data["tokens"], dtype=np.int64),
                        num_bins=num_bins,
                        source_faces=int(source_faces) if source_faces is not None else None,
                        within_face_order="canonical",
                        repair_modes=repair_modes,
                    )
                )
            if "paper" in families and "paper_tokens" in data:
                rows.append(
                    _row_from_tokens(
                        source_name=source_name,
                        path=str(path),
                        family="paper",
                        tokens=np.asarray(data["paper_tokens"], dtype=np.int64),
                        num_bins=num_bins,
                        source_faces=int(source_faces) if source_faces is not None else None,
                        within_face_order=order,
                        repair_modes=repair_modes,
                    )
                )
            if "indexed" in families and "indexed_vertices" in data and "indexed_faces" in data:
                vertices = np.asarray(data["indexed_vertices"], dtype=np.int64)
                faces = np.asarray(data["indexed_faces"], dtype=np.int64)
                tokens = vertices[faces].reshape(-1, 9) if len(faces) else np.zeros((0, 9), dtype=np.int64)
                rows.append(
                    _row_from_tokens(
                        source_name=source_name,
                        path=str(path),
                        family="indexed",
                        tokens=tokens,
                        num_bins=num_bins,
                        source_faces=int(source_faces) if source_faces is not None else None,
                        within_face_order="indexed",
                        repair_modes=repair_modes,
                    )
                )
        return rows
    except Exception as exc:
        return [
            {
                "source_name": source_name,
                "path": str(path),
                "family": "unknown",
                "error": f"{type(exc).__name__}: {exc}",
                "failure_causes": ["decode_error"],
            }
        ]


def _load_mesh(path: Path):
    import trimesh

    loaded = trimesh.load(path, force="mesh", process=False, skip_materials=True)
    if isinstance(loaded, trimesh.Scene):
        pieces = [geom for geom in loaded.geometry.values() if isinstance(geom, trimesh.Trimesh)]
        if not pieces:
            raise ValueError("scene contains no mesh geometry")
        loaded = trimesh.util.concatenate(pieces)
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
        raise ValueError("not a triangular mesh")
    return loaded


def _analyze_mesh_payload(payload: tuple[str, list[int], list[str], list[str], list[str], int]) -> list[dict[str, Any]]:
    path_text, bins_values, orders, families, repair_modes, max_faces = payload
    path = Path(path_text)
    rows: list[dict[str, Any]] = []
    try:
        mesh = _load_mesh(path)
    except Exception as exc:
        return [
            {
                "source_name": path.stem,
                "path": str(path),
                "family": "unknown",
                "error": f"mesh_load_error: {type(exc).__name__}: {exc}",
                "failure_causes": ["mesh_load_error"],
            }
        ]

    for num_bins in bins_values:
        for family in families:
            family_orders = orders if family == "paper" else ["canonical" if family == "coordinate" else "indexed"]
            for order in family_orders:
                try:
                    if family == "coordinate":
                        sequence = encode_mesh_to_face_tokens(mesh, num_bins=num_bins, max_faces=max_faces)
                        tokens = sequence.tokens
                    elif family == "paper":
                        sequence = encode_mesh_to_paper_face_tokens(
                            mesh,
                            num_bins=num_bins,
                            max_faces=max_faces,
                            within_face_order=order,
                        )
                        tokens = sequence.tokens
                    elif family == "indexed":
                        indexed = encode_mesh_to_indexed_face_tokens(mesh, num_bins=num_bins, max_faces=max_faces)
                        tokens = indexed_to_coordinate_tokens(indexed)
                    else:
                        raise ValueError(f"unknown family: {family}")
                    rows.append(
                        _row_from_tokens(
                            source_name=path.stem,
                            path=str(path),
                            family=family,
                            tokens=tokens,
                            num_bins=num_bins,
                            source_faces=len(mesh.faces),
                            within_face_order=order,
                            repair_modes=repair_modes,
                            source_watertight=bool(mesh.is_watertight),
                        )
                    )
                except Exception as exc:
                    rows.append(
                        {
                            "source_name": path.stem,
                            "path": str(path),
                            "family": family,
                            "num_bins": int(num_bins),
                            "within_face_order": order,
                            "source_faces": int(len(mesh.faces)),
                            "source_watertight": bool(mesh.is_watertight),
                            "error": f"{type(exc).__name__}: {exc}",
                            "failure_causes": ["encode_error"],
                        }
                    )
    return rows


def _run_parallel(payloads: list[Any], worker, workers: int) -> list[dict[str, Any]]:
    if workers <= 1 or len(payloads) <= 1:
        groups = [worker(payload) for payload in payloads]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            groups = list(pool.map(worker, payloads))
    rows: list[dict[str, Any]] = []
    for group in groups:
        rows.extend(group)
    return rows


def _collect_mesh_paths(mesh_dir: Path, limit: int) -> list[Path]:
    paths = sorted(path for path in mesh_dir.rglob("*") if path.suffix.lower() in MESH_SUFFIXES)
    return paths[:limit] if limit > 0 else paths


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _summarize(rows: list[dict[str, Any]], repair_modes: list[str]) -> dict[str, Any]:
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("error"):
            continue
        key = (
            str(row.get("family", "unknown")),
            int(row.get("num_bins", 0)),
            str(row.get("within_face_order", "unknown")),
        )
        groups[key].append(row)

    group_summaries = []
    for (family, num_bins, order), group in sorted(groups.items()):
        causes = Counter(cause for row in group for cause in row.get("failure_causes", []))
        summary: dict[str, Any] = {
            "family": family,
            "num_bins": num_bins,
            "within_face_order": order,
            "samples": len(group),
            "token_watertight": sum(1 for row in group if row.get("watertight_edge_graph")),
            "token_watertight_rate": _mean([1.0 if row.get("watertight_edge_graph") else 0.0 for row in group]),
            "mean_boundary_edges": _mean([float(row.get("boundary_edge_count", 0)) for row in group]),
            "mean_nonmanifold_edges": _mean([float(row.get("nonmanifold_edge_count", 0)) for row in group]),
            "mean_edge_pairing_ratio": _mean([float(row.get("edge_pairing_ratio", 0.0)) for row in group]),
            "mean_quantization_face_loss": _mean([float(row.get("quantization_face_loss", 0)) for row in group]),
            "failure_causes": dict(causes),
        }
        for mode in repair_modes:
            repair_rows = [row.get("repair", {}).get(mode, {}) for row in group]
            summary[f"repair_{mode}_watertight"] = sum(1 for item in repair_rows if item.get("watertight_edge_graph"))
            summary[f"repair_{mode}_watertight_rate"] = _mean([1.0 if item.get("watertight_edge_graph") else 0.0 for item in repair_rows])
            summary[f"repair_{mode}_mean_boundary_edges"] = _mean([float(item.get("boundary_edge_count", 0)) for item in repair_rows])
        group_summaries.append(summary)

    error_count = sum(1 for row in rows if row.get("error"))
    worst = sorted(
        [row for row in rows if not row.get("error")],
        key=lambda row: (
            int(row.get("boundary_edge_count", 0))
            + 4 * int(row.get("nonmanifold_edge_count", 0))
            + int(row.get("quantization_face_loss", 0)),
            1.0 - float(row.get("edge_pairing_ratio", 0.0)),
        ),
        reverse=True,
    )[:20]
    return {
        "rows": len(rows),
        "errors": int(error_count),
        "groups": group_summaries,
        "worst_failures": [
            {
                "source_name": row.get("source_name"),
                "family": row.get("family"),
                "num_bins": row.get("num_bins"),
                "within_face_order": row.get("within_face_order"),
                "source_faces": row.get("source_faces"),
                "token_faces": row.get("token_faces"),
                "quantization_face_loss": row.get("quantization_face_loss"),
                "boundary_edge_count": row.get("boundary_edge_count"),
                "nonmanifold_edge_count": row.get("nonmanifold_edge_count"),
                "edge_pairing_ratio": row.get("edge_pairing_ratio"),
                "failure_causes": row.get("failure_causes"),
                "path": row.get("path"),
            }
            for row in worst
        ],
    }


def _flatten_for_csv(row: dict[str, Any], repair_modes: list[str]) -> dict[str, Any]:
    flat = {key: value for key, value in row.items() if key not in {"repair", "failure_causes"}}
    flat["failure_causes"] = ",".join(row.get("failure_causes", []))
    for mode in repair_modes:
        repair = row.get("repair", {}).get(mode, {})
        for key, value in repair.items():
            flat[f"repair_{mode}_{key}"] = value
    return flat


def _write_outputs(rows: list[dict[str, Any]], summary: dict[str, Any], output_dir: Path, repair_modes: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "oracle_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    with (output_dir / "oracle_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    flat_rows = [_flatten_for_csv(row, repair_modes) for row in rows]
    fieldnames = sorted({key for row in flat_rows for key in row})
    with (output_dir / "oracle_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(flat_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset-dir", type=Path, default=None, help="Directory containing FACE NPZ shards.")
    source.add_argument("--manifest", type=Path, default=None, help="FACE manifest JSONL.")
    source.add_argument("--mesh-dir", type=Path, default=None, help="Directory of source meshes for bin/order scans.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--families", default="paper,indexed", help="Comma list: coordinate,paper,indexed.")
    parser.add_argument("--repair-modes", default="none,dedupe,manifold", help="Comma list: none,dedupe,manifold.")
    parser.add_argument("--bins", default="128,256,512", help="Mesh mode only. Comma-separated quantization bins.")
    parser.add_argument("--orders", default="preserve,rotate_min_zyx,sort_zyx", help="Paper mesh mode only.")
    parser.add_argument("--max-faces", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    args = parser.parse_args()

    started = time.time()
    families = _parse_csv(args.families)
    repair_modes = _parse_csv(args.repair_modes)
    bins_values = _parse_csv(args.bins, cast=int)
    orders = _parse_csv(args.orders)
    invalid_families = sorted(set(families) - {"coordinate", "paper", "indexed"})
    invalid_repairs = sorted(set(repair_modes) - {"none", "dedupe", "manifold"})
    if invalid_families:
        raise SystemExit(f"unknown families: {invalid_families}")
    if invalid_repairs:
        raise SystemExit(f"unknown repair modes: {invalid_repairs}")

    if args.mesh_dir is not None:
        paths = _collect_mesh_paths(args.mesh_dir, args.limit)
        payloads = [(str(path), bins_values, orders, families, repair_modes, args.max_faces) for path in paths]
        rows = _run_parallel(payloads, _analyze_mesh_payload, args.workers)
        source_desc = str(args.mesh_dir)
    else:
        dataset_dir = args.dataset_dir or args.manifest.parent
        records = _records_from_dataset(dataset_dir, args.manifest)
        if args.limit > 0:
            records = records[: args.limit]
        payloads = [(record, families, repair_modes) for record in records]
        rows = _run_parallel(payloads, _analyze_npz_record, args.workers)
        source_desc = str(args.dataset_dir or args.manifest)

    summary = _summarize(rows, repair_modes)
    summary.update(
        {
            "source": source_desc,
            "families": families,
            "repair_modes": repair_modes,
            "elapsed_sec": time.time() - started,
            "workers": int(args.workers),
        }
    )
    _write_outputs(rows, summary, args.output_dir, repair_modes)
    print(json.dumps({key: summary[key] for key in ["source", "rows", "errors", "elapsed_sec", "workers"]}, indent=2))
    print("Group summary:")
    for group in summary["groups"]:
        print(
            f"  {group['family']} bins={group['num_bins']} order={group['within_face_order']} "
            f"token={group['token_watertight']}/{group['samples']} "
            f"dedupe={group.get('repair_dedupe_watertight', 0)}/{group['samples']} "
            f"manifold={group.get('repair_manifold_watertight', 0)}/{group['samples']} "
            f"mean_boundary={group['mean_boundary_edges']:.2f}"
        )
    print(f"Wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
