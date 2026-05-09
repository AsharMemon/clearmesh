#!/usr/bin/env python3
"""Build a curated FACE training corpus manifest from downloaded 3D assets.

This is the practical replacement for "train on 130K curated meshes". It does
not assume raw public assets are already clean. Instead it ranks candidates for
UltraShape/manifoldization and strict FACE token-gate promotion.

Inputs can be either:
  - Objaverse download manifest: {uid: local_path}
  - candidate JSON list with {uid/objaverse_uid, path, quality_score, ...}

The output JSONL contains only candidates that pass cheap semantic/geometric
filters. Downstream steps should run:
  prepare_face_strict_targets.py -> face_token_oracle.py ->
  select_face_oracle_passes.py -> build_face_token_dataset.py
  --indexed-face-order boundary_growth
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import queue
import struct
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import trimesh


QUALITY_MAP = {
    "low": 0,
    "medium": 1,
    "high": 2,
    "superior": 3,
    "excellent": 3,
}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_jsonl(path: Path):
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            yield json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number} is not valid JSON") from exc


def _load_candidate_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return [dict(row) for row in _iter_jsonl(path)]
    data = _load_json(path)
    if isinstance(data, dict):
        records = []
        for uid, local_path in data.items():
            if isinstance(local_path, dict):
                row = dict(local_path)
                row.setdefault("uid", uid)
            else:
                row = {"uid": uid, "path": str(local_path)}
            records.append(row)
        return records
    if isinstance(data, list):
        return [dict(row) for row in data]
    raise ValueError(f"unsupported candidate manifest format: {path}")


def _record_uid(record: dict[str, Any]) -> str:
    for key in ("uid", "UID", "objaverse_uid", "sha256"):
        value = record.get(key)
        if value:
            return str(value)
    return ""


def _load_annotations(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    if path.suffix.lower() == ".jsonl":
        rows = list(_iter_jsonl(path))
    else:
        data = _load_json(path)
        if isinstance(data, dict):
            rows = []
            for uid, value in data.items():
                row = dict(value) if isinstance(value, dict) else {"score": value}
                row.setdefault("uid", uid)
                rows.append(row)
        elif isinstance(data, list):
            rows = data
        else:
            raise ValueError(f"unsupported annotation format: {path}")
    annotations = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        uid = _record_uid(row)
        if uid:
            annotations[uid] = row
    return annotations


def _quality_score(record: dict[str, Any], annotation: dict[str, Any] | None) -> int:
    merged = {}
    if annotation:
        merged.update(annotation)
    merged.update(record)
    for key in ("quality_score", "score", "quality", "Quality", "quality_label"):
        if key not in merged or merged[key] is None:
            continue
        value = merged[key]
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return int(value)
        label = str(value).strip().lower().replace(" quality", "")
        if label in QUALITY_MAP:
            return QUALITY_MAP[label]
    return -1


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _annotation_rejects(record: dict[str, Any], annotation: dict[str, Any] | None) -> list[str]:
    merged = {}
    if annotation:
        merged.update(annotation)
    merged.update(record)
    rejects = []
    reject_keys = {
        "transparent": "transparent asset",
        "transparency": "transparent asset",
        "is_transparent": "transparent asset",
        "scene": "scene/environment asset",
        "is_scene": "scene/environment asset",
        "not_single_object": "not a single object",
        "not_a_single_object": "not a single object",
        "multi_object": "not a single object",
        "is_multi_object": "not a single object",
        "multiple_objects": "not a single object",
    }
    for key, reason in reject_keys.items():
        if _truthy(merged.get(key)):
            rejects.append(reason)
    return sorted(set(rejects))


def _load_mesh(path: Path) -> trimesh.Trimesh | None:
    loaded = trimesh.load(path, force="scene", process=False, skip_materials=True)
    if isinstance(loaded, trimesh.Scene):
        meshes = [geom for geom in loaded.geometry.values() if isinstance(geom, trimesh.Trimesh) and len(geom.faces)]
        if not meshes:
            return None
        return trimesh.util.concatenate(meshes)
    if isinstance(loaded, trimesh.Trimesh) and len(loaded.faces):
        return loaded
    return None


def _cheap_mesh_report(path: Path, args: argparse.Namespace) -> tuple[dict[str, Any] | None, list[str]]:
    suffix = path.suffix.lower()
    if suffix not in {".glb", ".gltf"}:
        return None, [f"cheap metadata unsupported for {suffix or 'file'}"]
    try:
        document = _read_gltf_json(path)
    except Exception as exc:  # noqa: BLE001 - public assets can be malformed.
        return None, [f"cheap metadata failed: {type(exc).__name__}: {exc}"]

    accessors = document.get("accessors") or []
    face_count = 0
    vertex_count = 0
    primitive_count = 0
    unsupported_modes = 0
    for mesh in document.get("meshes") or []:
        for primitive in mesh.get("primitives") or []:
            primitive_count += 1
            mode = int(primitive.get("mode", 4))
            if mode != 4:
                unsupported_modes += 1
                continue
            attributes = primitive.get("attributes") or {}
            position_count = _accessor_count(accessors, attributes.get("POSITION"))
            indices_count = _accessor_count(accessors, primitive.get("indices"))
            vertex_count += position_count
            face_count += (indices_count // 3) if indices_count else (position_count // 3)

    report = {
        "vertex_count": int(vertex_count),
        "face_count": int(face_count),
        "component_count": 1,
        "largest_component_area_ratio": 1.0,
        "watertight_raw": False,
        "winding_consistent_raw": False,
        "aspect_ratio": 1.0,
        "degenerate_face_ratio": 0.0,
        "surface_area": 0.0,
        "cheap_metadata_only": True,
        "primitive_count": int(primitive_count),
        "unsupported_primitive_modes": int(unsupported_modes),
    }
    rejects = []
    if face_count <= 0 or vertex_count <= 0:
        rejects.append("cheap metadata found no triangle geometry")
    if face_count < args.min_faces:
        rejects.append(f"too few faces: {face_count} < {args.min_faces}")
    if face_count > args.max_faces:
        rejects.append(f"too many faces: {face_count} > {args.max_faces}")
    return report, rejects


def _read_gltf_json(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".gltf":
        return json.loads(path.read_text(encoding="utf-8"))
    with path.open("rb") as handle:
        header = handle.read(12)
        if len(header) != 12:
            raise ValueError("short GLB header")
        magic, _version, _length = struct.unpack("<III", header)
        if magic != 0x46546C67:
            raise ValueError("not a GLB file")
        while True:
            chunk_header = handle.read(8)
            if not chunk_header:
                break
            if len(chunk_header) != 8:
                raise ValueError("short GLB chunk header")
            chunk_length, chunk_type = struct.unpack("<II", chunk_header)
            chunk = handle.read(chunk_length)
            if chunk_type == 0x4E4F534A:
                return json.loads(chunk.decode("utf-8").rstrip("\x00 "))
    raise ValueError("GLB JSON chunk not found")


def _accessor_count(accessors: list[Any], index: Any) -> int:
    if index is None:
        return 0
    try:
        accessor = accessors[int(index)]
    except Exception:
        return 0
    if not isinstance(accessor, dict):
        return 0
    return int(accessor.get("count") or 0)


def _component_stats(mesh: trimesh.Trimesh) -> tuple[int, float]:
    try:
        pieces = mesh.split(only_watertight=False)
    except Exception:
        return 1, 1.0
    if not pieces:
        return 0, 0.0
    areas = np.asarray([max(float(piece.area), 0.0) for piece in pieces], dtype=np.float64)
    total = float(areas.sum())
    largest_ratio = float(areas.max() / total) if total > 0 else 0.0
    return int(len(pieces)), largest_ratio


def _mesh_report(path: Path, args: argparse.Namespace) -> tuple[dict[str, Any] | None, list[str]]:
    try:
        mesh = _load_mesh(path)
    except Exception as exc:  # noqa: BLE001 - batch curation should keep moving.
        return None, [f"load failed: {type(exc).__name__}: {exc}"]
    if mesh is None:
        return None, ["empty or unsupported mesh"]
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces)
    if not len(vertices) or not len(faces):
        return None, ["empty geometry"]
    extents = np.asarray(mesh.extents, dtype=np.float64)
    finite = bool(np.isfinite(vertices).all() and np.isfinite(extents).all())
    positive_extents = extents[extents > 1e-8]
    aspect = float(positive_extents.max() / positive_extents.min()) if len(positive_extents) else math.inf
    face_count = int(len(faces))
    vertex_count = int(len(vertices))
    component_count, largest_component_area_ratio = _component_stats(mesh)
    degenerate_ratio = _degenerate_ratio(vertices, faces)
    report = {
        "vertex_count": vertex_count,
        "face_count": face_count,
        "component_count": component_count,
        "largest_component_area_ratio": largest_component_area_ratio,
        "watertight_raw": bool(mesh.is_watertight),
        "winding_consistent_raw": bool(mesh.is_winding_consistent),
        "aspect_ratio": aspect,
        "degenerate_face_ratio": degenerate_ratio,
        "surface_area": float(mesh.area),
    }
    rejects = []
    if not finite:
        rejects.append("non-finite geometry")
    if face_count < args.min_faces:
        rejects.append(f"too few faces: {face_count} < {args.min_faces}")
    if face_count > args.max_faces:
        rejects.append(f"too many faces: {face_count} > {args.max_faces}")
    if aspect > args.max_aspect_ratio:
        rejects.append(f"aspect ratio too high: {aspect:.3f} > {args.max_aspect_ratio}")
    if component_count > args.max_components and largest_component_area_ratio < args.min_largest_component_area_ratio:
        rejects.append("fragmented mesh")
    if degenerate_ratio > args.max_degenerate_ratio:
        rejects.append(f"too many degenerates: {degenerate_ratio:.6f} > {args.max_degenerate_ratio}")
    return report, rejects


def _mesh_report_worker(path: str, args_dict: dict[str, Any], output_queue: mp.Queue) -> None:
    args = argparse.Namespace(**args_dict)
    try:
        _apply_memory_limit(float(getattr(args, "mesh_memory_limit_gb", 0.0)))
        output_queue.put(("ok", _mesh_report(Path(path), args)))
    except BaseException as exc:  # noqa: BLE001 - the parent converts this asset to a reject.
        output_queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _apply_memory_limit(limit_gb: float) -> None:
    if limit_gb <= 0:
        return
    try:
        import resource
    except Exception:
        return
    limit_bytes = int(limit_gb * 1024 * 1024 * 1024)
    try:
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
    except Exception:
        pass


def _mesh_report_with_timeout(path: Path, args: argparse.Namespace) -> tuple[dict[str, Any] | None, list[str]]:
    timeout = float(args.mesh_timeout_seconds)
    if timeout <= 0:
        return _mesh_report(path, args)

    # A few public GLBs can hang in loader/scene processing. Run each asset in a
    # child so the corpus builder keeps moving and records a normal reject.
    context = mp.get_context(str(args.mesh_worker_start_method))
    output_queue: mp.Queue = context.Queue(maxsize=1)
    process = context.Process(target=_mesh_report_worker, args=(str(path), vars(args), output_queue))
    process.daemon = True
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join(5)
        if process.is_alive():
            process.kill()
            process.join(5)
        return None, [f"mesh report timeout after {timeout:g}s"]
    try:
        status, payload = output_queue.get_nowait()
    except queue.Empty:
        return None, [f"mesh report process exited without output: {process.exitcode}"]
    if status == "error":
        return None, [f"mesh report failed: {payload}"]
    return payload


def _degenerate_ratio(vertices: np.ndarray, faces: np.ndarray) -> float:
    if len(faces) == 0:
        return 1.0
    tris = vertices[faces]
    areas = np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1) * 0.5
    return float(np.mean(areas <= 1e-12))


def _score_candidate(record: dict[str, Any], quality: int, report: dict[str, Any]) -> float:
    quality_score = {3: 1.0, 2: 0.82, 1: 0.45, 0: 0.1, -1: 0.55}.get(int(quality), 0.55)
    component_score = min(1.0, max(0.0, float(report["largest_component_area_ratio"])))
    watertight_score = 1.0 if report["watertight_raw"] else 0.72
    face_count = max(1, int(report["face_count"]))
    density_score = 1.0 - min(1.0, abs(math.log(face_count / 4000.0)) / 6.0)
    aesthetic = record.get("aesthetic_score")
    if aesthetic is None:
        aesthetic_score = 0.5
    else:
        aesthetic_score = min(1.0, max(0.0, (float(aesthetic) - 4.0) / 5.0))
    return float(
        0.38 * quality_score
        + 0.22 * component_score
        + 0.16 * watertight_score
        + 0.14 * density_score
        + 0.10 * aesthetic_score
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True, help="Objaverse manifest JSON, candidate JSON, or JSONL.")
    parser.add_argument("--objaversepp-annotations", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rejects-output", type=Path, default=None)
    parser.add_argument("--target", type=int, default=130_000)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--min-quality", type=int, default=2, help="2 keeps Objaverse++ High/Superior.")
    parser.add_argument("--allow-unscored", action="store_true", help="Allow unscored assets if geometry is strong.")
    parser.add_argument("--min-faces", type=int, default=64)
    parser.add_argument("--max-faces", type=int, default=250_000, help="Raw source cap before manifoldization/decimation.")
    parser.add_argument("--max-aspect-ratio", type=float, default=80.0)
    parser.add_argument("--max-components", type=int, default=12)
    parser.add_argument("--min-largest-component-area-ratio", type=float, default=0.85)
    parser.add_argument("--max-degenerate-ratio", type=float, default=0.02)
    parser.add_argument("--max-file-mb", type=float, default=256.0, help="Reject very large raw files before mesh loading.")
    parser.add_argument("--mesh-timeout-seconds", type=float, default=60.0, help="Per-asset mesh inspection timeout; <=0 disables.")
    parser.add_argument("--mesh-memory-limit-gb", type=float, default=12.0, help="Per-asset child memory cap; <=0 disables.")
    parser.add_argument(
        "--mesh-worker-start-method",
        choices=["spawn", "fork", "forkserver"],
        default="spawn",
        help="Use spawn by default so memory limits apply to a clean child process.",
    )
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--skip-mesh-report", action="store_true", help="Use cheap GLB/GLTF metadata only; avoids full scene loading.")
    parser.add_argument("--require-existing-path", action="store_true", default=True)
    args = parser.parse_args()

    records = _load_candidate_records(args.candidates)
    if args.limit:
        records = records[: args.limit]
    annotations = _load_annotations(args.objaversepp_annotations)
    accepted: list[dict[str, Any]] = []
    rejects: list[dict[str, Any]] = []
    reject_reasons: Counter[str] = Counter()

    for index, record in enumerate(records):
        uid = _record_uid(record)
        path_value = record.get("path") or record.get("local_path")
        path = Path(str(path_value)) if path_value else None
        reasons = []
        annotation = annotations.get(uid, {}) if uid else {}
        quality = _quality_score(record, annotation)
        reasons.extend(_annotation_rejects(record, annotation))
        if quality < args.min_quality and not (args.allow_unscored and quality < 0):
            reasons.append(f"quality {quality} < {args.min_quality}")
        if path is None:
            reasons.append("missing local path")
        elif args.require_existing_path and not path.exists():
            reasons.append("local path does not exist")
        elif path.exists() and args.max_file_mb > 0:
            size_mb = path.stat().st_size / (1024.0 * 1024.0)
            if size_mb > args.max_file_mb:
                reasons.append(f"file too large: {size_mb:.2f} MB > {args.max_file_mb:.2f} MB")
        report = None
        if not reasons and path is not None:
            if args.skip_mesh_report:
                report, mesh_reasons = _cheap_mesh_report(path, args)
            else:
                cheap_report, cheap_reasons = _cheap_mesh_report(path, args)
                if cheap_reasons and any(reason.startswith("too many faces") for reason in cheap_reasons):
                    report, mesh_reasons = cheap_report, cheap_reasons
                else:
                    report, mesh_reasons = _mesh_report_with_timeout(path, args)
            reasons.extend(mesh_reasons)
        if reasons:
            for reason in reasons:
                reject_reasons[reason] += 1
            rejects.append({"index": index, "uid": uid, "path": str(path) if path else None, "quality_score": quality, "reasons": reasons})
            if args.progress_every > 0 and (index + 1) % args.progress_every == 0:
                print(
                    json.dumps(
                        {
                            "processed": index + 1,
                            "accepted": len(accepted),
                            "rejected": len(rejects),
                            "top_reject_reasons": reject_reasons.most_common(5),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            continue
        assert path is not None and report is not None
        row = {
            "uid": uid,
            "path": str(path),
            "quality_score": quality,
            "curation_score": _score_candidate(record, quality, report),
            "source_record": {key: value for key, value in record.items() if key not in {"path", "local_path"}},
            **report,
        }
        accepted.append(row)
        if args.progress_every > 0 and (index + 1) % args.progress_every == 0:
            print(
                json.dumps(
                    {
                        "processed": index + 1,
                        "accepted": len(accepted),
                        "rejected": len(rejects),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    accepted.sort(key=lambda row: row["curation_score"], reverse=True)
    if args.target > 0:
        accepted = accepted[: args.target]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in accepted:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    if args.rejects_output:
        args.rejects_output.parent.mkdir(parents=True, exist_ok=True)
        args.rejects_output.write_text(json.dumps({"rejects": rejects, "reason_counts": reject_reasons}, indent=2, sort_keys=True), encoding="utf-8")

    summary = {
        "input_records": len(records),
        "accepted": len(accepted),
        "rejected": len(rejects),
        "target": args.target,
        "output": str(args.output),
        "top_reject_reasons": reject_reasons.most_common(20),
        "quality_distribution": Counter(str(row["quality_score"]) for row in accepted),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
