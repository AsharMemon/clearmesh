#!/usr/bin/env python3
"""Evaluate voxel-local FACE patches after stitching back to whole meshes.

Patch-level token accuracy is not enough for the Localizable FACE decision. This
probe answers the production-relevant question: if every voxel-local decoder
emits its patch independently, does the stitched mesh preserve coherent global
faces/topology, or do seams/invalid indices destroy the object?

Two decode strategies are supported:

* teacher_identity: stitch the packed target patches back together. This should
  exactly reconstruct the source indexed faces and is a sanity check for the
  packed dataset/evaluator.
* greedy: load a tiny local patch decoder checkpoint, greedily generate each
  patch to the target patch face count, map local vertex indices back to the
  source vertex table, then stitch all generated patches into one mesh.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clearmesh.eval.mesh_quality import evaluate_mesh, evaluate_mesh_pair
from clearmesh.mesh_heads.face_arae import build_tiny_point_conditioned_indexed_face_decoder
from clearmesh.mesh_heads.face_indexed import (
    FaceIndexedSequence,
    decode_indexed_face_tokens_to_mesh,
    indexed_to_coordinate_tokens,
)
from clearmesh.mesh_heads.face_tokens import FaceTokenTransform
from clearmesh.mesh_heads.face_topology import face_token_topology_report
from clearmesh.mesh_heads.localizable_face import reconstruct_source_faces_from_packed_arrays
from scripts.research.train_face_indexed_conditioned_tiny import _make_batch
from scripts.research.train_localizable_face_patch_tiny import (
    PackedPatchMeta,
    PackedPatchRef,
    _load_patch_sample,
    _scan_packed_patches,
)


def _load_checkpoint(path: Path, *, map_location):  # type: ignore[no-untyped-def]
    import torch

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _checkpoint_arg(checkpoint: dict[str, Any], name: str, default: Any) -> Any:
    args = checkpoint.get("args") or {}
    return args.get(name, default)


def _device(name: str):  # type: ignore[no-untyped-def]
    import torch

    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def _build_model(checkpoint: dict[str, Any], meta: PackedPatchMeta, *, device):  # type: ignore[no-untyped-def]
    model = build_tiny_point_conditioned_indexed_face_decoder(
        num_bins=int(meta.num_bins),
        max_vertices=int(meta.max_vertices),
        max_faces=int(meta.max_faces),
        point_feature_dim=6,
        hidden_size=int(_checkpoint_arg(checkpoint, "hidden_size", 128)),
        layers=int(_checkpoint_arg(checkpoint, "layers", 4)),
        heads=int(_checkpoint_arg(checkpoint, "heads", 4)),
        condition_tokens=int(_checkpoint_arg(checkpoint, "condition_tokens", 8)),
        condition_backend="pooled",
        decoder_backend="prefix",
        face_output_mode="geometry",
    ).to(device)
    state = checkpoint.get("model_state") or checkpoint
    model.load_state_dict(state)
    model.eval()
    return model



def _meta_with_checkpoint_shape(checkpoint: dict[str, Any], meta: PackedPatchMeta) -> PackedPatchMeta:
    """Use training-time patch caps when loading a checkpoint.

    Eval may inspect only a subset of sources whose max patch shape is smaller
    than the training set. The model heads are sized by the training-time max
    vertices/faces, so rebuilding from the subset alone can make checkpoint load
    fail even though every eval patch is valid for the model.
    """

    summary = checkpoint.get("summary") or {}
    return PackedPatchMeta(
        refs=meta.refs,
        num_bins=int(summary.get("num_bins", meta.num_bins)),
        max_vertices=max(int(meta.max_vertices), int(summary.get("max_vertices", meta.max_vertices))),
        max_faces=max(int(meta.max_faces), int(summary.get("max_faces", meta.max_faces))),
        max_points=max(int(meta.max_points), int(summary.get("max_points", meta.max_points))),
    )

def _manifest_rows(manifest: Path, *, limit_sources: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    root = manifest.parent
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        path = Path(str(row["path"]))
        if not path.is_absolute() and not path.exists():
            path = root / path
        row["resolved_path"] = str(path)
        source_path = row.get("source_path")
        if source_path:
            source = Path(str(source_path))
            if not source.is_absolute() and not source.exists():
                source = root / source
            row["resolved_source_path"] = str(source)
        rows.append(row)
        if limit_sources and len(rows) >= int(limit_sources):
            break
    return rows


def _load_transform(source_path: Path | None) -> FaceTokenTransform:
    if source_path is not None and source_path.exists():
        try:
            with np.load(source_path) as data:
                if "center" in data and "scale" in data:
                    return FaceTokenTransform(
                        center=tuple(float(x) for x in np.asarray(data["center"]).reshape(3)),
                        scale=float(np.asarray(data["scale"]).reshape(-1)[0]),
                    )
        except Exception:
            pass
    return FaceTokenTransform(center=(0.0, 0.0, 0.0), scale=1.0)


def _load_source_sequence(row: dict[str, Any]) -> tuple[FaceIndexedSequence, dict[str, np.ndarray]]:
    path = Path(str(row["resolved_path"]))
    with np.load(path) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}
    face_count = int(np.asarray(arrays["source_face_count"]).reshape(-1)[0])
    faces = reconstruct_source_faces_from_packed_arrays(
        patch_faces_flat=np.asarray(arrays["patch_faces_flat"], dtype=np.int64),
        patch_face_offsets=np.asarray(arrays["patch_face_offsets"], dtype=np.int64),
        global_vertex_indices_flat=np.asarray(arrays["global_vertex_indices_flat"], dtype=np.int64),
        patch_vertex_offsets=np.asarray(arrays["patch_vertex_offsets"], dtype=np.int64),
        source_face_indices_flat=np.asarray(arrays["source_face_indices_flat"], dtype=np.int64),
        face_count=face_count,
    )
    source_path_text = row.get("resolved_source_path")
    transform = _load_transform(Path(str(source_path_text)) if source_path_text else None)
    sequence = FaceIndexedSequence(
        vertices=np.asarray(arrays["source_vertices"], dtype=np.int64),
        faces=np.asarray(faces, dtype=np.int64),
        num_bins=int(np.asarray(arrays["num_bins"]).reshape(-1)[0]),
        transform=transform,
    )
    return sequence, arrays


def _face_key(face: np.ndarray) -> tuple[int, int, int]:
    return tuple(int(value) for value in np.asarray(face, dtype=np.int64).reshape(3))


def _unordered_face_key(face: np.ndarray) -> tuple[int, int, int]:
    return tuple(sorted(int(value) for value in np.asarray(face, dtype=np.int64).reshape(3)))


def _dedupe_valid_faces(faces: list[np.ndarray]) -> np.ndarray:
    kept: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    for face in faces:
        values = tuple(int(v) for v in np.asarray(face, dtype=np.int64).reshape(3))
        if len(set(values)) != 3:
            continue
        key = _unordered_face_key(np.asarray(values, dtype=np.int64))
        if key in seen:
            continue
        seen.add(key)
        kept.append(values)
    if not kept:
        return np.zeros((0, 3), dtype=np.int64)
    return np.asarray(kept, dtype=np.int64)


def _coverage_metrics(pred_faces: np.ndarray, source_faces: np.ndarray) -> dict[str, Any]:
    pred = np.asarray(pred_faces, dtype=np.int64).reshape(-1, 3)
    src = np.asarray(source_faces, dtype=np.int64).reshape(-1, 3)
    pred_ordered = [_face_key(face) for face in pred]
    src_ordered = [_face_key(face) for face in src]
    pred_unordered = [_unordered_face_key(face) for face in pred]
    src_unordered = [_unordered_face_key(face) for face in src]
    pred_ordered_set = set(pred_ordered)
    src_ordered_set = set(src_ordered)
    pred_unordered_set = set(pred_unordered)
    src_unordered_set = set(src_unordered)
    ordered_hits = sum(1 for key in src_ordered if key in pred_ordered_set)
    unordered_hits = sum(1 for key in src_unordered if key in pred_unordered_set)
    duplicate_unordered = len(pred_unordered) - len(pred_unordered_set)
    return {
        "predicted_faces": int(len(pred)),
        "source_faces": int(len(src)),
        "ordered_source_face_coverage": float(ordered_hits / max(1, len(src_ordered))),
        "unordered_source_face_coverage": float(unordered_hits / max(1, len(src_unordered))),
        "predicted_extra_unordered_faces": int(len(pred_unordered_set - src_unordered_set)),
        "predicted_duplicate_unordered_faces": int(duplicate_unordered),
    }


def _write_mesh(sequence: FaceIndexedSequence, path: Path) -> dict[str, Any]:
    mesh = decode_indexed_face_tokens_to_mesh(sequence, denormalize=True, process=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path)
    return evaluate_mesh(path)


def _pair_metrics(pred_path: Path, ref_path: Path, *, samples: int, seed: int) -> dict[str, Any] | None:
    if samples <= 0:
        return None
    try:
        return evaluate_mesh_pair(pred_path, ref_path, samples=int(samples), seed=int(seed))
    except Exception as exc:  # noqa: BLE001 - report, don't crash the probe.
        return {"error": f"{type(exc).__name__}: {exc}"}


def _teacher_faces_from_arrays(arrays: dict[str, np.ndarray]) -> np.ndarray:
    return reconstruct_source_faces_from_packed_arrays(
        patch_faces_flat=np.asarray(arrays["patch_faces_flat"], dtype=np.int64),
        patch_face_offsets=np.asarray(arrays["patch_face_offsets"], dtype=np.int64),
        global_vertex_indices_flat=np.asarray(arrays["global_vertex_indices_flat"], dtype=np.int64),
        patch_vertex_offsets=np.asarray(arrays["patch_vertex_offsets"], dtype=np.int64),
        source_face_indices_flat=np.asarray(arrays["source_face_indices_flat"], dtype=np.int64),
        face_count=int(np.asarray(arrays["source_face_count"]).reshape(-1)[0]),
    )


def _generate_patch_faces(model, ref: PackedPatchRef, meta: PackedPatchMeta, *, point_samples: int, device):  # type: ignore[no-untyped-def]
    import torch

    sample = _load_patch_sample(ref)
    (
        point_features,
        vertex_table,
        _input_faces,
        target_faces,
        _target_weights,
        *_rest,
        face_counts,
    ) = _make_batch(
        [sample],
        max_vertices=meta.max_vertices,
        max_faces=meta.max_faces,
        device=device,
        point_samples=point_samples,
    )
    face_count = int(face_counts.reshape(-1)[0].detach().cpu())
    generated = torch.full_like(target_faces, -1)
    with torch.no_grad():
        for face_index in range(face_count):
            logits = model(point_features, vertex_table, generated)
            generated[:, face_index] = logits[:, face_index].argmax(dim=-1)
    return np.asarray(generated[0, :face_count].detach().cpu(), dtype=np.int64)


def _greedy_stitched_faces(
    *,
    model,
    row: dict[str, Any],
    arrays: dict[str, np.ndarray],
    meta: PackedPatchMeta,
    point_samples: int,
    device,
) -> tuple[np.ndarray, dict[str, Any]]:
    path = Path(str(row["resolved_path"]))
    face_offsets = np.asarray(arrays["patch_face_offsets"], dtype=np.int64)
    vertex_offsets = np.asarray(arrays["patch_vertex_offsets"], dtype=np.int64)
    global_vertices_flat = np.asarray(arrays["global_vertex_indices_flat"], dtype=np.int64)
    patch_count = int(len(face_offsets) - 1)
    generated_global_faces: list[np.ndarray] = []
    invalid_faces = 0
    degenerate_faces = 0
    empty_patch_outputs = 0
    for patch_index in range(patch_count):
        f0, f1 = int(face_offsets[patch_index]), int(face_offsets[patch_index + 1])
        v0, v1 = int(vertex_offsets[patch_index]), int(vertex_offsets[patch_index + 1])
        local_vertex_count = max(0, v1 - v0)
        target_count = max(0, f1 - f0)
        if target_count <= 0:
            continue
        local_faces = _generate_patch_faces(
            model,
            PackedPatchRef(path=path, patch_index=patch_index),
            meta,
            point_samples=point_samples,
            device=device,
        )
        if len(local_faces) == 0:
            empty_patch_outputs += 1
            continue
        valid_mask = np.all((local_faces >= 0) & (local_faces < local_vertex_count), axis=1)
        invalid_faces += int(np.sum(~valid_mask))
        local_faces = local_faces[valid_mask]
        if len(local_faces) == 0:
            empty_patch_outputs += 1
            continue
        nondegenerate = np.asarray([len(set(int(v) for v in face)) == 3 for face in local_faces], dtype=bool)
        degenerate_faces += int(np.sum(~nondegenerate))
        local_faces = local_faces[nondegenerate]
        if len(local_faces) == 0:
            empty_patch_outputs += 1
            continue
        patch_global_vertices = global_vertices_flat[v0:v1]
        generated_global_faces.extend(np.asarray(patch_global_vertices[local_faces], dtype=np.int64))
    pred_faces = _dedupe_valid_faces(generated_global_faces)
    stats = {
        "patches": int(patch_count),
        "invalid_generated_faces": int(invalid_faces),
        "degenerate_generated_faces": int(degenerate_faces),
        "empty_patch_outputs": int(empty_patch_outputs),
    }
    return pred_faces, stats


def _evaluate_one(
    row: dict[str, Any],
    *,
    strategy: str,
    model,
    meta: PackedPatchMeta | None,
    point_samples: int,
    device,
    export_dir: Path | None,
    temp_dir: Path,
    pair_samples: int,
    seed: int,
) -> dict[str, Any]:
    source_sequence, arrays = _load_source_sequence(row)
    if strategy == "teacher_identity":
        pred_faces = _teacher_faces_from_arrays(arrays)
        generation_stats = {"patches": int(len(np.asarray(arrays["patch_face_offsets"])) - 1)}
    elif strategy == "greedy":
        if model is None or meta is None:
            raise ValueError("greedy strategy requires --checkpoint")
        pred_faces, generation_stats = _greedy_stitched_faces(
            model=model,
            row=row,
            arrays=arrays,
            meta=meta,
            point_samples=point_samples,
            device=device,
        )
    else:  # pragma: no cover - argparse prevents this.
        raise ValueError(f"unknown strategy {strategy!r}")

    pred_sequence = FaceIndexedSequence(
        vertices=source_sequence.vertices,
        faces=pred_faces,
        num_bins=source_sequence.num_bins,
        transform=source_sequence.transform,
    )
    source_topology = face_token_topology_report(indexed_to_coordinate_tokens(source_sequence)).to_dict()
    pred_topology = face_token_topology_report(indexed_to_coordinate_tokens(pred_sequence)).to_dict()
    source_id = Path(str(row["resolved_path"])).stem
    ref_path = temp_dir / f"{source_id}.reference.obj"
    pred_path = temp_dir / f"{source_id}.{strategy}.obj"
    ref_mesh = _write_mesh(source_sequence, ref_path)
    pred_mesh = None
    pair = None
    if len(pred_faces) > 0:
        pred_mesh = _write_mesh(pred_sequence, pred_path)
        pair = _pair_metrics(pred_path, ref_path, samples=pair_samples, seed=seed)
        if export_dir is not None:
            export_dir.mkdir(parents=True, exist_ok=True)
            (export_dir / ref_path.name).write_bytes(ref_path.read_bytes())
            (export_dir / pred_path.name).write_bytes(pred_path.read_bytes())
    else:
        pred_mesh = {"ok": False, "error": "empty predicted face set"}

    coverage = _coverage_metrics(pred_faces, source_sequence.faces)
    return {
        "path": str(row["resolved_path"]),
        "source_path": str(row.get("resolved_source_path", "")),
        "strategy": strategy,
        **generation_stats,
        **coverage,
        "teacher_identity_exact_faces": bool(np.array_equal(pred_faces, source_sequence.faces)) if strategy == "teacher_identity" else None,
        "source_topology": source_topology,
        "predicted_topology": pred_topology,
        "source_mesh": ref_mesh,
        "predicted_mesh": pred_mesh,
        "pair_metrics": pair,
    }


def _mean(values: list[float]) -> float | None:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not clean:
        return None
    return float(sum(clean) / len(clean))


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sources": int(len(results)),
        "mean_ordered_source_face_coverage": _mean([r["ordered_source_face_coverage"] for r in results]),
        "mean_unordered_source_face_coverage": _mean([r["unordered_source_face_coverage"] for r in results]),
        "mean_predicted_extra_unordered_faces": _mean([r["predicted_extra_unordered_faces"] for r in results]),
        "mean_predicted_duplicate_unordered_faces": _mean([r["predicted_duplicate_unordered_faces"] for r in results]),
        "predicted_watertight_count": int(sum(bool((r.get("predicted_mesh") or {}).get("watertight")) for r in results)),
        "predicted_ok_count": int(sum(bool((r.get("predicted_mesh") or {}).get("ok")) for r in results)),
        "mean_predicted_boundary_edges": _mean([
            (r.get("predicted_mesh") or {}).get("boundary_edge_count") for r in results
        ]),
        "mean_predicted_nonmanifold_edges": _mean([
            (r.get("predicted_mesh") or {}).get("nonmanifold_edge_count") for r in results
        ]),
        "mean_chamfer_l2_normalized": _mean([
            ((r.get("pair_metrics") or {}).get("chamfer_l2_normalized")) for r in results
        ]),
        "mean_normal_consistency": _mean([
            ((r.get("pair_metrics") or {}).get("normal_consistency")) for r in results
        ]),
        "teacher_identity_exact_count": int(sum(bool(r.get("teacher_identity_exact_faces")) for r in results)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--strategy", choices=["teacher_identity", "greedy", "both"], default="teacher_identity")
    parser.add_argument("--limit-sources", type=int, default=0)
    parser.add_argument("--limit-patches", type=int, default=0, help="Patch scan cap for building greedy checkpoint metadata.")
    parser.add_argument("--point-samples", type=int, default=32)
    parser.add_argument("--pair-samples", type=int, default=5000)
    parser.add_argument("--export-mesh-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    rows = _manifest_rows(args.manifest, limit_sources=args.limit_sources)
    if not rows:
        raise SystemExit(f"No packed sources found in {args.manifest}")
    strategies = ["teacher_identity", "greedy"] if args.strategy == "both" else [args.strategy]
    device = None
    model = None
    meta = None
    checkpoint_summary = None
    if "greedy" in strategies:
        device = _device(args.device)
        if args.checkpoint is None:
            raise SystemExit("--checkpoint is required for greedy stitch evaluation")
        checkpoint = _load_checkpoint(args.checkpoint, map_location=device)
        scanned_meta = _scan_packed_patches(
            args.manifest,
            limit_sources=args.limit_sources,
            limit_patches=args.limit_patches,
        )
        meta = _meta_with_checkpoint_shape(checkpoint, scanned_meta)
        model = _build_model(checkpoint, meta, device=device)
        checkpoint_summary = {
            "checkpoint": str(args.checkpoint),
            "checkpoint_best_loss": checkpoint.get("best_loss"),
            "checkpoint_best_step": checkpoint.get("best_step"),
            "checkpoint_summary": checkpoint.get("summary"),
            "meta_max_vertices": int(meta.max_vertices),
            "meta_max_faces": int(meta.max_faces),
        }

    all_results: dict[str, list[dict[str, Any]]] = {}
    with tempfile.TemporaryDirectory(prefix="clearmesh_localizable_stitch_") as temp_text:
        temp_dir = Path(temp_text)
        for strategy in strategies:
            results = []
            for row_index, row in enumerate(rows):
                results.append(
                    _evaluate_one(
                        row,
                        strategy=strategy,
                        model=model,
                        meta=meta,
                        point_samples=args.point_samples,
                        device=device,
                        export_dir=args.export_mesh_dir,
                        temp_dir=temp_dir,
                        pair_samples=args.pair_samples,
                        seed=args.seed + row_index,
                    )
                )
            all_results[strategy] = results

    report = {
        "manifest": str(args.manifest),
        "device": str(device),
        "strategies": strategies,
        "source_count": len(rows),
        "point_samples": int(args.point_samples),
        "pair_samples": int(args.pair_samples),
        "checkpoint": checkpoint_summary,
        "summary": {strategy: _aggregate(results) for strategy, results in all_results.items()},
        "results": all_results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(args.report), "summary": report["summary"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
