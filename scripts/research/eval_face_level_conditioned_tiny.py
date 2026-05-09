#!/usr/bin/env python3
"""Evaluate a tiny point-conditioned, face-level FACE checkpoint."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clearmesh.eval.mesh_quality import evaluate_mesh, evaluate_mesh_pair
from clearmesh.mesh_heads.face_arae import build_tiny_point_conditioned_face_level_decoder
from clearmesh.mesh_heads.face_tokens import (
    FaceTokenSequence,
    FaceTokenTransform,
    decode_face_tokens_to_mesh,
)
from clearmesh.mesh_heads.face_topology import face_token_topology_report, repair_face_tokens
from scripts.research.sample_face_level_conditioned_tiny import _load_checkpoint, _load_conditioning


def _generate_tokens(
    model,  # type: ignore[no-untyped-def]
    point_features: np.ndarray,
    face_count: int,
    num_bins: int,
    device,  # type: ignore[no-untyped-def]
    temperature: float,
    *,
    max_face_count: int | None = None,
    closure_extra_faces: int = 0,
    closure_repair_mode: str = "manifold",
):
    import torch

    stop_at = int(face_count)
    if closure_extra_faces > 0:
        stop_at = min(int(max_face_count or face_count), int(face_count) + int(closure_extra_faces))
    point_tensor = torch.as_tensor(point_features, dtype=torch.float32, device=device).unsqueeze(0)
    input_faces = torch.zeros((1, 1, 9), dtype=torch.long, device=device)
    generated: list[np.ndarray] = []
    with torch.no_grad():
        for _ in range(stop_at):
            logits = model(point_tensor, input_faces)[:, -1, :, :]
            if temperature and temperature > 0.0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_face = torch.multinomial(probs.reshape(-1, num_bins), num_samples=1).reshape(1, 9)
            else:
                next_face = torch.argmax(logits, dim=-1)
            generated.append(next_face.squeeze(0).detach().cpu().numpy().astype(np.int64))
            if len(generated) >= face_count and _tokens_are_closed(np.stack(generated, axis=0), closure_repair_mode):
                break
            if len(generated) < stop_at:
                input_faces = torch.cat([input_faces, next_face.unsqueeze(1)], dim=1)
    return np.stack(generated, axis=0)


def _tokens_are_closed(tokens: np.ndarray, repair_mode: str) -> bool:
    if len(tokens) == 0:
        return False
    inspected = tokens
    if repair_mode != "none":
        inspected, _ = repair_face_tokens(tokens, mode=repair_mode)
    if len(inspected) == 0:
        return False
    return bool(face_token_topology_report(inspected).watertight_edge_graph)


def _select_face_count(
    *,
    model,  # type: ignore[no-untyped-def]
    point_features: np.ndarray,
    gt_face_count: int,
    max_faces: int,
    mode: str,
    device,  # type: ignore[no-untyped-def]
) -> tuple[int, int | None]:
    """Choose the generation length for evaluation.

    ``gt`` is useful for isolating token quality from sequence-length mistakes.
    ``predicted`` is the production setting: the model has to choose how long
    the mesh should be from the conditioning points alone.
    """

    if mode == "gt":
        return max(1, min(int(gt_face_count), int(max_faces))), None
    if mode == "max":
        return int(max_faces), None
    if mode != "predicted":
        raise ValueError(f"unknown face-count mode {mode}")

    import torch

    point_tensor = torch.as_tensor(point_features, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        predicted = int(torch.argmax(model.predict_face_count_logits(point_tensor), dim=-1).item())
    return max(1, min(int(max_faces), predicted)), predicted


def _aggregate(
    results: list[dict[str, Any]],
    *,
    mesh_key: str = "mesh_metrics",
    pair_key: str = "pair_metrics",
) -> dict[str, Any]:
    attempted = len(results)
    ok_results = [result for result in results if result.get(mesh_key, {}).get("ok")]
    watertight = [result for result in ok_results if result.get(mesh_key, {}).get("watertight")]
    pair_results = [
        result[pair_key]
        for result in results
        if isinstance(result.get(pair_key), dict) and result[pair_key].get("chamfer_l2") is not None
    ]
    return {
        "attempted": attempted,
        "mesh_ok": len(ok_results),
        "mesh_ok_rate": len(ok_results) / max(1, attempted),
        "watertight": len(watertight),
        "watertight_rate": len(watertight) / max(1, attempted),
        "mean_chamfer_l2": _mean(pair_results, "chamfer_l2"),
        "mean_hausdorff_l2": _mean(pair_results, "hausdorff_l2"),
        "mean_normal_consistency": _mean(pair_results, "normal_consistency"),
    }


def _mean(items: list[dict[str, Any]], key: str) -> float | None:
    values = [float(item[key]) for item in items if item.get(key) is not None]
    if not values:
        return None
    return float(np.mean(values))


def _aggregate_token_topology(results: list[dict[str, Any]], key: str) -> dict[str, Any]:
    reports = [result[key] for result in results if isinstance(result.get(key), dict)]
    if not reports:
        return {
            "attempted": len(results),
            "reported": 0,
        }
    watertight = [report for report in reports if report.get("watertight_edge_graph")]
    return {
        "attempted": len(results),
        "reported": len(reports),
        "watertight_edge_graph": len(watertight),
        "watertight_edge_graph_rate": len(watertight) / max(1, len(reports)),
        "mean_boundary_edge_count": _mean(reports, "boundary_edge_count"),
        "mean_nonmanifold_edge_count": _mean(reports, "nonmanifold_edge_count"),
        "mean_duplicate_face_count": _mean(reports, "duplicate_face_count"),
        "mean_degenerate_face_count": _mean(reports, "degenerate_face_count"),
        "mean_edge_pairing_ratio": _mean(reports, "edge_pairing_ratio"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--export-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--point-samples", type=int, default=0)
    parser.add_argument("--pair-samples", type=int, default=2000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--face-count-mode",
        choices=["gt", "predicted", "max"],
        default="gt",
        help="Use ground-truth count, model-predicted count, or max_faces for generation length.",
    )
    parser.add_argument("--cleanup-export-dir", type=Path, default=None)
    parser.add_argument("--cleanup-fill-holes", action="store_true")
    parser.add_argument("--cleanup-min-component-faces", type=int, default=1)
    parser.add_argument(
        "--token-repair-mode",
        choices=["none", "dedupe", "manifold"],
        default="none",
        help="Conservative pre-decode repair for generated FACE tokens.",
    )
    parser.add_argument(
        "--closure-extra-faces",
        type=int,
        default=0,
        help="Allow this many extra AR faces beyond the selected count until token edges close.",
    )
    parser.add_argument(
        "--closure-repair-mode",
        choices=["none", "dedupe", "manifold"],
        default="manifold",
        help="Token view used to decide whether closure-extra generation can stop.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = parser.parse_args()

    import torch

    checkpoint = _load_checkpoint(args.checkpoint)
    train_args = checkpoint.get("args", {})
    num_bins = int(checkpoint["num_bins"])
    max_faces = int(checkpoint["max_faces"])

    paths = sorted(args.dataset_dir.glob("*.npz"))
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit(f"No FACE shards found in {args.dataset_dir}")

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    model = build_tiny_point_conditioned_face_level_decoder(
        num_bins=num_bins,
        max_faces=max_faces,
        point_feature_dim=6,
        hidden_size=int(train_args.get("hidden_size", 192)),
        layers=int(train_args.get("layers", 4)),
        heads=int(train_args.get("heads", 6)),
        condition_tokens=int(train_args.get("condition_tokens", 8)),
    ).to(device)
    incompatible = model.load_state_dict(checkpoint["model_state"], strict=False)
    model.eval()

    if args.export_dir is not None:
        args.export_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for index, path in enumerate(paths):
        try:
            point_features, transform, sample_bins, gt_tokens = _load_conditioning(
                path,
                point_samples=args.point_samples or train_args.get("point_samples") or None,
            )
            if int(sample_bins) != num_bins:
                raise ValueError(f"{path} num_bins={sample_bins}, checkpoint num_bins={num_bins}")
            gt_face_count = min(int(len(gt_tokens)), max_faces)
            face_count, predicted_face_count = _select_face_count(
                model=model,
                point_features=point_features,
                gt_face_count=gt_face_count,
                max_faces=max_faces,
                mode=args.face_count_mode,
                device=device,
            )
            generated_tokens = _generate_tokens(
                model=model,
                point_features=point_features,
                face_count=face_count,
                num_bins=num_bins,
                device=device,
                temperature=args.temperature,
                max_face_count=max_faces,
                closure_extra_faces=max(0, int(args.closure_extra_faces)),
                closure_repair_mode=args.closure_repair_mode,
            )
            raw_generated_tokens = generated_tokens
            token_repair_report = None
            if args.token_repair_mode != "none":
                generated_tokens, token_repair_report = repair_face_tokens(generated_tokens, mode=args.token_repair_mode)
                if len(generated_tokens) == 0:
                    generated_tokens = raw_generated_tokens
            generated = decode_face_tokens_to_mesh(
                FaceTokenSequence(tokens=generated_tokens, num_bins=num_bins, transform=transform)
            )
            teacher_tokens = np.asarray(gt_tokens[:face_count], dtype=np.int64)
            teacher = decode_face_tokens_to_mesh(
                FaceTokenSequence(tokens=teacher_tokens, num_bins=num_bins, transform=transform)
            )
            if args.export_dir is None:
                generated_path = args.output.parent / f"{index:04d}_generated.glb"
                teacher_path = args.output.parent / f"{index:04d}_teacher.glb"
            else:
                generated_path = args.export_dir / f"{index:04d}_{path.stem}_generated.glb"
                teacher_path = args.export_dir / f"{index:04d}_{path.stem}_teacher.glb"
            generated_path.parent.mkdir(parents=True, exist_ok=True)
            generated.export(generated_path)
            teacher.export(teacher_path)
            mesh_metrics = evaluate_mesh(generated_path)
            try:
                pair_metrics = evaluate_mesh_pair(generated_path, teacher_path, samples=args.pair_samples)
            except Exception as exc:  # noqa: BLE001 - report per-shard pair failures.
                pair_metrics = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            item = {
                "index": index,
                "sample": str(path),
                "generated_path": str(generated_path),
                "teacher_path": str(teacher_path),
                "face_count_mode": args.face_count_mode,
                "face_count": face_count,
                "gt_face_count": gt_face_count,
                "predicted_face_count": predicted_face_count,
                "closure_extra_faces": int(args.closure_extra_faces),
                "closure_repair_mode": args.closure_repair_mode,
                "coordinate_tokens": int(len(raw_generated_tokens) * 9),
                "autoregressive_steps": int(len(raw_generated_tokens)),
                "token_repair_mode": args.token_repair_mode,
                "raw_generated_face_count": int(len(raw_generated_tokens)),
                "decoded_generated_face_count": int(len(generated_tokens)),
                "raw_token_topology": face_token_topology_report(raw_generated_tokens).to_dict(),
                "decoded_token_topology": face_token_topology_report(generated_tokens).to_dict(),
                "teacher_token_topology": face_token_topology_report(teacher_tokens).to_dict(),
                "mesh_metrics": mesh_metrics,
                "pair_metrics": pair_metrics,
            }
            if token_repair_report is not None:
                item["token_repair_report"] = token_repair_report.to_dict()
            if args.cleanup_export_dir is not None:
                from clearmesh.mesh.cleanup import CleanupOptions, cleanup_mesh_file

                args.cleanup_export_dir.mkdir(parents=True, exist_ok=True)
                cleanup_path = args.cleanup_export_dir / f"{index:04d}_{path.stem}_cleaned.glb"
                try:
                    cleanup_report = cleanup_mesh_file(
                        generated_path,
                        cleanup_path,
                        CleanupOptions(
                            min_component_faces=args.cleanup_min_component_faces,
                            fill_holes=args.cleanup_fill_holes,
                            merge_vertices=True,
                            fix_normals=True,
                        ),
                    )
                    item["cleanup_generated_path"] = str(cleanup_path)
                    item["cleanup_report"] = asdict(cleanup_report)
                    item["cleanup_mesh_metrics"] = evaluate_mesh(cleanup_path)
                    try:
                        item["cleanup_pair_metrics"] = evaluate_mesh_pair(
                            cleanup_path,
                            teacher_path,
                            samples=args.pair_samples,
                        )
                    except Exception as exc:  # noqa: BLE001 - report per-shard cleanup pair failures.
                        item["cleanup_pair_metrics"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                except Exception as exc:  # noqa: BLE001 - cleanup is diagnostic, not batch-fatal.
                    item["cleanup_generated_path"] = str(cleanup_path)
                    item["cleanup_error"] = f"{type(exc).__name__}: {exc}"
                    item["cleanup_mesh_metrics"] = {"ok": False, "error": item["cleanup_error"]}
            results.append(item)
        except Exception as exc:  # noqa: BLE001 - keep batch evaluation moving.
            results.append(
                {
                    "index": index,
                    "sample": str(path),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    report = {
        "checkpoint": str(args.checkpoint),
        "dataset_dir": str(args.dataset_dir),
        "device": str(device),
        "face_count_mode": args.face_count_mode,
        "token_repair_mode": args.token_repair_mode,
        "missing_checkpoint_keys": list(incompatible.missing_keys),
        "unexpected_checkpoint_keys": list(incompatible.unexpected_keys),
        "summary": _aggregate(results),
        "token_summary": {
            "raw_generated": _aggregate_token_topology(results, "raw_token_topology"),
            "decoded_generated": _aggregate_token_topology(results, "decoded_token_topology"),
            "teacher": _aggregate_token_topology(results, "teacher_token_topology"),
        },
        "results": results,
    }
    if args.cleanup_export_dir is not None:
        report["cleanup_summary"] = _aggregate(
            results,
            mesh_key="cleanup_mesh_metrics",
            pair_key="cleanup_pair_metrics",
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
