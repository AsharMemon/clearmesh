#!/usr/bin/env python3
"""Evaluate a tiny Localizable FACE patch decoder checkpoint.

This is intentionally a bounded probe, not a production benchmark. It answers
the first questions we need before spending more money:

* does the model learn local patch tokens under teacher forcing?
* can it greedily reconstruct local patches without immediate collapse?
* is count/first-face prediction improving, or are we only lowering loss?
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clearmesh.mesh_heads.face_arae import build_tiny_point_conditioned_indexed_face_decoder
from scripts.research.train_face_indexed_conditioned_tiny import _make_batch
from scripts.research.train_localizable_face_patch_tiny import (
    PackedPatchRef,
    _load_patch_sample,
    _scan_packed_patches,
)


def _device(name: str):
    import torch

    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def _json_float(value) -> float:
    return float(value.detach().cpu() if hasattr(value, "detach") else value)


def _load_checkpoint(path: Path, *, map_location):
    import torch

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _checkpoint_arg(checkpoint: dict, name: str, default):
    args = checkpoint.get("args") or {}
    return args.get(name, default)


def _build_model(checkpoint: dict, meta, *, device):
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


def _batched(refs: list[PackedPatchRef], batch_size: int):
    for start in range(0, len(refs), int(batch_size)):
        yield refs[start : start + int(batch_size)]


def _teacher_forced_metrics(model, refs: list[PackedPatchRef], meta, *, batch_size: int, point_samples: int, device):
    import torch
    import torch.nn.functional as F

    total_tokens = 0
    correct_tokens = 0
    total_faces = 0
    exact_faces = 0
    exact_patches = 0
    total_patches = 0
    count_correct = 0
    count_total = 0
    loss_sum = 0.0
    loss_weight = 0.0
    first_face_exact = 0
    first_face_total = 0
    with torch.no_grad():
        for batch_refs in _batched(refs, batch_size):
            samples = [_load_patch_sample(ref) for ref in batch_refs]
            (
                point_features,
                vertex_table,
                input_faces,
                target_faces,
                target_weights,
                *_rest,
                face_counts,
            ) = _make_batch(
                samples,
                max_vertices=meta.max_vertices,
                max_faces=meta.max_faces,
                device=device,
                point_samples=point_samples,
            )
            logits = model(point_features, vertex_table, input_faces)
            losses = F.cross_entropy(
                logits.reshape(-1, meta.max_vertices),
                target_faces.reshape(-1),
                ignore_index=-100,
                reduction="none",
            ).reshape_as(target_weights)
            valid = target_faces.ge(0)
            valid_float = valid.to(losses.dtype)
            batch_weight = (target_weights * valid_float).sum()
            if float(batch_weight.detach().cpu()) > 0:
                loss_sum += _json_float((losses * target_weights * valid_float).sum())
                loss_weight += _json_float(batch_weight)
            pred = logits.argmax(dim=-1)
            correct = pred.eq(target_faces) & valid
            total_tokens += int(valid.sum().detach().cpu())
            correct_tokens += int(correct.sum().detach().cpu())
            face_valid = valid.all(dim=-1)
            face_exact = pred.eq(target_faces).all(dim=-1) & face_valid
            total_faces += int(face_valid.sum().detach().cpu())
            exact_faces += int(face_exact.sum().detach().cpu())
            for row, count in enumerate(face_counts.detach().cpu().tolist()):
                count = int(count)
                if count <= 0:
                    continue
                total_patches += 1
                if bool(face_exact[row, :count].all().detach().cpu()):
                    exact_patches += 1
                first_face_total += 1
                if bool(face_exact[row, 0].detach().cpu()):
                    first_face_exact += 1
            count_logits = model.predict_face_count_logits(point_features, vertex_table)
            count_pred = count_logits.argmax(dim=-1)
            count_total += int(face_counts.numel())
            count_correct += int(count_pred.eq(face_counts).sum().detach().cpu())
    return {
        "teacher_forced_token_accuracy": correct_tokens / max(1, total_tokens),
        "teacher_forced_face_exact_accuracy": exact_faces / max(1, total_faces),
        "teacher_forced_patch_exact_accuracy": exact_patches / max(1, total_patches),
        "teacher_forced_first_face_exact_accuracy": first_face_exact / max(1, first_face_total),
        "teacher_forced_count_accuracy": count_correct / max(1, count_total),
        "teacher_forced_token_loss": loss_sum / max(1.0, loss_weight),
        "teacher_forced_tokens": total_tokens,
        "teacher_forced_faces": total_faces,
        "teacher_forced_patches": total_patches,
    }


def _greedy_metrics(model, refs: list[PackedPatchRef], meta, *, batch_size: int, point_samples: int, device):
    import torch

    total_tokens = 0
    correct_tokens = 0
    total_faces = 0
    exact_faces = 0
    exact_patches = 0
    total_patches = 0
    first_face_exact = 0
    first_face_total = 0
    invalid_tokens = 0
    with torch.no_grad():
        for batch_refs in _batched(refs, batch_size):
            samples = [_load_patch_sample(ref) for ref in batch_refs]
            (
                point_features,
                vertex_table,
                _input_faces,
                target_faces,
                _target_weights,
                *_rest,
                face_counts,
            ) = _make_batch(
                samples,
                max_vertices=meta.max_vertices,
                max_faces=meta.max_faces,
                device=device,
                point_samples=point_samples,
            )
            generated = torch.full_like(target_faces, -1)
            max_count = int(face_counts.max().detach().cpu()) if face_counts.numel() else 0
            for face_index in range(max_count):
                logits = model(point_features, vertex_table, generated)
                next_face = logits[:, face_index].argmax(dim=-1)
                generated[:, face_index] = next_face
            valid = target_faces.ge(0)
            if max_count > 0:
                invalid_tokens += int((generated[:, :max_count] < 0).sum().detach().cpu())
                invalid_tokens += int((generated[:, :max_count] >= meta.max_vertices).sum().detach().cpu())
            correct = generated.eq(target_faces) & valid
            total_tokens += int(valid.sum().detach().cpu())
            correct_tokens += int(correct.sum().detach().cpu())
            face_valid = valid.all(dim=-1)
            face_exact = generated.eq(target_faces).all(dim=-1) & face_valid
            total_faces += int(face_valid.sum().detach().cpu())
            exact_faces += int(face_exact.sum().detach().cpu())
            for row, count in enumerate(face_counts.detach().cpu().tolist()):
                count = int(count)
                if count <= 0:
                    continue
                total_patches += 1
                if bool(face_exact[row, :count].all().detach().cpu()):
                    exact_patches += 1
                first_face_total += 1
                if bool(face_exact[row, 0].detach().cpu()):
                    first_face_exact += 1
    return {
        "greedy_target_count_token_accuracy": correct_tokens / max(1, total_tokens),
        "greedy_target_count_face_exact_accuracy": exact_faces / max(1, total_faces),
        "greedy_target_count_patch_exact_accuracy": exact_patches / max(1, total_patches),
        "greedy_target_count_first_face_exact_accuracy": first_face_exact / max(1, first_face_total),
        "greedy_invalid_token_rate": invalid_tokens / max(1, total_tokens),
        "greedy_tokens": total_tokens,
        "greedy_faces": total_faces,
        "greedy_patches": total_patches,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--limit-sources", type=int, default=0)
    parser.add_argument("--limit-patches", type=int, default=4096)
    parser.add_argument("--greedy-patches", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--greedy-batch-size", type=int, default=8)
    parser.add_argument("--point-samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    device = _device(args.device)
    checkpoint = _load_checkpoint(args.checkpoint, map_location=device)
    meta = _scan_packed_patches(
        args.manifest,
        limit_sources=args.limit_sources,
        limit_patches=args.limit_patches,
    )
    refs = list(meta.refs)
    rng = random.Random(args.seed)
    rng.shuffle(refs)
    model = _build_model(checkpoint, meta, device=device)
    teacher = _teacher_forced_metrics(
        model,
        refs,
        meta,
        batch_size=args.batch_size,
        point_samples=args.point_samples,
        device=device,
    )
    greedy_refs = refs[: max(0, min(int(args.greedy_patches), len(refs)))]
    greedy = _greedy_metrics(
        model,
        greedy_refs,
        meta,
        batch_size=args.greedy_batch_size,
        point_samples=args.point_samples,
        device=device,
    )
    report = {
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "device": str(device),
        "checkpoint_best_loss": checkpoint.get("best_loss"),
        "checkpoint_best_step": checkpoint.get("best_step"),
        "patches_scanned": len(refs),
        "packed_sources_scanned": len({str(ref.path) for ref in refs}),
        "num_bins": meta.num_bins,
        "max_vertices": meta.max_vertices,
        "max_faces": meta.max_faces,
        "max_points": meta.max_points,
        **teacher,
        **greedy,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
