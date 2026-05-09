#!/usr/bin/env python3
"""Evaluate the paper-faithful FACE reconstruction path."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clearmesh.eval.mesh_quality import evaluate_mesh, evaluate_mesh_pair
from clearmesh.mesh_heads.face_paper import build_paper_face_arae
from clearmesh.mesh_heads.face_tokens import (
    FaceTokenSequence,
    FaceTokenTransform,
    decode_paper_face_tokens_to_mesh,
)
from clearmesh.mesh_heads.face_topology import face_token_topology_report


FACE_COORD_SLOT_LABELS = ("z0", "y0", "x0", "z1", "y1", "x1", "z2", "y2", "x2")


def _load_checkpoint(path: Path):  # type: ignore[no-untyped-def]
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def _load_sample(path: Path, point_samples: int | None = None) -> tuple[np.ndarray, FaceTokenTransform, int, np.ndarray]:
    data = np.load(path)
    points = np.asarray(data["surface_points"], dtype=np.float32)
    normals = np.asarray(data["surface_normals"], dtype=np.float32)
    if point_samples is not None and point_samples > 0:
        if len(points) >= point_samples:
            points = points[:point_samples]
            normals = normals[:point_samples]
        else:
            repeat = int(np.ceil(point_samples / len(points)))
            points = np.tile(points, (repeat, 1))[:point_samples]
            normals = np.tile(normals, (repeat, 1))[:point_samples]
    transform = FaceTokenTransform(
        center=tuple(np.asarray(data["center"], dtype=np.float64).reshape(3).tolist()),
        scale=float(np.asarray(data["scale"], dtype=np.float64).reshape(-1)[0]),
    )
    num_bins = int(np.asarray(data["num_bins"]).reshape(-1)[0])
    tokens = np.asarray(data["paper_tokens"], dtype=np.int64)
    return np.concatenate([points, normals], axis=1), transform, num_bins, tokens


def _generate_tokens(
    model,
    point_features,
    face_count: int,
    num_bins: int,
    device,
    decode_head: str,
    *,
    incremental: bool = True,
    stop_on_eos: bool = False,
    eos_threshold: float = 0.5,
    min_faces: int = 1,
    teacher_prefix_tokens: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:  # type: ignore[no-untyped-def]
    import torch

    point_tensor = torch.as_tensor(point_features, dtype=torch.float32, device=device).unsqueeze(0)
    teacher_prefix = None
    teacher_prefix_count = 0
    if teacher_prefix_tokens is not None:
        teacher_prefix_array = np.asarray(teacher_prefix_tokens, dtype=np.int64).reshape(-1, 9)[:face_count]
        if len(teacher_prefix_array):
            teacher_prefix = torch.as_tensor(teacher_prefix_array, dtype=torch.long, device=device)
            teacher_prefix_count = int(len(teacher_prefix_array))
    generated: list[np.ndarray] = []
    eos_probs: list[float] = []
    stopped_on_eos = False
    with torch.no_grad():
        if incremental:
            cache = model.init_incremental_cache(point_tensor)
            previous_face = torch.full((1, 9), -1, dtype=torch.long, device=device)
            for position in range(face_count):
                hidden = model.incremental_hidden_step(previous_face, position, cache)
                if teacher_prefix is not None and position < teacher_prefix_count:
                    next_face = teacher_prefix[position : position + 1]
                else:
                    next_face = _decode_next_face_from_hidden(model, hidden, num_bins, device, decode_head)
                generated.append(next_face.squeeze(0).detach().cpu().numpy().astype(np.int64))
                eos_prob = _eos_probability(model, hidden)
                if eos_prob is not None:
                    eos_probs.append(eos_prob)
                if stop_on_eos and eos_prob is not None and position + 1 >= min_faces and eos_prob >= eos_threshold:
                    stopped_on_eos = True
                    break
                previous_face = next_face
        else:
            input_faces = torch.full((1, 1, 9), -1, dtype=torch.long, device=device)
            for position in range(face_count):
                if decode_head == "parallel":
                    hidden = model.hidden(point_tensor, input_faces)[:, -1:, :]
                    logits = model.parallel_head(hidden).reshape(hidden.shape[0], hidden.shape[1], 9, num_bins)[:, -1, :, :]
                    next_face = torch.argmax(logits, dim=-1)
                else:
                    hidden = model.hidden(point_tensor, input_faces)[:, -1:, :]
                    next_face = _decode_next_face_from_hidden(model, hidden, num_bins, device, decode_head)
                if teacher_prefix is not None and position < teacher_prefix_count:
                    next_face = teacher_prefix[position : position + 1]
                generated.append(next_face.squeeze(0).detach().cpu().numpy().astype(np.int64))
                eos_prob = _eos_probability(model, hidden)
                if eos_prob is not None:
                    eos_probs.append(eos_prob)
                if stop_on_eos and eos_prob is not None and position + 1 >= min_faces and eos_prob >= eos_threshold:
                    stopped_on_eos = True
                    break
                if len(generated) < face_count:
                    input_faces = torch.cat([input_faces, next_face.unsqueeze(1)], dim=1)
    if not generated:
        generated.append(np.zeros((9,), dtype=np.int64))
    return np.stack(generated, axis=0), {
        "stopped_on_eos": bool(stopped_on_eos),
        "predicted_face_count": int(len(generated)),
        "teacher_prefix_faces": int(min(teacher_prefix_count, len(generated))),
        "eos_probs": eos_probs,
        "last_eos_prob": eos_probs[-1] if eos_probs else None,
    }


def _eos_probability(model, hidden) -> float | None:  # type: ignore[no-untyped-def]
    import torch

    if not hasattr(model, "eos_logits_from_hidden"):
        return None
    logits = model.eos_logits_from_hidden(hidden)
    if logits is None:
        return None
    return float(torch.sigmoid(logits[:, -1]).detach().cpu().reshape(-1)[0])


def _decode_next_face_from_hidden(model, hidden, num_bins: int, device, decode_head: str):  # type: ignore[no-untyped-def]
    import torch

    if decode_head == "parallel":
        logits = model.parallel_head(hidden).reshape(hidden.shape[0], hidden.shape[1], 9, num_bins)[:, -1, :, :]
        return torch.argmax(logits[:, :, :num_bins], dim=-1)
    if hasattr(model, "greedy_face_from_hidden"):
        return model.greedy_face_from_hidden(hidden, limit_bins=num_bins)
    prefix = torch.full((hidden.shape[0], 9), -1, dtype=torch.long, device=device)
    for coord in range(9):
        logits = model._causal_logits_from_hidden(hidden, prefix.reshape(hidden.shape[0], 1, 9))[:, 0, :, :]
        token = torch.argmax(logits[:, coord, :num_bins], dim=-1)
        prefix[:, coord] = token
    return prefix


def _generate_tokens_teacher_forced(model, point_features, teacher_tokens, face_count: int, num_bins: int, device, decode_head: str):  # type: ignore[no-untyped-def]
    import torch

    tokens = np.asarray(teacher_tokens[:face_count], dtype=np.int64)
    point_tensor = torch.as_tensor(point_features, dtype=torch.float32, device=device).unsqueeze(0)
    input_faces = torch.full((1, len(tokens), 9), -1, dtype=torch.long, device=device)
    target_faces = torch.as_tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    if len(tokens) > 1:
        input_faces[:, 1:] = target_faces[:, :-1]
    with torch.no_grad():
        if decode_head == "parallel":
            logits = model.forward_parallel(point_tensor, input_faces)
        else:
            logits = model.forward_causal(point_tensor, input_faces, target_faces)
        generated = torch.argmax(logits[:, :, :, :num_bins], dim=-1).squeeze(0).detach().cpu().numpy().astype(np.int64)
    return generated, {"stopped_on_eos": False, "predicted_face_count": int(len(generated)), "eos_probs": [], "last_eos_prob": None}


def _aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {"attempted": 0}
    total_sample_sec = float(sum(float(item.get("sample_sec") or 0.0) for item in items))
    total_faces = int(sum(int(item.get("face_count") or 0) for item in items))
    first_slot_hist: Counter[str] = Counter()
    first_face_bucket_hist: Counter[str] = Counter()
    for item in items:
        first_slot = item.get("first_divergent_coord_slot")
        if first_slot is None:
            first_slot_hist["none"] += 1
        else:
            first_slot_int = int(first_slot)
            label = FACE_COORD_SLOT_LABELS[first_slot_int] if 0 <= first_slot_int < len(FACE_COORD_SLOT_LABELS) else str(first_slot_int)
            first_slot_hist[label] += 1
        first_face = item.get("first_divergent_face_index")
        if first_face is None:
            first_face_bucket_hist["none"] += 1
        else:
            face_index = int(first_face)
            if face_index < 2:
                bucket = str(face_index)
            elif face_index < 4:
                bucket = "2-3"
            elif face_index < 8:
                bucket = "4-7"
            elif face_index < 16:
                bucket = "8-15"
            elif face_index < 32:
                bucket = "16-31"
            elif face_index < 64:
                bucket = "32-63"
            elif face_index < 128:
                bucket = "64-127"
            else:
                bucket = "128+"
            first_face_bucket_hist[bucket] += 1

    def _mean_slot_metric(key: str) -> list[float | None] | None:
        slots: list[float | None] = []
        observed = False
        for slot in range(9):
            values = [
                float(item[key][slot])
                for item in items
                if isinstance(item.get(key), list) and len(item[key]) == 9 and item[key][slot] is not None
            ]
            if values:
                observed = True
                slots.append(float(np.mean(values)))
            else:
                slots.append(None)
        return slots if observed else None

    payload = {
        "attempted": len(items),
        "watertight": int(sum(1 for item in items if item.get("watertight"))),
        "total_faces": total_faces,
        "total_sample_sec": total_sample_sec,
        "mean_sample_sec": float(np.mean([float(item.get("sample_sec") or 0.0) for item in items])),
        "faces_per_sec": float(total_faces / total_sample_sec) if total_sample_sec > 0 else None,
        "mean_boundary_edges": float(np.mean([item.get("token_boundary_edge_count", 0) for item in items])),
        "mean_edge_pairing_ratio": float(np.mean([item.get("token_edge_pairing_ratio", 0.0) for item in items])),
        "mean_chamfer_l2": float(np.mean([item["chamfer_l2"] for item in items if item.get("chamfer_l2") is not None])) if any(item.get("chamfer_l2") is not None for item in items) else None,
        "mean_hausdorff_l2": float(np.mean([item["hausdorff_l2"] for item in items if item.get("hausdorff_l2") is not None])) if any(item.get("hausdorff_l2") is not None for item in items) else None,
        "mean_teacher_forced_accuracy": float(np.mean([item["teacher_forced_accuracy"] for item in items if item.get("teacher_forced_accuracy") is not None])) if any(item.get("teacher_forced_accuracy") is not None for item in items) else None,
        "mean_teacher_forced_loss": float(np.mean([item["teacher_forced_loss"] for item in items if item.get("teacher_forced_loss") is not None])) if any(item.get("teacher_forced_loss") is not None for item in items) else None,
        "mean_teacher_forced_eos_accuracy": float(np.mean([item["teacher_forced_eos_accuracy"] for item in items if item.get("teacher_forced_eos_accuracy") is not None])) if any(item.get("teacher_forced_eos_accuracy") is not None for item in items) else None,
        "mean_generated_token_accuracy": float(np.mean([item["generated_token_accuracy"] for item in items if item.get("generated_token_accuracy") is not None])) if any(item.get("generated_token_accuracy") is not None for item in items) else None,
        "mean_generated_vertex_exact_ratio": float(np.mean([item["generated_vertex_exact_ratio"] for item in items if item.get("generated_vertex_exact_ratio") is not None])) if any(item.get("generated_vertex_exact_ratio") is not None for item in items) else None,
        "mean_generated_edge_exact_ratio": float(np.mean([item["generated_edge_exact_ratio"] for item in items if item.get("generated_edge_exact_ratio") is not None])) if any(item.get("generated_edge_exact_ratio") is not None for item in items) else None,
        "mean_generated_face_exact_ratio": float(np.mean([item["generated_face_exact_ratio"] for item in items if item.get("generated_face_exact_ratio") is not None])) if any(item.get("generated_face_exact_ratio") is not None for item in items) else None,
        "mean_generated_edge_set_precision": float(np.mean([item["generated_edge_set_precision"] for item in items if item.get("generated_edge_set_precision") is not None])) if any(item.get("generated_edge_set_precision") is not None for item in items) else None,
        "mean_generated_edge_set_recall": float(np.mean([item["generated_edge_set_recall"] for item in items if item.get("generated_edge_set_recall") is not None])) if any(item.get("generated_edge_set_recall") is not None for item in items) else None,
        "mean_generated_edge_set_f1": float(np.mean([item["generated_edge_set_f1"] for item in items if item.get("generated_edge_set_f1") is not None])) if any(item.get("generated_edge_set_f1") is not None for item in items) else None,
        "mean_predicted_to_reference_face_ratio": float(np.mean([item["predicted_to_reference_face_ratio"] for item in items if item.get("predicted_to_reference_face_ratio") is not None])) if any(item.get("predicted_to_reference_face_ratio") is not None for item in items) else None,
        "first_divergent_coord_slot_histogram": dict(sorted(first_slot_hist.items())),
        "first_divergent_face_bucket_histogram": dict(sorted(first_face_bucket_hist.items())),
        "coord_slot_labels": list(FACE_COORD_SLOT_LABELS),
    }
    for key in (
        "teacher_forced_slot_accuracy",
        "teacher_forced_slot_loss",
        "generated_slot_accuracy",
        "first_face_teacher_rank_by_slot",
        "first_face_teacher_target_prob_by_slot",
        "first_face_teacher_entropy_by_slot",
    ):
        mean_value = _mean_slot_metric(key)
        if mean_value is not None:
            payload[f"mean_{key}"] = mean_value
    for key in (
        "first_face_teacher_rank_mean",
        "first_face_teacher_top1_accuracy",
        "first_face_teacher_target_prob_mean",
        "first_face_teacher_entropy_mean",
        "teacher_prefix_faces",
    ):
        values = [float(item[key]) for item in items if item.get(key) is not None]
        if values:
            payload[f"mean_{key}"] = float(np.mean(values))
    return payload


def _face_edges(face: np.ndarray) -> list[tuple[tuple[int, int, int], tuple[int, int, int]]]:
    vertices = [tuple(int(coord) for coord in vertex) for vertex in face.reshape(3, 3)]
    edges = []
    for a_idx, b_idx in ((0, 1), (1, 2), (2, 0)):
        a = vertices[a_idx]
        b = vertices[b_idx]
        edges.append((a, b) if a <= b else (b, a))
    return edges


def _edge_counter(tokens: np.ndarray) -> Counter[tuple[tuple[int, int, int], tuple[int, int, int]]]:
    arr = np.asarray(tokens, dtype=np.int64).reshape(-1, 3, 3)
    counter: Counter[tuple[tuple[int, int, int], tuple[int, int, int]]] = Counter()
    for face in arr:
        counter.update(_face_edges(face))
    return counter


def _counter_intersection_total(
    left: Counter[tuple[tuple[int, int, int], tuple[int, int, int]]],
    right: Counter[tuple[tuple[int, int, int], tuple[int, int, int]]],
) -> int:
    return int(sum(min(count, right.get(edge, 0)) for edge, count in left.items()))


def _generated_vs_teacher_metrics(generated_tokens: np.ndarray, teacher_tokens: np.ndarray) -> dict[str, Any]:
    """Compare free-running tokens with the teacher sequence.

    Teacher-forced loss/accuracy answers "can the model predict under a clean
    prefix?" These metrics answer the nastier production question: "did the
    generated AR sequence actually match the target token graph?"
    """

    generated = np.asarray(generated_tokens, dtype=np.int64).reshape(-1, 9)
    teacher = np.asarray(teacher_tokens, dtype=np.int64).reshape(-1, 9)
    compared_faces = int(min(len(generated), len(teacher)))
    payload: dict[str, Any] = {
        "generated_compared_face_count": compared_faces,
        "generated_face_count_delta": int(len(generated) - len(teacher)),
        "generated_to_teacher_face_count_ratio": float(len(generated) / max(1, len(teacher))),
    }
    if compared_faces <= 0:
        return {
            **payload,
            "generated_token_accuracy": None,
            "generated_vertex_exact_ratio": None,
            "generated_edge_exact_ratio": None,
            "generated_face_exact_ratio": None,
            "generated_edge_set_precision": None,
            "generated_edge_set_recall": None,
            "generated_edge_set_f1": None,
            "first_divergent_face_index": None,
            "first_divergent_token_index": None,
            "first_divergent_coord_slot": None,
            "generated_slot_accuracy": None,
        }

    generated_aligned = generated[:compared_faces]
    teacher_aligned = teacher[:compared_faces]
    matches = generated_aligned == teacher_aligned
    face_matches = np.all(matches, axis=1)
    vertex_matches = np.all(
        generated_aligned.reshape(compared_faces, 3, 3) == teacher_aligned.reshape(compared_faces, 3, 3),
        axis=2,
    )
    edge_matches = []
    for generated_face, teacher_face in zip(generated_aligned.reshape(compared_faces, 3, 3), teacher_aligned.reshape(compared_faces, 3, 3)):
        teacher_edges = set(_face_edges(teacher_face))
        edge_matches.extend(edge in teacher_edges for edge in _face_edges(generated_face))

    divergent = np.argwhere(~matches)
    if len(divergent):
        first_face = int(divergent[0][0])
        first_slot = int(divergent[0][1])
        first_token = int(first_face * 9 + first_slot)
    else:
        first_face = None
        first_slot = None
        first_token = None

    generated_edges = _edge_counter(generated)
    teacher_edges = _edge_counter(teacher)
    matched_generated_edges = _counter_intersection_total(generated_edges, teacher_edges)
    generated_edge_uses = int(sum(generated_edges.values()))
    teacher_edge_uses = int(sum(teacher_edges.values()))
    precision = matched_generated_edges / generated_edge_uses if generated_edge_uses else None
    recall = matched_generated_edges / teacher_edge_uses if teacher_edge_uses else None
    f1 = None
    if precision is not None and recall is not None and precision + recall > 0:
        f1 = 2.0 * precision * recall / (precision + recall)

    return {
        **payload,
        "generated_token_accuracy": float(np.mean(matches)),
        "generated_vertex_exact_ratio": float(np.mean(vertex_matches)),
        "generated_edge_exact_ratio": float(np.mean(edge_matches)) if edge_matches else None,
        "generated_face_exact_ratio": float(np.mean(face_matches)),
        "generated_edge_set_precision": None if precision is None else float(precision),
        "generated_edge_set_recall": None if recall is None else float(recall),
        "generated_edge_set_f1": None if f1 is None else float(f1),
        "first_divergent_face_index": first_face,
        "first_divergent_token_index": first_token,
        "first_divergent_coord_slot": first_slot,
        "generated_slot_accuracy": [float(np.mean(matches[:, slot])) for slot in range(9)],
    }


def _first_face_logit_metrics(logits, target_faces, num_bins: int) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    import torch
    import torch.nn.functional as F

    if target_faces.shape[1] <= 0:
        return {}
    first_logits = logits[0, 0, :, :num_bins].to(dtype=torch.float32)
    first_targets = target_faces[0, 0, :].to(dtype=torch.long)
    valid = first_targets.ge(0) & first_targets.lt(num_bins)
    if not bool(valid.all()):
        return {}
    probs = F.softmax(first_logits, dim=-1)
    log_probs = F.log_softmax(first_logits, dim=-1)
    target_scores = first_logits.gather(1, first_targets.unsqueeze(1)).squeeze(1)
    ranks = first_logits.gt(target_scores.unsqueeze(1)).sum(dim=1) + 1
    top1 = torch.argmax(first_logits, dim=-1)
    target_probs = probs.gather(1, first_targets.unsqueeze(1)).squeeze(1)
    entropy = -(probs * log_probs).sum(dim=-1)
    return {
        "first_face_teacher_rank_by_slot": [int(value) for value in ranks.detach().cpu().tolist()],
        "first_face_teacher_rank_mean": float(ranks.to(torch.float32).mean().detach().cpu()),
        "first_face_teacher_top1_accuracy": float(top1.eq(first_targets).to(torch.float32).mean().detach().cpu()),
        "first_face_teacher_target_prob_by_slot": [float(value) for value in target_probs.detach().cpu().tolist()],
        "first_face_teacher_target_prob_mean": float(target_probs.mean().detach().cpu()),
        "first_face_teacher_entropy_by_slot": [float(value) for value in entropy.detach().cpu().tolist()],
        "first_face_teacher_entropy_mean": float(entropy.mean().detach().cpu()),
        "first_face_teacher_tokens": [int(value) for value in first_targets.detach().cpu().tolist()],
        "first_face_top1_tokens": [int(value) for value in top1.detach().cpu().tolist()],
    }


def _teacher_forced_metrics(model, point_features, teacher_tokens, max_faces: int, num_bins: int, device, decode_head: str):  # type: ignore[no-untyped-def]
    import torch
    import torch.nn.functional as F

    tokens = np.asarray(teacher_tokens[:max_faces], dtype=np.int64)
    if len(tokens) == 0:
        return {"teacher_forced_loss": None, "teacher_forced_accuracy": None}
    point_tensor = torch.as_tensor(point_features, dtype=torch.float32, device=device).unsqueeze(0)
    input_faces = torch.full((1, len(tokens), 9), -1, dtype=torch.long, device=device)
    target_faces = torch.as_tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    if len(tokens) > 1:
        input_faces[:, 1:] = target_faces[:, :-1]
    with torch.no_grad():
        hidden = model.hidden(point_tensor, input_faces)
        if decode_head == "parallel":
            logits = model.parallel_head(hidden).reshape(hidden.shape[0], hidden.shape[1], 9, num_bins)
        else:
            logits = model._causal_logits_from_hidden(hidden, target_faces)
        losses = F.cross_entropy(logits.reshape(-1, num_bins), target_faces.reshape(-1), reduction="none").reshape_as(target_faces)
        loss = losses.mean()
        pred = torch.argmax(logits, dim=-1)
        accuracy = pred.eq(target_faces).to(torch.float32).mean()
        slot_accuracy = pred.eq(target_faces).to(torch.float32).mean(dim=(0, 1))
        slot_loss = losses.mean(dim=(0, 1))
        first_face_metrics = _first_face_logit_metrics(logits, target_faces, num_bins)
        eos_loss = None
        eos_accuracy = None
        eos_logits = model.eos_logits_from_hidden(hidden) if hasattr(model, "eos_logits_from_hidden") else None
        if eos_logits is not None:
            eos_targets = torch.zeros((1, len(tokens)), dtype=torch.float32, device=device)
            eos_targets[:, len(tokens) - 1] = 1.0
            eos_loss = F.binary_cross_entropy_with_logits(eos_logits, eos_targets, reduction="mean")
            eos_accuracy = torch.sigmoid(eos_logits).ge(0.5).eq(eos_targets.bool()).to(torch.float32).mean()
    return {
        "teacher_forced_loss": float(loss.detach().cpu()),
        "teacher_forced_accuracy": float(accuracy.detach().cpu()),
        "teacher_forced_slot_accuracy": [float(value) for value in slot_accuracy.detach().cpu().tolist()],
        "teacher_forced_slot_loss": [float(value) for value in slot_loss.detach().cpu().tolist()],
        **first_face_metrics,
        "teacher_forced_eos_loss": None if eos_loss is None else float(eos_loss.detach().cpu()),
        "teacher_forced_eos_accuracy": None if eos_accuracy is None else float(eos_accuracy.detach().cpu()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--export-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--point-samples", type=int, default=0)
    parser.add_argument("--face-count-mode", choices=["gt", "max", "predicted"], default="gt")
    parser.add_argument("--generation-mode", choices=["autoregressive", "teacher_forced"], default="autoregressive")
    parser.add_argument("--disable-incremental-generation", action="store_true")
    parser.add_argument(
        "--teacher-prefix-faces",
        type=int,
        default=0,
        help=(
            "For autoregressive diagnostics, feed the first N ground-truth faces "
            "before free-running. This isolates first-face/order collapse from later exposure bias."
        ),
    )
    parser.add_argument("--eos-threshold", type=float, default=0.5)
    parser.add_argument("--min-generated-faces", type=int, default=1)
    parser.add_argument(
        "--generation-face-limit",
        type=int,
        default=0,
        help="Optional face cap for generation/eval. Useful for quick autoregressive probes on long meshes.",
    )
    parser.add_argument("--pair-samples", type=int, default=1000)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--log-every", type=int, default=1)
    args = parser.parse_args()
    if not 0.0 <= args.eos_threshold <= 1.0:
        raise SystemExit("--eos-threshold must be in [0, 1]")
    if args.min_generated_faces < 1:
        raise SystemExit("--min-generated-faces must be >= 1")
    if args.teacher_prefix_faces < 0:
        raise SystemExit("--teacher-prefix-faces must be >= 0")

    import torch

    checkpoint = _load_checkpoint(args.checkpoint)
    train_args = checkpoint.get("args", {})
    num_bins = int(checkpoint["num_bins"])
    max_faces = int(checkpoint["max_faces"])
    decode_head = str(checkpoint.get("decode_head") or train_args.get("decode_head") or "causal")
    encoder_backend = str(checkpoint.get("encoder_backend") or train_args.get("encoder_backend") or "native")
    causal_mlp_variant = str(checkpoint.get("causal_mlp_variant") or train_args.get("causal_mlp_variant") or "legacy_concat")
    face_embedding_variant = str(
        checkpoint.get("face_embedding_variant") or train_args.get("face_embedding_variant") or "token_concat_project"
    )
    has_eos_head = bool(checkpoint.get("has_eos_head", False))
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    model = build_paper_face_arae(
        num_bins=num_bins,
        max_faces=max_faces,
        point_feature_dim=6,
        hidden_size=int(train_args.get("hidden_size", 256)),
        encoder_hidden_size=int(train_args.get("encoder_hidden_size", train_args.get("hidden_size", 256))),
        encoder_layers=int(train_args.get("encoder_layers", 4)),
        decoder_layers=int(train_args.get("decoder_layers", 4)),
        heads=int(train_args.get("heads", 8)),
        vecset_tokens=int(train_args.get("vecset_tokens", 128)),
        latent_dim=int(train_args.get("latent_dim", 64)),
        encoder_backend=encoder_backend,
        causal_mlp_variant=causal_mlp_variant,
        face_embedding_variant=face_embedding_variant,
        enable_eos_head=has_eos_head,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    if args.face_count_mode == "predicted" and not has_eos_head:
        raise SystemExit("checkpoint has no EOS head; use --face-count-mode gt or max")

    paths = sorted(args.dataset_dir.glob("*.npz"))
    if args.limit:
        paths = paths[: args.limit]
    if args.export_dir:
        args.export_dir.mkdir(parents=True, exist_ok=True)
    results = []
    started_at = time.perf_counter()
    for idx, path in enumerate(paths):
        sample_started_at = time.perf_counter()
        point_features, transform, sample_bins, teacher_tokens = _load_sample(
            path,
            point_samples=args.point_samples or int(train_args.get("point_samples", 0)) or None,
        )
        if int(sample_bins) != num_bins:
            raise ValueError(f"{path} num_bins={sample_bins}, checkpoint num_bins={num_bins}")
        uncapped_reference_face_count = max_faces if args.face_count_mode == "max" else min(len(teacher_tokens), max_faces)
        reference_face_count = uncapped_reference_face_count
        if args.generation_face_limit > 0:
            reference_face_count = min(reference_face_count, args.generation_face_limit)
        generation_cap = max_faces if args.face_count_mode == "predicted" else reference_face_count
        if args.generation_face_limit > 0:
            generation_cap = min(generation_cap, args.generation_face_limit)
        if args.generation_mode == "teacher_forced":
            generated_tokens, generation_meta = _generate_tokens_teacher_forced(
                model,
                point_features,
                teacher_tokens,
                reference_face_count,
                num_bins,
                device,
                decode_head,
            )
        else:
            teacher_prefix = None
            if args.teacher_prefix_faces > 0:
                teacher_prefix = teacher_tokens[: min(int(args.teacher_prefix_faces), int(generation_cap), len(teacher_tokens))]
            generated_tokens, generation_meta = _generate_tokens(
                model,
                point_features,
                generation_cap,
                num_bins,
                device,
                decode_head,
                incremental=not args.disable_incremental_generation,
                stop_on_eos=args.face_count_mode == "predicted",
                eos_threshold=args.eos_threshold,
                min_faces=args.min_generated_faces,
                teacher_prefix_tokens=teacher_prefix,
            )
        generated_face_count = int(len(generated_tokens))
        teacher_tokens_capped = teacher_tokens[:reference_face_count]
        generated_metrics = _generated_vs_teacher_metrics(generated_tokens, teacher_tokens_capped)
        teacher_metrics = _teacher_forced_metrics(
            model,
            point_features,
            teacher_tokens,
            reference_face_count,
            num_bins,
            device,
            decode_head,
        )
        generated_seq = FaceTokenSequence(tokens=generated_tokens, num_bins=num_bins, transform=transform)
        teacher_seq = FaceTokenSequence(tokens=teacher_tokens[:reference_face_count], num_bins=num_bins, transform=transform)
        generated = decode_paper_face_tokens_to_mesh(generated_seq)
        teacher = decode_paper_face_tokens_to_mesh(teacher_seq)
        if args.export_dir:
            gen_path = args.export_dir / f"{idx:04d}_{path.stem}_generated.glb"
            teacher_path = args.export_dir / f"{idx:04d}_{path.stem}_teacher.glb"
        else:
            gen_path = args.output.parent / f".tmp_{idx:04d}_{path.stem}_generated.glb"
            teacher_path = args.output.parent / f".tmp_{idx:04d}_{path.stem}_teacher.glb"
        gen_path.parent.mkdir(parents=True, exist_ok=True)
        generated.export(gen_path)
        teacher.export(teacher_path)
        try:
            pair = evaluate_mesh_pair(gen_path, teacher_path, samples=args.pair_samples)
        except Exception as exc:  # noqa: BLE001 - a bad sample should not kill the whole eval.
            pair = {
                "chamfer_l2": None,
                "hausdorff_l2": None,
                "normal_consistency": None,
                "surface_area_ratio": None,
                "volume_ratio": None,
                "pair_error": f"{type(exc).__name__}: {exc}",
            }
        quality = evaluate_mesh(gen_path)
        token_report = face_token_topology_report(generated_tokens)
        teacher_token_report = face_token_topology_report(teacher_tokens_capped)
        sample_sec = max(time.perf_counter() - sample_started_at, 1e-6)
        elapsed_sec = max(time.perf_counter() - started_at, 1e-6)
        item = {
            "path": str(path),
            "decode_head": decode_head,
            "generation_mode": args.generation_mode,
            "face_count": int(generated_face_count),
            "reference_face_count": int(reference_face_count),
            "uncapped_reference_face_count": int(uncapped_reference_face_count),
            "face_count_mode": args.face_count_mode,
            "generation_face_limit": int(args.generation_face_limit),
            "truncated_by_generation_face_limit": bool(
                args.generation_face_limit > 0 and reference_face_count < uncapped_reference_face_count
            ),
            "predicted_face_count": int(generation_meta.get("predicted_face_count") or generated_face_count),
            "predicted_to_reference_face_ratio": float(generated_face_count / max(1, reference_face_count)),
            "stopped_on_eos": bool(generation_meta.get("stopped_on_eos")),
            "teacher_prefix_faces": int(generation_meta.get("teacher_prefix_faces") or 0),
            "last_eos_prob": generation_meta.get("last_eos_prob"),
            "sample_sec": float(sample_sec),
            "faces_per_sec": float(generated_face_count / sample_sec),
            "elapsed_sec": float(elapsed_sec),
            "watertight": bool(quality.get("watertight")),
            "boundary_edges": int(quality.get("boundary_edge_count") or 0),
            "nonmanifold_edges": int(quality.get("nonmanifold_edge_count") or 0),
            "token_watertight_edge_graph": bool(token_report.watertight_edge_graph),
            "token_boundary_edge_count": int(token_report.boundary_edge_count),
            "token_edge_pairing_ratio": float(token_report.edge_pairing_ratio),
            "teacher_token_watertight_edge_graph": bool(teacher_token_report.watertight_edge_graph),
            "teacher_token_boundary_edge_count": int(teacher_token_report.boundary_edge_count),
            "teacher_token_edge_pairing_ratio": float(teacher_token_report.edge_pairing_ratio),
            **generated_metrics,
            **teacher_metrics,
            **pair,
        }
        results.append(item)
        if not args.export_dir:
            gen_path.unlink(missing_ok=True)
            teacher_path.unlink(missing_ok=True)
        if args.log_every > 0 and (idx == 0 or (idx + 1) % args.log_every == 0 or idx + 1 == len(paths)):
            print(
                json.dumps(
                    {
                        "sample": idx + 1,
                        "total": len(paths),
                        "path": path.name,
                        "generation_mode": args.generation_mode,
                        "incremental_generation": not args.disable_incremental_generation,
                        "teacher_prefix_faces": int(generation_meta.get("teacher_prefix_faces") or 0),
                        "face_count": int(generated_face_count),
                        "reference_face_count": int(reference_face_count),
                        "stopped_on_eos": bool(generation_meta.get("stopped_on_eos")),
                        "sample_sec": sample_sec,
                        "faces_per_sec": generated_face_count / sample_sec,
                        "elapsed_sec": elapsed_sec,
                    }
                ),
                flush=True,
            )

    report = {
        "checkpoint": str(args.checkpoint),
        "dataset_dir": str(args.dataset_dir),
        "representation": "paper-face-arae",
        "decode_head": decode_head,
        "encoder_backend": encoder_backend,
        "causal_mlp_variant": causal_mlp_variant,
        "face_embedding_variant": face_embedding_variant,
        "has_eos_head": has_eos_head,
        "face_count_mode": args.face_count_mode,
        "generation_face_limit": int(args.generation_face_limit),
        "generation_mode": args.generation_mode,
        "incremental_generation": bool(not args.disable_incremental_generation),
        "teacher_prefix_faces": int(args.teacher_prefix_faces),
        "summary": _aggregate(results),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
