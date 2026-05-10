#!/usr/bin/env python3
"""Train the paper-faithful FACE reconstruction path."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
import os
import random
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clearmesh.mesh_heads.face_paper import _farthest_point_indices, build_paper_face_arae
from clearmesh.mesh_heads.face_tokens import (
    canonicalize_mesh_faces_paper_zyx,
    dequantize_normalized_points,
    fit_face_token_transform,
)
from clearmesh.utils.checkpoint import args_to_json_safe
from clearmesh.utils.muon_fallback import build_muon_fallback


@dataclass(frozen=True)
class PaperFaceSample:
    path: Path
    point_features: np.ndarray
    tokens: np.ndarray
    paper_within_face_order: str = "preserve"


@dataclass
class AugmentDiagnostics:
    attempts: int = 0
    successes: int = 0
    fallbacks: int = 0
    mirrored_affine_attempts: int = 0
    original_faces_total: int = 0
    augmented_faces_total: int = 0
    min_augmented_faces: int | None = None
    max_augmented_faces: int = 0
    fallback_reasons: Counter[str] = field(default_factory=Counter)
    exception_types: Counter[str] = field(default_factory=Counter)
    last_exception_type: str | None = None
    last_exception_message: str | None = None
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def record_attempt(self, *, original_faces: int, mirrored_affine: bool) -> None:
        with self._lock:
            self.attempts += 1
            self.original_faces_total += int(original_faces)
            if mirrored_affine:
                self.mirrored_affine_attempts += 1

    def record_success(self, *, augmented_faces: int) -> None:
        augmented_faces = int(augmented_faces)
        with self._lock:
            self.successes += 1
            self.augmented_faces_total += augmented_faces
            if self.min_augmented_faces is None:
                self.min_augmented_faces = augmented_faces
            else:
                self.min_augmented_faces = min(self.min_augmented_faces, augmented_faces)
            self.max_augmented_faces = max(self.max_augmented_faces, augmented_faces)

    def record_fallback(self, reason: str, exc: Exception | None = None) -> None:
        with self._lock:
            self.fallbacks += 1
            self.fallback_reasons[str(reason)] += 1
            if exc is not None:
                exc_type = type(exc).__name__
                self.exception_types[exc_type] += 1
                self.last_exception_type = exc_type
                self.last_exception_message = str(exc)[:240]

    def snapshot(self) -> dict[str, int | float | str | None | dict[str, int]]:
        with self._lock:
            attempts = int(self.attempts)
            successes = int(self.successes)
            fallbacks = int(self.fallbacks)
            original_faces_total = int(self.original_faces_total)
            augmented_faces_total = int(self.augmented_faces_total)
            min_augmented_faces = self.min_augmented_faces
            max_augmented_faces = int(self.max_augmented_faces)
            mirrored_affine_attempts = int(self.mirrored_affine_attempts)
            fallback_reasons = dict(sorted(self.fallback_reasons.items()))
            exception_types = dict(sorted(self.exception_types.items()))
            last_exception_type = self.last_exception_type
            last_exception_message = self.last_exception_message
        return {
            "attempts": attempts,
            "successes": successes,
            "fallbacks": fallbacks,
            "success_rate": successes / attempts if attempts else 0.0,
            "fallback_rate": fallbacks / attempts if attempts else 0.0,
            "mirrored_affine_attempts": mirrored_affine_attempts,
            "avg_original_faces": original_faces_total / attempts if attempts else 0.0,
            "avg_augmented_faces": augmented_faces_total / successes if successes else 0.0,
            "min_augmented_faces": min_augmented_faces,
            "max_augmented_faces": max_augmented_faces,
            "fallback_reasons": fallback_reasons,
            "exception_types": exception_types,
            "last_exception_type": last_exception_type,
            "last_exception_message": last_exception_message,
        }


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _load_sample(path: Path) -> tuple[PaperFaceSample, int]:
    data = np.load(path)
    required = {"surface_points", "surface_normals", "paper_tokens", "num_bins"}
    missing = sorted(required.difference(data.files))
    if missing:
        raise ValueError(f"{path} is missing paper-faithful FACE arrays: {missing}")
    points = np.asarray(data["surface_points"], dtype=np.float32)
    normals = np.asarray(data["surface_normals"], dtype=np.float32)
    tokens = np.asarray(data["paper_tokens"], dtype=np.int64)
    if tokens.ndim != 2 or tokens.shape[1] != 9:
        raise ValueError(f"{path} has invalid paper_tokens shape {tokens.shape}")
    num_bins = int(np.asarray(data["num_bins"]).reshape(-1)[0])
    paper_within_face_order = "preserve"
    if "paper_within_face_order" in data.files:
        paper_within_face_order = str(np.asarray(data["paper_within_face_order"]).reshape(-1)[0])
    return (
        PaperFaceSample(
            path=path,
            point_features=np.concatenate([points, normals], axis=1),
            tokens=tokens,
            paper_within_face_order=paper_within_face_order,
        ),
        num_bins,
    )


def _load_dataset(dataset_dir: Path, limit: int = 0) -> tuple[list[PaperFaceSample], int]:
    samples: list[PaperFaceSample] = []
    num_bins: int | None = None
    for path in sorted(dataset_dir.glob("*.npz"))[: limit or None]:
        try:
            sample, bins = _load_sample(path)
        except Exception:
            continue
        if num_bins is None:
            num_bins = bins
        elif bins != num_bins:
            raise ValueError(f"Mixed num_bins: {num_bins} and {bins}")
        samples.append(sample)
    if not samples or num_bins is None:
        raise SystemExit(f"No paper-faithful FACE shards found in {dataset_dir}")
    return samples, num_bins


def _sample_point_features(points: np.ndarray, point_samples: int | None) -> np.ndarray:
    if point_samples is not None and point_samples > 0:
        if len(points) >= point_samples:
            return points[:point_samples]
        repeat = int(np.ceil(point_samples / len(points)))
        return np.tile(points, (repeat, 1))[:point_samples]
    return points


def _precompute_fps_indices(
    samples: list[PaperFaceSample],
    *,
    point_samples: int,
    vecset_tokens: int,
    device,
) -> dict[Path, np.ndarray]:
    """Cache deterministic FPS query indices for fixed no-augmentation inputs."""

    import torch

    cache: dict[Path, np.ndarray] = {}
    with torch.no_grad():
        for sample in samples:
            points = _sample_point_features(sample.point_features, point_samples)
            xyz = torch.as_tensor(points[:, :3], dtype=torch.float32, device=device).unsqueeze(0)
            cache[sample.path] = _farthest_point_indices(xyz, int(vecset_tokens)).squeeze(0).cpu().numpy().astype(np.int64)
    return cache


def _random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    q = rng.normal(size=4)
    q /= np.linalg.norm(q) + 1e-12
    w, x, y, z = q
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _augment_sample(
    sample: PaperFaceSample,
    *,
    num_bins: int,
    rng: np.random.Generator,
    rotation: str,
    scale_min: float,
    scale_max: float,
    flip_prob: float,
    diagnostics: AugmentDiagnostics | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply FACE paper train augmentation and re-tokenize in paper order."""

    q_faces_zyx = np.asarray(sample.tokens, dtype=np.int64).reshape(-1, 3, 3)
    face_vertices = dequantize_normalized_points(
        q_faces_zyx.reshape(-1, 3)[:, [2, 1, 0]],
        num_bins=num_bins,
    ).reshape(-1, 3, 3)
    points = np.asarray(sample.point_features[:, :3], dtype=np.float64)
    normals = np.asarray(sample.point_features[:, 3:6], dtype=np.float64)

    if rotation == "none":
        rotation_matrix = np.eye(3, dtype=np.float64)
    elif rotation == "z":
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        rotation_matrix = np.asarray(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    elif rotation == "so3":
        rotation_matrix = _random_rotation_matrix(rng)
    else:  # pragma: no cover - guarded by argparse choices.
        raise ValueError(f"unknown augmentation rotation mode {rotation!r}")
    flips = np.where(rng.random(3) < flip_prob, -1.0, 1.0).astype(np.float64)
    scales = rng.uniform(scale_min, scale_max, size=3)
    affine = rotation_matrix @ np.diag(flips * scales)
    mirrored_affine = bool(np.linalg.det(affine) < 0.0)
    if diagnostics is not None:
        diagnostics.record_attempt(original_faces=len(sample.tokens), mirrored_affine=mirrored_affine)

    flat_vertices = face_vertices.reshape(-1, 3) @ affine.T
    augmented_points = points @ affine.T
    normal_matrix = np.linalg.inv(affine).T
    augmented_normals = normals @ normal_matrix.T
    augmented_normals /= np.linalg.norm(augmented_normals, axis=1, keepdims=True) + 1e-12

    transform = fit_face_token_transform(flat_vertices)
    normalized_vertices = transform.normalize(flat_vertices)
    normalized_points = transform.normalize(augmented_points)
    faces = np.arange(len(flat_vertices), dtype=np.int64).reshape(-1, 3)
    if mirrored_affine:
        faces[:, [1, 2]] = faces[:, [2, 1]]
    try:
        tokens, _ = canonicalize_mesh_faces_paper_zyx(
            normalized_vertices,
            faces,
            num_bins=num_bins,
            within_face_order=sample.paper_within_face_order,
        )
    except Exception as exc:
        if diagnostics is not None:
            diagnostics.record_fallback("tokenize_exception", exc)
        return sample.point_features, sample.tokens
    if len(tokens) == 0:
        if diagnostics is not None:
            diagnostics.record_fallback("empty_tokens")
        return sample.point_features, sample.tokens
    if diagnostics is not None:
        diagnostics.record_success(augmented_faces=len(tokens))
    point_features = np.concatenate(
        [normalized_points.astype(np.float32), augmented_normals.astype(np.float32)],
        axis=1,
    )
    return point_features, tokens.astype(np.int64)


def _make_batch(
    samples: list[PaperFaceSample],
    max_faces: int,
    device,
    point_samples: int | None = None,
    *,
    num_bins: int,
    augment: bool = False,
    rng: np.random.Generator | None = None,
    augment_rotation: str = "so3",
    augment_scale_min: float = 0.75,
    augment_scale_max: float = 1.25,
    augment_flip_prob: float = 0.5,
    augment_diagnostics: AugmentDiagnostics | None = None,
    fps_index_cache: dict[Path, np.ndarray] | None = None,
):  # type: ignore[no-untyped-def]
    import torch

    point_batches = []
    query_index_batches = []
    input_faces = torch.full((len(samples), max_faces, 9), -1, dtype=torch.long, device=device)
    target_faces = torch.full((len(samples), max_faces, 9), -100, dtype=torch.long, device=device)
    valid_weights = torch.zeros((len(samples), max_faces, 9), dtype=torch.float32, device=device)
    eos_targets = torch.zeros((len(samples), max_faces), dtype=torch.float32, device=device)
    eos_weights = torch.zeros((len(samples), max_faces), dtype=torch.float32, device=device)
    for row, sample in enumerate(samples):
        points = sample.point_features
        tokens = sample.tokens
        if augment:
            if rng is None:
                raise ValueError("augment=True requires an rng")
            points, tokens = _augment_sample(
                sample,
                num_bins=num_bins,
                rng=rng,
                rotation=augment_rotation,
                scale_min=augment_scale_min,
                scale_max=augment_scale_max,
                flip_prob=augment_flip_prob,
                diagnostics=augment_diagnostics,
            )
        tokens = tokens[:max_faces]
        if len(tokens):
            target_faces[row, : len(tokens)] = torch.as_tensor(tokens, dtype=torch.long, device=device)
            valid_weights[row, : len(tokens)] = 1.0
            eos_weights[row, : len(tokens)] = 1.0
            eos_targets[row, len(tokens) - 1] = 1.0
        if len(tokens) > 1:
            input_faces[row, 1 : len(tokens)] = torch.as_tensor(tokens[:-1], dtype=torch.long, device=device)
        points = _sample_point_features(points, point_samples)
        point_batches.append(torch.as_tensor(points, dtype=torch.float32, device=device))
        if fps_index_cache is not None and not augment:
            query_index_batches.append(torch.as_tensor(fps_index_cache[sample.path], dtype=torch.long, device=device))
    query_indices = torch.stack(query_index_batches, dim=0) if query_index_batches else None
    return torch.stack(point_batches, dim=0), input_faces, target_faces, valid_weights, eos_targets, eos_weights, query_indices


def _move_batch_to_device(batch, device, *, non_blocking: bool = True):  # type: ignore[no-untyped-def]
    return tuple(item.to(device, non_blocking=non_blocking) if hasattr(item, "to") else item for item in batch)


def _build_optimizer(torch, model, args):  # type: ignore[no-untyped-def]
    if args.optimizer != "muon":
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    muon_cls = torch.optim.Muon if hasattr(torch.optim, "Muon") else build_muon_fallback(torch)
    matrix_params = []
    scalar_or_embedding_params = []
    adamw_name_markers = (
        "embedding",
        "embed",
        "bos_face",
        "norm",
        "bias",
        "eos_head",
    )
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lower_name = name.lower()
        # Muon is intended for hidden-layer matrices. Discrete token embeddings,
        # positional tables, normalization scales, biases, and small heads are
        # more stable under AdamW while preserving the paper's Muon path for the
        # main matrix weights.
        use_adamw = param.ndim < 2 or any(marker in lower_name for marker in adamw_name_markers)
        if param.ndim == 2 and not use_adamw:
            matrix_params.append(param)
        else:
            scalar_or_embedding_params.append(param)
    optimizers = []
    if matrix_params:
        optimizers.append(muon_cls(matrix_params, lr=args.lr, weight_decay=args.weight_decay))
    if scalar_or_embedding_params:
        optimizers.append(torch.optim.AdamW(scalar_or_embedding_params, lr=args.lr, weight_decay=args.weight_decay))
    return _OptimizerGroup(optimizers)


def _compute_loss(F, logits, target_faces, valid_weights, num_bins: int):  # type: ignore[no-untyped-def]
    losses_raw = F.cross_entropy(
        logits.reshape(-1, num_bins),
        target_faces.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape_as(target_faces)
    valid = target_faces.ne(-100).to(losses_raw.dtype)
    denom = (valid_weights * valid).sum().clamp_min(1.0)
    return (losses_raw * valid_weights * valid).sum() / denom, denom


def _compute_eos_loss(F, eos_logits, eos_targets, eos_weights):  # type: ignore[no-untyped-def]
    if eos_logits is None:
        return None
    raw = F.binary_cross_entropy_with_logits(eos_logits, eos_targets, reduction="none")
    denom = eos_weights.sum().clamp_min(1.0)
    return (raw * eos_weights).sum() / denom


def _forward_outputs(model, decode_head: str, point_features, input_faces, target_faces, query_indices=None):  # type: ignore[no-untyped-def]
    hidden = model.hidden(point_features, input_faces, query_indices=query_indices)
    if decode_head == "parallel":
        logits = model.parallel_head(hidden).reshape(hidden.shape[0], hidden.shape[1], 9, model.num_bins)
    else:
        logits = model._causal_logits_from_hidden(hidden, target_faces)
    eos_logits = model.eos_logits_from_hidden(hidden) if hasattr(model, "eos_logits_from_hidden") else None
    return logits, eos_logits


def _forward_logits(model, decode_head: str, point_features, input_faces, target_faces):  # type: ignore[no-untyped-def]
    logits, _ = _forward_outputs(model, decode_head, point_features, input_faces, target_faces)
    return logits


def _evaluate_dataset_loss(
    F,
    model,
    samples: list[PaperFaceSample],
    *,
    max_faces: int,
    device,
    point_samples: int,
    num_bins: int,
    decode_head: str,
    batch_size: int,
    eos_loss_weight: float,
    precision: str,
    fps_index_cache: dict[Path, np.ndarray] | None = None,
) -> float:
    import torch

    was_training = bool(model.training)
    model.eval()
    total_loss = 0.0
    total_weight = 0.0
    with torch.no_grad():
        for start in range(0, len(samples), max(1, int(batch_size))):
            batch = samples[start : start + max(1, int(batch_size))]
            point_features, input_faces, target_faces, valid_weights, eos_targets, eos_weights, query_indices = _make_batch(
                batch,
                max_faces=max_faces,
                device=device,
                point_samples=point_samples,
                num_bins=num_bins,
                augment=False,
                fps_index_cache=fps_index_cache,
            )
            with _autocast_context(torch, device, precision):
                logits, eos_logits = _forward_outputs(
                    model,
                    decode_head,
                    point_features,
                    input_faces,
                    target_faces,
                    query_indices=query_indices,
                )
                coord_loss, weight = _compute_loss(F, logits, target_faces, valid_weights, num_bins)
                eos_loss = _compute_eos_loss(F, eos_logits, eos_targets, eos_weights)
            loss = coord_loss if eos_loss is None else coord_loss + float(eos_loss_weight) * eos_loss
            weight_value = float(weight.detach().cpu())
            total_loss += float(loss.detach().cpu()) * weight_value
            total_weight += weight_value
    if was_training:
        model.train()
    return total_loss / max(total_weight, 1.0)


class _OptimizerGroup:
    def __init__(self, optimizers):  # type: ignore[no-untyped-def]
        self.optimizers = list(optimizers)

    def zero_grad(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        for optimizer in self.optimizers:
            optimizer.zero_grad(*args, **kwargs)

    def step(self):  # type: ignore[no-untyped-def]
        for optimizer in self.optimizers:
            optimizer.step()


def _autocast_context(torch, device, precision: str):  # type: ignore[no-untyped-def]
    if precision == "fp32" or device.type not in {"cuda", "cpu"}:
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    try:
        return torch.amp.autocast(device_type=device.type, dtype=dtype)
    except AttributeError:  # pragma: no cover - older PyTorch compatibility.
        return torch.cuda.amp.autocast(dtype=dtype)


def _checkpoint_payload(
    *,
    args,
    model_state,
    losses: list[float],
    selection_losses: list[dict[str, float | int]],
    selection_eval_every: int,
    num_bins: int,
    max_faces: int,
    best_loss: float,
    best_step: int,
    augment_diagnostics: dict | None = None,
) -> dict:
    args_payload = args_to_json_safe(args)
    if not getattr(args, "augment_diagnostics", False):
        args_payload.pop("augment_diagnostics", None)
    payload = {
        "model_state": model_state,
        "args": args_payload,
        "losses": losses,
        "selection_losses": selection_losses,
        "selection_eval_every": int(selection_eval_every),
        "num_bins": num_bins,
        "max_faces": max_faces,
        "best_loss": best_loss,
        "best_step": best_step,
        "representation": "paper-face-arae",
        "has_vecset_encoder": True,
        "has_decoder_cross_attention": True,
        "encoder_backend": args.encoder_backend,
        "causal_mlp_variant": args.causal_mlp_variant,
        "face_embedding_variant": args.face_embedding_variant,
        "has_eos_head": not args.disable_eos_head,
        "eos_loss_weight": float(args.eos_loss_weight),
        "decode_head": args.decode_head,
        "precision": args.precision,
        "augmentation": {
            "enabled": not args.disable_augment,
            "rotation": args.augment_rotation,
            "scale_min": args.augment_scale_min,
            "scale_max": args.augment_scale_max,
            "flip_prob": args.augment_flip_prob,
        },
    }
    if augment_diagnostics is not None:
        payload["augmentation_diagnostics"] = augment_diagnostics
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Optional FACE checkpoint to load model weights from before training. Optimizer state is not resumed.",
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--point-samples", type=int, default=8192)
    parser.add_argument(
        "--model-max-faces",
        type=int,
        default=0,
        help="Optional decoder context cap. Defaults to the largest training shard; set for held-out meshes with more faces than the train split.",
    )
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--encoder-hidden-size", type=int, default=768)
    parser.add_argument("--encoder-layers", type=int, default=8)
    parser.add_argument("--decoder-layers", type=int, default=24)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--vecset-tokens", type=int, default=2048)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--encoder-backend", choices=["native", "shape2vecset"], default="shape2vecset")
    parser.add_argument(
        "--causal-mlp-variant",
        choices=["legacy_concat", "paper_chain"],
        default="legacy_concat",
        help=(
            "Coordinate decoder. legacy_concat is the cited TreeMeshGPT-style "
            "stagewise CausalMLP lane; paper_chain is an experimental ablation."
        ),
    )
    parser.add_argument(
        "--face-embedding-variant",
        choices=["continuous_mlp", "discrete_sum", "token_concat_project"],
        default="token_concat_project",
        help=(
            "Previous-face embedding. token_concat_project is the strict paper lane; "
            "continuous_mlp/discrete_sum are deprecated checkpoint-compatible ablations."
        ),
    )
    parser.add_argument(
        "--allow-deprecated-face-embedding",
        action="store_true",
        help="Allow continuous_mlp or discrete_sum for checkpoint compatibility/ablations. Strict paper gates should not use this.",
    )
    parser.add_argument("--disable-eos-head", action="store_true")
    parser.add_argument("--eos-loss-weight", type=float, default=0.05)
    parser.add_argument("--decode-head", choices=["causal", "parallel"], default="causal")
    parser.add_argument("--optimizer", choices=["muon", "adamw"], default="muon")
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument(
        "--precision",
        choices=["fp32", "bf16", "fp16"],
        default="fp32",
        help="Autocast precision. bf16 is the practical A100 paper-scale setting; fp32 preserves old behavior.",
    )
    parser.add_argument("--disable-augment", action="store_true")
    parser.add_argument("--augment-rotation", choices=["none", "z", "so3"], default="so3")
    parser.add_argument("--augment-scale-min", type=float, default=0.75)
    parser.add_argument("--augment-scale-max", type=float, default=1.25)
    parser.add_argument("--augment-flip-prob", type=float, default=0.5)
    parser.add_argument(
        "--augment-diagnostics",
        action="store_true",
        default=_env_flag("CLEARMESH_FACE_AUGMENT_DIAGNOSTICS"),
        help="Log online augmentation attempt/success/fallback counters. Also enabled by CLEARMESH_FACE_AUGMENT_DIAGNOSTICS=1.",
    )
    parser.add_argument("--log-every", type=int, default=0, help="Training log interval; defaults to five logs per run.")
    parser.add_argument(
        "--selection-eval-every",
        type=int,
        default=0,
        help="If set, select the saved checkpoint by full unaugmented dataset loss every N steps.",
    )
    parser.add_argument("--selection-eval-batch-size", type=int, default=1)
    parser.add_argument(
        "--skip-initial-selection-eval",
        action="store_true",
        help="Skip the step-1 selection pass; useful for large corpora where the first full eval dominates wall time.",
    )
    parser.add_argument(
        "--prefetch-batches",
        type=int,
        default=0,
        help="If >0 on CUDA, prepare the next augmented CPU batch in a background thread while the GPU trains.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="If >0, periodically write the current best checkpoint next to --output as <stem>.latest<suffix>.",
    )
    parser.add_argument(
        "--save-current-checkpoint",
        action="store_true",
        help="When checkpointing periodically, also write <stem>.current<suffix> with the current weights.",
    )
    parser.add_argument(
        "--cache-fps-indices",
        action="store_true",
        help="Precompute deterministic FPS query indices for fixed no-augmentation point clouds.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.augment_scale_min <= 0.0 or args.augment_scale_max <= 0.0:
        raise SystemExit("augmentation scale bounds must be positive")
    if args.augment_scale_min > args.augment_scale_max:
        raise SystemExit("--augment-scale-min cannot exceed --augment-scale-max")
    if not 0.0 <= args.augment_flip_prob <= 1.0:
        raise SystemExit("--augment-flip-prob must be in [0, 1]")
    if args.eos_loss_weight < 0.0:
        raise SystemExit("--eos-loss-weight must be >= 0")
    if args.checkpoint_every < 0:
        raise SystemExit("--checkpoint-every must be >= 0")
    if args.prefetch_batches < 0:
        raise SystemExit("--prefetch-batches must be >= 0")
    if args.cache_fps_indices and not args.disable_augment:
        raise SystemExit("--cache-fps-indices is only valid with --disable-augment")
    if args.face_embedding_variant in {"continuous_mlp", "discrete_sum"} and not args.allow_deprecated_face_embedding:
        raise SystemExit(
            f"{args.face_embedding_variant!r} is deprecated for new FACE training; "
            "use --face-embedding-variant token_concat_project, or pass "
            "--allow-deprecated-face-embedding for checkpoint compatibility/ablation."
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    samples, num_bins = _load_dataset(args.dataset_dir, limit=args.limit)
    observed_max_faces = max(len(sample.tokens) for sample in samples)
    max_faces = max(observed_max_faces, int(args.model_max_faces or 0))
    summary = {
        "samples": len(samples),
        "num_bins": num_bins,
        "max_faces": max_faces,
        "observed_max_faces": observed_max_faces,
        "paper_tokens_per_sample": int(max_faces * 9),
        "point_samples": int(args.point_samples),
        "architecture": "paper-face-arae",
        "encoder_backend": args.encoder_backend,
        "causal_mlp_variant": args.causal_mlp_variant,
        "face_embedding_variant": args.face_embedding_variant,
        "eos_head": not args.disable_eos_head,
        "precision": args.precision,
    }
    if args.augment_diagnostics:
        summary["augmentation_diagnostics_enabled"] = True
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.dry_run:
        return 0

    import torch
    import torch.nn.functional as F

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if args.precision == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise SystemExit("bf16 requested, but this CUDA device does not report bf16 support")
    fps_index_cache = None
    if args.cache_fps_indices:
        cache_started_at = time.perf_counter()
        fps_index_cache = _precompute_fps_indices(
            samples,
            point_samples=args.point_samples,
            vecset_tokens=args.vecset_tokens,
            device=device,
        )
        summary["fps_index_cache"] = {
            "enabled": True,
            "samples": len(fps_index_cache),
            "elapsed_sec": time.perf_counter() - cache_started_at,
        }
        print(json.dumps(summary["fps_index_cache"], sort_keys=True))
    model = build_paper_face_arae(
        num_bins=num_bins,
        max_faces=max_faces,
        point_feature_dim=6,
        hidden_size=args.hidden_size,
        encoder_hidden_size=args.encoder_hidden_size,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        heads=args.heads,
        vecset_tokens=args.vecset_tokens,
        latent_dim=args.latent_dim,
        encoder_backend=args.encoder_backend,
        causal_mlp_variant=args.causal_mlp_variant,
        face_embedding_variant=args.face_embedding_variant,
        enable_eos_head=not args.disable_eos_head,
    ).to(device)
    if args.init_checkpoint is not None:
        if not args.init_checkpoint.exists():
            raise SystemExit(f"--init-checkpoint not found: {args.init_checkpoint}")
        checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        model_state = checkpoint.get("model_state", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        if not isinstance(model_state, dict):
            raise SystemExit(f"--init-checkpoint has no model_state dict: {args.init_checkpoint}")
        missing, unexpected = model.load_state_dict(model_state, strict=False)
        print(
            json.dumps(
                {
                    "init_checkpoint": str(args.init_checkpoint),
                    "init_checkpoint_best_loss": checkpoint.get("best_loss") if isinstance(checkpoint, dict) else None,
                    "init_checkpoint_best_step": checkpoint.get("best_step") if isinstance(checkpoint, dict) else None,
                    "init_missing_keys": list(missing),
                    "init_unexpected_keys": list(unexpected),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    optimizer = _build_optimizer(torch, model, args)
    augment_rng = np.random.default_rng(args.seed + 1009)
    augment_diagnostics = AugmentDiagnostics() if args.augment_diagnostics else None
    best_loss = float("inf")
    best_step = 0
    losses: list[float] = []
    selection_losses: list[dict[str, float | int]] = []
    best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
    log_every = int(args.log_every or max(1, args.steps // 5))
    selection_eval_every = int(args.selection_eval_every or 0)
    prefetch_enabled = bool(args.prefetch_batches > 0 and device.type == "cuda")
    cpu_device = torch.device("cpu")

    def build_train_batch(batch_device):  # type: ignore[no-untyped-def]
        batch = random.choices(samples, k=args.batch_size)
        return _make_batch(
            batch,
            max_faces=max_faces,
            device=batch_device,
            point_samples=args.point_samples,
            num_bins=num_bins,
            augment=not args.disable_augment,
            rng=augment_rng,
            augment_rotation=args.augment_rotation,
            augment_scale_min=args.augment_scale_min,
            augment_scale_max=args.augment_scale_max,
            augment_flip_prob=args.augment_flip_prob,
            augment_diagnostics=augment_diagnostics,
            fps_index_cache=fps_index_cache,
        )

    executor: ThreadPoolExecutor | None = None
    pending_batch: Future | None = None
    if prefetch_enabled:
        # One worker keeps RNG use deterministic while overlapping CPU
        # augmentation with GPU kernels from the previous step.
        executor = ThreadPoolExecutor(max_workers=1)
        pending_batch = executor.submit(build_train_batch, cpu_device)
    started_at = time.perf_counter()
    selection_eval_elapsed_sec = 0.0
    try:
        for step in range(1, args.steps + 1):
            if prefetch_enabled:
                assert pending_batch is not None
                cpu_batch = pending_batch.result()
                pending_batch = executor.submit(build_train_batch, cpu_device) if executor is not None else None
                point_features, input_faces, target_faces, valid_weights, eos_targets, eos_weights, query_indices = _move_batch_to_device(
                    cpu_batch,
                    device,
                    non_blocking=True,
                )
            else:
                point_features, input_faces, target_faces, valid_weights, eos_targets, eos_weights, query_indices = build_train_batch(device)
            with _autocast_context(torch, device, args.precision):
                logits, eos_logits = _forward_outputs(
                    model,
                    args.decode_head,
                    point_features,
                    input_faces,
                    target_faces,
                    query_indices=query_indices,
                )
                coord_loss, _ = _compute_loss(F, logits, target_faces, valid_weights, num_bins)
                eos_loss = _compute_eos_loss(F, eos_logits, eos_targets, eos_weights)
                loss = coord_loss if eos_loss is None else coord_loss + args.eos_loss_weight * eos_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_value = float(loss.detach().cpu())
            losses.append(loss_value)
            selection_loss_value = None
            selection_loss_elapsed_sec = None
            run_initial_selection = step == 1 and not args.skip_initial_selection_eval
            run_periodic_selection = step == args.steps or step % selection_eval_every == 0
            if selection_eval_every > 0 and (run_initial_selection or run_periodic_selection):
                selection_started_at = time.perf_counter()
                selection_loss_value = _evaluate_dataset_loss(
                    F,
                    model,
                    samples,
                    max_faces=max_faces,
                    device=device,
                    point_samples=args.point_samples,
                    num_bins=num_bins,
                    decode_head=args.decode_head,
                    batch_size=args.selection_eval_batch_size,
                    eos_loss_weight=args.eos_loss_weight,
                    precision=args.precision,
                    fps_index_cache=fps_index_cache,
                )
                selection_loss_elapsed_sec = time.perf_counter() - selection_started_at
                selection_eval_elapsed_sec += selection_loss_elapsed_sec
                selection_losses.append({"step": step, "loss": selection_loss_value})
            if selection_eval_every > 0:
                should_consider_checkpoint = selection_loss_value is not None
                score_loss = selection_loss_value if selection_loss_value is not None else float("inf")
            else:
                should_consider_checkpoint = True
                score_loss = loss_value
            if should_consider_checkpoint and score_loss < best_loss:
                best_loss = score_loss
                best_step = step
                best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            if step == 1 or step == args.steps or step % log_every == 0:
                elapsed_sec = max(time.perf_counter() - started_at, 1e-6)
                train_elapsed_sec = max(elapsed_sec - selection_eval_elapsed_sec, 1e-6)
                steps_per_sec = step / elapsed_sec
                train_steps_per_sec_ex_selection = step / train_elapsed_sec
                remaining_sec = (args.steps - step) / max(steps_per_sec, 1e-6)
                remaining_sec_ex_selection = (args.steps - step) / max(train_steps_per_sec_ex_selection, 1e-6)
                print(
                    json.dumps(
                        {
                            "step": step,
                            "loss": loss_value,
                            "coord_loss": float(coord_loss.detach().cpu()),
                            **({} if eos_loss is None else {"eos_loss": float(eos_loss.detach().cpu())}),
                            "best_loss": best_loss,
                            "best_step": best_step,
                            **({} if selection_loss_value is None else {"selection_loss": selection_loss_value}),
                            **({} if selection_loss_elapsed_sec is None else {"selection_eval_sec": selection_loss_elapsed_sec}),
                            "selection_eval_elapsed_sec": selection_eval_elapsed_sec,
                            "elapsed_sec": elapsed_sec,
                            "steps_per_sec": steps_per_sec,
                            "train_steps_per_sec_ex_selection": train_steps_per_sec_ex_selection,
                            "eta_sec": remaining_sec,
                            "eta_sec_ex_selection": remaining_sec_ex_selection,
                            "prefetch_enabled": prefetch_enabled,
                            **(
                                {}
                                if augment_diagnostics is None
                                else {
                                    "augmentation_diagnostics": augment_diagnostics.snapshot(),
                                    "augmentation_diagnostics_prefetch_may_be_ahead": prefetch_enabled,
                                }
                            ),
                        }
                    ),
                    flush=True,
                )
            if args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                latest_output = args.output.with_name(f"{args.output.stem}.latest{args.output.suffix}")
                torch.save(
                    _checkpoint_payload(
                        args=args,
                        model_state=best_state,
                        losses=losses,
                        selection_losses=selection_losses,
                        selection_eval_every=selection_eval_every,
                        num_bins=num_bins,
                        max_faces=max_faces,
                        best_loss=best_loss,
                        best_step=best_step,
                        augment_diagnostics=augment_diagnostics.snapshot() if augment_diagnostics is not None else None,
                    ),
                    latest_output,
                )
                if args.save_current_checkpoint:
                    current_output = args.output.with_name(f"{args.output.stem}.current{args.output.suffix}")
                    current_score = selection_loss_value if selection_loss_value is not None else loss_value
                    current_payload = _checkpoint_payload(
                        args=args,
                        model_state={key: value.detach().cpu() for key, value in model.state_dict().items()},
                        losses=losses,
                        selection_losses=selection_losses,
                        selection_eval_every=selection_eval_every,
                        num_bins=num_bins,
                        max_faces=max_faces,
                        best_loss=float(current_score),
                        best_step=step,
                        augment_diagnostics=augment_diagnostics.snapshot() if augment_diagnostics is not None else None,
                    )
                    current_payload["checkpoint_kind"] = "current"
                    current_payload["current_step"] = step
                    current_payload["current_loss"] = loss_value
                    if selection_loss_value is not None:
                        current_payload["current_selection_loss"] = selection_loss_value
                    torch.save(current_payload, current_output)
    finally:
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        _checkpoint_payload(
            args=args,
            model_state=best_state,
            losses=losses,
            selection_losses=selection_losses,
            selection_eval_every=selection_eval_every,
            num_bins=num_bins,
            max_faces=max_faces,
            best_loss=best_loss,
            best_step=best_step,
            augment_diagnostics=augment_diagnostics.snapshot() if augment_diagnostics is not None else None,
        ),
        args.output,
    )
    final_log = {"checkpoint": str(args.output), "best_loss": best_loss, "best_step": best_step, "device": str(device)}
    if augment_diagnostics is not None:
        final_log["augmentation_diagnostics"] = augment_diagnostics.snapshot()
    print(json.dumps(final_log))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
