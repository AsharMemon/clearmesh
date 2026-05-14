#!/usr/bin/env python3
"""Train FACE-lite v2: point-conditioned triangle-index decoder."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clearmesh.mesh_heads.face_arae import build_tiny_point_conditioned_indexed_face_decoder
from clearmesh.mesh_heads.face_indexed import indexed_face_closure_counts
from clearmesh.utils.checkpoint import args_to_json_safe
from clearmesh.utils.muon_fallback import build_muon_fallback


@dataclass(frozen=True)
class FaceIndexedSample:
    path: Path
    point_features: np.ndarray
    vertices: np.ndarray
    faces: np.ndarray


def _load_sample(path: Path) -> tuple[FaceIndexedSample, int]:
    data = np.load(path)
    required = {"surface_points", "surface_normals", "indexed_vertices", "indexed_faces", "num_bins"}
    missing = sorted(required.difference(data.files))
    if missing:
        raise ValueError(f"{path} is missing indexed FACE arrays: {missing}")
    points = np.asarray(data["surface_points"], dtype=np.float32)
    normals = np.asarray(data["surface_normals"], dtype=np.float32)
    vertices = np.asarray(data["indexed_vertices"], dtype=np.int64)
    faces = np.asarray(data["indexed_faces"], dtype=np.int64)
    num_bins = int(np.asarray(data["num_bins"]).reshape(-1)[0])
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"{path} has invalid indexed_vertices shape {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"{path} has invalid indexed_faces shape {faces.shape}")
    if len(points) == 0:
        raise ValueError(f"{path} has no point conditioning samples")
    return (
        FaceIndexedSample(
            path=path,
            point_features=np.concatenate([points, normals], axis=1).astype(np.float32),
            vertices=vertices,
            faces=faces,
        ),
        num_bins,
    )


def _load_dataset(dataset_dir: Path, limit: int = 0) -> tuple[list[FaceIndexedSample], int]:
    paths = sorted(path for path in dataset_dir.glob("*.npz") if not path.name.startswith("._"))
    if limit:
        paths = paths[:limit]
    samples: list[FaceIndexedSample] = []
    failures: list[str] = []
    num_bins: int | None = None
    for path in paths:
        try:
            sample, sample_bins = _load_sample(path)
        except Exception as exc:
            if len(failures) < 8:
                failures.append(f"{path.name}: {type(exc).__name__}: {exc}")
            continue
        if num_bins is None:
            num_bins = sample_bins
        elif num_bins != sample_bins:
            raise ValueError(f"Mixed num_bins in dataset: {num_bins} and {sample_bins}")
        samples.append(sample)
    if not samples or num_bins is None:
        details = "\n".join(f"  - {failure}" for failure in failures)
        suffix = f"\nFirst load failures:\n{details}" if details else ""
        raise SystemExit(
            f"No indexed FACE shards found in {dataset_dir} "
            f"(npz_files={len(paths)}, limit={limit}).{suffix}"
        )
    return samples, num_bins


def _closure_count_targets(faces: np.ndarray) -> np.ndarray:
    """Return teacher edge-closure counts for each next face.

    The label is the number of currently open boundary edges consumed by the
    target face at that autoregressive step. A closed tetrahedron, for example,
    produces the rhythm ``0, 1, 2, 3``.
    """

    return indexed_face_closure_counts(faces)


def _edge_key(a: int, b: int) -> tuple[int, int]:
    left = int(a)
    right = int(b)
    return (left, right) if left <= right else (right, left)


def _edge_action_targets(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return target open-edge action and third vertex for each target face.

    The first disconnected seed face has no boundary action target. For later
    faces, if the teacher face consumes at least one currently open edge, we use
    that edge as the conditioning action and train the model to select the
    remaining third vertex. This is the supervised version of the boundary-edge
    completion decode heuristic.
    """

    edge_counts: dict[tuple[int, int], int] = {}
    edges_out = np.full((len(faces), 2), -1, dtype=np.int64)
    thirds_out = np.full((len(faces),), -100, dtype=np.int64)
    for idx, face_arr in enumerate(np.asarray(faces, dtype=np.int64).reshape(-1, 3)):
        face = tuple(int(value) for value in face_arr)
        face_edges = (
            (_edge_key(face[0], face[1]), face[2]),
            (_edge_key(face[1], face[2]), face[0]),
            (_edge_key(face[2], face[0]), face[1]),
        )
        closing = [(edge, third) for edge, third in face_edges if edge_counts.get(edge, 0) == 1]
        if closing:
            edge, third = max(closing, key=lambda item: (edge_counts.get(item[0], 0), -item[0][0], -item[0][1]))
            edges_out[idx] = np.asarray(edge, dtype=np.int64)
            thirds_out[idx] = int(third)
        for edge, _ in face_edges:
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    return edges_out, thirds_out


def _edge_choice_targets(faces: np.ndarray, max_choices: int) -> tuple[np.ndarray, np.ndarray]:
    """Return candidate open edges and target edge labels for each next face."""

    max_choices = max(1, int(max_choices))
    edge_counts: dict[tuple[int, int], int] = {}
    candidates_out = np.full((len(faces), max_choices, 2), -1, dtype=np.int64)
    labels_out = np.full((len(faces),), -100, dtype=np.int64)
    for idx, face_arr in enumerate(np.asarray(faces, dtype=np.int64).reshape(-1, 3)):
        face = tuple(int(value) for value in face_arr)
        boundary_edges = sorted(edge for edge, count in edge_counts.items() if count == 1)
        face_edges = (
            (_edge_key(face[0], face[1]), face[2]),
            (_edge_key(face[1], face[2]), face[0]),
            (_edge_key(face[2], face[0]), face[1]),
        )
        closing = [(edge, third) for edge, third in face_edges if edge_counts.get(edge, 0) == 1]
        if closing:
            target_edge, _ = max(closing, key=lambda item: (edge_counts.get(item[0], 0), -item[0][0], -item[0][1]))
            negatives = [edge for edge in boundary_edges if edge != target_edge]
            row_edges = [target_edge] + negatives[: max_choices - 1]
            candidates_out[idx, : len(row_edges)] = np.asarray(row_edges, dtype=np.int64)
            labels_out[idx] = 0
        for edge, _ in face_edges:
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    return candidates_out, labels_out


def _make_batch(
    samples: list[FaceIndexedSample],
    max_vertices: int,
    max_faces: int,
    device,
    point_samples: int | None = None,
    edge_choice_candidates: int = 64,
    early_face_count: int = 0,
    early_face_loss_weight: float = 1.0,
):  # type: ignore[no-untyped-def]
    import torch

    point_batches = []
    vertex_table = torch.full((len(samples), max_vertices, 3), -1, dtype=torch.long, device=device)
    input_faces = torch.full((len(samples), max_faces, 3), -1, dtype=torch.long, device=device)
    target_faces = torch.full((len(samples), max_faces, 3), -100, dtype=torch.long, device=device)
    target_weights = torch.zeros((len(samples), max_faces, 3), dtype=torch.float32, device=device)
    target_closure_counts = torch.full((len(samples), max_faces), -100, dtype=torch.long, device=device)
    target_edge_actions = torch.full((len(samples), max_faces, 2), -1, dtype=torch.long, device=device)
    target_edge_thirds = torch.full((len(samples), max_faces), -100, dtype=torch.long, device=device)
    target_edge_choice_candidates = torch.full(
        (len(samples), max_faces, max(1, int(edge_choice_candidates)), 2),
        -1,
        dtype=torch.long,
        device=device,
    )
    target_edge_choice_labels = torch.full((len(samples), max_faces), -100, dtype=torch.long, device=device)
    face_counts = torch.zeros((len(samples),), dtype=torch.long, device=device)
    for row, sample in enumerate(samples):
        vertices = sample.vertices[:max_vertices]
        faces = sample.faces[:max_faces]
        valid_faces = faces[np.all((faces >= 0) & (faces < len(vertices)), axis=1)]
        vertex_table[row, : len(vertices)] = torch.as_tensor(vertices, dtype=torch.long, device=device)
        face_count = len(valid_faces)
        face_counts[row] = int(face_count)
        if face_count:
            target_faces[row, :face_count] = torch.as_tensor(valid_faces, dtype=torch.long, device=device)
            target_weights[row, :face_count] = 1.0
            if early_face_count > 0 and float(early_face_loss_weight) != 1.0:
                boosted = min(face_count, int(early_face_count))
                target_weights[row, :boosted] *= float(early_face_loss_weight)
            target_closure_counts[row, :face_count] = torch.as_tensor(
                _closure_count_targets(valid_faces),
                dtype=torch.long,
                device=device,
            )
            edge_actions, edge_thirds = _edge_action_targets(valid_faces)
            target_edge_actions[row, :face_count] = torch.as_tensor(edge_actions, dtype=torch.long, device=device)
            target_edge_thirds[row, :face_count] = torch.as_tensor(edge_thirds, dtype=torch.long, device=device)
            edge_choice_edges, edge_choice_labels = _edge_choice_targets(valid_faces, edge_choice_candidates)
            target_edge_choice_candidates[row, :face_count] = torch.as_tensor(
                edge_choice_edges,
                dtype=torch.long,
                device=device,
            )
            target_edge_choice_labels[row, :face_count] = torch.as_tensor(
                edge_choice_labels,
                dtype=torch.long,
                device=device,
            )
        if face_count > 1:
            input_faces[row, 1:face_count] = torch.as_tensor(valid_faces[:-1], dtype=torch.long, device=device)
        points = sample.point_features
        if point_samples is not None and point_samples > 0:
            if len(points) >= point_samples:
                points = points[:point_samples]
            else:
                repeat = int(np.ceil(point_samples / len(points)))
                points = np.tile(points, (repeat, 1))[:point_samples]
        point_batches.append(torch.as_tensor(points, dtype=torch.float32, device=device))
    return (
        torch.stack(point_batches, dim=0),
        vertex_table,
        input_faces,
        target_faces,
        target_weights,
        target_closure_counts,
        target_edge_actions,
        target_edge_thirds,
        target_edge_choice_candidates,
        target_edge_choice_labels,
        face_counts,
    )


class _OptimizerGroup:
    def __init__(self, optimizers):  # type: ignore[no-untyped-def]
        self.optimizers = list(optimizers)

    def zero_grad(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        for optimizer in self.optimizers:
            optimizer.zero_grad(*args, **kwargs)

    def step(self) -> None:
        for optimizer in self.optimizers:
            optimizer.step()

    def state_dict(self) -> list[dict]:  # type: ignore[type-arg]
        return [optimizer.state_dict() for optimizer in self.optimizers]

    def load_state_dict(self, states: list[dict]) -> None:  # type: ignore[type-arg]
        if len(states) != len(self.optimizers):
            raise ValueError(f"optimizer state count mismatch: {len(states)} != {len(self.optimizers)}")
        for optimizer, state in zip(self.optimizers, states, strict=True):
            optimizer.load_state_dict(state)


def _build_optimizer(torch, model, args):  # type: ignore[no-untyped-def]
    if args.optimizer != "muon":
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    muon_cls = torch.optim.Muon if hasattr(torch.optim, "Muon") else build_muon_fallback(torch)
    matrix_params = []
    adamw_params = []
    adamw_name_markers = (
        "embedding",
        "embed",
        "vertex_table",
        "bos_face",
        "norm",
        "bias",
        "count_head",
        "topology_output",
        "seed_face",
    )
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lower_name = name.lower()
        use_adamw = param.ndim < 2 or any(marker in lower_name for marker in adamw_name_markers)
        if param.ndim == 2 and not use_adamw:
            matrix_params.append(param)
        else:
            adamw_params.append(param)

    optimizers = []
    if matrix_params:
        optimizers.append(muon_cls(matrix_params, lr=args.lr, weight_decay=args.weight_decay))
    if adamw_params:
        optimizers.append(torch.optim.AdamW(adamw_params, lr=args.lr, weight_decay=args.weight_decay))
    return _OptimizerGroup(optimizers)


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
    optimizer_state,
    losses: list[float],
    num_bins: int,
    max_vertices: int,
    max_faces: int,
    best_loss: float,
    best_step: int,
    step: int,
) -> dict:
    return {
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "args": args_to_json_safe(args),
        "losses": losses,
        "num_bins": num_bins,
        "max_vertices": max_vertices,
        "max_faces": max_faces,
        "best_loss": best_loss,
        "best_step": best_step,
        "step": int(step),
        "representation": "face-indexed-v2",
        "has_count_head": True,
        "has_corner_causal_head": bool(args.corner_head == "causal"),
        "has_topology_head": bool(args.topology_loss_weight > 0),
        "has_edge_action_head": bool(args.edge_action_loss_weight > 0),
        "has_edge_choice_head": bool(args.edge_choice_loss_weight > 0),
        "has_seed_face_head": bool(args.seed_face_loss_weight > 0),
        "optimizer": args.optimizer,
        "precision": args.precision,
    }


def _save_checkpoint(path: Path, payload: dict, torch) -> None:  # type: ignore[no-untyped-def]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/face_indexed_conditioned_tiny.pt"))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--point-samples", type=int, default=0)
    parser.add_argument("--hidden-size", type=int, default=192)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--condition-tokens", type=int, default=8)
    parser.add_argument("--edge-head-mode", choices=["index", "geometry"], default="geometry")
    parser.add_argument("--condition-backend", choices=["pooled", "vecset"], default="pooled")
    parser.add_argument("--decoder-backend", choices=["prefix", "cross_attn"], default="prefix")
    parser.add_argument("--encoder-layers", type=int, default=4)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--face-output-mode", choices=["linear", "geometry"], default="linear")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--optimizer", choices=["adamw", "muon"], default="adamw")
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="fp32")
    parser.add_argument("--init-checkpoint", type=Path, default=None, help="Load model weights before training.")
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="Resume model, optimizer, losses, and step from a previous indexed checkpoint.",
    )
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--save-current-checkpoint", action="store_true")
    parser.add_argument("--log-every", type=int, default=0)
    parser.add_argument("--grad-clip-norm", type=float, default=0.0)
    parser.add_argument("--corner-head", choices=["causal", "parallel"], default="parallel")
    parser.add_argument("--count-loss-weight", type=float, default=0.05)
    parser.add_argument("--topology-loss-weight", type=float, default=0.0)
    parser.add_argument("--edge-action-loss-weight", type=float, default=0.0)
    parser.add_argument("--edge-choice-loss-weight", type=float, default=0.0)
    parser.add_argument("--edge-choice-candidates", type=int, default=64)
    parser.add_argument("--seed-face-loss-weight", type=float, default=0.0)
    parser.add_argument("--seed-face-loss-stop-step", type=int, default=0)
    parser.add_argument("--early-face-count", type=int, default=0)
    parser.add_argument("--early-face-loss-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.checkpoint_every < 0:
        raise SystemExit("--checkpoint-every must be >= 0")
    if args.init_checkpoint is not None and args.resume_checkpoint is not None:
        raise SystemExit("Use only one of --init-checkpoint or --resume-checkpoint.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    samples, num_bins = _load_dataset(args.dataset_dir, limit=args.limit)
    max_vertices = max(len(sample.vertices) for sample in samples)
    max_faces = max(len(sample.faces) for sample in samples)
    point_count = int(samples[0].point_features.shape[0] if args.point_samples <= 0 else args.point_samples)
    summary = {
        "samples": len(samples),
        "num_bins": num_bins,
        "max_vertices": max_vertices,
        "max_faces": max_faces,
        "index_tokens_per_sample": int(max_faces * 3),
        "vertex_coordinate_tokens_per_sample": int(max_vertices * 3),
        "point_samples": point_count,
        "edge_head_mode": args.edge_head_mode,
        "condition_backend": args.condition_backend,
        "decoder_backend": args.decoder_backend,
        "encoder_layers": args.encoder_layers,
        "latent_dim": args.latent_dim,
        "face_output_mode": args.face_output_mode,
    }
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
        raise SystemExit("bf16 requested but CUDA device does not report bf16 support")
    model = build_tiny_point_conditioned_indexed_face_decoder(
        num_bins=num_bins,
        max_vertices=max_vertices,
        max_faces=max_faces,
        point_feature_dim=6,
        hidden_size=args.hidden_size,
        layers=args.layers,
        heads=args.heads,
        condition_tokens=args.condition_tokens,
        edge_head_mode=args.edge_head_mode,
        condition_backend=args.condition_backend,
        decoder_backend=args.decoder_backend,
        encoder_layers=args.encoder_layers,
        latent_dim=args.latent_dim,
        face_output_mode=args.face_output_mode,
    ).to(device)
    if args.init_checkpoint is not None:
        if not args.init_checkpoint.exists():
            raise SystemExit(f"--init-checkpoint not found: {args.init_checkpoint}")
        checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        state = checkpoint.get("model_state")
        if not isinstance(state, dict):
            raise SystemExit(f"--init-checkpoint has no model_state dict: {args.init_checkpoint}")
        model.load_state_dict(state)

    optimizer = _build_optimizer(torch, model, args)
    losses: list[float] = []
    best_loss = float("inf")
    best_step = 0
    start_step = 1

    if args.resume_checkpoint is not None:
        if not args.resume_checkpoint.exists():
            raise SystemExit(f"--resume-checkpoint not found: {args.resume_checkpoint}")
        checkpoint = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
        state = checkpoint.get("model_state")
        if not isinstance(state, dict):
            raise SystemExit(f"--resume-checkpoint has no model_state dict: {args.resume_checkpoint}")
        model.load_state_dict(state)
        optimizer_state = checkpoint.get("optimizer_state")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        losses = [float(value) for value in checkpoint.get("losses", [])]
        best_loss = float(checkpoint.get("best_loss", best_loss))
        best_step = int(checkpoint.get("best_step", 0))
        start_step = int(checkpoint.get("step", len(losses))) + 1

    best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
    if start_step > args.steps:
        raise SystemExit(f"resume checkpoint step {start_step - 1} is already >= requested --steps {args.steps}")
    log_every = int(args.log_every or max(1, args.steps // 5))
    for step in range(start_step, args.steps + 1):
        batch = random.choices(samples, k=args.batch_size)
        (
            point_features,
            vertex_table,
            input_faces,
            target_faces,
            target_weights,
            target_closure_counts,
            target_edge_actions,
            target_edge_thirds,
            target_edge_choice_candidates,
            target_edge_choice_labels,
            face_counts,
        ) = _make_batch(
            batch,
            max_vertices=max_vertices,
            max_faces=max_faces,
            device=device,
            point_samples=args.point_samples if args.point_samples > 0 else None,
            edge_choice_candidates=args.edge_choice_candidates,
            early_face_count=args.early_face_count,
            early_face_loss_weight=args.early_face_loss_weight,
        )
        topology_loss = None
        edge_action_loss = None
        edge_choice_loss = None
        seed_face_loss = None
        hidden_for_aux = None
        with _autocast_context(torch, device, args.precision):
            if args.corner_head == "causal" and hasattr(model, "_corner_causal_logits_from_hidden"):
                hidden = model._hidden(point_features, vertex_table, input_faces)
                hidden_for_aux = hidden
                prefix = target_faces.masked_fill(target_faces.lt(0), -1)
                logits = model._corner_causal_logits_from_hidden(hidden, prefix)
                if args.topology_loss_weight > 0 and hasattr(model, "topology_output"):
                    topology_loss = F.cross_entropy(
                        model.topology_output(hidden).reshape(-1, 4),
                        target_closure_counts.reshape(-1),
                        ignore_index=-100,
                    )
            elif args.topology_loss_weight > 0 and hasattr(model, "forward_with_topology"):
                outputs = model.forward_with_topology(point_features, vertex_table, input_faces)
                logits = outputs["face_logits"]
                topology_loss = F.cross_entropy(
                    outputs["closure_logits"].reshape(-1, 4),
                    target_closure_counts.reshape(-1),
                    ignore_index=-100,
                )
            else:
                logits = model(point_features, vertex_table, input_faces)
            token_losses = F.cross_entropy(
                logits.reshape(-1, max_vertices),
                target_faces.reshape(-1),
                ignore_index=-100,
                reduction="none",
            ).reshape_as(target_faces)
            valid = target_faces.ne(-100)
            token_loss = (token_losses * target_weights * valid.to(token_losses.dtype)).sum() / torch.clamp(
                (target_weights * valid.to(target_weights.dtype)).sum(),
                min=1.0,
            )
            if args.count_loss_weight > 0:
                count_loss = F.cross_entropy(model.predict_face_count_logits(point_features, vertex_table), face_counts)
                loss = token_loss + float(args.count_loss_weight) * count_loss
            else:
                count_loss = token_loss.new_tensor(0.0)
                loss = token_loss
            seed_face_loss_active = bool(
                args.seed_face_loss_weight > 0
                and hasattr(model, "seed_face_logits")
                and (args.seed_face_loss_stop_step <= 0 or step <= int(args.seed_face_loss_stop_step))
            )
            if seed_face_loss_active:
                seed_logits = model.seed_face_logits(point_features, vertex_table)
                seed_targets = target_faces[:, 0, :]
                seed_face_loss = F.cross_entropy(
                    seed_logits.reshape(-1, max_vertices),
                    seed_targets.reshape(-1),
                    ignore_index=-100,
                )
                loss = loss + float(args.seed_face_loss_weight) * seed_face_loss
            if topology_loss is not None:
                loss = loss + float(args.topology_loss_weight) * topology_loss
            if args.edge_action_loss_weight > 0 and hasattr(model, "forward_edge_action"):
                if hidden_for_aux is not None and hasattr(model, "_edge_action_logits_from_hidden"):
                    edge_logits = model._edge_action_logits_from_hidden(hidden_for_aux, target_edge_actions, vertex_table)
                else:
                    edge_logits = model.forward_edge_action(point_features, vertex_table, input_faces, target_edge_actions)
                edge_action_loss = F.cross_entropy(
                    edge_logits.reshape(-1, max_vertices),
                    target_edge_thirds.reshape(-1),
                    ignore_index=-100,
                )
                loss = loss + float(args.edge_action_loss_weight) * edge_action_loss
            if args.edge_choice_loss_weight > 0 and hasattr(model, "forward_edge_choice"):
                if hidden_for_aux is not None and hasattr(model, "_edge_choice_logits_from_hidden"):
                    edge_choice_logits = model._edge_choice_logits_from_hidden(
                        hidden_for_aux,
                        target_edge_choice_candidates,
                    )
                else:
                    edge_choice_logits = model.forward_edge_choice(
                        point_features,
                        vertex_table,
                        input_faces,
                        target_edge_choice_candidates,
                    )
                invalid_edges = target_edge_choice_candidates[..., 0].lt(0)
                edge_choice_logits = edge_choice_logits.masked_fill(invalid_edges, -1e9)
                edge_choice_loss = F.cross_entropy(
                    edge_choice_logits.reshape(-1, edge_choice_logits.shape[-1]),
                    target_edge_choice_labels.reshape(-1),
                    ignore_index=-100,
                )
                loss = loss + float(args.edge_choice_loss_weight) * edge_choice_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip_norm))
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if losses[-1] < best_loss:
            best_loss = losses[-1]
            best_step = step
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
        if args.checkpoint_every and step % int(args.checkpoint_every) == 0:
            payload = _checkpoint_payload(
                args=args,
                model_state={key: value.detach().cpu() for key, value in model.state_dict().items()},
                optimizer_state=optimizer.state_dict(),
                losses=losses,
                num_bins=num_bins,
                max_vertices=max_vertices,
                max_faces=max_faces,
                best_loss=best_loss,
                best_step=best_step,
                step=step,
            )
            _save_checkpoint(args.output.with_suffix(f".step{step:06d}.pt"), payload, torch)
            if args.save_current_checkpoint:
                _save_checkpoint(args.output.parent / "checkpoint.current.pt", payload, torch)

        if step == 1 or step == args.steps or step % log_every == 0:
            log_item = {
                "step": step,
                "loss": losses[-1],
                "token_loss": float(token_loss.detach().cpu()),
                "count_loss": float(count_loss.detach().cpu()),
            }
            if topology_loss is not None:
                log_item["topology_loss"] = float(topology_loss.detach().cpu())
            if edge_action_loss is not None:
                log_item["edge_action_loss"] = float(edge_action_loss.detach().cpu())
            if edge_choice_loss is not None:
                log_item["edge_choice_loss"] = float(edge_choice_loss.detach().cpu())
            if seed_face_loss is not None:
                log_item["seed_face_loss"] = float(seed_face_loss.detach().cpu())
                log_item["seed_face_loss_active"] = bool(seed_face_loss_active)
            print(json.dumps(log_item))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    _save_checkpoint(
        args.output,
        _checkpoint_payload(
            args=args,
            model_state=best_state,
            optimizer_state=optimizer.state_dict(),
            losses=losses,
            num_bins=num_bins,
            max_vertices=max_vertices,
            max_faces=max_faces,
            best_loss=best_loss,
            best_step=best_step,
            step=args.steps,
        ),
        torch,
    )
    print(json.dumps({"checkpoint": str(args.output), "final_loss": losses[-1], "best_loss": best_loss, "best_step": best_step, "device": str(device)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
