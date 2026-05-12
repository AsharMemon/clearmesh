#!/usr/bin/env python3
"""Probe FACE first-face ambiguity and constrained decoding.

The FACE paper specifies an autoregressive face decoder and a CausalMLP that
factorizes one face into nine quantized coordinate tokens. It does not specify a
beam/geometric decoder. This script keeps the trained model unchanged and asks a
bounded diagnostic question:

- Is the teacher first face present in top-k / within-face beam candidates?
- Does point-cloud geometric rescoring recover a better first face than greedy?
- Does conservative topology-aware rollout improve edge pairing/boundaries?

It is intentionally a probe, not a new training claim.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for item in (REPO_ROOT, SCRIPT_DIR):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from clearmesh.mesh_heads.face_paper import build_paper_face_arae
from clearmesh.mesh_heads.face_tokens import dequantize_normalized_points, quantize_normalized_points
from clearmesh.mesh_heads.face_topology import face_token_topology_report

_EVAL_SPEC = importlib.util.spec_from_file_location(
    "eval_face_paper_faithful",
    SCRIPT_DIR / "eval_face_paper_faithful.py",
)
if _EVAL_SPEC is None or _EVAL_SPEC.loader is None:  # pragma: no cover - defensive import guard.
    raise RuntimeError("could not import eval_face_paper_faithful.py")
face_eval = importlib.util.module_from_spec(_EVAL_SPEC)
sys.modules[_EVAL_SPEC.name] = face_eval
_EVAL_SPEC.loader.exec_module(face_eval)

FACE_COORD_SLOT_LABELS = ("z0", "y0", "x0", "z1", "y1", "x1", "z2", "y2", "x2")


@dataclass(frozen=True)
class FaceCandidate:
    tokens: tuple[int, ...]
    logprob: float
    rank: int = 0
    surface_rmse_bins: float | None = None
    min_vertex_l1_bins: float | None = None
    area2_bins: float | None = None
    topology_score: float | None = None
    hybrid_score: float | None = None
    teacher_l1: float | None = None
    teacher_token_accuracy: float | None = None
    teacher_vertex_exact_ratio: float | None = None
    teacher_exact: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tokens": list(self.tokens),
            "logprob": float(self.logprob),
            "rank": int(self.rank),
            "surface_rmse_bins": self.surface_rmse_bins,
            "min_vertex_l1_bins": self.min_vertex_l1_bins,
            "area2_bins": self.area2_bins,
            "topology_score": self.topology_score,
            "hybrid_score": self.hybrid_score,
            "teacher_l1": self.teacher_l1,
            "teacher_token_accuracy": self.teacher_token_accuracy,
            "teacher_vertex_exact_ratio": self.teacher_vertex_exact_ratio,
            "teacher_exact": self.teacher_exact,
        }


@dataclass
class TopologyDecodeState:
    edge_counts: Counter[tuple[tuple[int, int, int], tuple[int, int, int]]] = field(default_factory=Counter)
    seen_faces: set[tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]] = field(default_factory=set)
    seen_vertices: set[tuple[int, int, int]] = field(default_factory=set)
    accepted_faces: int = 0

    @property
    def boundary_edges(self) -> int:
        return int(sum(1 for count in self.edge_counts.values() if count == 1))

    def clone(self) -> "TopologyDecodeState":
        return TopologyDecodeState(
            edge_counts=Counter(self.edge_counts),
            seen_faces=set(self.seen_faces),
            seen_vertices=set(self.seen_vertices),
            accepted_faces=int(self.accepted_faces),
        )

    def add(self, tokens: Iterable[int]) -> None:
        face = _tokens_to_face(tokens)
        if _is_degenerate(face) or _area2(face) <= 0.0:
            return
        self.seen_faces.add(_face_key(face))
        self.seen_vertices.update(face)
        self.edge_counts.update(_face_edges(face))
        self.accepted_faces += 1


def _load_checkpoint(path: Path):  # type: ignore[no-untyped-def]
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def _load_model(checkpoint_path: Path, device):  # type: ignore[no-untyped-def]
    checkpoint = _load_checkpoint(checkpoint_path)
    train_args = checkpoint.get("args", {})
    num_bins = int(checkpoint["num_bins"])
    max_faces = int(checkpoint["max_faces"])
    decode_head = str(checkpoint.get("decode_head") or train_args.get("decode_head") or "causal")
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
        encoder_backend=str(checkpoint.get("encoder_backend") or train_args.get("encoder_backend") or "native"),
        causal_mlp_variant=str(checkpoint.get("causal_mlp_variant") or train_args.get("causal_mlp_variant") or "legacy_concat"),
        face_embedding_variant=str(checkpoint.get("face_embedding_variant") or train_args.get("face_embedding_variant") or "token_concat_project"),
        enable_eos_head=bool(checkpoint.get("has_eos_head", False)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model, checkpoint, train_args, num_bins, max_faces, decode_head


def _sample_paths(dataset_dir: Path, limit: int) -> list[Path]:
    paths = sorted(dataset_dir.glob("*.npz"))
    return paths[:limit] if limit else paths


def _point_features_to_tensor(point_features: np.ndarray, device):  # type: ignore[no-untyped-def]
    import torch

    return torch.as_tensor(point_features, dtype=torch.float32, device=device).unsqueeze(0)


def _decode_candidate_beam(
    model,
    hidden,
    *,
    num_bins: int,
    beam_width: int,
    slot_topk: int,
) -> list[FaceCandidate]:  # type: ignore[no-untyped-def]
    import torch
    import torch.nn.functional as F

    beam_width = max(1, int(beam_width))
    slot_topk = max(1, min(int(slot_topk), int(num_bins)))
    device = hidden.device
    beams: list[tuple[list[int], float]] = [([], 0.0)]
    with torch.no_grad():
        for slot in range(9):
            expanded: list[tuple[list[int], float]] = []
            for prefix_tokens, prefix_logprob in beams:
                prefix = torch.full((1, 9), -1, dtype=torch.long, device=device)
                if prefix_tokens:
                    prefix[0, : len(prefix_tokens)] = torch.as_tensor(prefix_tokens, dtype=torch.long, device=device)
                logits = model._causal_logits_from_hidden(hidden, prefix.reshape(1, 1, 9))[0, 0, slot, :num_bins]
                log_probs = F.log_softmax(logits.to(dtype=torch.float32), dim=-1)
                values, indices = torch.topk(log_probs, k=slot_topk)
                for value, index in zip(values.detach().cpu().tolist(), indices.detach().cpu().tolist()):
                    expanded.append((prefix_tokens + [int(index)], float(prefix_logprob + value)))
            expanded.sort(key=lambda item: item[1], reverse=True)
            beams = expanded[:beam_width]
    return [FaceCandidate(tokens=tuple(tokens), logprob=float(logprob), rank=rank + 1) for rank, (tokens, logprob) in enumerate(beams)]


def _tokens_to_face(tokens: Iterable[int]) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
    vals = tuple(int(value) for value in tokens)
    if len(vals) != 9:
        raise ValueError(f"FACE candidate must contain 9 tokens, got {len(vals)}")
    return (vals[0:3], vals[3:6], vals[6:9])  # type: ignore[return-value]


def _edge_key(a: tuple[int, int, int], b: tuple[int, int, int]) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    return (a, b) if a <= b else (b, a)


def _face_edges(face) -> tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...]:  # type: ignore[no-untyped-def]
    return (_edge_key(face[0], face[1]), _edge_key(face[1], face[2]), _edge_key(face[2], face[0]))


def _face_key(face) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:  # type: ignore[no-untyped-def]
    return tuple(sorted(face))  # type: ignore[return-value]


def _is_degenerate(face) -> bool:  # type: ignore[no-untyped-def]
    return len(set(face)) != 3


def _area2(face) -> float:  # type: ignore[no-untyped-def]
    q = np.asarray(face, dtype=np.float64)
    cross = np.cross(q[1] - q[0], q[2] - q[0])
    return float(np.linalg.norm(cross))


def _candidate_xyz(tokens: Iterable[int], num_bins: int) -> np.ndarray:
    q_zyx = np.asarray(list(tokens), dtype=np.int64).reshape(3, 3)
    q_xyz = q_zyx[:, [2, 1, 0]]
    return dequantize_normalized_points(q_xyz, num_bins=num_bins)


def _canonical_min_vertex_zyx(points_xyz: np.ndarray, num_bins: int) -> np.ndarray:
    q_xyz = quantize_normalized_points(np.asarray(points_xyz, dtype=np.float64), num_bins=num_bins)
    q_zyx = q_xyz[:, [2, 1, 0]]
    order = np.lexsort((q_zyx[:, 2], q_zyx[:, 1], q_zyx[:, 0]))
    return q_zyx[order[0]].astype(np.float64)


def _surface_rmse_bins(tokens: Iterable[int], points_xyz: np.ndarray, num_bins: int, max_points: int = 8192) -> float:
    verts = _candidate_xyz(tokens, num_bins=num_bins)
    support = np.concatenate(
        [
            verts,
            (verts + np.roll(verts, -1, axis=0)) * 0.5,
            verts.mean(axis=0, keepdims=True),
        ],
        axis=0,
    )
    pts = np.asarray(points_xyz, dtype=np.float64)
    if len(pts) > max_points:
        pts = pts[:max_points]
    d2 = np.sum((support[:, None, :] - pts[None, :, :]) ** 2, axis=-1)
    rmse = float(np.sqrt(np.mean(np.min(d2, axis=1))))
    return rmse * float(max(1, num_bins - 1)) * 0.5


def _min_vertex_l1_bins(tokens: Iterable[int], anchor_zyx: np.ndarray) -> float:
    face = np.asarray(list(tokens), dtype=np.float64).reshape(3, 3)
    order = np.lexsort((face[:, 2], face[:, 1], face[:, 0]))
    return float(np.mean(np.abs(face[order[0]] - anchor_zyx)))


def _teacher_metrics(tokens: Iterable[int], teacher_first: np.ndarray | None) -> dict[str, float | bool | None]:
    if teacher_first is None:
        return {"teacher_l1": None, "teacher_token_accuracy": None, "teacher_vertex_exact_ratio": None, "teacher_exact": None}
    candidate = np.asarray(list(tokens), dtype=np.int64).reshape(9)
    teacher = np.asarray(teacher_first, dtype=np.int64).reshape(9)
    matches = candidate == teacher
    vertex_matches = np.all(candidate.reshape(3, 3) == teacher.reshape(3, 3), axis=1)
    return {
        "teacher_l1": float(np.mean(np.abs(candidate - teacher))),
        "teacher_token_accuracy": float(np.mean(matches)),
        "teacher_vertex_exact_ratio": float(np.mean(vertex_matches)),
        "teacher_exact": bool(np.all(matches)),
    }


def _topology_score(
    tokens: Iterable[int],
    state: TopologyDecodeState,
    *,
    position: int,
    closure_weight: float = 1.25,
    boundary_delta_weight: float = 0.75,
    vertex_reuse_weight: float = 0.15,
    new_edge_penalty: float = 0.08,
    disconnected_penalty: float = 2.0,
    duplicate_penalty: float = 16.0,
    nonmanifold_penalty: float = 32.0,
) -> float:
    face = _tokens_to_face(tokens)
    if _is_degenerate(face) or _area2(face) <= 0.0:
        return -1.0e6
    key = _face_key(face)
    if key in state.seen_faces:
        return -duplicate_penalty
    before_boundary = state.boundary_edges
    edge_counts = Counter(state.edge_counts)
    closures = 0
    new_edges = 0
    nonmanifold = 0
    for edge in _face_edges(face):
        count = edge_counts[edge]
        if count == 0:
            new_edges += 1
        elif count == 1:
            closures += 1
        else:
            nonmanifold += 1
        edge_counts[edge] += 1
    after_boundary = int(sum(1 for count in edge_counts.values() if count == 1))
    boundary_delta = before_boundary - after_boundary
    reused_vertices = sum(1 for vertex in face if vertex in state.seen_vertices)
    disconnected = bool(position > 0 and closures == 0 and reused_vertices == 0)
    return float(
        closure_weight * closures
        + boundary_delta_weight * boundary_delta
        + vertex_reuse_weight * reused_vertices
        - new_edge_penalty * new_edges
        - disconnected_penalty * int(disconnected)
        - nonmanifold_penalty * nonmanifold
    )


def _annotate_candidates(
    candidates: list[FaceCandidate],
    *,
    points_xyz: np.ndarray,
    num_bins: int,
    teacher_first: np.ndarray | None,
    topology_state: TopologyDecodeState | None = None,
    position: int = 0,
    geometry_weight: float = 0.12,
    min_anchor_weight: float = 0.025,
    topology_weight: float = 0.25,
) -> list[FaceCandidate]:
    anchor = _canonical_min_vertex_zyx(points_xyz, num_bins=num_bins)
    state = topology_state or TopologyDecodeState()
    annotated: list[FaceCandidate] = []
    for candidate in candidates:
        face = _tokens_to_face(candidate.tokens)
        area2 = _area2(face)
        surface = _surface_rmse_bins(candidate.tokens, points_xyz, num_bins=num_bins)
        min_l1 = _min_vertex_l1_bins(candidate.tokens, anchor)
        topo = _topology_score(candidate.tokens, state, position=position)
        teacher = _teacher_metrics(candidate.tokens, teacher_first)
        area_penalty = 0.0 if area2 > 0.0 else 1.0e6
        hybrid = (
            float(candidate.logprob) / 9.0
            - geometry_weight * float(surface)
            - min_anchor_weight * float(min_l1)
            + topology_weight * float(topo)
            - area_penalty
        )
        annotated.append(
            FaceCandidate(
                tokens=candidate.tokens,
                logprob=candidate.logprob,
                rank=candidate.rank,
                surface_rmse_bins=surface,
                min_vertex_l1_bins=min_l1,
                area2_bins=area2,
                topology_score=topo,
                hybrid_score=hybrid,
                teacher_l1=teacher["teacher_l1"],
                teacher_token_accuracy=teacher["teacher_token_accuracy"],
                teacher_vertex_exact_ratio=teacher["teacher_vertex_exact_ratio"],
                teacher_exact=teacher["teacher_exact"],
            )
        )
    return annotated


def _best(candidates: list[FaceCandidate], key: str, reverse: bool = True) -> FaceCandidate:
    values = [candidate for candidate in candidates if getattr(candidate, key) is not None]
    if not values:
        raise ValueError(f"no candidates have {key}")
    return sorted(values, key=lambda candidate: float(getattr(candidate, key)), reverse=reverse)[0]


def _candidate_selection_summary(candidates: list[FaceCandidate]) -> dict[str, Any]:
    logprob = _best(candidates, "logprob", reverse=True)
    surface = _best(candidates, "surface_rmse_bins", reverse=False)
    anchor = _best(candidates, "min_vertex_l1_bins", reverse=False)
    topology = _best(candidates, "topology_score", reverse=True)
    hybrid = _best(candidates, "hybrid_score", reverse=True)
    oracle = _best(candidates, "teacher_l1", reverse=False) if any(c.teacher_l1 is not None for c in candidates) else None
    exact = [c for c in candidates if c.teacher_exact]
    return {
        "candidate_count": len(candidates),
        "teacher_exact_in_beam": bool(exact),
        "teacher_exact_best_rank": min((c.rank for c in exact), default=None),
        "best_logprob": logprob.to_dict(),
        "best_surface": surface.to_dict(),
        "best_anchor": anchor.to_dict(),
        "best_topology": topology.to_dict(),
        "best_hybrid": hybrid.to_dict(),
        "best_oracle": None if oracle is None else oracle.to_dict(),
        "top5_logprob": [c.to_dict() for c in sorted(candidates, key=lambda c: c.logprob, reverse=True)[:5]],
        "top5_hybrid": [c.to_dict() for c in sorted(candidates, key=lambda c: float(c.hybrid_score or -1.0e18), reverse=True)[:5]],
    }



def _teacher_path_metrics(model, hidden, teacher_first: np.ndarray, num_bins: int) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    import torch
    import torch.nn.functional as F

    teacher = np.asarray(teacher_first, dtype=np.int64).reshape(9)
    device = hidden.device
    prefix = torch.full((1, 9), -1, dtype=torch.long, device=device)
    ranks: list[int] = []
    probs: list[float] = []
    log_probs_out: list[float] = []
    top1: list[int] = []
    with torch.no_grad():
        for slot in range(9):
            logits = model._causal_logits_from_hidden(hidden, prefix.reshape(1, 1, 9))[0, 0, slot, :num_bins].to(dtype=torch.float32)
            target = int(teacher[slot])
            log_probs = F.log_softmax(logits, dim=-1)
            probs_t = torch.exp(log_probs)
            target_score = logits[target]
            rank = int(logits.gt(target_score).sum().detach().cpu()) + 1
            ranks.append(rank)
            probs.append(float(probs_t[target].detach().cpu()))
            log_probs_out.append(float(log_probs[target].detach().cpu()))
            top1.append(int(torch.argmax(logits).detach().cpu()))
            prefix[0, slot] = target
    return {
        "teacher_path_rank_by_slot": ranks,
        "teacher_path_rank_mean": float(np.mean(ranks)),
        "teacher_path_rank_max": int(max(ranks)),
        "teacher_path_target_prob_by_slot": probs,
        "teacher_path_target_prob_mean": float(np.mean(probs)),
        "teacher_path_logprob": float(np.sum(log_probs_out)),
        "teacher_path_top1_tokens": top1,
        "teacher_path_top1_accuracy": float(np.mean(np.asarray(top1, dtype=np.int64) == teacher)),
    }

def _first_hidden(model, point_features: np.ndarray, device):  # type: ignore[no-untyped-def]
    import torch

    point_tensor = _point_features_to_tensor(point_features, device)
    with torch.no_grad():
        cache = model.init_incremental_cache(point_tensor)
        previous_face = torch.full((1, 9), -1, dtype=torch.long, device=device)
        return model.incremental_hidden_step(previous_face, 0, cache)


def _rollout(
    model,
    point_features: np.ndarray,
    teacher_tokens: np.ndarray,
    *,
    num_bins: int,
    device,
    face_count: int,
    beam_width: int,
    slot_topk: int,
    strategy: str,
    geometry_weight: float,
    min_anchor_weight: float,
    topology_weight: float,
) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    import torch

    if strategy not in {"greedy", "hybrid_constrained", "topology_constrained", "surface_constrained"}:
        raise ValueError(f"unknown rollout strategy: {strategy}")
    point_tensor = _point_features_to_tensor(point_features, device)
    points_xyz = np.asarray(point_features[:, :3], dtype=np.float64)
    generated: list[np.ndarray] = []
    state = TopologyDecodeState()
    with torch.no_grad():
        cache = model.init_incremental_cache(point_tensor)
        previous_face = torch.full((1, 9), -1, dtype=torch.long, device=device)
        for position in range(face_count):
            hidden = model.incremental_hidden_step(previous_face, position, cache)
            if strategy == "greedy":
                next_face = model.greedy_face_from_hidden(hidden, limit_bins=num_bins).squeeze(0).detach().cpu().numpy().astype(np.int64)
            else:
                raw = _decode_candidate_beam(model, hidden, num_bins=num_bins, beam_width=beam_width, slot_topk=slot_topk)
                teacher_first = teacher_tokens[position] if position < len(teacher_tokens) else None
                annotated = _annotate_candidates(
                    raw,
                    points_xyz=points_xyz,
                    num_bins=num_bins,
                    teacher_first=teacher_first,
                    topology_state=state,
                    position=position,
                    geometry_weight=geometry_weight,
                    min_anchor_weight=min_anchor_weight if position == 0 else 0.0,
                    topology_weight=topology_weight,
                )
                if strategy == "surface_constrained":
                    chosen = _best(annotated, "surface_rmse_bins", reverse=False)
                elif strategy == "topology_constrained":
                    chosen = _best(annotated, "topology_score", reverse=True)
                else:
                    chosen = _best(annotated, "hybrid_score", reverse=True)
                next_face = np.asarray(chosen.tokens, dtype=np.int64)
            generated.append(next_face)
            state.add(next_face)
            previous_face = torch.as_tensor(next_face.reshape(1, 9), dtype=torch.long, device=device)
    generated_tokens = np.stack(generated, axis=0) if generated else np.zeros((0, 9), dtype=np.int64)
    teacher_capped = np.asarray(teacher_tokens[:face_count], dtype=np.int64)
    token_metrics = face_eval._generated_vs_teacher_metrics(generated_tokens, teacher_capped)
    topology = face_token_topology_report(generated_tokens).to_dict()
    return {
        "strategy": strategy,
        "face_count": int(len(generated_tokens)),
        "generated_token_accuracy": token_metrics.get("generated_token_accuracy"),
        "generated_vertex_exact_ratio": token_metrics.get("generated_vertex_exact_ratio"),
        "generated_edge_exact_ratio": token_metrics.get("generated_edge_exact_ratio"),
        "generated_face_exact_ratio": token_metrics.get("generated_face_exact_ratio"),
        "generated_edge_set_f1": token_metrics.get("generated_edge_set_f1"),
        "first_divergent_face_index": token_metrics.get("first_divergent_face_index"),
        "first_divergent_coord_slot": token_metrics.get("first_divergent_coord_slot"),
        "topology": topology,
    }


def _mean(values: list[float | int | None]) -> float | None:
    nums = [float(v) for v in values if v is not None]
    return float(np.mean(nums)) if nums else None


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {"attempted": 0}
    selections = [item["first_face"] for item in results]
    rollouts: dict[str, list[dict[str, Any]]] = {}
    for item in results:
        for rollout in item.get("rollouts", []):
            rollouts.setdefault(rollout["strategy"], []).append(rollout)
    return {
        "attempted": len(results),
        "teacher_exact_in_beam_rate": float(np.mean([bool(s.get("teacher_exact_in_beam")) for s in selections])),
        "teacher_exact_best_rank_mean": _mean([s.get("teacher_exact_best_rank") for s in selections]),
        "teacher_path_rank_max_mean": _mean([item.get("teacher_path", {}).get("teacher_path_rank_max") for item in results]),
        "teacher_path_rank_mean_mean": _mean([item.get("teacher_path", {}).get("teacher_path_rank_mean") for item in results]),
        "teacher_path_all_ranks_within_slot_topk_rate": float(np.mean([bool(item.get("teacher_path", {}).get("all_ranks_within_slot_topk")) for item in results])),
        "teacher_path_top1_accuracy_mean": _mean([item.get("teacher_path", {}).get("teacher_path_top1_accuracy") for item in results]),
        "best_logprob_teacher_l1_mean": _mean([s["best_logprob"].get("teacher_l1") for s in selections]),
        "best_hybrid_teacher_l1_mean": _mean([s["best_hybrid"].get("teacher_l1") for s in selections]),
        "best_surface_teacher_l1_mean": _mean([s["best_surface"].get("teacher_l1") for s in selections]),
        "best_oracle_teacher_l1_mean": _mean([(s.get("best_oracle") or {}).get("teacher_l1") for s in selections]),
        "best_logprob_token_accuracy_mean": _mean([s["best_logprob"].get("teacher_token_accuracy") for s in selections]),
        "best_hybrid_token_accuracy_mean": _mean([s["best_hybrid"].get("teacher_token_accuracy") for s in selections]),
        "rollouts": {
            name: {
                "attempted": len(rows),
                "mean_generated_token_accuracy": _mean([r.get("generated_token_accuracy") for r in rows]),
                "mean_generated_edge_set_f1": _mean([r.get("generated_edge_set_f1") for r in rows]),
                "mean_boundary_edges": _mean([r.get("topology", {}).get("boundary_edge_count") for r in rows]),
                "mean_edge_pairing_ratio": _mean([r.get("topology", {}).get("edge_pairing_ratio") for r in rows]),
                "watertight_edge_graph_rate": float(np.mean([bool(r.get("topology", {}).get("watertight_edge_graph")) for r in rows])) if rows else 0.0,
                "first_face_divergence_rate": float(np.mean([r.get("first_divergent_face_index") == 0 for r in rows])) if rows else 0.0,
            }
            for name, rows in sorted(rollouts.items())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--point-samples", type=int, default=0)
    parser.add_argument("--beam-width", type=int, default=128)
    parser.add_argument("--slot-topk", type=int, default=8)
    parser.add_argument("--rollout-faces", type=int, default=64)
    parser.add_argument("--rollout-strategies", default="greedy,hybrid_constrained,topology_constrained")
    parser.add_argument("--geometry-weight", type=float, default=0.12)
    parser.add_argument("--min-anchor-weight", type=float, default=0.025)
    parser.add_argument("--topology-weight", type=float, default=0.25)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--log-every", type=int, default=1)
    args = parser.parse_args()

    import torch

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    model, checkpoint, train_args, num_bins, max_faces, decode_head = _load_model(args.checkpoint, device)
    if decode_head != "causal":
        raise SystemExit(f"first-face probe currently supports causal decode_head only, got {decode_head!r}")
    point_samples = args.point_samples or int(train_args.get("point_samples", 0)) or None
    strategies = [item.strip() for item in args.rollout_strategies.split(",") if item.strip()]
    paths = _sample_paths(args.dataset_dir, args.limit)
    results: list[dict[str, Any]] = []
    for index, path in enumerate(paths):
        point_features, _transform, sample_bins, teacher_tokens = face_eval._load_sample(path, point_samples=point_samples)
        if int(sample_bins) != int(num_bins):
            raise ValueError(f"{path} num_bins={sample_bins}, checkpoint num_bins={num_bins}")
        teacher_tokens = np.asarray(teacher_tokens[:max_faces], dtype=np.int64)
        if len(teacher_tokens) == 0:
            continue
        hidden0 = _first_hidden(model, point_features, device)
        raw_candidates = _decode_candidate_beam(
            model,
            hidden0,
            num_bins=num_bins,
            beam_width=args.beam_width,
            slot_topk=args.slot_topk,
        )
        annotated = _annotate_candidates(
            raw_candidates,
            points_xyz=point_features[:, :3],
            num_bins=num_bins,
            teacher_first=teacher_tokens[0],
            topology_state=TopologyDecodeState(),
            position=0,
            geometry_weight=args.geometry_weight,
            min_anchor_weight=args.min_anchor_weight,
            topology_weight=args.topology_weight,
        )
        face_count = min(int(args.rollout_faces), int(len(teacher_tokens)), int(max_faces))
        rollouts = [
            _rollout(
                model,
                point_features,
                teacher_tokens,
                num_bins=num_bins,
                device=device,
                face_count=face_count,
                beam_width=min(args.beam_width, 32),
                slot_topk=min(args.slot_topk, 6),
                strategy=strategy,
                geometry_weight=args.geometry_weight,
                min_anchor_weight=args.min_anchor_weight,
                topology_weight=args.topology_weight,
            )
            for strategy in strategies
        ]
        teacher_path = _teacher_path_metrics(model, hidden0, teacher_tokens[0], num_bins)
        teacher_path["all_ranks_within_slot_topk"] = bool(max(teacher_path["teacher_path_rank_by_slot"]) <= int(args.slot_topk))
        item = {
            "index": int(index),
            "path": str(path),
            "reference_face_count": int(len(teacher_tokens)),
            "rollout_face_count": int(face_count),
            "teacher_path": teacher_path,
            "first_face": _candidate_selection_summary(annotated),
            "rollouts": rollouts,
        }
        results.append(item)
        if args.log_every > 0 and (index == 0 or (index + 1) % args.log_every == 0 or index + 1 == len(paths)):
            print(
                json.dumps(
                    {
                        "sample": index + 1,
                        "total": len(paths),
                        "path": path.name,
                        "teacher_exact_in_beam": item["first_face"]["teacher_exact_in_beam"],
                        "best_logprob_l1": item["first_face"]["best_logprob"].get("teacher_l1"),
                        "best_hybrid_l1": item["first_face"]["best_hybrid"].get("teacher_l1"),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    report = {
        "checkpoint": str(args.checkpoint),
        "dataset_dir": str(args.dataset_dir),
        "num_bins": int(num_bins),
        "max_faces": int(max_faces),
        "point_samples": None if point_samples is None else int(point_samples),
        "beam_width": int(args.beam_width),
        "slot_topk": int(args.slot_topk),
        "rollout_faces": int(args.rollout_faces),
        "geometry_weight": float(args.geometry_weight),
        "min_anchor_weight": float(args.min_anchor_weight),
        "topology_weight": float(args.topology_weight),
        "summary": _aggregate(results),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
