#!/usr/bin/env python3
"""Diagnose FACE-indexed free-run divergence against teacher topology.

This is a fast hostile diagnostic for the current FACE-indexed branch. A model
can be perfect under teacher forcing while still drifting during free-run decode;
this script stops at the first topology divergence and reports whether the
teacher next face was present/ranked in the constrained candidate set.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clearmesh.mesh_heads.face_arae import build_tiny_point_conditioned_indexed_face_decoder
from clearmesh.mesh_heads.face_indexed import IndexedDecodeState
from scripts.research.eval_face_indexed_conditioned_tiny import (  # noqa: E402
    _append_face_tensor,
    _load_checkpoint,
    _load_sample,
    _select_corner_causal_face_candidates,
)


def _face_key(face: np.ndarray | tuple[int, int, int] | list[int]) -> tuple[int, int, int]:
    return tuple(sorted(int(value) for value in np.asarray(face, dtype=np.int64).reshape(3)))


def _face_exact(face: np.ndarray | tuple[int, int, int] | list[int]) -> tuple[int, int, int]:
    return tuple(int(value) for value in np.asarray(face, dtype=np.int64).reshape(3))


def _top_faces(candidates: list[tuple[np.ndarray, float]], limit: int = 5) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for face, score in candidates[: max(0, int(limit))]:
        arr = np.asarray(face, dtype=np.int64).reshape(3)
        out.append(
            {
                "face": [int(value) for value in arr.tolist()],
                "key": [int(value) for value in _face_key(arr)],
                "score": float(score),
            }
        )
    return out


def _rank_teacher(
    candidates: list[tuple[np.ndarray, float]],
    teacher_face: np.ndarray,
) -> tuple[int | None, int | None, float | None]:
    teacher_exact = _face_exact(teacher_face)
    teacher_key = _face_key(teacher_face)
    exact_rank: int | None = None
    key_rank: int | None = None
    key_score: float | None = None
    for idx, (face, score) in enumerate(candidates):
        rank = idx + 1
        if exact_rank is None and _face_exact(face) == teacher_exact:
            exact_rank = rank
        if key_rank is None and _face_key(face) == teacher_key:
            key_rank = rank
            key_score = float(score)
        if exact_rank is not None and key_rank is not None:
            break
    return exact_rank, key_rank, key_score


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--point-samples", type=int, default=2048)
    parser.add_argument("--max-steps", type=int, default=512)
    parser.add_argument("--constraint-top-k", type=int, default=16)
    parser.add_argument("--local-candidate-neighbors", type=int, default=24)
    parser.add_argument("--closure-bonus", type=float, default=12.0)
    parser.add_argument("--new-edge-penalty", type=float, default=2.0)
    parser.add_argument("--edge-length-penalty", type=float, default=20.0)
    parser.add_argument("--aspect-penalty", type=float, default=0.25)
    parser.add_argument("--edge-action-bonus", type=float, default=2.0)
    parser.add_argument("--edge-action-candidate-top-k", type=int, default=0)
    parser.add_argument("--edge-choice-bonus", type=float, default=0.0)
    parser.add_argument("--edge-choice-candidate-top-k", type=int, default=0)
    parser.add_argument("--seed-face-bonus", type=float, default=4.0)
    parser.add_argument("--require-boundary-closure-after", type=int, default=1)
    parser.add_argument("--closure-target-bonus", type=float, default=6.0)
    parser.add_argument("--max-candidates", type=int, default=10000)
    parser.add_argument("--top-preview", type=int, default=5)
    parser.add_argument("--continue-after-divergence", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = parser.parse_args()

    import torch

    checkpoint = _load_checkpoint(args.checkpoint)
    train_args = checkpoint.get("args", {})
    num_bins = int(checkpoint["num_bins"])
    max_vertices = int(checkpoint["max_vertices"])
    max_faces = int(checkpoint["max_faces"])
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    model = build_tiny_point_conditioned_indexed_face_decoder(
        num_bins=num_bins,
        max_vertices=max_vertices,
        max_faces=max_faces,
        point_feature_dim=6,
        hidden_size=int(train_args.get("hidden_size", 192)),
        layers=int(train_args.get("layers", 4)),
        heads=int(train_args.get("heads", 6)),
        condition_tokens=int(train_args.get("condition_tokens", 8)),
        edge_head_mode=str(train_args.get("edge_head_mode", "index")),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=False)
    model.eval()
    if not bool(checkpoint.get("has_corner_causal_head")):
        raise SystemExit("checkpoint does not contain the corner-causal head required by this diagnostic")

    use_topology_head = bool(checkpoint.get("has_topology_head"))
    use_edge_choice_head = bool(checkpoint.get("has_edge_choice_head"))
    edge_choice_bonus = float(args.edge_choice_bonus) if use_edge_choice_head else 0.0
    use_seed_face_head = bool(checkpoint.get("has_seed_face_head"))
    seed_face_bonus = float(args.seed_face_bonus) if use_seed_face_head else 0.0

    paths = sorted(path for path in args.dataset_dir.glob("*.npz") if not path.name.startswith("._"))
    if args.offset:
        paths = paths[args.offset :]
    if args.limit:
        paths = paths[: args.limit]

    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    for path in paths:
        sample_started = time.perf_counter()
        point_features_np, teacher_seq = _load_sample(path, args.point_samples if args.point_samples > 0 else None)
        vertex_count = min(len(teacher_seq.vertices), max_vertices)
        face_count = min(len(teacher_seq.faces), max_faces, max(1, int(args.max_steps)))
        vertex_table_np = np.full((max_vertices, 3), -1, dtype=np.int64)
        vertex_table_np[:vertex_count] = teacher_seq.vertices[:vertex_count]
        point_tensor = torch.as_tensor(point_features_np, dtype=torch.float32, device=device).unsqueeze(0)
        vertex_tensor = torch.as_tensor(vertex_table_np, dtype=torch.long, device=device).unsqueeze(0)
        vertex_table_for_scoring = vertex_tensor.detach().cpu().numpy()[0]
        input_faces = torch.full((1, 1, 3), -1, dtype=torch.long, device=device)
        state = IndexedDecodeState.empty()
        first_divergence: dict[str, Any] | None = None
        matched_steps = 0
        trace: list[dict[str, Any]] = []
        with torch.no_grad():
            for step in range(face_count):
                hidden = model._hidden(point_tensor, vertex_tensor, input_faces)[:, -1:, :]
                closure_target_scores = None
                if use_topology_head and args.closure_target_bonus != 0.0 and hasattr(model, "topology_output"):
                    closure_target_scores = torch.log_softmax(model.topology_output(hidden)[0, -1], dim=0).detach().cpu().numpy()
                prefix0 = torch.full((1, 1, 3), -1, dtype=torch.long, device=device)
                logits0 = model._corner_causal_logits_from_hidden(hidden, prefix0)[0, 0, 0, :vertex_count].detach().cpu().numpy()
                if state.accepted_faces == 0 and seed_face_bonus != 0.0 and hasattr(model, "seed_face_logits"):
                    seed_logits = model.seed_face_logits(point_tensor, vertex_tensor)[0, :, :vertex_count].detach().cpu().numpy()
                    logits0 = logits0 + seed_face_bonus * seed_logits[0]
                candidates = _select_corner_causal_face_candidates(
                    model,
                    point_tensor,
                    vertex_tensor,
                    input_faces,
                    state,
                    vertex_count=vertex_count,
                    vertices=vertex_table_for_scoring,
                    top_k=args.constraint_top_k,
                    local_candidate_neighbors=args.local_candidate_neighbors,
                    closure_bonus=args.closure_bonus,
                    new_edge_penalty=args.new_edge_penalty,
                    edge_length_penalty=args.edge_length_penalty,
                    aspect_penalty=args.aspect_penalty,
                    edge_action_bonus=args.edge_action_bonus,
                    edge_action_candidate_top_k=args.edge_action_candidate_top_k,
                    edge_choice_bonus=edge_choice_bonus,
                    edge_choice_candidate_top_k=args.edge_choice_candidate_top_k,
                    seed_face_bonus=seed_face_bonus,
                    require_boundary_closure_after=args.require_boundary_closure_after,
                    use_topology_head=use_topology_head,
                    closure_target_bonus=args.closure_target_bonus,
                    boundary_action=True,
                    strict_manifold=True,
                    enforce_vertex_link_manifold=False,
                    max_candidates=args.max_candidates,
                )
                if not candidates:
                    first_divergence = {
                        "step": int(step),
                        "reason": "no_candidates",
                        "boundary_edges_before": int(state.boundary_edge_count),
                    }
                    break
                selected, selected_score = candidates[0]
                teacher_face = np.asarray(teacher_seq.faces[step], dtype=np.int64)
                exact_rank, key_rank, key_score = _rank_teacher(candidates, teacher_face)
                selected_key = _face_key(selected)
                teacher_key = _face_key(teacher_face)
                event = {
                    "step": int(step),
                    "boundary_edges_before": int(state.boundary_edge_count),
                    "candidate_count": int(len(candidates)),
                    "selected_face": [int(value) for value in np.asarray(selected, dtype=np.int64).reshape(3).tolist()],
                    "selected_key": [int(value) for value in selected_key],
                    "selected_score": float(selected_score),
                    "teacher_face": [int(value) for value in teacher_face.reshape(3).tolist()],
                    "teacher_key": [int(value) for value in teacher_key],
                    "teacher_exact_rank": exact_rank,
                    "teacher_key_rank": key_rank,
                    "teacher_key_score": key_score,
                    "top_candidates": _top_faces(candidates, args.top_preview),
                }
                trace.append(event)
                if selected_key != teacher_key:
                    first_divergence = {
                        **event,
                        "reason": "selected_topology_differs_from_teacher",
                    }
                    if not args.continue_after_divergence:
                        break
                else:
                    matched_steps += 1
                state.add_face(selected)
                input_faces = _append_face_tensor(input_faces, np.asarray(selected, dtype=np.int64))
        results.append(
            {
                "sample": path.name,
                "face_count": int(face_count),
                "matched_prefix_steps": int(matched_steps),
                "first_divergence": first_divergence,
                "elapsed_sec": float(time.perf_counter() - sample_started),
                "trace": trace if args.continue_after_divergence else trace[-3:],
            }
        )

    summary = {
        "attempted": int(len(results)),
        "all_prefix_matched": int(sum(1 for item in results if item["first_divergence"] is None)),
        "mean_matched_prefix_steps": float(np.mean([item["matched_prefix_steps"] for item in results])) if results else None,
        "median_matched_prefix_steps": float(np.median([item["matched_prefix_steps"] for item in results])) if results else None,
        "elapsed_sec": float(time.perf_counter() - started),
    }
    serializable_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    payload = {
        "args": serializable_args,
        "device": str(device),
        "edge_head_mode": str(getattr(model, "edge_head_mode", "index")),
        "checkpoint": str(args.checkpoint),
        "offset": int(args.offset),
        "summary": summary,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
