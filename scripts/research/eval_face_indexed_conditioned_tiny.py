#!/usr/bin/env python3
"""Evaluate FACE-lite v2 indexed triangle decoder."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clearmesh.eval.mesh_quality import evaluate_mesh, evaluate_mesh_pair
from clearmesh.mesh.cleanup import CleanupOptions, cleanup_mesh_file, split_pinched_vertices
from clearmesh.mesh_heads.face_arae import build_tiny_point_conditioned_indexed_face_decoder
from clearmesh.mesh_heads.face_indexed import (
    FaceIndexedSequence,
    IndexedDecodeState,
    canonical_indexed_face_orientations,
    coordinate_tokens_to_indexed,
    decode_indexed_face_tokens_to_mesh,
    drop_geometric_degenerate_indexed_faces,
    fill_indexed_boundary_loops,
    indexed_to_coordinate_tokens,
    select_boundary_edge_action_face,
    select_constrained_indexed_face,
    select_topology_fallback_indexed_face,
    score_indexed_face_candidate,
)
from clearmesh.mesh_heads.face_tokens import FaceTokenTransform
from clearmesh.mesh_heads.face_topology import face_token_topology_report, repair_face_tokens


def _load_checkpoint(path: Path):  # type: ignore[no-untyped-def]
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        reason = str(exc).splitlines()[0]
        print(f"Falling back to trusted checkpoint load for legacy metadata: {reason}", file=sys.stderr)
        return torch.load(path, map_location="cpu", weights_only=False)


def _load_sample(path: Path, point_samples: int | None = None) -> tuple[np.ndarray, FaceIndexedSequence]:
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
        center=tuple(float(x) for x in np.asarray(data["center"]).reshape(3)),
        scale=float(np.asarray(data["scale"]).reshape(-1)[0]),
    )
    sequence = FaceIndexedSequence(
        vertices=np.asarray(data["indexed_vertices"], dtype=np.int64),
        faces=np.asarray(data["indexed_faces"], dtype=np.int64),
        num_bins=int(np.asarray(data["num_bins"]).reshape(-1)[0]),
        transform=transform,
    )
    return np.concatenate([points, normals], axis=1).astype(np.float32), sequence


def _clone_decode_state(state: IndexedDecodeState) -> IndexedDecodeState:
    return IndexedDecodeState(
        edge_counts=Counter(state.edge_counts),
        seen_faces=set(state.seen_faces),
        accepted_faces=int(state.accepted_faces),
        vertex_links={int(vertex): list(links) for vertex, links in state.vertex_links.items()},
    )


def _append_face_tensor(input_faces, face: np.ndarray):  # type: ignore[no-untyped-def]
    import torch

    next_face = torch.as_tensor(face, dtype=torch.long, device=input_faces.device)
    return torch.cat([input_faces, next_face.reshape(1, 1, 3)], dim=1)


def _validate_or_fallback_face(
    face: np.ndarray,
    state: IndexedDecodeState,
    *,
    vertex_count: int,
    target_face_count: int | None,
    enforce_vertex_link_manifold: bool,
) -> tuple[np.ndarray | None, bool]:
    face_arr = np.asarray(face, dtype=np.int64).reshape(3)
    if np.any(face_arr < 0) or np.any(face_arr >= int(vertex_count)):
        score = None
    else:
        score = score_indexed_face_candidate(
            tuple(int(value) for value in face_arr),
            0.0,
            state,
            target_face_count=target_face_count,
            enforce_vertex_link_manifold=enforce_vertex_link_manifold,
            strict_manifold=target_face_count is not None,
        )
    if score is not None:
        return face_arr, False
    fallback = select_topology_fallback_indexed_face(
        state,
        vertex_count=vertex_count,
        target_face_count=target_face_count,
        enforce_vertex_link_manifold=enforce_vertex_link_manifold,
    )
    return fallback, fallback is not None


def _generate_faces(
    model,
    point_features,
    vertex_table,
    *,
    face_count: int,
    vertex_count: int,
    device,
    decode_mode: str = "edge_constrained",
    constraint_top_k: int = 12,
    local_candidate_neighbors: int = 0,
    closure_bonus: float = 2.0,
    new_edge_penalty: float = 0.15,
    edge_length_penalty: float = 0.0,
    aspect_penalty: float = 0.0,
    edge_action_bonus: float = 0.0,
    edge_action_candidate_top_k: int = 0,
    edge_choice_bonus: float = 0.0,
    edge_choice_candidate_top_k: int = 0,
    seed_face_bonus: float = 0.0,
    require_boundary_closure_after: int = 0,
    closure_target_bonus: float = 0.0,
    use_topology_head: bool = False,
    use_corner_causal: bool = False,
    enforce_vertex_link_manifold: bool = False,
    boundary_budget_constraint: bool = False,
    seed_faces: np.ndarray | None = None,
    beam_width: int = 1,
    beam_candidates: int = 4,
    decode_stats: dict[str, Any] | None = None,
):  # type: ignore[no-untyped-def]
    import torch

    input_faces = torch.full((1, 1, 3), -1, dtype=torch.long, device=device)
    state = IndexedDecodeState.empty()
    generated = []
    if decode_stats is not None:
        decode_stats.setdefault("topology_fallbacks", 0)
        decode_stats.setdefault("topology_stop_early", 0)
        decode_stats.setdefault("topology_requested_faces", int(face_count))
    target_face_count = int(face_count) if boundary_budget_constraint else None
    vertex_table_np = vertex_table.detach().cpu().numpy()[0] if hasattr(vertex_table, "detach") else None
    if seed_faces is not None:
        for seed_face in np.asarray(seed_faces, dtype=np.int64).reshape(-1, 3):
            if len(generated) >= face_count:
                break
            if np.any(seed_face < 0) or np.any(seed_face >= vertex_count) or len(set(int(v) for v in seed_face)) != 3:
                continue
            state.add_face(seed_face)
            generated.append(seed_face.copy())
            input_faces = _append_face_tensor(input_faces, seed_face)
    if int(beam_width) > 1 and use_corner_causal and hasattr(model, "_corner_causal_logits_from_hidden"):
        return _generate_faces_beam(
            model,
            point_features,
            vertex_table,
            input_faces,
            state,
            generated,
            face_count=face_count,
            vertex_count=vertex_count,
            device=device,
            decode_mode=decode_mode,
            constraint_top_k=constraint_top_k,
            local_candidate_neighbors=local_candidate_neighbors,
            closure_bonus=closure_bonus,
            new_edge_penalty=new_edge_penalty,
                edge_length_penalty=edge_length_penalty,
                aspect_penalty=aspect_penalty,
                edge_action_bonus=edge_action_bonus,
                edge_action_candidate_top_k=edge_action_candidate_top_k,
                edge_choice_bonus=edge_choice_bonus,
                edge_choice_candidate_top_k=edge_choice_candidate_top_k,
            seed_face_bonus=seed_face_bonus,
            require_boundary_closure_after=require_boundary_closure_after,
            closure_target_bonus=closure_target_bonus,
            use_topology_head=use_topology_head,
            enforce_vertex_link_manifold=enforce_vertex_link_manifold,
            target_face_count=target_face_count,
            beam_width=beam_width,
            beam_candidates=beam_candidates,
            decode_stats=decode_stats,
        )
    with torch.no_grad():
        for _ in range(max(0, face_count - len(generated))):
            if use_corner_causal and hasattr(model, "_corner_causal_logits_from_hidden"):
                next_face_np = _select_corner_causal_face(
                    model,
                    point_features,
                    vertex_table,
                    input_faces,
                    state,
                    vertex_count=vertex_count,
                    vertices=vertex_table_np,
                    top_k=constraint_top_k,
                    local_candidate_neighbors=local_candidate_neighbors,
                    closure_bonus=closure_bonus,
                    new_edge_penalty=new_edge_penalty,
                    edge_length_penalty=edge_length_penalty,
                    aspect_penalty=aspect_penalty,
                    edge_action_bonus=edge_action_bonus,
                    edge_action_candidate_top_k=edge_action_candidate_top_k,
                    edge_choice_bonus=edge_choice_bonus,
                    edge_choice_candidate_top_k=edge_choice_candidate_top_k,
                    seed_face_bonus=seed_face_bonus,
                    require_boundary_closure_after=require_boundary_closure_after if decode_mode in {"edge_constrained", "boundary_edge"} else 0,
                    use_topology_head=use_topology_head,
                    closure_target_bonus=closure_target_bonus,
                    boundary_action=decode_mode == "boundary_edge",
                    strict_manifold=decode_mode in {"edge_constrained", "boundary_edge"},
                    enforce_vertex_link_manifold=enforce_vertex_link_manifold,
                    target_face_count=target_face_count,
                )
                next_face = torch.as_tensor(next_face_np, dtype=torch.long, device=device)
            else:
                closure_target_scores = None
                if use_topology_head and closure_target_bonus != 0.0 and hasattr(model, "forward_with_topology"):
                    outputs = model.forward_with_topology(point_features, vertex_table, input_faces)
                    logits = outputs["face_logits"][:, -1, :, :]
                    closure_target_scores = torch.log_softmax(outputs["closure_logits"][0, -1], dim=0).detach().cpu().numpy()
                else:
                    logits = model(point_features, vertex_table, input_faces)[:, -1, :, :]
                if (
                    state.accepted_faces == 0
                    and seed_face_bonus != 0.0
                    and hasattr(model, "seed_face_logits")
                ):
                    seed_logits = model.seed_face_logits(point_features, vertex_table)[0, :, :vertex_count]
                    logits = logits.clone()
                    logits[0, :, :vertex_count] = logits[0, :, :vertex_count] + float(seed_face_bonus) * seed_logits
                if decode_mode == "boundary_edge":
                    next_face_np = select_boundary_edge_action_face(
                        logits[0].detach().cpu().numpy(),
                        state,
                        vertex_count=vertex_count,
                        vertices=vertex_table_np,
                        top_k=constraint_top_k,
                        local_candidate_neighbors=local_candidate_neighbors,
                        closure_bonus=closure_bonus,
                        new_edge_penalty=new_edge_penalty,
                        edge_length_penalty=edge_length_penalty,
                        aspect_penalty=aspect_penalty,
                        require_boundary_closure_after=require_boundary_closure_after,
                        closure_target_scores=closure_target_scores,
                        closure_target_bonus=closure_target_bonus,
                        enforce_vertex_link_manifold=enforce_vertex_link_manifold,
                        target_face_count=target_face_count,
                    )
                    next_face = torch.as_tensor(next_face_np, dtype=torch.long, device=device)
                elif decode_mode == "edge_constrained":
                    next_face_np = select_constrained_indexed_face(
                        logits[0].detach().cpu().numpy(),
                        state,
                    vertex_count=vertex_count,
                    vertices=vertex_table_np,
                    top_k=constraint_top_k,
                    local_candidate_neighbors=local_candidate_neighbors,
                    closure_bonus=closure_bonus,
                        new_edge_penalty=new_edge_penalty,
                        edge_length_penalty=edge_length_penalty,
                        aspect_penalty=aspect_penalty,
                        require_boundary_closure_after=require_boundary_closure_after,
                        closure_target_scores=closure_target_scores,
                        closure_target_bonus=closure_target_bonus,
                        enforce_vertex_link_manifold=enforce_vertex_link_manifold,
                        target_face_count=target_face_count,
                    )
                    next_face = torch.as_tensor(next_face_np, dtype=torch.long, device=device)
                else:
                    next_face = logits.argmax(dim=-1)[0]
                    # Avoid degenerate index triples at decode time. This is conservative:
                    # keep model order but replace duplicates with highest unused logits.
                    repaired = []
                    used = set()
                    for corner in range(3):
                        ranking = torch.argsort(logits[0, corner], descending=True).tolist()
                        chosen = int(next((idx for idx in ranking if idx not in used), int(next_face[corner].item())))
                        repaired.append(chosen)
                        used.add(chosen)
                    next_face = torch.as_tensor(repaired, dtype=torch.long, device=device)
            next_face_np, used_fallback = _validate_or_fallback_face(
                next_face.detach().cpu().numpy(),
                state,
                vertex_count=vertex_count,
                target_face_count=target_face_count,
                enforce_vertex_link_manifold=enforce_vertex_link_manifold,
            )
            if next_face_np is None:
                if decode_stats is not None:
                    decode_stats["topology_stop_early"] = int(decode_stats.get("topology_stop_early", 0)) + 1
                break
            if used_fallback and decode_stats is not None:
                decode_stats["topology_fallbacks"] = int(decode_stats.get("topology_fallbacks", 0)) + 1
            state.add_face(next_face_np)
            generated.append(next_face_np)
            input_faces = _append_face_tensor(input_faces, next_face_np)
    if decode_stats is not None:
        decode_stats["topology_generated_faces"] = int(len(generated))
    return np.asarray(generated, dtype=np.int64)


def _generate_faces_beam(
    model,
    point_features,
    vertex_table,
    input_faces,
    state: IndexedDecodeState,
    generated: list[np.ndarray],
    *,
    face_count: int,
    vertex_count: int,
    device,
    decode_mode: str,
    constraint_top_k: int,
    local_candidate_neighbors: int,
    closure_bonus: float,
    new_edge_penalty: float,
    edge_length_penalty: float,
    aspect_penalty: float,
    edge_action_bonus: float,
    edge_action_candidate_top_k: int,
    edge_choice_bonus: float,
    edge_choice_candidate_top_k: int,
    seed_face_bonus: float,
    require_boundary_closure_after: int,
    closure_target_bonus: float,
    use_topology_head: bool,
    enforce_vertex_link_manifold: bool,
    target_face_count: int | None,
    beam_width: int,
    beam_candidates: int,
    decode_stats: dict[str, Any] | None = None,
):  # type: ignore[no-untyped-def]
    import torch

    vertex_table_np = vertex_table.detach().cpu().numpy()[0] if hasattr(vertex_table, "detach") else None
    beams: list[tuple[float, IndexedDecodeState, list[np.ndarray], Any]] = [
        (0.0, _clone_decode_state(state), [np.asarray(face, dtype=np.int64).copy() for face in generated], input_faces)
    ]
    beam_width = max(1, int(beam_width))
    beam_candidates = max(1, int(beam_candidates))
    with torch.no_grad():
        for _ in range(max(0, face_count - len(generated))):
            expanded: list[tuple[float, IndexedDecodeState, list[np.ndarray], Any]] = []
            for beam_score, beam_state, beam_generated, beam_input in beams:
                candidates = _select_corner_causal_face_candidates(
                    model,
                    point_features,
                    vertex_table,
                    beam_input,
                    beam_state,
                    vertex_count=vertex_count,
                    vertices=vertex_table_np,
                    top_k=constraint_top_k,
                    local_candidate_neighbors=local_candidate_neighbors,
                    closure_bonus=closure_bonus,
                    new_edge_penalty=new_edge_penalty,
                    edge_length_penalty=edge_length_penalty,
                    aspect_penalty=aspect_penalty,
                    edge_action_bonus=edge_action_bonus,
                    edge_action_candidate_top_k=edge_action_candidate_top_k,
                    edge_choice_bonus=edge_choice_bonus,
                    edge_choice_candidate_top_k=edge_choice_candidate_top_k,
                    seed_face_bonus=seed_face_bonus,
                    require_boundary_closure_after=require_boundary_closure_after if decode_mode in {"edge_constrained", "boundary_edge"} else 0,
                    use_topology_head=use_topology_head,
                    closure_target_bonus=closure_target_bonus,
                    boundary_action=decode_mode == "boundary_edge",
                    strict_manifold=decode_mode in {"edge_constrained", "boundary_edge"},
                    enforce_vertex_link_manifold=enforce_vertex_link_manifold,
                    target_face_count=target_face_count,
                    max_candidates=beam_candidates,
                )
                if not candidates:
                    fallback = _select_corner_causal_face(
                        model,
                        point_features,
                        vertex_table,
                        beam_input,
                        beam_state,
                        vertex_count=vertex_count,
                        vertices=vertex_table_np,
                        top_k=constraint_top_k,
                        local_candidate_neighbors=local_candidate_neighbors,
                        closure_bonus=closure_bonus,
                        new_edge_penalty=new_edge_penalty,
                        edge_length_penalty=edge_length_penalty,
                        aspect_penalty=aspect_penalty,
                        edge_action_bonus=edge_action_bonus,
                        edge_action_candidate_top_k=edge_action_candidate_top_k,
                        edge_choice_bonus=edge_choice_bonus,
                        edge_choice_candidate_top_k=edge_choice_candidate_top_k,
                        seed_face_bonus=seed_face_bonus,
                        require_boundary_closure_after=require_boundary_closure_after if decode_mode in {"edge_constrained", "boundary_edge"} else 0,
                        use_topology_head=use_topology_head,
                        closure_target_bonus=closure_target_bonus,
                        boundary_action=decode_mode == "boundary_edge",
                        strict_manifold=decode_mode in {"edge_constrained", "boundary_edge"},
                        enforce_vertex_link_manifold=enforce_vertex_link_manifold,
                        target_face_count=target_face_count,
                    )
                    candidates = [(np.asarray(fallback, dtype=np.int64), 0.0)]
                for face, face_score in candidates[:beam_candidates]:
                    face_arr = np.asarray(face, dtype=np.int64).reshape(3)
                    face_arr, used_fallback = _validate_or_fallback_face(
                        face_arr,
                        beam_state,
                        vertex_count=vertex_count,
                        target_face_count=target_face_count,
                        enforce_vertex_link_manifold=enforce_vertex_link_manifold,
                    )
                    if face_arr is None:
                        if decode_stats is not None:
                            decode_stats["topology_stop_early"] = int(decode_stats.get("topology_stop_early", 0)) + 1
                        continue
                    if used_fallback and decode_stats is not None:
                        decode_stats["topology_fallbacks"] = int(decode_stats.get("topology_fallbacks", 0)) + 1
                    next_state = _clone_decode_state(beam_state)
                    next_state.add_face(face_arr)
                    next_generated = beam_generated + [face_arr.copy()]
                    next_input = _append_face_tensor(beam_input, face_arr)
                    expanded.append((float(beam_score) + float(face_score), next_state, next_generated, next_input))
            if not expanded:
                break
            expanded.sort(key=lambda item: (len(item[2]), item[0], -item[1].boundary_edge_count), reverse=True)
            beams = expanded[:beam_width]
    if not beams:
        return np.asarray(generated, dtype=np.int64)
    best = max(beams, key=lambda item: (len(item[2]), item[0], -item[1].boundary_edge_count))
    if decode_stats is not None:
        decode_stats["topology_generated_faces"] = int(len(best[2]))
    return np.asarray(best[2], dtype=np.int64)


def _teacher_forced_faces(
    model,
    point_features,
    vertex_table,
    teacher_faces: np.ndarray,
    *,
    vertex_count: int,
    device,
    use_corner_causal: bool = False,
):  # type: ignore[no-untyped-def]
    """Decode all face positions in one teacher-forced pass.

    This is not a replacement for free-running AR. It is a fast circuit test for
    target/token learning before spending time on sequential decoding.
    """

    import torch

    target = np.asarray(teacher_faces, dtype=np.int64).reshape(-1, 3)
    face_count = int(len(target))
    if face_count == 0:
        return np.zeros((0, 3), dtype=np.int64), {
            "token_accuracy": None,
            "face_exact_ratio": None,
        }
    input_faces = torch.full((1, face_count, 3), -1, dtype=torch.long, device=device)
    target_tensor = torch.as_tensor(target, dtype=torch.long, device=device).reshape(1, face_count, 3)
    if face_count > 1:
        input_faces[:, 1:, :] = target_tensor[:, :-1, :]
    with torch.no_grad():
        if use_corner_causal and hasattr(model, "forward_corner_causal"):
            logits = model.forward_corner_causal(point_features, vertex_table, input_faces, target_tensor)
        else:
            logits = model(point_features, vertex_table, input_faces)
        pred = logits[:, :, :, :vertex_count].argmax(dim=-1)[0].detach().cpu().numpy().astype(np.int64)
    valid = np.all((target >= 0) & (target < vertex_count), axis=1)
    if np.any(valid):
        token_accuracy = float(np.mean(pred[valid] == target[valid]))
        face_exact_ratio = float(np.mean(np.all(pred[valid] == target[valid], axis=1)))
    else:
        token_accuracy = None
        face_exact_ratio = None
    return pred, {
        "token_accuracy": token_accuracy,
        "face_exact_ratio": face_exact_ratio,
    }


def _teacher_identity_faces(teacher_faces: np.ndarray) -> tuple[np.ndarray, dict[str, float | None]]:
    """Return the target faces unchanged for an evaluator circuit test.

    This mode is deliberately not a model-quality metric. It sends the exact
    FACE-Q target sequence through the same post-processing, mesh decode, and
    mesh-quality checks used by model outputs. If this path is not watertight,
    a low teacher-forced watertight rate is caused by target/eval/tokenization
    issues rather than by the network.
    """

    target = np.asarray(teacher_faces, dtype=np.int64).reshape(-1, 3)
    if len(target) == 0:
        return target.copy(), {
            "token_accuracy": None,
            "face_exact_ratio": None,
        }
    return target.copy(), {
        "token_accuracy": 1.0,
        "face_exact_ratio": 1.0,
    }


def _prefilter_boundary_edge_rows(
    boundary_edges: list[tuple[int, int]],
    logits0: np.ndarray,
    *,
    edge_choice_by_boundary: np.ndarray | None,
    edge_choice_bonus: float,
    edge_limit: int,
    edge_action_candidate_top_k: int,
    edge_choice_candidate_top_k: int,
) -> list[int]:
    """Cheaply prune boundary rows before expensive per-edge action logits."""

    prefilter_limit = min(
        len(boundary_edges),
        max(
            int(edge_limit) * 8,
            int(edge_action_candidate_top_k) * 4,
            int(edge_choice_candidate_top_k) * 4,
            32,
        ),
    )
    rows = sorted(
        range(len(boundary_edges)),
        key=lambda row: (
            max(float(logits0[int(boundary_edges[row][0])]), float(logits0[int(boundary_edges[row][1])]))
            + (float(edge_choice_bonus) * float(edge_choice_by_boundary[row]) if edge_choice_by_boundary is not None else 0.0)
        ),
        reverse=True,
    )[:prefilter_limit]
    if edge_choice_by_boundary is not None and edge_choice_candidate_top_k > 0:
        for row in np.argsort(edge_choice_by_boundary)[-edge_choice_candidate_top_k:][::-1].tolist():
            if int(row) not in rows:
                rows.append(int(row))
    return rows


def _select_corner_causal_face(
    model,
    point_features,
    vertex_table,
    input_faces,
    state: IndexedDecodeState,
    *,
    vertex_count: int,
    vertices: np.ndarray | None,
    top_k: int,
    local_candidate_neighbors: int,
    closure_bonus: float,
    new_edge_penalty: float,
    edge_length_penalty: float,
    aspect_penalty: float,
    edge_action_bonus: float,
    edge_action_candidate_top_k: int,
    edge_choice_bonus: float,
    edge_choice_candidate_top_k: int,
    seed_face_bonus: float,
    require_boundary_closure_after: int,
    use_topology_head: bool,
    closure_target_bonus: float,
    boundary_action: bool,
    strict_manifold: bool,
    enforce_vertex_link_manifold: bool,
    target_face_count: int | None,
):  # type: ignore[no-untyped-def]
    import torch

    vertex_count = max(3, int(vertex_count))
    top_k = max(3, min(int(top_k), vertex_count))
    hidden = model._hidden(point_features, vertex_table, input_faces)[:, -1:, :]
    device = input_faces.device
    closure_target_scores = None
    if use_topology_head and closure_target_bonus != 0.0 and hasattr(model, "topology_output"):
        closure_target_scores = torch.log_softmax(model.topology_output(hidden)[0, -1], dim=0).detach().cpu().numpy()
    prefix0 = torch.full((1, 1, 3), -1, dtype=torch.long, device=device)
    logits0 = model._corner_causal_logits_from_hidden(hidden, prefix0, vertex_table=vertex_table)[
        0, 0, 0, :vertex_count
    ].detach().cpu().numpy()
    seed_logits: np.ndarray | None = None
    if state.accepted_faces == 0 and seed_face_bonus != 0.0 and hasattr(model, "seed_face_logits"):
        seed_logits = model.seed_face_logits(point_features, vertex_table)[0, :, :vertex_count].detach().cpu().numpy()
        logits0 = logits0 + float(seed_face_bonus) * seed_logits[0]
    if boundary_action:
        boundary_face = _select_corner_causal_boundary_face(
            model,
            hidden,
            vertex_table,
            logits0,
            device,
            state,
            vertex_count=vertex_count,
            vertices=vertices,
            top_k=top_k,
            local_candidate_neighbors=local_candidate_neighbors,
            closure_bonus=closure_bonus,
            new_edge_penalty=new_edge_penalty,
            edge_length_penalty=edge_length_penalty,
            aspect_penalty=aspect_penalty,
            edge_action_bonus=edge_action_bonus,
            edge_action_candidate_top_k=edge_action_candidate_top_k,
            edge_choice_bonus=edge_choice_bonus,
            edge_choice_candidate_top_k=edge_choice_candidate_top_k,
            require_boundary_closure_after=require_boundary_closure_after,
            closure_target_scores=closure_target_scores,
            closure_target_bonus=closure_target_bonus,
            enforce_vertex_link_manifold=enforce_vertex_link_manifold,
            target_face_count=target_face_count,
        )
        if boundary_face is not None:
            return boundary_face

    top0 = np.argsort(logits0)[-top_k:][::-1].tolist()

    prefix1 = np.full((len(top0), 1, 3), -1, dtype=np.int64)
    for row, a in enumerate(top0):
        prefix1[row, 0, 0] = int(a)
    logits1 = model._corner_causal_logits_from_hidden(
        hidden.expand(len(top0), -1, -1),
        torch.as_tensor(prefix1, dtype=torch.long, device=device),
        vertex_table=vertex_table,
    )[:, 0, 1, :vertex_count].detach().cpu().numpy()
    if seed_logits is not None:
        logits1 = logits1 + float(seed_face_bonus) * seed_logits[1].reshape(1, -1)

    pairs: list[tuple[int, int, float]] = []
    for row, a in enumerate(top0):
        for b in np.argsort(logits1[row])[-top_k:][::-1].tolist():
            if int(b) == int(a):
                continue
            pairs.append((int(a), int(b), float(logits0[a] + logits1[row, b])))
    if not pairs:
        fallback = select_topology_fallback_indexed_face(
            state,
            vertex_count=vertex_count,
            target_face_count=target_face_count,
            enforce_vertex_link_manifold=enforce_vertex_link_manifold,
        )
        return fallback if fallback is not None else np.asarray([0, 1, 2], dtype=np.int64)

    prefix2 = np.full((len(pairs), 1, 3), -1, dtype=np.int64)
    for row, (a, b, _) in enumerate(pairs):
        prefix2[row, 0, 0] = a
        prefix2[row, 0, 1] = b
    logits2 = model._corner_causal_logits_from_hidden(
        hidden.expand(len(pairs), -1, -1),
        torch.as_tensor(prefix2, dtype=torch.long, device=device),
        vertex_table=vertex_table,
    )[:, 0, 2, :vertex_count].detach().cpu().numpy()
    if seed_logits is not None:
        logits2 = logits2 + float(seed_face_bonus) * seed_logits[2].reshape(1, -1)

    best_face: tuple[int, int, int] | None = None
    best_score = float("-inf")
    relaxed_best: tuple[int, int, int] | None = None
    relaxed_score = float("-inf")
    for row, (a, b, prefix_score) in enumerate(pairs):
        for c in np.argsort(logits2[row])[-top_k:][::-1].tolist():
            face = (int(a), int(b), int(c))
            model_score = float(prefix_score + logits2[row, c])
            score = score_indexed_face_candidate(
                face,
                model_score,
                state,
                closure_bonus=closure_bonus,
                new_edge_penalty=new_edge_penalty,
                edge_length_penalty=edge_length_penalty,
                aspect_penalty=aspect_penalty,
                require_boundary_closure_after=require_boundary_closure_after,
                closure_target_scores=closure_target_scores,
                closure_target_bonus=closure_target_bonus,
                enforce_vertex_link_manifold=enforce_vertex_link_manifold,
                target_face_count=target_face_count,
                vertices=vertices,
                strict_manifold=strict_manifold,
            )
            if score is not None and score > best_score:
                best_score = score
                best_face = face
            relaxed = score_indexed_face_candidate(
                face,
                model_score,
                state,
                closure_bonus=closure_bonus,
                new_edge_penalty=new_edge_penalty,
                edge_length_penalty=edge_length_penalty,
                aspect_penalty=aspect_penalty,
                require_boundary_closure_after=0,
                closure_target_scores=closure_target_scores,
                closure_target_bonus=closure_target_bonus,
                enforce_vertex_link_manifold=enforce_vertex_link_manifold,
                target_face_count=target_face_count,
                vertices=vertices,
                strict_manifold=False,
            )
            if relaxed is not None and relaxed > relaxed_score:
                relaxed_score = relaxed
                relaxed_best = face
    if best_face is not None:
        return np.asarray(best_face, dtype=np.int64)
    if relaxed_best is not None:
        return np.asarray(relaxed_best, dtype=np.int64)
    for row, (a, b, _) in enumerate(pairs):
        for c in np.argsort(logits2[row])[-top_k:][::-1].tolist():
            face = (int(a), int(b), int(c))
            if len(set(face)) == 3 and score_indexed_face_candidate(
                face,
                0.0,
                state,
                require_boundary_closure_after=0,
                enforce_vertex_link_manifold=enforce_vertex_link_manifold,
                target_face_count=target_face_count,
                strict_manifold=False,
            ) is not None:
                return np.asarray([a, b, int(c)], dtype=np.int64)
    fallback = select_topology_fallback_indexed_face(
        state,
        vertex_count=vertex_count,
        target_face_count=target_face_count,
        enforce_vertex_link_manifold=enforce_vertex_link_manifold,
    )
    return fallback if fallback is not None else np.asarray([0, 1, 2], dtype=np.int64)


def _select_corner_causal_face_candidates(
    model,
    point_features,
    vertex_table,
    input_faces,
    state: IndexedDecodeState,
    *,
    vertex_count: int,
    vertices: np.ndarray | None,
    top_k: int,
    local_candidate_neighbors: int,
    closure_bonus: float,
    new_edge_penalty: float,
    edge_length_penalty: float,
    aspect_penalty: float,
    edge_action_bonus: float,
    edge_action_candidate_top_k: int,
    edge_choice_bonus: float,
    edge_choice_candidate_top_k: int,
    seed_face_bonus: float,
    require_boundary_closure_after: int,
    use_topology_head: bool,
    closure_target_bonus: float,
    boundary_action: bool,
    strict_manifold: bool,
    enforce_vertex_link_manifold: bool,
    target_face_count: int | None,
    max_candidates: int,
) -> list[tuple[np.ndarray, float]]:  # type: ignore[no-untyped-def]
    import torch

    vertex_count = max(3, int(vertex_count))
    top_k = max(3, min(int(top_k), vertex_count))
    max_candidates = max(1, int(max_candidates))
    hidden = model._hidden(point_features, vertex_table, input_faces)[:, -1:, :]
    device = input_faces.device
    closure_target_scores = None
    if use_topology_head and closure_target_bonus != 0.0 and hasattr(model, "topology_output"):
        closure_target_scores = torch.log_softmax(model.topology_output(hidden)[0, -1], dim=0).detach().cpu().numpy()
    prefix0 = torch.full((1, 1, 3), -1, dtype=torch.long, device=device)
    logits0 = model._corner_causal_logits_from_hidden(hidden, prefix0, vertex_table=vertex_table)[
        0, 0, 0, :vertex_count
    ].detach().cpu().numpy()
    seed_logits: np.ndarray | None = None
    if state.accepted_faces == 0 and seed_face_bonus != 0.0 and hasattr(model, "seed_face_logits"):
        seed_logits = model.seed_face_logits(point_features, vertex_table)[0, :, :vertex_count].detach().cpu().numpy()
        logits0 = logits0 + float(seed_face_bonus) * seed_logits[0]

    if boundary_action:
        boundary_candidates = _select_corner_causal_boundary_face_candidates(
            model,
            hidden,
            vertex_table,
            logits0,
            device,
            state,
            vertex_count=vertex_count,
            vertices=vertices,
            top_k=top_k,
            local_candidate_neighbors=local_candidate_neighbors,
            closure_bonus=closure_bonus,
            new_edge_penalty=new_edge_penalty,
            edge_length_penalty=edge_length_penalty,
            aspect_penalty=aspect_penalty,
            edge_action_bonus=edge_action_bonus,
            edge_action_candidate_top_k=edge_action_candidate_top_k,
            edge_choice_bonus=edge_choice_bonus,
            edge_choice_candidate_top_k=edge_choice_candidate_top_k,
            require_boundary_closure_after=require_boundary_closure_after,
            closure_target_scores=closure_target_scores,
            closure_target_bonus=closure_target_bonus,
            enforce_vertex_link_manifold=enforce_vertex_link_manifold,
            target_face_count=target_face_count,
            max_candidates=max_candidates,
        )
        if boundary_candidates:
            return boundary_candidates

    top0 = np.argsort(logits0)[-top_k:][::-1].tolist()
    prefix1 = np.full((len(top0), 1, 3), -1, dtype=np.int64)
    for row, a in enumerate(top0):
        prefix1[row, 0, 0] = int(a)
    logits1 = model._corner_causal_logits_from_hidden(
        hidden.expand(len(top0), -1, -1),
        torch.as_tensor(prefix1, dtype=torch.long, device=device),
        vertex_table=vertex_table,
    )[:, 0, 1, :vertex_count].detach().cpu().numpy()
    if seed_logits is not None:
        logits1 = logits1 + float(seed_face_bonus) * seed_logits[1].reshape(1, -1)

    pairs: list[tuple[int, int, float]] = []
    for row, a in enumerate(top0):
        for b in np.argsort(logits1[row])[-top_k:][::-1].tolist():
            if int(b) == int(a):
                continue
            pairs.append((int(a), int(b), float(logits0[a] + logits1[row, b])))
    if not pairs:
        fallback = select_topology_fallback_indexed_face(
            state,
            vertex_count=vertex_count,
            target_face_count=target_face_count,
            enforce_vertex_link_manifold=enforce_vertex_link_manifold,
        )
        return [(fallback, 0.0)] if fallback is not None else [(np.asarray([0, 1, 2], dtype=np.int64), 0.0)]

    prefix2 = np.full((len(pairs), 1, 3), -1, dtype=np.int64)
    for row, (a, b, _) in enumerate(pairs):
        prefix2[row, 0, 0] = a
        prefix2[row, 0, 1] = b
    logits2 = model._corner_causal_logits_from_hidden(
        hidden.expand(len(pairs), -1, -1),
        torch.as_tensor(prefix2, dtype=torch.long, device=device),
        vertex_table=vertex_table,
    )[:, 0, 2, :vertex_count].detach().cpu().numpy()
    if seed_logits is not None:
        logits2 = logits2 + float(seed_face_bonus) * seed_logits[2].reshape(1, -1)

    strict_scores: list[tuple[tuple[int, int, int], float]] = []
    relaxed_scores: list[tuple[tuple[int, int, int], float]] = []
    for row, (a, b, prefix_score) in enumerate(pairs):
        for c in np.argsort(logits2[row])[-top_k:][::-1].tolist():
            face = (int(a), int(b), int(c))
            model_score = float(prefix_score + logits2[row, c])
            score = score_indexed_face_candidate(
                face,
                model_score,
                state,
                closure_bonus=closure_bonus,
                new_edge_penalty=new_edge_penalty,
                edge_length_penalty=edge_length_penalty,
                aspect_penalty=aspect_penalty,
                require_boundary_closure_after=require_boundary_closure_after,
                closure_target_scores=closure_target_scores,
                closure_target_bonus=closure_target_bonus,
                enforce_vertex_link_manifold=enforce_vertex_link_manifold,
                target_face_count=target_face_count,
                vertices=vertices,
                strict_manifold=strict_manifold,
            )
            if score is not None:
                strict_scores.append((face, float(score)))
            relaxed = score_indexed_face_candidate(
                face,
                model_score,
                state,
                closure_bonus=closure_bonus,
                new_edge_penalty=new_edge_penalty,
                edge_length_penalty=edge_length_penalty,
                aspect_penalty=aspect_penalty,
                require_boundary_closure_after=0,
                closure_target_scores=closure_target_scores,
                closure_target_bonus=closure_target_bonus,
                enforce_vertex_link_manifold=enforce_vertex_link_manifold,
                target_face_count=target_face_count,
                vertices=vertices,
                strict_manifold=False,
            )
            if relaxed is not None:
                relaxed_scores.append((face, float(relaxed)))
    ranked = strict_scores if strict_scores else relaxed_scores
    if ranked:
        ranked.sort(key=lambda item: item[1], reverse=True)
        return [(np.asarray(face, dtype=np.int64), score) for face, score in ranked[:max_candidates]]
    for row, (a, b, _) in enumerate(pairs):
        for c in np.argsort(logits2[row])[-top_k:][::-1].tolist():
            face = (int(a), int(b), int(c))
            if len(set(face)) == 3 and score_indexed_face_candidate(
                face,
                0.0,
                state,
                require_boundary_closure_after=0,
                enforce_vertex_link_manifold=enforce_vertex_link_manifold,
                target_face_count=target_face_count,
                strict_manifold=False,
            ) is not None:
                return [(np.asarray(face, dtype=np.int64), 0.0)]
    fallback = select_topology_fallback_indexed_face(
        state,
        vertex_count=vertex_count,
        target_face_count=target_face_count,
        enforce_vertex_link_manifold=enforce_vertex_link_manifold,
    )
    return [(fallback, 0.0)] if fallback is not None else [(np.asarray([0, 1, 2], dtype=np.int64), 0.0)]


def _select_corner_causal_boundary_face(
    model,
    hidden,
    vertex_table,
    logits0: np.ndarray,
    device,
    state: IndexedDecodeState,
    *,
    vertex_count: int,
    vertices: np.ndarray | None,
    top_k: int,
    local_candidate_neighbors: int,
    closure_bonus: float,
    new_edge_penalty: float,
    edge_length_penalty: float,
    aspect_penalty: float,
    edge_action_bonus: float,
    edge_action_candidate_top_k: int,
    edge_choice_bonus: float,
    edge_choice_candidate_top_k: int,
    require_boundary_closure_after: int,
    closure_target_scores: np.ndarray | None,
    closure_target_bonus: float,
    enforce_vertex_link_manifold: bool,
    target_face_count: int | None,
) -> np.ndarray | None:  # type: ignore[no-untyped-def]
    """Score causal FACE candidates as boundary-edge completion actions."""

    import torch

    boundary_edges = state.boundary_edges
    if not boundary_edges:
        return None
    if require_boundary_closure_after and state.accepted_faces < int(require_boundary_closure_after):
        return None

    top_k = max(3, min(int(top_k), vertex_count))
    third_vertices = np.argsort(logits0[:vertex_count])[-top_k:][::-1].tolist()
    edge_limit = min(len(boundary_edges), max(top_k, top_k * 2))

    edge_action_candidate_top_k = max(0, int(edge_action_candidate_top_k))
    use_edge_action_candidates = edge_action_candidate_top_k > 0
    edge_choice_by_boundary: np.ndarray | None = None
    edge_choice_candidate_top_k = max(0, int(edge_choice_candidate_top_k))
    use_edge_choice_candidates = edge_choice_candidate_top_k > 0
    if (edge_choice_bonus != 0.0 or use_edge_choice_candidates) and hasattr(model, "_edge_choice_logits_from_hidden"):
        edge_prefix = np.asarray(boundary_edges, dtype=np.int64).reshape(1, 1, len(boundary_edges), 2)
        edge_choice_by_boundary = model._edge_choice_logits_from_hidden(
            hidden,
            torch.as_tensor(edge_prefix, dtype=torch.long, device=device),
            vertex_table=vertex_table,
        )[0, 0].detach().cpu().numpy()

    prefilter_edge_rows = _prefilter_boundary_edge_rows(
        boundary_edges,
        logits0,
        edge_choice_by_boundary=edge_choice_by_boundary,
        edge_choice_bonus=edge_choice_bonus,
        edge_limit=edge_limit,
        edge_action_candidate_top_k=edge_action_candidate_top_k,
        edge_choice_candidate_top_k=edge_choice_candidate_top_k,
    )
    edge_action_by_row: dict[int, np.ndarray] = {}
    if (edge_action_bonus != 0.0 or use_edge_action_candidates) and hasattr(model, "_edge_action_logits_from_hidden"):
        edge_prefix = np.asarray([boundary_edges[row] for row in prefilter_edge_rows], dtype=np.int64).reshape(len(prefilter_edge_rows), 1, 2)
        edge_action_subset = model._edge_action_logits_from_hidden(
            hidden.expand(len(prefilter_edge_rows), -1, -1),
            torch.as_tensor(edge_prefix, dtype=torch.long, device=device),
            vertex_table=vertex_table,
        )[:, 0, :vertex_count].detach().cpu().numpy()
        edge_action_by_row = {int(row): edge_action_subset[idx] for idx, row in enumerate(prefilter_edge_rows)}

    ranked_edge_rows = sorted(
        prefilter_edge_rows,
        key=lambda row: (
            max(float(logits0[int(boundary_edges[row][0])]), float(logits0[int(boundary_edges[row][1])]))
            + (float(edge_action_bonus) * float(np.max(edge_action_by_row[row])) if row in edge_action_by_row else 0.0)
            + (float(edge_choice_bonus) * float(edge_choice_by_boundary[row]) if edge_choice_by_boundary is not None else 0.0)
        ),
        reverse=True,
    )[:edge_limit]
    if edge_choice_by_boundary is not None and edge_choice_candidate_top_k > 0:
        for row in np.argsort(edge_choice_by_boundary)[-edge_choice_candidate_top_k:][::-1].tolist():
            if int(row) not in ranked_edge_rows:
                ranked_edge_rows.append(int(row))

    candidates: list[tuple[int, int, int]] = []
    candidate_edge_rows: list[int] = []
    seen: set[tuple[int, int, int]] = set()
    for edge_row in ranked_edge_rows:
        edge = boundary_edges[edge_row]
        edge_vertices = {int(edge[0]), int(edge[1])}
        edge_thirds = list(third_vertices)
        edge_action_logits = edge_action_by_row.get(int(edge_row))
        if edge_action_logits is not None and edge_action_candidate_top_k > 0:
            edge_action_top = np.argsort(edge_action_logits)[-edge_action_candidate_top_k:][::-1].tolist()
            edge_thirds = edge_action_top + edge_thirds
        if vertices is not None and local_candidate_neighbors > 0:
            q_vertices = np.asarray(vertices, dtype=np.float64)[:vertex_count]
            midpoint = (q_vertices[int(edge[0])] + q_vertices[int(edge[1])]) * 0.5
            edge_thirds.extend(int(index) for index in np.argsort(np.linalg.norm(q_vertices - midpoint, axis=1))[: local_candidate_neighbors + 2])
        seen_thirds = []
        for value in edge_thirds:
            if int(value) not in seen_thirds:
                seen_thirds.append(int(value))
        for third in seen_thirds:
            third = int(third)
            if third in edge_vertices:
                continue
            for face in canonical_indexed_face_orientations(int(edge[0]), int(edge[1]), third):
                if face in seen:
                    continue
                seen.add(face)
                candidates.append(face)
                candidate_edge_rows.append(edge_row)
    if not candidates:
        return None

    candidate_arr = np.asarray(candidates, dtype=np.int64)
    # Candidate permutations create many repeated prefixes. Cache the exact
    # corner-causal logits for each unique prefix; this preserves scores while
    # removing a large source of CPU/GPU churn in the boundary decoder.
    unique_first, first_inverse = np.unique(candidate_arr[:, 0], return_inverse=True)
    prefix1 = np.full((len(unique_first), 1, 3), -1, dtype=np.int64)
    prefix1[:, 0, 0] = unique_first
    logits1_unique = model._corner_causal_logits_from_hidden(
        hidden.expand(len(unique_first), -1, -1),
        torch.as_tensor(prefix1, dtype=torch.long, device=device),
        vertex_table=vertex_table,
    )[:, 0, 1, :vertex_count].detach().cpu().numpy()

    unique_pairs, pair_inverse = np.unique(candidate_arr[:, :2], axis=0, return_inverse=True)
    prefix2 = np.full((len(unique_pairs), 1, 3), -1, dtype=np.int64)
    prefix2[:, 0, 0] = unique_pairs[:, 0]
    prefix2[:, 0, 1] = unique_pairs[:, 1]
    logits2_unique = model._corner_causal_logits_from_hidden(
        hidden.expand(len(unique_pairs), -1, -1),
        torch.as_tensor(prefix2, dtype=torch.long, device=device),
        vertex_table=vertex_table,
    )[:, 0, 2, :vertex_count].detach().cpu().numpy()
    best_face: tuple[int, int, int] | None = None
    best_score = float("-inf")
    for row, face_values in enumerate(candidate_arr):
        face = tuple(int(value) for value in face_values)
        model_score = float(
            logits0[face[0]]
            + logits1_unique[first_inverse[row], face[1]]
            + logits2_unique[pair_inverse[row], face[2]]
        )
        edge_action_logits = edge_action_by_row.get(int(candidate_edge_rows[row]))
        if edge_action_logits is not None:
            model_score += float(edge_action_bonus) * float(edge_action_logits[face[2]])
        if edge_choice_by_boundary is not None:
            model_score += float(edge_choice_bonus) * float(edge_choice_by_boundary[candidate_edge_rows[row]])
        score = score_indexed_face_candidate(
            face,
            model_score,
            state,
            closure_bonus=closure_bonus,
            new_edge_penalty=new_edge_penalty,
            edge_length_penalty=edge_length_penalty,
            aspect_penalty=aspect_penalty,
            require_boundary_closure_after=require_boundary_closure_after,
            closure_target_scores=closure_target_scores,
            closure_target_bonus=closure_target_bonus,
            enforce_vertex_link_manifold=enforce_vertex_link_manifold,
            target_face_count=target_face_count,
            vertices=vertices,
            strict_manifold=True,
        )
        if score is not None and score > best_score:
            best_score = score
            best_face = face
    return np.asarray(best_face, dtype=np.int64) if best_face is not None else None


def _select_corner_causal_boundary_face_candidates(
    model,
    hidden,
    vertex_table,
    logits0: np.ndarray,
    device,
    state: IndexedDecodeState,
    *,
    vertex_count: int,
    vertices: np.ndarray | None,
    top_k: int,
    local_candidate_neighbors: int,
    closure_bonus: float,
    new_edge_penalty: float,
    edge_length_penalty: float,
    aspect_penalty: float,
    edge_action_bonus: float,
    edge_action_candidate_top_k: int,
    edge_choice_bonus: float,
    edge_choice_candidate_top_k: int,
    require_boundary_closure_after: int,
    closure_target_scores: np.ndarray | None,
    closure_target_bonus: float,
    enforce_vertex_link_manifold: bool,
    target_face_count: int | None,
    max_candidates: int,
) -> list[tuple[np.ndarray, float]]:  # type: ignore[no-untyped-def]
    import torch

    boundary_edges = state.boundary_edges
    if not boundary_edges:
        return []
    if require_boundary_closure_after and state.accepted_faces < int(require_boundary_closure_after):
        return []

    top_k = max(3, min(int(top_k), vertex_count))
    max_candidates = max(1, int(max_candidates))
    third_vertices = np.argsort(logits0[:vertex_count])[-top_k:][::-1].tolist()
    edge_limit = min(len(boundary_edges), max(top_k, top_k * 2))

    edge_action_candidate_top_k = max(0, int(edge_action_candidate_top_k))
    use_edge_action_candidates = edge_action_candidate_top_k > 0
    edge_choice_by_boundary: np.ndarray | None = None
    edge_choice_candidate_top_k = max(0, int(edge_choice_candidate_top_k))
    use_edge_choice_candidates = edge_choice_candidate_top_k > 0
    if (edge_choice_bonus != 0.0 or use_edge_choice_candidates) and hasattr(model, "_edge_choice_logits_from_hidden"):
        edge_prefix = np.asarray(boundary_edges, dtype=np.int64).reshape(1, 1, len(boundary_edges), 2)
        edge_choice_by_boundary = model._edge_choice_logits_from_hidden(
            hidden,
            torch.as_tensor(edge_prefix, dtype=torch.long, device=device),
            vertex_table=vertex_table,
        )[0, 0].detach().cpu().numpy()

    prefilter_edge_rows = _prefilter_boundary_edge_rows(
        boundary_edges,
        logits0,
        edge_choice_by_boundary=edge_choice_by_boundary,
        edge_choice_bonus=edge_choice_bonus,
        edge_limit=edge_limit,
        edge_action_candidate_top_k=edge_action_candidate_top_k,
        edge_choice_candidate_top_k=edge_choice_candidate_top_k,
    )
    edge_action_by_row: dict[int, np.ndarray] = {}
    if (edge_action_bonus != 0.0 or use_edge_action_candidates) and hasattr(model, "_edge_action_logits_from_hidden"):
        edge_prefix = np.asarray([boundary_edges[row] for row in prefilter_edge_rows], dtype=np.int64).reshape(len(prefilter_edge_rows), 1, 2)
        edge_action_subset = model._edge_action_logits_from_hidden(
            hidden.expand(len(prefilter_edge_rows), -1, -1),
            torch.as_tensor(edge_prefix, dtype=torch.long, device=device),
            vertex_table=vertex_table,
        )[:, 0, :vertex_count].detach().cpu().numpy()
        edge_action_by_row = {int(row): edge_action_subset[idx] for idx, row in enumerate(prefilter_edge_rows)}

    ranked_edge_rows = sorted(
        prefilter_edge_rows,
        key=lambda row: (
            max(float(logits0[int(boundary_edges[row][0])]), float(logits0[int(boundary_edges[row][1])]))
            + (float(edge_action_bonus) * float(np.max(edge_action_by_row[row])) if row in edge_action_by_row else 0.0)
            + (float(edge_choice_bonus) * float(edge_choice_by_boundary[row]) if edge_choice_by_boundary is not None else 0.0)
        ),
        reverse=True,
    )[:edge_limit]
    if edge_choice_by_boundary is not None and edge_choice_candidate_top_k > 0:
        for row in np.argsort(edge_choice_by_boundary)[-edge_choice_candidate_top_k:][::-1].tolist():
            if int(row) not in ranked_edge_rows:
                ranked_edge_rows.append(int(row))

    candidates: list[tuple[int, int, int]] = []
    candidate_edge_rows: list[int] = []
    seen: set[tuple[int, int, int]] = set()
    for edge_row in ranked_edge_rows:
        edge = boundary_edges[edge_row]
        edge_vertices = {int(edge[0]), int(edge[1])}
        edge_thirds = list(third_vertices)
        edge_action_logits = edge_action_by_row.get(int(edge_row))
        if edge_action_logits is not None and edge_action_candidate_top_k > 0:
            edge_action_top = np.argsort(edge_action_logits)[-edge_action_candidate_top_k:][::-1].tolist()
            edge_thirds = edge_action_top + edge_thirds
        if vertices is not None and local_candidate_neighbors > 0:
            q_vertices = np.asarray(vertices, dtype=np.float64)[:vertex_count]
            midpoint = (q_vertices[int(edge[0])] + q_vertices[int(edge[1])]) * 0.5
            edge_thirds.extend(int(index) for index in np.argsort(np.linalg.norm(q_vertices - midpoint, axis=1))[: local_candidate_neighbors + 2])
        seen_thirds = []
        for value in edge_thirds:
            if int(value) not in seen_thirds:
                seen_thirds.append(int(value))
        for third in seen_thirds:
            third = int(third)
            if third in edge_vertices:
                continue
            for face in canonical_indexed_face_orientations(int(edge[0]), int(edge[1]), third):
                if face in seen:
                    continue
                seen.add(face)
                candidates.append(face)
                candidate_edge_rows.append(edge_row)
    if not candidates:
        return []

    candidate_arr = np.asarray(candidates, dtype=np.int64)
    unique_first, first_inverse = np.unique(candidate_arr[:, 0], return_inverse=True)
    prefix1 = np.full((len(unique_first), 1, 3), -1, dtype=np.int64)
    prefix1[:, 0, 0] = unique_first
    logits1_unique = model._corner_causal_logits_from_hidden(
        hidden.expand(len(unique_first), -1, -1),
        torch.as_tensor(prefix1, dtype=torch.long, device=device),
        vertex_table=vertex_table,
    )[:, 0, 1, :vertex_count].detach().cpu().numpy()

    unique_pairs, pair_inverse = np.unique(candidate_arr[:, :2], axis=0, return_inverse=True)
    prefix2 = np.full((len(unique_pairs), 1, 3), -1, dtype=np.int64)
    prefix2[:, 0, 0] = unique_pairs[:, 0]
    prefix2[:, 0, 1] = unique_pairs[:, 1]
    logits2_unique = model._corner_causal_logits_from_hidden(
        hidden.expand(len(unique_pairs), -1, -1),
        torch.as_tensor(prefix2, dtype=torch.long, device=device),
        vertex_table=vertex_table,
    )[:, 0, 2, :vertex_count].detach().cpu().numpy()

    scored: list[tuple[tuple[int, int, int], float]] = []
    for row, face_values in enumerate(candidate_arr):
        face = tuple(int(value) for value in face_values)
        model_score = float(
            logits0[face[0]]
            + logits1_unique[first_inverse[row], face[1]]
            + logits2_unique[pair_inverse[row], face[2]]
        )
        edge_action_logits = edge_action_by_row.get(int(candidate_edge_rows[row]))
        if edge_action_logits is not None:
            model_score += float(edge_action_bonus) * float(edge_action_logits[face[2]])
        if edge_choice_by_boundary is not None:
            model_score += float(edge_choice_bonus) * float(edge_choice_by_boundary[candidate_edge_rows[row]])
        score = score_indexed_face_candidate(
            face,
            model_score,
            state,
            closure_bonus=closure_bonus,
            new_edge_penalty=new_edge_penalty,
            edge_length_penalty=edge_length_penalty,
            aspect_penalty=aspect_penalty,
            require_boundary_closure_after=require_boundary_closure_after,
            closure_target_scores=closure_target_scores,
            closure_target_bonus=closure_target_bonus,
            enforce_vertex_link_manifold=enforce_vertex_link_manifold,
            target_face_count=target_face_count,
            vertices=vertices,
            strict_manifold=True,
        )
        if score is not None:
            scored.append((face, float(score)))
    scored.sort(key=lambda item: item[1], reverse=True)
    return [(np.asarray(face, dtype=np.int64), score) for face, score in scored[:max_candidates]]


def _aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {"attempted": 0}
    token_accuracies = [item["teacher_forced_token_accuracy"] for item in items if item.get("teacher_forced_token_accuracy") is not None]
    face_exact_ratios = [item["teacher_forced_face_exact_ratio"] for item in items if item.get("teacher_forced_face_exact_ratio") is not None]
    raw_items = [item for item in items if item.get("raw_watertight") is not None]
    raw_chamfer = [item["raw_chamfer_l2_normalized"] for item in items if item.get("raw_chamfer_l2_normalized") is not None]
    raw_normal = [item["raw_normal_consistency"] for item in items if item.get("raw_normal_consistency") is not None]
    topology_stats = [item.get("topology_decode_stats") or {} for item in items]
    requested_faces = [int(stats.get("topology_requested_faces", item.get("selected_face_count", 0)) or 0) for item, stats in zip(items, topology_stats)]
    generated_faces = [int(stats.get("topology_generated_faces", item.get("generated_faces", 0)) or 0) for item, stats in zip(items, topology_stats)]
    fallback_counts = [int(stats.get("topology_fallbacks", 0) or 0) for stats in topology_stats]
    stop_counts = [int(stats.get("topology_stop_early", 0) or 0) for stats in topology_stats]
    requested_total = int(sum(requested_faces))
    generated_total = int(sum(generated_faces))
    fallback_total = int(sum(fallback_counts))
    return {
        "attempted": len(items),
        "watertight": int(sum(1 for item in items if item.get("watertight"))),
        "mean_boundary_edges": float(np.mean([item.get("boundary_edges", 0) for item in items])),
        "mean_nonmanifold_edges": float(np.mean([item.get("nonmanifold_edges", 0) for item in items])),
        "mean_nonmanifold_vertices": float(np.mean([item.get("nonmanifold_vertices", 0) for item in items])),
        "raw_watertight": int(sum(1 for item in raw_items if item.get("raw_watertight"))),
        "raw_mean_boundary_edges": float(np.mean([item.get("raw_boundary_edges", 0) for item in raw_items])) if raw_items else None,
        "raw_mean_nonmanifold_edges": float(np.mean([item.get("raw_nonmanifold_edges", 0) for item in raw_items])) if raw_items else None,
        "raw_mean_nonmanifold_vertices": float(np.mean([item.get("raw_nonmanifold_vertices", 0) for item in raw_items])) if raw_items else None,
        "raw_mean_chamfer_l2_normalized": float(np.mean(raw_chamfer)) if raw_chamfer else None,
        "raw_mean_normal_consistency": float(np.mean(raw_normal)) if raw_normal else None,
        "mean_chamfer_l2": float(np.mean([item["chamfer_l2"] for item in items if item.get("chamfer_l2") is not None])) if any(item.get("chamfer_l2") is not None for item in items) else None,
        "mean_chamfer_l2_normalized": float(np.mean([item["chamfer_l2_normalized"] for item in items if item.get("chamfer_l2_normalized") is not None])) if any(item.get("chamfer_l2_normalized") is not None for item in items) else None,
        "mean_hausdorff_l2_normalized": float(np.mean([item["hausdorff_l2_normalized"] for item in items if item.get("hausdorff_l2_normalized") is not None])) if any(item.get("hausdorff_l2_normalized") is not None for item in items) else None,
        "mean_edge_pairing_ratio": float(np.mean([item.get("token_edge_pairing_ratio", 0.0) for item in items])),
        "raw_mean_edge_pairing_ratio": float(np.mean([item.get("raw_token_edge_pairing_ratio", 0.0) for item in raw_items])) if raw_items else None,
        "mean_teacher_forced_token_accuracy": float(np.mean(token_accuracies)) if token_accuracies else None,
        "mean_teacher_forced_face_exact_ratio": float(np.mean(face_exact_ratios)) if face_exact_ratios else None,
        "mean_topology_fallbacks": float(np.mean(fallback_counts)) if fallback_counts else 0.0,
        "topology_fallback_rate_per_generated_face": float(fallback_total / generated_total) if generated_total > 0 else None,
        "topology_stop_early_count": int(sum(1 for count in stop_counts if count > 0)),
        "mean_topology_generated_face_ratio": float(np.mean([generated / requested for generated, requested in zip(generated_faces, requested_faces) if requested > 0])) if any(requested > 0 for requested in requested_faces) else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Model checkpoint. Optional only for --decode-strategy teacher_identity.",
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--export-dir", type=Path, default=None)
    parser.add_argument("--cleanup-export-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0, help="Skip this many sorted dataset shards before eval.")
    parser.add_argument("--point-samples", type=int, default=0)
    parser.add_argument("--face-count-mode", choices=["gt", "predicted", "max"], default="predicted")
    parser.add_argument("--pair-samples", type=int, default=1000)
    parser.add_argument("--cleanup-min-component-faces", type=int, default=1)
    parser.add_argument("--cleanup-split-nonmanifold-vertices", action="store_true")
    parser.add_argument("--split-pinched-vertices", action="store_true")
    parser.add_argument("--token-repair-mode", choices=["none", "dedupe", "manifold"], default="none")
    parser.add_argument("--boundary-fill", choices=["none", "fan", "centroid"], default="none")
    parser.add_argument("--boundary-fill-max-loop-edges", type=int, default=128)
    parser.add_argument(
        "--raw-pair-metrics",
        action="store_true",
        help="Also compute generated-vs-reference geometry metrics before boundary fill/repair. Useful for short repair probes.",
    )
    parser.add_argument(
        "--decode-strategy",
        choices=["free_run", "teacher_forced", "teacher_identity"],
        default="free_run",
        help=(
            "teacher_identity bypasses the model and evaluates exact target faces "
            "through the same decode/quality path as model outputs."
        ),
    )
    parser.add_argument("--decode-mode", choices=["edge_constrained", "boundary_edge", "unconstrained"], default="edge_constrained")
    parser.add_argument("--corner-decode", choices=["auto", "causal", "parallel"], default="auto")
    parser.add_argument("--constraint-top-k", type=int, default=24)
    parser.add_argument("--local-candidate-neighbors", type=int, default=0)
    parser.add_argument("--closure-bonus", type=float, default=1.0)
    parser.add_argument("--new-edge-penalty", type=float, default=0.1)
    parser.add_argument("--edge-length-penalty", type=float, default=0.0)
    parser.add_argument("--aspect-penalty", type=float, default=0.0)
    parser.add_argument("--edge-action-bonus", type=float, default=0.0)
    parser.add_argument("--edge-action-candidate-top-k", type=int, default=0)
    parser.add_argument("--edge-choice-bonus", type=float, default=0.0)
    parser.add_argument("--edge-choice-candidate-top-k", type=int, default=0)
    parser.add_argument("--seed-face-bonus", type=float, default=0.0)
    parser.add_argument("--require-boundary-closure-after", type=int, default=1)
    parser.add_argument("--closure-target-bonus", type=float, default=0.0)
    parser.add_argument("--boundary-budget-constraint", action="store_true")
    parser.add_argument("--teacher-seed-faces", type=int, default=0)
    parser.add_argument("--beam-width", type=int, default=1)
    parser.add_argument("--beam-candidates", type=int, default=4)
    parser.add_argument("--vertex-link-constraint", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = parser.parse_args()

    paths = sorted(path for path in args.dataset_dir.glob("*.npz") if not path.name.startswith("._"))
    if args.offset:
        paths = paths[max(0, int(args.offset)) :]
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit(f"no FACE-Q npz samples found in {args.dataset_dir}")

    checkpoint = None
    train_args: dict[str, Any] = {}
    if args.checkpoint is not None:
        checkpoint = _load_checkpoint(args.checkpoint)
        train_args = checkpoint.get("args", {})
        num_bins = int(checkpoint["num_bins"])
        max_vertices = int(checkpoint["max_vertices"])
        max_faces = int(checkpoint["max_faces"])
    elif args.decode_strategy == "teacher_identity":
        # Model-free target/evaluator sanity check. Infer the shape envelope from
        # the selected samples so the rest of the evaluator uses the same path.
        max_vertices = 0
        max_faces = 0
        num_bins = None
        for path in paths:
            data = np.load(path)
            max_vertices = max(max_vertices, int(len(data["indexed_vertices"])))
            max_faces = max(max_faces, int(len(data["indexed_faces"])))
            sample_bins = int(np.asarray(data["num_bins"]).reshape(-1)[0])
            num_bins = sample_bins if num_bins is None else num_bins
            if sample_bins != num_bins:
                raise SystemExit(f"mixed num_bins in teacher_identity eval: {path} has {sample_bins}, expected {num_bins}")
        if max_vertices <= 0 or max_faces <= 0 or num_bins is None:
            raise SystemExit("teacher_identity eval could not infer non-empty FACE-Q shapes")
    else:
        raise SystemExit("--checkpoint is required unless --decode-strategy teacher_identity")

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
    model = None
    if checkpoint is not None:
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
            condition_backend=str(train_args.get("condition_backend", "pooled")),
            decoder_backend=str(train_args.get("decoder_backend", "prefix")),
            encoder_layers=int(train_args.get("encoder_layers", 4)),
            latent_dim=int(train_args.get("latent_dim", 64)),
            face_output_mode=str(train_args.get("face_output_mode", "linear")),
        ).to(device)
        load_result = model.load_state_dict(checkpoint["model_state"], strict=False)
        if load_result.missing_keys or load_result.unexpected_keys:
            print(
                json.dumps(
                    {
                        "checkpoint_load_note": "non-strict load for evolving FACE indexed research checkpoints",
                        "missing_keys": sorted(load_result.missing_keys),
                        "unexpected_keys": sorted(load_result.unexpected_keys),
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
            )
        model.eval()
    if args.corner_decode == "causal":
        use_corner_causal = True
    elif args.corner_decode == "parallel":
        use_corner_causal = False
    else:
        use_corner_causal = bool(checkpoint.get("has_corner_causal_head")) if checkpoint is not None else False
    use_topology_head = bool(checkpoint.get("has_topology_head")) if checkpoint is not None else False
    use_edge_choice_head = bool(checkpoint.get("has_edge_choice_head")) if checkpoint is not None else False
    edge_choice_bonus = float(args.edge_choice_bonus) if use_edge_choice_head else 0.0
    use_seed_face_head = bool(checkpoint.get("has_seed_face_head")) if checkpoint is not None else False
    seed_face_bonus = float(args.seed_face_bonus) if use_seed_face_head else 0.0
    if args.decode_strategy == "teacher_identity" and args.face_count_mode == "predicted":
        raise SystemExit("teacher_identity requires --face-count-mode gt or max; predicted count needs a model")
    if args.export_dir:
        args.export_dir.mkdir(parents=True, exist_ok=True)
    if args.cleanup_export_dir:
        args.cleanup_export_dir.mkdir(parents=True, exist_ok=True)

    results = []
    cleanup_results = []
    for idx, path in enumerate(paths):
        point_features_np, teacher_seq = _load_sample(path, args.point_samples if args.point_samples > 0 else None)
        vertex_count = min(len(teacher_seq.vertices), max_vertices)
        vertex_table_np = np.full((max_vertices, 3), -1, dtype=np.int64)
        vertex_table_np[:vertex_count] = teacher_seq.vertices[:vertex_count]
        point_tensor = None
        vertex_tensor = None
        if args.decode_strategy != "teacher_identity":
            point_tensor = torch.as_tensor(point_features_np, dtype=torch.float32, device=device).unsqueeze(0)
            vertex_tensor = torch.as_tensor(vertex_table_np, dtype=torch.long, device=device).unsqueeze(0)
        if args.face_count_mode == "gt":
            face_count = min(len(teacher_seq.faces), max_faces)
            predicted_count = None
        elif args.face_count_mode == "max":
            face_count = max_faces
            predicted_count = None
        else:
            if model is None or point_tensor is None or vertex_tensor is None:
                raise SystemExit("predicted face-count mode requires a checkpoint-backed model")
            with torch.no_grad():
                predicted_count = int(model.predict_face_count_logits(point_tensor, vertex_tensor).argmax(dim=-1).item())
            face_count = max(1, min(max_faces, predicted_count))
        decode_started = time.perf_counter()
        teacher_forced_stats: dict[str, Any] = {
            "token_accuracy": None,
            "face_exact_ratio": None,
        }
        decode_stats: dict[str, Any] = {}
        if args.decode_strategy == "teacher_identity":
            generated_faces, teacher_forced_stats = _teacher_identity_faces(teacher_seq.faces[:face_count])
        elif args.decode_strategy == "teacher_forced":
            if model is None or point_tensor is None or vertex_tensor is None:
                raise SystemExit("teacher_forced decode requires a checkpoint-backed model")
            generated_faces, teacher_forced_stats = _teacher_forced_faces(
                model,
                point_tensor,
                vertex_tensor,
                teacher_seq.faces[:face_count],
                vertex_count=vertex_count,
                device=device,
                use_corner_causal=use_corner_causal,
            )
        else:
            if model is None or point_tensor is None or vertex_tensor is None:
                raise SystemExit("free_run decode requires a checkpoint-backed model")
            generated_faces = _generate_faces(
                model,
                point_tensor,
                vertex_tensor,
                face_count=face_count,
                vertex_count=vertex_count,
                device=device,
                decode_mode=args.decode_mode,
                constraint_top_k=args.constraint_top_k,
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
                closure_target_bonus=args.closure_target_bonus,
                use_topology_head=use_topology_head,
                use_corner_causal=use_corner_causal,
                enforce_vertex_link_manifold=args.vertex_link_constraint,
                boundary_budget_constraint=args.boundary_budget_constraint,
                seed_faces=teacher_seq.faces[: max(0, int(args.teacher_seed_faces))] if args.teacher_seed_faces > 0 else None,
                beam_width=args.beam_width,
                beam_candidates=args.beam_candidates,
                decode_stats=decode_stats,
            )
        decode_elapsed_sec = time.perf_counter() - decode_started
        generated_faces = generated_faces[np.all(generated_faces < vertex_count, axis=1)]
        generated_seq = FaceIndexedSequence(
            vertices=teacher_seq.vertices[:vertex_count],
            faces=generated_faces,
            num_bins=teacher_seq.num_bins,
            transform=teacher_seq.transform,
        )
        token_repair_report = None
        boundary_fill_report = None
        degenerate_drop_report = None
        if args.token_repair_mode != "none":
            repaired_tokens, repair_report = repair_face_tokens(
                indexed_to_coordinate_tokens(generated_seq),
                mode=args.token_repair_mode,
            )
            token_repair_report = asdict(repair_report)
            generated_seq = coordinate_tokens_to_indexed(
                repaired_tokens,
                num_bins=teacher_seq.num_bins,
                transform=teacher_seq.transform,
                max_vertices=max_vertices,
            )
        generated_seq, dropped_degenerate_faces = drop_geometric_degenerate_indexed_faces(generated_seq)
        if dropped_degenerate_faces:
            degenerate_drop_report = {"dropped_faces": int(dropped_degenerate_faces)}
        raw_generated_seq = generated_seq
        raw_generated = decode_indexed_face_tokens_to_mesh(raw_generated_seq)
        raw_token_report = face_token_topology_report(indexed_to_coordinate_tokens(raw_generated_seq))
        if args.boundary_fill in {"fan", "centroid"}:
            generated_seq, fill_report = fill_indexed_boundary_loops(
                generated_seq,
                max_loop_edges=args.boundary_fill_max_loop_edges,
                strategy=args.boundary_fill,
            )
            boundary_fill_report = fill_report.to_dict()
        teacher_seq = FaceIndexedSequence(
            vertices=teacher_seq.vertices[:vertex_count],
            faces=teacher_seq.faces[:face_count],
            num_bins=teacher_seq.num_bins,
            transform=teacher_seq.transform,
        )
        generated = decode_indexed_face_tokens_to_mesh(generated_seq)
        teacher = decode_indexed_face_tokens_to_mesh(teacher_seq)
        pinch_split_report = None
        if args.split_pinched_vertices:
            generated, pinch_split_report = split_pinched_vertices(generated)
        token_report = face_token_topology_report(indexed_to_coordinate_tokens(generated_seq))
        if args.export_dir:
            raw_path = args.export_dir / f"{idx:04d}_{path.stem}_raw.glb"
            generated_path = args.export_dir / f"{idx:04d}_{path.stem}_generated.glb"
            teacher_path = args.export_dir / f"{idx:04d}_{path.stem}_teacher.glb"
        else:
            raw_path = args.output.parent / f".tmp_{idx:04d}_{path.stem}_raw.glb"
            generated_path = args.output.parent / f".tmp_{idx:04d}_{path.stem}_generated.glb"
            teacher_path = args.output.parent / f".tmp_{idx:04d}_{path.stem}_teacher.glb"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_generated.export(raw_path)
        generated.export(generated_path)
        teacher.export(teacher_path)
        raw_quality = evaluate_mesh(raw_path)
        raw_metrics = evaluate_mesh_pair(raw_path, teacher_path, samples=args.pair_samples) if args.raw_pair_metrics else {}
        metrics = evaluate_mesh_pair(generated_path, teacher_path, samples=args.pair_samples)
        quality = evaluate_mesh(generated_path)
        item = {
            "path": str(path),
            "selected_face_count": int(face_count),
            "predicted_face_count": predicted_count,
            "decode_mode": args.decode_mode,
            "decode_strategy": args.decode_strategy,
            "decode_elapsed_sec": float(decode_elapsed_sec),
            "vertex_link_constraint": bool(args.vertex_link_constraint),
            "teacher_seed_faces": int(args.teacher_seed_faces),
            "beam_width": int(args.beam_width),
            "beam_candidates": int(args.beam_candidates),
            "split_pinched_vertices": bool(args.split_pinched_vertices),
            "pinch_split_report": pinch_split_report,
            "token_repair_mode": args.token_repair_mode,
            "token_repair_report": token_repair_report,
            "degenerate_drop_report": degenerate_drop_report,
            "boundary_fill": args.boundary_fill,
            "boundary_fill_report": boundary_fill_report,
            "corner_decode": "causal" if use_corner_causal else "parallel",
            "topology_decode": bool(use_topology_head and args.closure_target_bonus != 0.0),
            "topology_decode_stats": decode_stats,
            "generated_faces": int(len(generated.faces)),
            "generated_vertices": int(len(generated.vertices)),
            "raw_generated_faces": int(len(raw_generated.faces)),
            "raw_generated_vertices": int(len(raw_generated.vertices)),
            "teacher_faces": int(len(teacher.faces)),
            "teacher_vertices": int(len(teacher.vertices)),
            "raw_watertight": bool(raw_quality.get("watertight")),
            "raw_boundary_edges": int(raw_quality.get("boundary_edge_count") or 0),
            "raw_nonmanifold_edges": int(raw_quality.get("nonmanifold_edge_count") or 0),
            "raw_nonmanifold_vertices": int(raw_quality.get("nonmanifold_vertex_count") or 0),
            "raw_token_watertight_edge_graph": bool(raw_token_report.watertight_edge_graph),
            "raw_token_boundary_edge_count": int(raw_token_report.boundary_edge_count),
            "raw_token_nonmanifold_edge_count": int(raw_token_report.nonmanifold_edge_count),
            "raw_token_edge_pairing_ratio": float(raw_token_report.edge_pairing_ratio),
            "raw_chamfer_l2_normalized": raw_metrics.get("chamfer_l2_normalized"),
            "raw_normal_consistency": raw_metrics.get("normal_consistency"),
            "watertight": bool(quality.get("watertight")),
            "boundary_edges": int(quality.get("boundary_edge_count") or 0),
            "nonmanifold_edges": int(quality.get("nonmanifold_edge_count") or 0),
            "nonmanifold_vertices": int(quality.get("nonmanifold_vertex_count") or 0),
            "token_watertight_edge_graph": bool(token_report.watertight_edge_graph),
            "token_boundary_edge_count": int(token_report.boundary_edge_count),
            "token_nonmanifold_edge_count": int(token_report.nonmanifold_edge_count),
            "token_edge_pairing_ratio": float(token_report.edge_pairing_ratio),
            "teacher_forced_token_accuracy": teacher_forced_stats["token_accuracy"],
            "teacher_forced_face_exact_ratio": teacher_forced_stats["face_exact_ratio"],
            **metrics,
        }
        results.append(item)
        fill_report = item.get("boundary_fill_report") or {}
        print(
            json.dumps(
                {
                    "eval_progress": idx + 1,
                    "eval_total": len(paths),
                    "path": path.name,
                    "decode_strategy": args.decode_strategy,
                    "decode_elapsed_sec": item["decode_elapsed_sec"],
                    "raw_watertight": item["raw_watertight"],
                    "raw_boundary_edges": item["raw_boundary_edges"],
                    "raw_nonmanifold_vertices": item["raw_nonmanifold_vertices"],
                    "watertight": item["watertight"],
                    "chamfer_l2_normalized": item.get("chamfer_l2_normalized"),
                    "normal_consistency": item.get("normal_consistency"),
                    "boundary_fill_input_boundary_edges": fill_report.get("input_boundary_edges"),
                    "boundary_fill_filled_faces": fill_report.get("filled_faces"),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if args.export_dir:
            pass
        else:
            raw_path.unlink(missing_ok=True)
            generated_path.unlink(missing_ok=True)
            teacher_path.unlink(missing_ok=True)
        if args.cleanup_export_dir:
            raw_path = args.cleanup_export_dir / f"{idx:04d}_{path.stem}_raw.glb"
            clean_path = args.cleanup_export_dir / f"{idx:04d}_{path.stem}_cleaned.glb"
            generated.export(raw_path)
            cleanup_report = cleanup_mesh_file(
                raw_path,
                clean_path,
                CleanupOptions(
                    min_component_faces=args.cleanup_min_component_faces,
                    merge_vertices=True,
                    split_nonmanifold_vertices=args.cleanup_split_nonmanifold_vertices,
                    fill_holes=True,
                ),
            )
            teacher_clean_path = args.cleanup_export_dir / f"{idx:04d}_{path.stem}_teacher.glb"
            teacher.export(teacher_clean_path)
            clean_metrics = evaluate_mesh_pair(clean_path, teacher_clean_path, samples=args.pair_samples)
            clean_quality = evaluate_mesh(clean_path)
            cleanup_results.append({
                "path": str(path),
                "watertight": bool(clean_quality.get("watertight")),
                "boundary_edges": int(clean_quality.get("boundary_edge_count") or 0),
                "nonmanifold_edges": int(clean_quality.get("nonmanifold_edge_count") or 0),
                "nonmanifold_vertices": int(clean_quality.get("nonmanifold_vertex_count") or 0),
                "cleanup_report": asdict(cleanup_report),
                **clean_metrics,
            })

    report = {
        "checkpoint": str(args.checkpoint),
        "dataset_dir": str(args.dataset_dir),
        "offset": int(args.offset),
        "decode_strategy": args.decode_strategy,
        "decode_mode": args.decode_mode,
        "token_repair_mode": args.token_repair_mode,
        "boundary_fill": args.boundary_fill,
        "boundary_fill_max_loop_edges": int(args.boundary_fill_max_loop_edges),
        "raw_pair_metrics": bool(args.raw_pair_metrics),
        "corner_decode": "causal" if use_corner_causal else "parallel",
        "edge_head_mode": str(getattr(model, "edge_head_mode", "index")),
        "constraint_top_k": int(args.constraint_top_k),
        "local_candidate_neighbors": int(args.local_candidate_neighbors),
        "closure_bonus": float(args.closure_bonus),
        "new_edge_penalty": float(args.new_edge_penalty),
        "edge_length_penalty": float(args.edge_length_penalty),
        "aspect_penalty": float(args.aspect_penalty),
        "edge_action_bonus": float(args.edge_action_bonus),
        "edge_action_candidate_top_k": int(args.edge_action_candidate_top_k),
        "edge_choice_bonus": float(edge_choice_bonus),
        "edge_choice_candidate_top_k": int(args.edge_choice_candidate_top_k),
        "seed_face_bonus": float(seed_face_bonus),
        "require_boundary_closure_after": int(args.require_boundary_closure_after),
        "closure_target_bonus": float(args.closure_target_bonus),
        "boundary_budget_constraint": bool(args.boundary_budget_constraint),
        "vertex_link_constraint": bool(args.vertex_link_constraint),
        "teacher_seed_faces": int(args.teacher_seed_faces),
        "beam_width": int(args.beam_width),
        "beam_candidates": int(args.beam_candidates),
        "split_pinched_vertices": bool(args.split_pinched_vertices),
        "cleanup_split_nonmanifold_vertices": bool(args.cleanup_split_nonmanifold_vertices),
        "topology_decode": bool(use_topology_head and args.closure_target_bonus != 0.0),
        "representation": "face-indexed-v2",
        "summary": _aggregate(results),
        "cleanup_summary": _aggregate(cleanup_results) if cleanup_results else None,
        "results": results,
        "cleanup_results": cleanup_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
