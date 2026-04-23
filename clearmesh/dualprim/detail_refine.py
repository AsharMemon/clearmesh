"""Detail-refinement preparation for DualPrim.

This module does not replace DualPrim. It prepares a narrow-band
detail-refinement stage that can add local accuracy while preserving
DualPrim as the compact structural scaffold.

Design goals:
  - operate only near the coarse DualPrim mesh
  - quantify where detail is missing
  - emit explicit compactness budgets so refinement cannot explode
  - be backend-agnostic (residual SDF, displacement, FlexiCubes, etc.)

The output is:
  - a JSON manifest describing budgets + detail signal
  - an NPZ with sampled target/coarse points and normals
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import trimesh


@dataclass
class DetailRefineConfig:
    """Guardrails for narrow-band detail refinement.

    The defaults are intentionally conservative so the refinement stage
    cannot defeat the purpose of DualPrim by turning into an unconstrained
    high-vertex reconstruction.
    """

    num_surface_samples: int = 32768
    coarse_surface_samples: int = 16384
    band_radius_frac: float = 0.05
    max_displacement_frac: float = 0.03

    max_vertex_growth: float = 1.35
    max_face_growth: float = 1.35
    max_component_growth: float = 1.15

    chamfer_weight: float = 1.0
    normal_weight: float = 2.0
    structure_weight: float = 4.0

    extractor: str = "flexicubes_candidate"
    stage_name: str = "dualprim_narrow_band_detail"
    seed: int = 0


@dataclass
class DetailRefineBudget:
    max_vertices: int
    max_faces: int
    max_components: int
    max_displacement: float


@dataclass
class PrimitiveLocalRefineConfig:
    """Guardrails for primitive-local refinement setup.

    This stage stays attached to the surviving DualPrim scaffold instead of
    introducing a new global field. The defaults bias toward keeping the
    local refinement small, part-centric, and easy to budget.
    """

    num_surface_samples: int = 32768
    band_radius_frac: float = 0.05
    assignment_margin_frac: float = 0.35
    min_alpha: float = 0.1
    min_samples_per_primitive: int = 256
    max_samples_per_primitive: int = 4096
    max_vertex_growth: float = 1.35
    max_face_growth: float = 1.35
    max_component_growth: float = 1.15
    max_displacement_frac: float = 0.03
    stage_name: str = "dualprim_primitive_local_detail"
    seed: int = 0


@dataclass
class MeshSummary:
    vertices: int
    faces: int
    components: int


def _load_mesh(path: str | Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="mesh")
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            return trimesh.Trimesh()
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    return loaded


def _normalize_pair(
    coarse_mesh: trimesh.Trimesh,
    target_mesh: trimesh.Trimesh,
) -> tuple[trimesh.Trimesh, trimesh.Trimesh, dict]:
    """Normalize both meshes using the target frame.

    This keeps the detail stage aligned to the input/reference geometry
    while making budgets dimensionless and comparable across objects.
    """
    coarse = coarse_mesh.copy()
    target = target_mesh.copy()

    centroid = target.bounding_box.centroid
    extent = float(max(target.extents)) if len(target.vertices) else 1.0
    scale = 1.0 / max(extent, 1e-8)

    coarse.vertices = (coarse.vertices - centroid) * scale
    target.vertices = (target.vertices - centroid) * scale

    meta = {
        "target_centroid": centroid.tolist(),
        "target_extent": extent,
        "target_scale": scale,
    }
    return coarse, target, meta


def _mesh_summary(mesh: trimesh.Trimesh) -> MeshSummary:
    components = len(mesh.split(only_watertight=False)) if len(mesh.vertices) else 0
    return MeshSummary(
        vertices=int(len(mesh.vertices)),
        faces=int(len(mesh.faces)),
        components=int(components),
    )


def _sample_surface(mesh: trimesh.Trimesh, count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if len(mesh.faces) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    rng = np.random.default_rng(seed)
    points, face_idx = trimesh.sample.sample_surface(mesh, count, seed=rng)
    normals = mesh.face_normals[face_idx]
    normals = normals / np.clip(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8, None)
    return points.astype(np.float32), normals.astype(np.float32)


def _closest_on_mesh(
    mesh: trimesh.Trimesh,
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Closest-point query with a deterministic fallback.

    trimesh proximity is preferred because it returns surface points and
    triangle IDs. If the acceleration backend is unavailable, fall back to
    sampled-surface nearest neighbors. The fallback is coarser, but still
    useful for preparing the refinement stage.
    """
    try:
        closest, distance, tri_id = trimesh.proximity.closest_point(mesh, points)
        tri_id = np.asarray(tri_id, dtype=np.int64)
        return closest.astype(np.float32), distance.astype(np.float32), tri_id
    except Exception:
        coarse_points, coarse_normals = _sample_surface(mesh, max(len(points), 8192), seed=0)
        if len(coarse_points) == 0:
            return (
                np.zeros_like(points, dtype=np.float32),
                np.full((len(points),), np.inf, dtype=np.float32),
                np.zeros((len(points),), dtype=np.int64),
            )
        diff = points[:, None, :] - coarse_points[None, :, :]
        dist2 = np.sum(diff * diff, axis=-1)
        nn = np.argmin(dist2, axis=1)
        closest = coarse_points[nn]
        distance = np.sqrt(dist2[np.arange(len(points)), nn]).astype(np.float32)
        tri_id = np.zeros((len(points),), dtype=np.int64)
        return closest.astype(np.float32), distance, tri_id


def _normal_angles_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-8, None)
    b = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-8, None)
    dot = np.sum(a * b, axis=1)
    dot = np.clip(dot, -1.0, 1.0)
    return np.degrees(np.arccos(dot)).astype(np.float32)


def build_detail_refine_budget(
    coarse_summary: MeshSummary,
    cfg: DetailRefineConfig,
) -> DetailRefineBudget:
    return DetailRefineBudget(
        max_vertices=max(coarse_summary.vertices, int(round(coarse_summary.vertices * cfg.max_vertex_growth))),
        max_faces=max(coarse_summary.faces, int(round(coarse_summary.faces * cfg.max_face_growth))),
        max_components=max(coarse_summary.components, int(round(coarse_summary.components * cfg.max_component_growth))),
        max_displacement=cfg.max_displacement_frac,
    )


def build_primitive_local_budget(
    coarse_summary: MeshSummary,
    cfg: PrimitiveLocalRefineConfig,
) -> DetailRefineBudget:
    return DetailRefineBudget(
        max_vertices=max(coarse_summary.vertices, int(round(coarse_summary.vertices * cfg.max_vertex_growth))),
        max_faces=max(coarse_summary.faces, int(round(coarse_summary.faces * cfg.max_face_growth))),
        max_components=max(coarse_summary.components, int(round(coarse_summary.components * cfg.max_component_growth))),
        max_displacement=cfg.max_displacement_frac,
    )


def prepare_detail_refine_artifacts(
    coarse_mesh_path: str | Path,
    target_mesh_path: str | Path,
    out_dir: str | Path,
    cfg: DetailRefineConfig | None = None,
) -> dict:
    """Prepare narrow-band detail-refinement artifacts.

    Returns the manifest payload and writes:
      - detail_refine_manifest.json
      - detail_refine_samples.npz
    """
    cfg = cfg or DetailRefineConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    coarse_mesh = _load_mesh(coarse_mesh_path)
    target_mesh = _load_mesh(target_mesh_path)
    coarse_mesh, target_mesh, norm_meta = _normalize_pair(coarse_mesh, target_mesh)

    coarse_summary = _mesh_summary(coarse_mesh)
    target_summary = _mesh_summary(target_mesh)
    budget = build_detail_refine_budget(coarse_summary, cfg)

    target_points, target_normals = _sample_surface(
        target_mesh, cfg.num_surface_samples, seed=cfg.seed,
    )
    coarse_points, coarse_normals = _sample_surface(
        coarse_mesh, cfg.coarse_surface_samples, seed=cfg.seed + 1,
    )

    closest_on_coarse, target_to_coarse_dist, tri_id = _closest_on_mesh(coarse_mesh, target_points)
    if len(target_points):
        if len(coarse_mesh.face_normals) and len(tri_id):
            tri_id = np.clip(tri_id, 0, max(len(coarse_mesh.face_normals) - 1, 0))
            closest_normals = coarse_mesh.face_normals[tri_id].astype(np.float32)
        else:
            closest_normals = np.zeros_like(target_normals, dtype=np.float32)
    else:
        closest_normals = np.zeros_like(target_normals, dtype=np.float32)

    signed_offsets = np.sum((target_points - closest_on_coarse) * closest_normals, axis=1) \
        if len(target_points) else np.zeros((0,), dtype=np.float32)
    normal_error_deg = _normal_angles_deg(target_normals, closest_normals) \
        if len(target_points) else np.zeros((0,), dtype=np.float32)

    band_mask = target_to_coarse_dist <= cfg.band_radius_frac
    band_ratio = float(band_mask.mean()) if len(band_mask) else 0.0

    coarse_to_target_closest, coarse_to_target_dist, _ = _closest_on_mesh(target_mesh, coarse_points)
    chamfer_proxy = 0.0
    if len(target_to_coarse_dist) and len(coarse_to_target_dist):
        chamfer_proxy = float(target_to_coarse_dist.mean() + coarse_to_target_dist.mean())

    manifest = {
        "stage_name": cfg.stage_name,
        "coarse_mesh": str(Path(coarse_mesh_path)),
        "target_mesh": str(Path(target_mesh_path)),
        "normalization": norm_meta,
        "coarse_summary": asdict(coarse_summary),
        "target_summary": asdict(target_summary),
        "budget": asdict(budget),
        "objective": {
            "chamfer_weight": cfg.chamfer_weight,
            "normal_weight": cfg.normal_weight,
            "structure_weight": cfg.structure_weight,
            "extractor": cfg.extractor,
        },
        "detail_signal": {
            "band_radius_frac": cfg.band_radius_frac,
            "detail_band_ratio": band_ratio,
            "mean_target_to_coarse": float(target_to_coarse_dist.mean()) if len(target_to_coarse_dist) else 0.0,
            "p90_target_to_coarse": float(np.percentile(target_to_coarse_dist, 90)) if len(target_to_coarse_dist) else 0.0,
            "mean_normal_error_deg": float(normal_error_deg.mean()) if len(normal_error_deg) else 0.0,
            "p90_normal_error_deg": float(np.percentile(normal_error_deg, 90)) if len(normal_error_deg) else 0.0,
            "chamfer_proxy": chamfer_proxy,
        },
        "preserve_dualprim_structure": True,
    }

    np.savez_compressed(
        out_dir / "detail_refine_samples.npz",
        target_points=target_points.astype(np.float32),
        target_normals=target_normals.astype(np.float32),
        closest_on_coarse=closest_on_coarse.astype(np.float32),
        closest_normals=closest_normals.astype(np.float32),
        target_to_coarse_dist=target_to_coarse_dist.astype(np.float32),
        signed_offsets=signed_offsets.astype(np.float32),
        normal_error_deg=normal_error_deg.astype(np.float32),
        band_mask=band_mask.astype(np.bool_),
        coarse_points=coarse_points.astype(np.float32),
        coarse_normals=coarse_normals.astype(np.float32),
        coarse_to_target_closest=coarse_to_target_closest.astype(np.float32),
        coarse_to_target_dist=coarse_to_target_dist.astype(np.float32),
    )

    with open(out_dir / "detail_refine_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def _load_primitives_payload(path: str | Path) -> list[dict]:
    with open(path) as f:
        payload = json.load(f)
    return list(payload.get("primitives", []))


def _normalize_primitives(
    primitive_dicts: list[dict],
    centroid: np.ndarray,
    scale: float,
) -> list[dict]:
    normalized: list[dict] = []
    for idx, record in enumerate(primitive_dicts):
        psq_translation = (np.asarray(record["psq_translation"], dtype=np.float32) - centroid) * scale
        nsq_translation = (np.asarray(record["nsq_translation"], dtype=np.float32) - centroid) * scale
        psq_scale = np.asarray(record["psq_scale"], dtype=np.float32) * scale
        nsq_scale = np.asarray(record["nsq_scale"], dtype=np.float32) * scale
        psq_rotation = np.asarray(record["psq_rotation_rad"], dtype=np.float32)
        rotation = trimesh.transformations.euler_matrix(
            float(psq_rotation[0]),
            float(psq_rotation[1]),
            float(psq_rotation[2]),
            axes="sxyz",
        )[:3, :3].astype(np.float32)
        offset = np.abs(nsq_translation - psq_translation)
        support_scale = np.maximum(psq_scale, nsq_scale + offset)
        normalized.append(
            {
                "primitive_live_index": idx,
                "alpha": float(record["alpha"]),
                "theta": float(record["theta"]),
                "psq_translation": psq_translation,
                "nsq_translation": nsq_translation,
                "psq_scale": psq_scale,
                "nsq_scale": nsq_scale,
                "psq_shape": np.asarray(record["psq_shape"], dtype=np.float32),
                "nsq_shape": np.asarray(record["nsq_shape"], dtype=np.float32),
                "psq_rotation_rad": psq_rotation,
                "nsq_rotation_rad": np.asarray(record["nsq_rotation_rad"], dtype=np.float32),
                "rotation_matrix": rotation,
                "support_scale": np.clip(support_scale, 1e-5, None),
            }
        )
    return normalized


def _world_to_local(
    points: np.ndarray,
    translation: np.ndarray,
    rotation_matrix: np.ndarray,
) -> np.ndarray:
    centered = points - translation[None, :]
    return (rotation_matrix.T @ centered.T).T.astype(np.float32)


def prepare_primitive_local_refine_artifacts(
    coarse_mesh_path: str | Path,
    target_mesh_path: str | Path,
    primitives_json_path: str | Path,
    out_dir: str | Path,
    cfg: PrimitiveLocalRefineConfig | None = None,
) -> dict:
    """Prepare per-primitive local refinement artifacts.

    This is the first concrete Stage C handoff:
      - use the best coarse DualPrim mesh as the structural scaffold
      - assign narrow-band target samples to surviving primitives
      - save local-frame supervision bundles per primitive
      - emit explicit budgets so a later local refiner cannot overgrow
    """
    cfg = cfg or PrimitiveLocalRefineConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    coarse_mesh = _load_mesh(coarse_mesh_path)
    target_mesh = _load_mesh(target_mesh_path)
    coarse_mesh, target_mesh, norm_meta = _normalize_pair(coarse_mesh, target_mesh)
    centroid = np.asarray(norm_meta["target_centroid"], dtype=np.float32)
    target_scale = float(norm_meta["target_scale"])

    coarse_summary = _mesh_summary(coarse_mesh)
    target_summary = _mesh_summary(target_mesh)
    budget = build_primitive_local_budget(coarse_summary, cfg)

    primitive_dicts = _load_primitives_payload(primitives_json_path)
    primitives = [
        prim for prim in _normalize_primitives(primitive_dicts, centroid, target_scale)
        if prim["alpha"] >= cfg.min_alpha
    ]

    target_points, target_normals = _sample_surface(
        target_mesh, cfg.num_surface_samples, seed=cfg.seed,
    )
    closest_on_coarse, target_to_coarse_dist, tri_id = _closest_on_mesh(coarse_mesh, target_points)
    if len(target_points) and len(coarse_mesh.face_normals) and len(tri_id):
        tri_id = np.clip(tri_id, 0, max(len(coarse_mesh.face_normals) - 1, 0))
        closest_normals = coarse_mesh.face_normals[tri_id].astype(np.float32)
    else:
        closest_normals = np.zeros_like(target_normals, dtype=np.float32)
    signed_offsets = np.sum((target_points - closest_on_coarse) * closest_normals, axis=1) \
        if len(target_points) else np.zeros((0,), dtype=np.float32)
    normal_error_deg = _normal_angles_deg(target_normals, closest_normals) \
        if len(target_points) else np.zeros((0,), dtype=np.float32)

    band_mask = target_to_coarse_dist <= cfg.band_radius_frac
    band_indices = np.flatnonzero(band_mask)

    manifest = {
        "stage_name": cfg.stage_name,
        "coarse_mesh": str(Path(coarse_mesh_path)),
        "target_mesh": str(Path(target_mesh_path)),
        "primitives_json": str(Path(primitives_json_path)),
        "normalization": norm_meta,
        "coarse_summary": asdict(coarse_summary),
        "target_summary": asdict(target_summary),
        "budget": asdict(budget),
        "config": asdict(cfg),
        "detail_signal": {
            "band_radius_frac": cfg.band_radius_frac,
            "detail_band_ratio": float(band_mask.mean()) if len(band_mask) else 0.0,
            "mean_target_to_coarse": float(target_to_coarse_dist.mean()) if len(target_to_coarse_dist) else 0.0,
            "p90_target_to_coarse": float(np.percentile(target_to_coarse_dist, 90)) if len(target_to_coarse_dist) else 0.0,
            "mean_normal_error_deg": float(normal_error_deg.mean()) if len(normal_error_deg) else 0.0,
            "p90_normal_error_deg": float(np.percentile(normal_error_deg, 90)) if len(normal_error_deg) else 0.0,
        },
        "primitives": [],
        "preserve_dualprim_structure": True,
    }
    if not primitives or len(band_indices) == 0:
        with open(out_dir / "primitive_local_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        return manifest

    band_points = target_points[band_mask]
    assignment_scores = np.zeros((len(band_points), len(primitives)), dtype=np.float32)
    for prim_idx, prim in enumerate(primitives):
        local = _world_to_local(band_points, prim["psq_translation"], prim["rotation_matrix"])
        denom = np.clip(
            prim["support_scale"] * (1.0 + cfg.assignment_margin_frac),
            1e-5,
            None,
        )
        assignment_scores[:, prim_idx] = np.sqrt(np.sum((local / denom[None, :]) ** 2, axis=1))

    assigned_primitive = np.argmin(assignment_scores, axis=1)
    assigned_score = assignment_scores[np.arange(len(band_points)), assigned_primitive]
    assignable = assigned_score <= 1.0
    assignable_indices = band_indices[assignable]

    volume_weights = np.array(
        [float(np.prod(np.clip(prim["support_scale"], 1e-5, None))) for prim in primitives],
        dtype=np.float32,
    )
    total_volume = float(volume_weights.sum()) if len(volume_weights) else 0.0
    extra_vertices = max(budget.max_vertices - coarse_summary.vertices, 0)
    extra_faces = max(budget.max_faces - coarse_summary.faces, 0)

    rng = np.random.default_rng(cfg.seed)
    for prim_idx, prim in enumerate(primitives):
        local_mask = assignable & (assigned_primitive == prim_idx)
        point_indices = band_indices[local_mask]
        if len(point_indices) == 0:
            continue
        if len(point_indices) > cfg.max_samples_per_primitive:
            point_indices = np.sort(
                rng.choice(point_indices, size=cfg.max_samples_per_primitive, replace=False)
            )
        local_target_points = _world_to_local(
            target_points[point_indices], prim["psq_translation"], prim["rotation_matrix"],
        )
        local_target_normals = (prim["rotation_matrix"].T @ target_normals[point_indices].T).T.astype(np.float32)
        local_closest_points = _world_to_local(
            closest_on_coarse[point_indices], prim["psq_translation"], prim["rotation_matrix"],
        )
        local_closest_normals = (
            prim["rotation_matrix"].T @ closest_normals[point_indices].T
        ).T.astype(np.float32)

        volume_share = float(volume_weights[prim_idx] / total_volume) if total_volume > 0 else 0.0
        primitive_budget = {
            "max_vertices": max(128, int(round(extra_vertices * volume_share))) if extra_vertices > 0 else 0,
            "max_faces": max(128, int(round(extra_faces * volume_share))) if extra_faces > 0 else 0,
            "max_displacement": float(cfg.max_displacement_frac * np.max(prim["support_scale"])),
        }

        artifact_name = f"primitive_{prim['primitive_live_index']:03d}_samples.npz"
        np.savez_compressed(
            out_dir / artifact_name,
            target_points_world=target_points[point_indices].astype(np.float32),
            target_points_local=local_target_points.astype(np.float32),
            target_normals_world=target_normals[point_indices].astype(np.float32),
            target_normals_local=local_target_normals.astype(np.float32),
            closest_points_world=closest_on_coarse[point_indices].astype(np.float32),
            closest_points_local=local_closest_points.astype(np.float32),
            closest_normals_world=closest_normals[point_indices].astype(np.float32),
            closest_normals_local=local_closest_normals.astype(np.float32),
            target_to_coarse_dist=target_to_coarse_dist[point_indices].astype(np.float32),
            signed_offsets=signed_offsets[point_indices].astype(np.float32),
            normal_error_deg=normal_error_deg[point_indices].astype(np.float32),
        )

        manifest["primitives"].append(
            {
                "primitive_live_index": prim["primitive_live_index"],
                "alpha": prim["alpha"],
                "theta": prim["theta"],
                "sample_count": int(len(point_indices)),
                "budget": primitive_budget,
                "artifact": artifact_name,
                "assignment_score_mean": float(assignment_scores[local_mask, prim_idx].mean()),
                "assignment_score_p90": float(np.percentile(assignment_scores[local_mask, prim_idx], 90)),
                "mean_target_to_coarse": float(target_to_coarse_dist[point_indices].mean()),
                "mean_normal_error_deg": float(normal_error_deg[point_indices].mean()),
                "psq_translation_norm": prim["psq_translation"].tolist(),
                "psq_scale_norm": prim["psq_scale"].tolist(),
                "nsq_scale_norm": prim["nsq_scale"].tolist(),
                "support_scale_norm": prim["support_scale"].tolist(),
                "psq_rotation_rad": prim["psq_rotation_rad"].tolist(),
            }
        )

    manifest["summary"] = {
        "num_candidate_primitives": len(primitives),
        "num_local_artifacts": len(manifest["primitives"]),
        "assigned_band_points": int(len(assignable_indices)),
        "unassigned_band_points": int(len(band_indices) - len(assignable_indices)),
    }

    with open(out_dir / "primitive_local_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest
