"""Build visual previews for Stage C primitive-local refinement."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from trimesh.smoothing import filter_taubin
import torch

_CANDIDATE_ROOTS = [
    "/workspace/clearmesh",
    str(Path(__file__).resolve().parents[2]),
]
for _root in _CANDIDATE_ROOTS:
    if os.path.isdir(_root) and _root not in sys.path:
        sys.path.insert(0, _root)

from clearmesh.dualprim import DualPrimConfig, export_scene
from clearmesh.dualprim.io import load_scene_from_json
from clearmesh.dualprim.local_refine import LocalDisplacementMLP


def _load_mesh(path: str | Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        if not mesh.geometry:
            return trimesh.Trimesh()
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    return mesh


def _normalize_vertices(vertices_world: np.ndarray, normalization: dict) -> np.ndarray:
    centroid = np.asarray(normalization["target_centroid"], dtype=np.float32)
    scale = float(normalization["target_scale"])
    return (vertices_world - centroid[None, :]) * scale


def _denormalize_vertices(vertices_norm: np.ndarray, normalization: dict) -> np.ndarray:
    centroid = np.asarray(normalization["target_centroid"], dtype=np.float32)
    scale = float(normalization["target_scale"])
    return vertices_norm / max(scale, 1e-8) + centroid[None, :]


def _sample_mesh_points(mesh: trimesh.Trimesh, count: int, seed: int) -> np.ndarray:
    if len(mesh.faces) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    rng = np.random.default_rng(seed)
    points, _ = trimesh.sample.sample_surface(mesh, count, seed=rng)
    return np.asarray(points, dtype=np.float32)


def _rotation_matrix_xyz(rot_xyz_rad: list[float] | np.ndarray) -> np.ndarray:
    ax, ay, az = [float(v) for v in rot_xyz_rad]
    cx, cy, cz = np.cos([ax, ay, az])
    sx, sy, sz = np.sin([ax, ay, az])
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    return (rz @ ry @ rx).astype(np.float32)


def _plot_point_cloud(ax, points, color, title, s=0.4):
    if len(points):
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=s, c=[color], linewidths=0, alpha=0.9)
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()


def _equal_axes(ax, points: np.ndarray):
    if len(points) == 0:
        return
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins)) / 2.0
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _render_compare(target_pts, coarse_pts, preview_pts, out_path: Path, titles=None):
    titles = titles or ["target", "r6 / Stage B mesh", "Stage C preview"]
    non_empty = [arr for arr in (target_pts, coarse_pts, preview_pts) if len(arr)]
    combined = np.concatenate(non_empty, axis=0) if non_empty else np.zeros((0, 3), dtype=np.float32)
    fig = plt.figure(figsize=(13, 7))
    views = [(20, 35), (12, -60)]
    clouds = [target_pts, coarse_pts, preview_pts]
    colors = ["#5b6472", "#6f8fb7", "#c96f4b"]
    for row, (elev, azim) in enumerate(views):
        for col, (pts, title, color) in enumerate(zip(clouds, titles, colors)):
            ax = fig.add_subplot(2, 3, row * 3 + col + 1, projection="3d")
            _plot_point_cloud(ax, pts, color, title)
            _equal_axes(ax, combined)
            ax.view_init(elev=elev, azim=azim)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _crop_points(points: np.ndarray, bounds: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    if len(points) == 0:
        return points
    mins, maxs = bounds
    mask = np.all((points >= mins[None, :]) & (points <= maxs[None, :]), axis=1)
    return points[mask]


def _render_primitive_map(base_pts, primitive_groups, out_path: Path):
    fig = plt.figure(figsize=(12, 7))
    views = [(20, 35), (12, -60)]
    cmap = matplotlib.colormaps.get_cmap("tab10")
    combined_groups = [g["points"] for g in primitive_groups if len(g["points"])]
    combined = np.concatenate(([base_pts] + combined_groups), axis=0) if combined_groups else base_pts
    for row, (elev, azim) in enumerate(views):
        ax = fig.add_subplot(1, 2, row + 1, projection="3d")
        if len(base_pts):
            ax.scatter(base_pts[:, 0], base_pts[:, 1], base_pts[:, 2], s=0.15, c=["#cfd4dd"], linewidths=0, alpha=0.22)
        for i, group in enumerate(primitive_groups):
            pts = group["points"]
            if len(pts) == 0:
                continue
            color = cmap(i % 10)
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.8, c=[color], linewidths=0, alpha=0.92)
            centroid = pts.mean(axis=0)
            ax.text(centroid[0], centroid[1], centroid[2], str(group["primitive_live_index"]), color=color, fontsize=8)
        ax.set_title(f"refined primitive map ({elev}°, {azim}°)", fontsize=10)
        ax.set_axis_off()
        _equal_axes(ax, combined)
        ax.view_init(elev=elev, azim=azim)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _region_label(points_norm: np.ndarray) -> str:
    if len(points_norm) == 0:
        return "unknown"
    centroid = points_norm.mean(axis=0)
    x, y, z = centroid.tolist()
    if z < -0.18:
        return "lens/front"
    if z > 0.12:
        return "rear/body"
    if y > 0.10:
        return "top/body"
    if x > 0.10:
        return "side/grip"
    if x < -0.10:
        return "side/body"
    return "body/core"


def _load_refiners(
    primitive_manifest: dict,
    local_summary: dict,
    local_refine_dir: Path,
    device: str,
    focus_ids: set[int] | None,
) -> dict[int, dict]:
    refiners = {}
    for summary in local_summary["summaries"]:
        if "prediction_path" not in summary:
            continue
        primitive_id = int(summary["primitive_live_index"])
        if focus_ids is not None and primitive_id not in focus_ids:
            continue
        primitive_record = next(
            p for p in primitive_manifest["primitives"] if int(p["primitive_live_index"]) == primitive_id
        )
        checkpoint = torch.load(local_refine_dir / summary["refiner_path"], map_location=device)
        model_cfg = checkpoint["config"]
        model = LocalDisplacementMLP(
            hidden_dim=int(model_cfg["hidden_dim"]),
            layers=int(model_cfg["layers"]),
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        refiners[primitive_id] = {
            "summary": summary,
            "primitive_record": primitive_record,
            "checkpoint": checkpoint,
            "model": model,
        }
    return refiners


def _displace_mesh_with_refiner(
    mesh_world: trimesh.Trimesh,
    primitive_record: dict,
    checkpoint: dict,
    model: LocalDisplacementMLP,
    normalization: dict,
    assignment_margin_frac: float,
    preview_scale: float,
    device: str,
) -> trimesh.Trimesh:
    mesh = mesh_world.copy()
    if len(mesh.vertices) == 0:
        return mesh
    vertices_norm = _normalize_vertices(np.asarray(mesh.vertices, dtype=np.float32), normalization)
    vertex_normals_norm = np.asarray(mesh.vertex_normals, dtype=np.float32)
    translation = np.asarray(primitive_record["psq_translation_norm"], dtype=np.float32)
    support_scale = np.asarray(primitive_record["support_scale_norm"], dtype=np.float32)
    rot = _rotation_matrix_xyz(primitive_record["psq_rotation_rad"])
    local = (rot.T @ (vertices_norm - translation[None, :]).T).T.astype(np.float32)
    denom = np.clip(support_scale * (1.0 + assignment_margin_frac), 1e-5, None)
    score = np.sqrt(np.sum((local / denom[None, :]) ** 2, axis=1))
    weight = np.clip(1.0 - score, 0.0, 1.0) ** 2
    model_in = torch.from_numpy(local / np.clip(support_scale[None, :], 1e-5, None)).to(device)
    with torch.no_grad():
        disp = (
            model(model_in).detach().cpu().numpy().astype(np.float32)
            * float(checkpoint["max_displacement"])
            * float(preview_scale)
        )
    disp *= weight
    displaced_norm = vertices_norm + vertex_normals_norm * disp[:, None]
    mesh.vertices = _denormalize_vertices(displaced_norm, normalization)
    return mesh


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-mesh", required=True)
    ap.add_argument("--coarse-mesh", required=True)
    ap.add_argument("--primitive-manifest", required=True)
    ap.add_argument("--local-refine-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--mesh-samples", type=int, default=28000)
    ap.add_argument("--focus-primitive-ids", default=None,
                    help="Comma-separated primitive ids to visualize/emphasize.")
    ap.add_argument("--preview-scale", type=float, default=1.0,
                    help="Scale the learned local displacements for inspection.")
    ap.add_argument("--smooth-iters", type=int, default=0)
    ap.add_argument("--smooth-lambda", type=float, default=0.5)
    ap.add_argument("--smooth-nu", type=float, default=-0.53)
    ap.add_argument("--primitives-json", default=None,
                    help="If set, build a true per-primitive displaced Boolean preview.")
    ap.add_argument("--tessellation-resolution", type=int, default=48)
    ap.add_argument("--export-alpha-threshold", type=float, default=0.5)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.primitive_manifest) as f:
        primitive_manifest = json.load(f)
    with open(Path(args.local_refine_dir) / "local_refine_summary.json") as f:
        local_summary = json.load(f)

    focus_ids = None
    if args.focus_primitive_ids:
        focus_ids = {int(tok.strip()) for tok in args.focus_primitive_ids.split(",") if tok.strip()}

    normalization = primitive_manifest["normalization"]
    assignment_margin_frac = float(primitive_manifest["config"]["assignment_margin_frac"])
    local_refine_dir = Path(args.local_refine_dir)
    refiners = _load_refiners(
        primitive_manifest,
        local_summary,
        local_refine_dir,
        args.device,
        focus_ids,
    )

    target_mesh = _load_mesh(args.target_mesh)
    coarse_mesh_world = _load_mesh(args.coarse_mesh)
    preview_mode = "global_vertex_preview"

    if args.primitives_json:
        preview_mode = "per_primitive_boolean_preview"
        config = DualPrimConfig()
        config.export_alpha_threshold = args.export_alpha_threshold
        config.tessellation_resolution = args.tessellation_resolution
        scene = load_scene_from_json(args.primitives_json, config, device="cpu", pad_to_K=config.num_primitives_init)
        keep_idx = np.where(
            scene.alive.detach().cpu().numpy()
            & (scene.alpha().detach().cpu().numpy() >= config.export_alpha_threshold)
        )[0]
        _, per_prim = export_scene(scene, config, union_all=False)
        displaced_meshes = []
        for primitive_id, mesh in zip(keep_idx.tolist(), per_prim):
            if primitive_id in refiners:
                entry = refiners[primitive_id]
                mesh = _displace_mesh_with_refiner(
                    mesh,
                    entry["primitive_record"],
                    entry["checkpoint"],
                    entry["model"],
                    normalization,
                    assignment_margin_frac,
                    args.preview_scale,
                    args.device,
                )
            displaced_meshes.append(mesh)
        preview_mesh = trimesh.util.concatenate(displaced_meshes) if displaced_meshes else trimesh.Trimesh()
    else:
        coarse_mesh_norm = coarse_mesh_world.copy()
        coarse_mesh_norm.vertices = _normalize_vertices(np.asarray(coarse_mesh_world.vertices, dtype=np.float32), normalization)
        coarse_vertex_normals = np.asarray(coarse_mesh_norm.vertex_normals, dtype=np.float32)
        vertices_norm = np.asarray(coarse_mesh_norm.vertices, dtype=np.float32)
        total_weight = np.zeros((len(vertices_norm),), dtype=np.float32)
        total_disp = np.zeros((len(vertices_norm),), dtype=np.float32)
        for primitive_id, entry in refiners.items():
            primitive_record = entry["primitive_record"]
            checkpoint = entry["checkpoint"]
            model = entry["model"]
            translation = np.asarray(primitive_record["psq_translation_norm"], dtype=np.float32)
            support_scale = np.asarray(primitive_record["support_scale_norm"], dtype=np.float32)
            rot = _rotation_matrix_xyz(primitive_record["psq_rotation_rad"])
            local = (rot.T @ (vertices_norm - translation[None, :]).T).T.astype(np.float32)
            denom = np.clip(support_scale * (1.0 + assignment_margin_frac), 1e-5, None)
            score = np.sqrt(np.sum((local / denom[None, :]) ** 2, axis=1))
            weight = np.clip(1.0 - score, 0.0, 1.0) ** 2
            model_in = torch.from_numpy(local / np.clip(support_scale[None, :], 1e-5, None)).to(args.device)
            with torch.no_grad():
                disp = (
                    model(model_in).detach().cpu().numpy().astype(np.float32)
                    * float(checkpoint["max_displacement"])
                    * float(args.preview_scale)
                )
            total_weight += weight
            total_disp += weight * disp
        avg_disp = np.divide(total_disp, np.clip(total_weight, 1e-6, None))
        avg_disp[total_weight < 1e-5] = 0.0
        preview_vertices_norm = vertices_norm + coarse_vertex_normals * avg_disp[:, None]
        preview_mesh = coarse_mesh_world.copy()
        preview_mesh.vertices = _denormalize_vertices(preview_vertices_norm, normalization)

    if args.smooth_iters > 0 and len(preview_mesh.faces):
        filter_taubin(
            preview_mesh,
            lamb=float(args.smooth_lambda),
            nu=float(args.smooth_nu),
            iterations=int(args.smooth_iters),
        )
    preview_mesh.export(out_dir / "stage_c_preview.glb")

    target_pts = _sample_mesh_points(target_mesh, args.mesh_samples, seed=0)
    coarse_pts = _sample_mesh_points(coarse_mesh_world, args.mesh_samples, seed=1)
    preview_pts = _sample_mesh_points(preview_mesh, args.mesh_samples, seed=2)
    _render_compare(target_pts, coarse_pts, preview_pts, out_dir / "r6_vs_stagec_preview.png")

    primitive_groups = []
    lens_points = []
    for primitive_id, entry in refiners.items():
        summary = entry["summary"]
        data = np.load(local_refine_dir / summary["prediction_path"])
        points = np.asarray(data["refined_points_world"], dtype=np.float32)
        points_norm = _normalize_vertices(points, normalization)
        region_guess = _region_label(points_norm)
        primitive_groups.append(
            {
                "primitive_live_index": primitive_id,
                "points": points,
                "region_guess": region_guess,
                "improvement_mean": float(summary["improvement_mean"]),
            }
        )
        if region_guess == "lens/front":
            lens_points.append(points)
    primitive_groups.sort(key=lambda item: item["improvement_mean"], reverse=True)
    _render_primitive_map(target_pts, primitive_groups, out_dir / "primitive_local_map.png")

    if lens_points:
        lens_concat = np.concatenate(lens_points, axis=0)
        mins = lens_concat.min(axis=0)
        maxs = lens_concat.max(axis=0)
        margin = np.maximum((maxs - mins) * 0.35, 1e-3)
        crop_bounds = (mins - margin, maxs + margin)
        _render_compare(
            _crop_points(target_pts, crop_bounds),
            _crop_points(coarse_pts, crop_bounds),
            _crop_points(preview_pts, crop_bounds),
            out_dir / "lens_crop_compare.png",
            titles=["target lens crop", "r6 lens crop", "Stage C lens crop"],
        )

    summary_payload = {
        "stage_c_preview_mesh": str(out_dir / "stage_c_preview.glb"),
        "compare_image": str(out_dir / "r6_vs_stagec_preview.png"),
        "primitive_map_image": str(out_dir / "primitive_local_map.png"),
        "lens_crop_image": str(out_dir / "lens_crop_compare.png") if lens_points else None,
        "preview_mode": preview_mode,
        "primitive_regions": [
            {
                "primitive_live_index": p["primitive_live_index"],
                "region_guess": p["region_guess"],
                "improvement_mean": p["improvement_mean"],
            }
            for p in primitive_groups
        ],
    }
    with open(out_dir / "preview_summary.json", "w") as f:
        json.dump(summary_payload, f, indent=2)
    print(f"[preview-local] mesh={summary_payload['stage_c_preview_mesh']}")
    print(f"[preview-local] compare={summary_payload['compare_image']}")
    print(f"[preview-local] primitive_map={summary_payload['primitive_map_image']}")
    if summary_payload["lens_crop_image"]:
        print(f"[preview-local] lens_crop={summary_payload['lens_crop_image']}")


if __name__ == "__main__":
    main()
