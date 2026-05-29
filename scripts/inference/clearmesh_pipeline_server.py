#!/usr/bin/env python3
"""Small remote inference bridge for the ClearMesh product UI.

This is intentionally simple: one GPU host, local JSON job files, subprocess
stage execution, and artifact URLs. It is enough for product testing while the
production queue/worker layer is hardened.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import threading
import urllib.request
from uuid import uuid4

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

REPO_ROOT = Path(os.getenv("CLEARMESH_REPO_ROOT", Path(__file__).resolve().parents[2]))


def _normalize_reference_subject(prompt: str) -> str:
    subject = " ".join(str(prompt or "").split()).strip(" ,.;:")
    # We add these constraints ourselves. Strip duplicate leading phrasing from
    # user prompts so the image model gets one clean subject phrase.
    subject = re.sub(
        r"^(?:a\s+|an\s+|the\s+)?(?:single\s+|isolated\s+|single\s+isolated\s+|one\s+)+",
        "",
        subject,
        flags=re.IGNORECASE,
    ).strip(" ,.;:")
    return subject


def build_text_to_image_prompt(prompt: str) -> str:
    raw = " ".join(str(prompt or "").split())
    if os.getenv("CLEARMESH_DISABLE_REFERENCE_PROMPT_REWRITE", "0") == "1":
        return raw
    subject = _normalize_reference_subject(raw) or "a clean production-ready 3D asset"
    return (
        f"single isolated {subject}, exactly one object, complete object, centered three-quarter product view, "
        "clean silhouette, matte neutral material, seamless plain light background, studio lighting, "
        "object floating in empty studio space, no support surface, no tabletop, no props, no duplicate objects, "
        "no collage, no text, no logo, no cropping, no black glossy silhouette"
    )


def build_text_to_image_negative_prompt() -> str:
    return os.getenv(
        "CLEARMESH_TEXT_TO_IMAGE_NEGATIVE_PROMPT",
        "multiple objects, duplicate objects, collage, collection, grid, clutter, "
        "scene background, props, table, tabletop, wooden surface, floor, room, base, stand, "
        "text, logo, watermark, cropped, cut off, fragments, "
        "broken pieces, blurry, malformed, black glossy object"
    )


def inspect_reference_image(path: Path) -> dict:
    """Cheaply catch collage/multi-object references before expensive 3D inference."""
    try:
        from PIL import Image
        import numpy as np
    except Exception as exc:  # noqa: BLE001 - optional quality gate.
        return {"ok": True, "skipped": f"missing image deps: {exc}"}

    image = Image.open(path).convert("RGB")
    image.thumbnail((384, 384))
    arr = np.asarray(image).astype(np.int16)
    height, width = arr.shape[:2]
    patch = max(8, min(height, width) // 16)
    corners = np.concatenate(
        [
            arr[:patch, :patch].reshape(-1, 3),
            arr[:patch, -patch:].reshape(-1, 3),
            arr[-patch:, :patch].reshape(-1, 3),
            arr[-patch:, -patch:].reshape(-1, 3),
        ],
        axis=0,
    )
    background = np.median(corners, axis=0)
    image_area = float(height * width)
    threshold_reports = []

    def component_areas(mask: "np.ndarray") -> list[int]:
        seen = np.zeros((height, width), dtype=bool)
        areas: list[int] = []
        for y in range(height):
            for x in range(width):
                if not mask[y, x] or seen[y, x]:
                    continue
                stack = [(y, x)]
                seen[y, x] = True
                area = 0
                while stack:
                    cy, cx = stack.pop()
                    area += 1
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
                areas.append(area)
        return sorted(areas, reverse=True)

    diff = np.sqrt(((arr - background) ** 2).sum(axis=2))
    for threshold in (30, 42, 54, 66):
        areas = component_areas(diff > threshold)
        large = [area for area in areas if area / image_area >= 0.015]
        threshold_reports.append(
            {
                "threshold": threshold,
                "large_components": len(large),
                "largest_area_ratio": round((large[0] / image_area) if large else 0.0, 4),
                "second_area_ratio": round((large[1] / image_area) if len(large) > 1 else 0.0, 4),
                "large_area_ratios": [round(area / image_area, 4) for area in large[:6]],
            }
        )

    multi_thresholds = sum(1 for report in threshold_reports if report["large_components"] >= 3)
    split_thresholds = sum(
        1
        for report in threshold_reports
        if report["large_components"] >= 2 and report["largest_area_ratio"] < 0.65
    )
    second_object_thresholds = sum(1 for report in threshold_reports if report["second_area_ratio"] >= 0.06)
    # Low thresholds often split one glossy object into body/shadow/highlight islands.
    # Require repeated evidence across thresholds before rejecting the expensive 3D path.
    likely_collage = multi_thresholds >= 2 or split_thresholds >= 3 or second_object_thresholds >= 2
    return {
        "ok": not likely_collage,
        "likely_collage": likely_collage,
        "thresholds": threshold_reports,
    }


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def load_mesh_for_qc(path: Path):
    import trimesh

    loaded = trimesh.load(path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = [geom for geom in loaded.geometry.values() if isinstance(geom, trimesh.Trimesh)]
        if not meshes:
            return trimesh.Trimesh()
        return trimesh.util.concatenate(meshes)
    return loaded


def _apply_mesh_qc_resource_limits() -> None:
    """Best-effort child-process limits so malformed meshes cannot OOM the host."""
    memory_mb = _env_int("CLEARMESH_MESH_QC_MEMORY_MB", 4096)
    if memory_mb <= 0:
        return
    try:
        import resource

        limit = int(memory_mb) * 1024 * 1024
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        hard_limit = hard if hard != resource.RLIM_INFINITY else limit
        resource.setrlimit(resource.RLIMIT_AS, (min(limit, hard_limit), hard_limit))
    except Exception:
        # macOS/Linux/container support differs; subprocess isolation still helps.
        return


def _run_mesh_qc_child(args: list[str], timeout_seconds: int) -> dict:
    env = os.environ.copy()
    env["CLEARMESH_MESH_QC_CHILD"] = "1"
    try:
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=max(1, timeout_seconds),
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "load_error": f"mesh QC timed out after {timeout_seconds}s"}
    except Exception as exc:  # noqa: BLE001 - QC should never crash serving.
        return {"ok": False, "load_error": f"mesh QC subprocess failed: {type(exc).__name__}: {exc}"}

    stdout = (proc.stdout or "").strip()
    if proc.returncode != 0:
        stderr_tail = (proc.stderr or "").strip()[-800:]
        return {"ok": False, "load_error": f"mesh QC subprocess exited {proc.returncode}", "stderr_tail": stderr_tail}
    try:
        parsed = json.loads(stdout)
    except json.JSONDecodeError:
        return {"ok": False, "load_error": "mesh QC subprocess returned invalid JSON", "stdout_tail": stdout[-800:]}
    return parsed if isinstance(parsed, dict) else {"ok": False, "load_error": "mesh QC subprocess returned non-object JSON"}


def _mesh_basic_stats_direct(path: Path) -> dict:
    try:
        import numpy as np

        mesh = load_mesh_for_qc(path)
        vertices = int(len(mesh.vertices))
        faces = int(len(mesh.faces))
        extents = np.asarray(mesh.extents if vertices else [0.0, 0.0, 0.0], dtype=float)
        finite_extents = extents[np.isfinite(extents)]
        positive_extents = finite_extents[finite_extents > 1e-8]
        max_extent = float(positive_extents.max()) if positive_extents.size else 0.0
        min_extent = float(positive_extents.min()) if positive_extents.size else 0.0
        extent_ratio = float(max_extent / max(min_extent, 1e-8)) if max_extent else math.inf

        def split_face_counts(candidate) -> tuple[int | None, list[int]]:
            try:
                parts = candidate.split(only_watertight=False) if faces else []
                counts = sorted((int(len(part.faces)) for part in parts), reverse=True)
                return int(len(parts)), counts
            except Exception:
                return None, []

        raw_components, raw_component_face_counts = split_face_counts(mesh)

        # GLB exporters commonly duplicate vertices at UV/material/normal seams.
        # For product QC we care whether the visible surface is fragmented, not
        # whether every seam shares the same vertex index, so component checks use
        # a topology copy with seam duplicates merged by position.
        component_mesh = mesh.copy()
        component_vertices = vertices
        merge_digits = _env_int("CLEARMESH_MESH_QC_COMPONENT_MERGE_DIGITS", 6)
        try:
            component_mesh.merge_vertices(merge_tex=True, merge_norm=True, digits_vertex=merge_digits)
            component_mesh.remove_unreferenced_vertices()
            component_vertices = int(len(component_mesh.vertices))
        except Exception:
            component_mesh = mesh
        components, component_face_counts = split_face_counts(component_mesh)
        return {
            "ok": bool(vertices and faces),
            "vertices": vertices,
            "faces": faces,
            "watertight": bool(getattr(mesh, "is_watertight", False)),
            "raw_components": raw_components,
            "raw_largest_component_face_ratio": round(raw_component_face_counts[0] / max(faces, 1), 4)
            if raw_component_face_counts
            else 0.0,
            "raw_tiny_component_count": int(
                sum(1 for count in raw_component_face_counts if count / max(faces, 1) < 0.01)
            ),
            "raw_component_face_counts_top8": raw_component_face_counts[:8],
            "components": components,
            "largest_component_face_ratio": round(component_face_counts[0] / max(faces, 1), 4) if component_face_counts else 0.0,
            "tiny_component_count": int(sum(1 for count in component_face_counts if count / max(faces, 1) < 0.01)),
            "component_face_counts_top8": component_face_counts[:8],
            "component_vertices": component_vertices,
            "component_merge_digits": merge_digits,
            "extents": [round(float(v), 6) for v in extents.tolist()],
            "bbox_diag": round(float(np.linalg.norm(extents)), 6),
            "extent_ratio": round(extent_ratio, 4) if math.isfinite(extent_ratio) else "inf",
        }
    except Exception as exc:  # noqa: BLE001 - QC should never crash the serving path.
        return {"ok": False, "load_error": f"{type(exc).__name__}: {exc}"}


def mesh_basic_stats(path: Path) -> dict:
    if os.getenv("CLEARMESH_MESH_QC_SUBPROCESS", "1") == "0" or os.getenv("CLEARMESH_MESH_QC_CHILD") == "1":
        return _mesh_basic_stats_direct(path)
    return _run_mesh_qc_child(
        ["--mesh-stats-json", str(path)],
        _env_int("CLEARMESH_MESH_QC_TIMEOUT_SECONDS", 45),
    )


def _normalized_surface_chamfer_direct(source_path: Path, candidate_path: Path, samples: int) -> float | None:
    if samples <= 0:
        return None
    try:
        import numpy as np
        from scipy.spatial import cKDTree
        import trimesh

        source = load_mesh_for_qc(source_path)
        candidate = load_mesh_for_qc(candidate_path)
        if len(source.faces) == 0 or len(candidate.faces) == 0:
            return None
        count = max(128, samples)
        source_points, _ = trimesh.sample.sample_surface(source, count)
        candidate_points, _ = trimesh.sample.sample_surface(candidate, count)
        source_tree = cKDTree(source_points)
        candidate_tree = cKDTree(candidate_points)
        source_to_candidate = candidate_tree.query(source_points, workers=-1)[0]
        candidate_to_source = source_tree.query(candidate_points, workers=-1)[0]
        diag = float(np.linalg.norm(np.asarray(source.extents, dtype=float))) or 1.0
        return float((source_to_candidate.mean() + candidate_to_source.mean()) * 0.5 / max(diag, 1e-8))
    except Exception:
        return None


def normalized_surface_chamfer(source_path: Path, candidate_path: Path, samples: int) -> float | None:
    if samples <= 0:
        return None
    if os.getenv("CLEARMESH_MESH_QC_SUBPROCESS", "1") == "0" or os.getenv("CLEARMESH_MESH_QC_CHILD") == "1":
        return _normalized_surface_chamfer_direct(source_path, candidate_path, samples)
    result = _run_mesh_qc_child(
        [
            "--mesh-chamfer-json",
            "--source-mesh",
            str(source_path),
            "--candidate-mesh",
            str(candidate_path),
            "--samples",
            str(samples),
        ],
        _env_int("CLEARMESH_MESH_QC_TIMEOUT_SECONDS", 45),
    )
    if not result.get("ok"):
        return None
    value = result.get("chamfer_norm")
    return float(value) if isinstance(value, (float, int)) else None


def inspect_faceq_candidate(source_path: Path, candidate_path: Path) -> dict:
    source_stats = mesh_basic_stats(source_path)
    candidate_stats = mesh_basic_stats(candidate_path)
    reasons: list[str] = []
    if not candidate_stats.get("ok"):
        reasons.append("candidate mesh could not be loaded")
    min_faces = _env_int("CLEARMESH_FACEQ_ACCEPTANCE_MIN_FACES", 256)
    min_face_ratio = _env_float("CLEARMESH_FACEQ_ACCEPTANCE_MIN_FACE_RATIO", 0.06)
    max_extent_ratio = _env_float("CLEARMESH_FACEQ_ACCEPTANCE_MAX_EXTENT_RATIO", 35.0)
    max_chamfer_norm = _env_float("CLEARMESH_FACEQ_ACCEPTANCE_MAX_CHAMFER_NORM", 0.5)
    candidate_faces = int(candidate_stats.get("faces") or 0)
    source_faces = int(source_stats.get("faces") or 0)
    if candidate_faces < min_faces:
        reasons.append(f"candidate has too few faces ({candidate_faces} < {min_faces})")
    if source_faces and candidate_faces / max(source_faces, 1) < min_face_ratio:
        reasons.append(f"candidate face ratio too low ({candidate_faces / max(source_faces, 1):.3f} < {min_face_ratio:.3f})")
    extent_ratio = candidate_stats.get("extent_ratio")
    if extent_ratio == "inf" or (isinstance(extent_ratio, (float, int)) and float(extent_ratio) > max_extent_ratio):
        reasons.append(f"candidate bounding box is degenerate/needle-like (extent_ratio={extent_ratio})")
    samples = _env_int("CLEARMESH_FACEQ_ACCEPTANCE_SAMPLES", 2048)
    chamfer = normalized_surface_chamfer(source_path, candidate_path, samples)
    if chamfer is not None and chamfer > max_chamfer_norm:
        reasons.append(f"candidate moved too far from Trellis proxy (chamfer_norm={chamfer:.4f} > {max_chamfer_norm:.4f})")
    return {
        "accepted": not reasons,
        "reasons": reasons,
        "source_stats": source_stats,
        "candidate_stats": candidate_stats,
        "chamfer_norm": round(chamfer, 6) if chamfer is not None else None,
        "thresholds": {
            "min_faces": min_faces,
            "min_face_ratio": min_face_ratio,
            "max_extent_ratio": max_extent_ratio,
            "max_chamfer_norm": max_chamfer_norm,
            "samples": samples,
        },
    }


def inspect_trellis_candidate(mesh_path: Path) -> dict:
    """Gate Trellis outputs before FACE-Q sees them.

    A bad text-reference image can make Trellis produce a spiky, fragmented, or
    needle-like mesh. FACE-Q should not be blamed for those cases, and the app
    should retry the reference/Trellis stage before publishing a confusing
    fallback.
    """

    stats = mesh_basic_stats(mesh_path)
    reasons: list[str] = []
    min_faces = _env_int("CLEARMESH_TRELLIS_ACCEPTANCE_MIN_FACES", 1024)
    max_components = _env_int("CLEARMESH_TRELLIS_ACCEPTANCE_MAX_COMPONENTS", 24)
    max_tiny_components = _env_int("CLEARMESH_TRELLIS_ACCEPTANCE_MAX_TINY_COMPONENTS", 16)
    min_largest_ratio = _env_float("CLEARMESH_TRELLIS_ACCEPTANCE_MIN_LARGEST_COMPONENT_RATIO", 0.55)
    min_two_component_largest_ratio = _env_float("CLEARMESH_TRELLIS_ACCEPTANCE_MIN_TWO_COMPONENT_LARGEST_RATIO", 0.45)
    max_extent_ratio = _env_float("CLEARMESH_TRELLIS_ACCEPTANCE_MAX_EXTENT_RATIO", 45.0)
    min_bbox_diag = _env_float("CLEARMESH_TRELLIS_ACCEPTANCE_MIN_BBOX_DIAG", 0.05)
    if not stats.get("ok"):
        reasons.append("Trellis mesh could not be loaded")
    faces = int(stats.get("faces") or 0)
    if faces < min_faces:
        reasons.append(f"Trellis mesh has too few faces ({faces} < {min_faces})")
    components = stats.get("components")
    if isinstance(components, int) and components > max_components:
        reasons.append(f"Trellis mesh is too fragmented ({components} components > {max_components})")
    tiny_components = int(stats.get("tiny_component_count") or 0)
    if tiny_components > max_tiny_components:
        reasons.append(f"Trellis mesh has too much debris ({tiny_components} tiny components > {max_tiny_components})")
    largest_ratio = float(stats.get("largest_component_face_ratio") or 0.0)
    component_count = components if isinstance(components, int) else None
    required_largest_ratio = min_largest_ratio
    if component_count is not None and component_count <= 2 and tiny_components == 0:
        # A single coherent textured asset may split into body/handle or
        # inner/outer shells after seam merging. Keep rejecting many-part debris,
        # but do not reject a clean two-component mesh solely for being balanced.
        required_largest_ratio = min_two_component_largest_ratio
    if largest_ratio and largest_ratio < required_largest_ratio:
        reasons.append(f"Trellis mesh lacks a dominant object component ({largest_ratio:.3f} < {required_largest_ratio:.3f})")
    extent_ratio = stats.get("extent_ratio")
    if extent_ratio == "inf" or (isinstance(extent_ratio, (float, int)) and float(extent_ratio) > max_extent_ratio):
        reasons.append(f"Trellis mesh is degenerate/needle-like (extent_ratio={extent_ratio})")
    bbox_diag = float(stats.get("bbox_diag") or 0.0)
    if bbox_diag < min_bbox_diag:
        reasons.append(f"Trellis mesh bounding box is tiny/empty (bbox_diag={bbox_diag:.4f} < {min_bbox_diag:.4f})")

    # Larger, connected, non-degenerate meshes are better fallback candidates.
    face_score = min(1.0, faces / max(float(min_faces), 1.0))
    component_penalty = 0.0
    if isinstance(components, int):
        component_penalty += max(0.0, (components - 1) / max(float(max_components), 1.0))
    component_penalty += max(0.0, required_largest_ratio - largest_ratio)
    extent_penalty = 0.0
    if extent_ratio == "inf":
        extent_penalty = 1.0
    elif isinstance(extent_ratio, (float, int)):
        extent_penalty = max(0.0, (float(extent_ratio) - max_extent_ratio) / max(max_extent_ratio, 1.0))
    score = face_score + largest_ratio - component_penalty - extent_penalty
    return {
        "accepted": not reasons,
        "reasons": reasons,
        "stats": stats,
        "score": round(float(score), 6),
        "thresholds": {
            "min_faces": min_faces,
            "max_components": max_components,
            "max_tiny_components": max_tiny_components,
            "min_largest_component_ratio": min_largest_ratio,
            "min_two_component_largest_ratio": min_two_component_largest_ratio,
            "max_extent_ratio": max_extent_ratio,
            "min_bbox_diag": min_bbox_diag,
        },
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class PipelineRequest(BaseModel):
    prompt: str | None = None
    input_uri: str | None = None
    mode: str = Field(default="text_to_3d")
    output_formats: list[str] = Field(default_factory=lambda: ["glb", "obj"])
    quality_tier: str = "standard"
    faceq: bool | None = None
    pipeline: dict = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict)
    request_id: str | None = None


class PipelineServer:
    def __init__(self, *, work_root: Path, state_root: Path, checkpoint: Path, model_bundle: Path) -> None:
        self.work_root = work_root
        self.state_root = state_root
        self.checkpoint = checkpoint
        self.model_bundle = model_bundle
        self.work_root.mkdir(parents=True, exist_ok=True)
        self.state_root.mkdir(parents=True, exist_ok=True)
        self.app = FastAPI(title="ClearMesh Pipeline Bridge", version="0.1.0")
        self._install_routes()

    def require_auth(self, authorization: str | None) -> None:
        token = os.getenv("CLEARMESH_PIPELINE_TOKEN", "").strip()
        if not token:
            return
        expected = f"Bearer {token}"
        if authorization != expected:
            raise HTTPException(status_code=401, detail="missing or invalid bridge token")

    def job_path(self, job_id: str) -> Path:
        return self.state_root / f"{job_id}.json"

    def job_dir(self, job_id: str) -> Path:
        return self.work_root / job_id

    def read_job(self, job_id: str) -> dict:
        path = self.job_path(job_id)
        if not path.exists():
            raise KeyError(job_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def write_job(self, job: dict) -> None:
        path = self.job_path(job["id"])
        path.write_text(json.dumps(job, indent=2, sort_keys=True), encoding="utf-8")

    def update_job(self, job_id: str, **updates) -> dict:
        job = self.read_job(job_id)
        job.update(updates)
        job["updated_at"] = utc_now()
        self.write_job(job)
        return job

    def _install_routes(self) -> None:
        @self.app.get("/healthz")
        def healthz() -> dict:
            return {
                "ok": True,
                "checkpoint_exists": self.checkpoint.exists(),
                "model_bundle": str(self.model_bundle),
                "pipeline": self.pipeline_config(),
            }

        @self.app.get("/v1/pipeline/config")
        def config(authorization: str | None = Header(default=None)) -> dict:
            self.require_auth(authorization)
            return self.pipeline_config()

        @self.app.post("/v1/pipeline/jobs", status_code=202)
        def create_job(payload: PipelineRequest, authorization: str | None = Header(default=None)) -> dict:
            self.require_auth(authorization)
            job_id = f"pipe_{uuid4().hex[:16]}"
            job = {
                "id": job_id,
                "remote_job_id": job_id,
                "request_id": payload.request_id,
                "status": "queued",
                "stage": "queued",
                "created_at": utc_now(),
                "updated_at": utc_now(),
                "request": payload.model_dump() if hasattr(payload, "model_dump") else payload.dict(),
                "artifacts": {},
                "events": [],
                "pipeline": "Text/Image -> Trellis -> FACE-Q -> Texture/UV -> Easy3E-ready",
                "status_url": f"/v1/pipeline/jobs/{job_id}",
            }
            self.write_job(job)
            thread = threading.Thread(target=self.run_job, args=(job_id,), daemon=True)
            thread.start()
            return job

        @self.app.get("/v1/pipeline/jobs/{job_id}")
        def get_job(job_id: str, authorization: str | None = Header(default=None)) -> dict:
            self.require_auth(authorization)
            try:
                return self.read_job(job_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail="job not found") from exc

        @self.app.get("/v1/pipeline/jobs/{job_id}/artifacts/{name}")
        def get_artifact(job_id: str, name: str, authorization: str | None = Header(default=None)) -> FileResponse:
            self.require_auth(authorization)
            try:
                job = self.read_job(job_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail="job not found") from exc
            artifact = job.get("artifacts", {}).get(name)
            if not artifact:
                raise HTTPException(status_code=404, detail="artifact not found")
            path = Path(artifact)
            if not path.exists() or not path.is_file():
                raise HTTPException(status_code=404, detail="artifact file not found")
            return FileResponse(path, filename=path.name)

    def easy3e_readiness(self) -> dict:
        requested = os.getenv("CLEARMESH_EASY3E_ENABLED", "0") == "1"
        status = {
            "requested": requested,
            "enabled": False,
            "ready": False,
            "model_dir": os.getenv("CLEARMESH_TRELLIS_MODEL", "microsoft/TRELLIS.2-4B"),
            "reason": "disabled by CLEARMESH_EASY3E_ENABLED",
        }
        if not requested:
            return status
        try:
            from scripts.product.run_easy3e_edit import (
                Easy3EPreflightError,
                resolve_model_dir,
                validate_easy3e_model_dir,
            )

            resolved = resolve_model_dir(status["model_dir"])
            validate_easy3e_model_dir(resolved)
            status.update(
                {
                    "enabled": True,
                    "ready": True,
                    "resolved_model_dir": str(resolved),
                    "reason": "ready",
                }
            )
        except Easy3EPreflightError as exc:
            status["reason"] = str(exc)
        except Exception as exc:  # noqa: BLE001 - config must remain available.
            status["reason"] = f"{type(exc).__name__}: {exc}"
        return status

    def pipeline_config(self) -> dict:
        easy3e = self.easy3e_readiness()
        return {
            "text_to_image_model": os.getenv("CLEARMESH_TEXT_TO_IMAGE_MODEL", "HiDream-ai/HiDream-O1-Image-Dev-2604"),
            "text_to_image_backend": os.getenv("CLEARMESH_TEXT_TO_IMAGE_BACKEND", "hidream"),
            "text_to_image_fallback_model": os.getenv("CLEARMESH_TEXT_TO_IMAGE_FALLBACK_MODEL", "stabilityai/stable-diffusion-xl-base-1.0"),
            "trellis_model": os.getenv("CLEARMESH_TRELLIS_MODEL", "microsoft/TRELLIS.2-4B"),
            "faceq_checkpoint": str(self.checkpoint),
            "faceq_checkpoint_exists": self.checkpoint.exists(),
            "faceq_inference": {
                "generation_max_faces": os.getenv("CLEARMESH_FACEQ_GENERATION_MAX_FACES", "checkpoint"),
                "point_samples": os.getenv("CLEARMESH_FACEQ_POINT_SAMPLES", "checkpoint"),
                "decode_mode": os.getenv("CLEARMESH_FACEQ_DECODE_MODE", "boundary_edge"),
                "constraint_top_k": os.getenv("CLEARMESH_FACEQ_CONSTRAINT_TOP_K", "24"),
                "boundary_budget": os.getenv("CLEARMESH_FACEQ_NO_BOUNDARY_BUDGET", "0") != "1",
                "vertex_link_constraint": os.getenv("CLEARMESH_FACEQ_NO_VERTEX_LINK_CONSTRAINT", "0") != "1",
            },
            "faceq_acceptance": {
                "enabled": os.getenv("CLEARMESH_FACEQ_ACCEPTANCE_ENABLE", "1") != "0",
                "min_faces": _env_int("CLEARMESH_FACEQ_ACCEPTANCE_MIN_FACES", 256),
                "min_face_ratio": _env_float("CLEARMESH_FACEQ_ACCEPTANCE_MIN_FACE_RATIO", 0.06),
                "max_extent_ratio": _env_float("CLEARMESH_FACEQ_ACCEPTANCE_MAX_EXTENT_RATIO", 35.0),
                "max_chamfer_norm": _env_float("CLEARMESH_FACEQ_ACCEPTANCE_MAX_CHAMFER_NORM", 0.5),
            },
            "faceq_retry": {
                "enabled": os.getenv("CLEARMESH_FACEQ_RETRY_ON_REJECT", "1") == "1",
                "generation_max_faces": os.getenv("CLEARMESH_FACEQ_RETRY_GENERATION_MAX_FACES", "768"),
                "point_samples": os.getenv("CLEARMESH_FACEQ_RETRY_POINT_SAMPLES", "16384"),
                "constraint_top_k": os.getenv("CLEARMESH_FACEQ_RETRY_CONSTRAINT_TOP_K", "24"),
            },
            "faceq_timeout_seconds": _env_int("CLEARMESH_FACEQ_TIMEOUT_SECONDS", 360),
            "trellis_timeout_seconds": _env_int("CLEARMESH_TRELLIS_TIMEOUT_SECONDS", 1200),
            "trellis_acceptance": {
                "max_attempts": _env_int("CLEARMESH_TRELLIS_MAX_ATTEMPTS", 2),
                "min_faces": _env_int("CLEARMESH_TRELLIS_ACCEPTANCE_MIN_FACES", 1024),
                "max_components": _env_int("CLEARMESH_TRELLIS_ACCEPTANCE_MAX_COMPONENTS", 24),
                "min_largest_component_ratio": _env_float("CLEARMESH_TRELLIS_ACCEPTANCE_MIN_LARGEST_COMPONENT_RATIO", 0.55),
                "min_two_component_largest_ratio": _env_float("CLEARMESH_TRELLIS_ACCEPTANCE_MIN_TWO_COMPONENT_LARGEST_RATIO", 0.45),
            },
            "trellis_only_preview_enabled": os.getenv("CLEARMESH_ALLOW_TRELLIS_ONLY", "0") == "1",
            "publish_trellis_preview_before_faceq": os.getenv("CLEARMESH_PUBLISH_TRELLIS_PREVIEW_BEFORE_FACEQ", "1") == "1",
            "easy3e_enabled": bool(easy3e.get("enabled")),
            "easy3e": easy3e,
            "modes": ["text_to_3d", "image_to_3d", "edit_text", "edit_image"],
            "textures": {
                "trellis_pbr": True,
                "easy3e_ctrl_adapter": bool(os.getenv("CLEARMESH_EASY3E_CTRL_ADAPTER_CHECKPOINT")),
                "postprocess_enabled": os.getenv("CLEARMESH_TEXTURE_UV_ENABLED", "1") != "0",
                "postprocess_mode": os.getenv("CLEARMESH_TEXTURE_UV_MODE", "auto"),
                "ai_texture_command": bool(os.getenv("CLEARMESH_TEXTURE_UV_COMMAND")),
                "blender_uv": os.getenv("CLEARMESH_TEXTURE_UV_USE_BLENDER", "0") == "1",
            },
            "dry_run": os.getenv("CLEARMESH_PIPELINE_DRY_RUN", "0") == "1",
        }

    def event(self, job_id: str, stage: str, status: str, detail: str | None = None) -> None:
        job = self.read_job(job_id)
        job["events"].append({"time": utc_now(), "stage": stage, "status": status, "detail": detail})
        job["stage"] = stage
        job["status"] = status if status in {"succeeded", "failed"} else "running"
        job["updated_at"] = utc_now()
        self.write_job(job)

    def run_command(self, job_id: str, stage: str, command: list[str], cwd: Path | None = None, timeout_seconds: int | None = None) -> None:
        root = self.job_dir(job_id)
        logs = root / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        stdout_path = logs / f"{stage}.stdout.log"
        stderr_path = logs / f"{stage}.stderr.log"
        self.event(job_id, stage, "running", " ".join(command))
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            proc = subprocess.Popen(
                command,
                cwd=str(cwd or REPO_ROOT),
                text=True,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            try:
                proc.communicate(timeout=timeout_seconds)
            except subprocess.TimeoutExpired as exc:
                try:
                    os.killpg(proc.pid, 15)
                    proc.wait(timeout=10)
                except Exception:
                    try:
                        os.killpg(proc.pid, 9)
                    except Exception:
                        pass
                self.event(job_id, stage, "failed", f"timeout={timeout_seconds}s; stdout={stdout_path}; stderr={stderr_path}")
                raise TimeoutError(f"{stage} timed out after {timeout_seconds}s") from exc
        if proc.returncode != 0:
            self.event(job_id, stage, "failed", f"exit={proc.returncode}; stdout={stdout_path}; stderr={stderr_path}")
            raise RuntimeError(f"{stage} failed with exit={proc.returncode}")
        self.event(job_id, stage, "running", f"complete; stdout={stdout_path}; stderr={stderr_path}")

    def fetch_uri(self, job_id: str, value: str, subdir: str, default_name: str | None = None) -> Path:
        root = self.job_dir(job_id)
        if value.startswith(("http://", "https://")):
            url_path = Path(value.split("?", 1)[0]).name
            dest = root / subdir / (url_path or default_name or "download.bin")
            dest.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(value, dest)  # noqa: S310 - user-configured bridge host.
            return dest
        if value.startswith("file://"):
            return Path(value.removeprefix("file://"))
        if not value.startswith("local://"):
            candidate = Path(value).expanduser()
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"could not resolve input URI: {value}")

    def metadata_value(self, request: dict, *keys: str) -> str | None:
        metadata = request.get("metadata") or {}
        pipeline = request.get("pipeline") or {}
        for source in (metadata, pipeline, request):
            for key in keys:
                value = source.get(key) if isinstance(source, dict) else None
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return None

    def text_to_image_command(self, *, prompt: str, output: Path, seed: int) -> list[str]:
        return [
            sys.executable,
            str(REPO_ROOT / "scripts/product/run_text_to_image.py"),
            "--prompt",
            prompt,
            "--negative-prompt",
            build_text_to_image_negative_prompt(),
            "--output",
            str(output),
            "--model",
            os.getenv("CLEARMESH_TEXT_TO_IMAGE_MODEL", "HiDream-ai/HiDream-O1-Image-Dev-2604"),
            "--backend",
            os.getenv("CLEARMESH_TEXT_TO_IMAGE_BACKEND", "hidream"),
            "--steps",
            os.getenv("CLEARMESH_TEXT_TO_IMAGE_STEPS", "4"),
            "--guidance-scale",
            os.getenv("CLEARMESH_TEXT_TO_IMAGE_GUIDANCE_SCALE", "0.0"),
            "--width",
            os.getenv("CLEARMESH_TEXT_TO_IMAGE_WIDTH", "1024"),
            "--height",
            os.getenv("CLEARMESH_TEXT_TO_IMAGE_HEIGHT", "1024"),
            "--model-type",
            os.getenv("CLEARMESH_HIDREAM_MODEL_TYPE", "dev"),
            "--hidream-dir",
            os.getenv("CLEARMESH_HIDREAM_DIR", "/ephemeral/HiDream-O1-Image"),
            "--seed",
            str(seed),
        ]

    def generate_text_reference_image(self, job_id: str, request: dict, *, attempt: int) -> tuple[Path, dict | None]:
        root = self.job_dir(job_id)
        prompt = request.get("prompt") or "a clean production-ready 3D asset"
        reference_prompt = build_text_to_image_prompt(prompt)
        base_seed = int((request.get("metadata") or {}).get("seed", 0))
        image_name = "text_prompt.png" if attempt == 0 else f"text_prompt_retry_{attempt}.png"
        image_path = root / "input" / image_name
        stage = "text_to_image" if attempt == 0 else f"text_to_image_retry_{attempt}"
        self.run_command(
            job_id,
            stage,
            self.text_to_image_command(prompt=reference_prompt, output=image_path, seed=base_seed + attempt),
        )
        if os.getenv("CLEARMESH_DISABLE_REFERENCE_IMAGE_QC", "0") == "1":
            return image_path, None
        quality = inspect_reference_image(image_path)
        detail = f"attempt={attempt + 1}; quality={json.dumps(quality, sort_keys=True)}"
        if quality.get("ok", True):
            self.event(job_id, "reference_qc", "running", detail)
        else:
            self.event(job_id, "reference_qc", "running", f"rejected reference image; {detail}")
        return image_path, quality

    def resolve_input(self, job_id: str, request: dict) -> Path:
        root = self.job_dir(job_id)
        input_uri = request.get("input_uri")
        if input_uri:
            return self.fetch_uri(job_id, input_uri, "input", "input.png")
        max_attempts = max(1, int(os.getenv("CLEARMESH_TEXT_TO_IMAGE_MAX_ATTEMPTS", "5")))
        last_quality: dict | None = None
        for attempt in range(max_attempts):
            image_path, quality = self.generate_text_reference_image(job_id, request, attempt=attempt)
            if quality is None:
                return image_path
            last_quality = quality
            if quality.get("ok", True):
                return image_path
        if os.getenv("CLEARMESH_ALLOW_REFERENCE_IMAGE_QC_FAIL", "0") == "1":
            return image_path
        raise RuntimeError(
            "text-to-image produced a likely multi-object/collage reference after "
            f"{max_attempts} attempts; upload a reference image or switch the text-to-image model. "
            f"last_quality={json.dumps(last_quality, sort_keys=True)}"
        )

    def resolve_source_mesh(self, job_id: str, request: dict) -> Path:
        source_job_id = self.metadata_value(request, "source_job_id")
        source_artifact = self.metadata_value(request, "source_artifact") or "final_mesh"
        if source_job_id:
            try:
                source_job = self.read_job(source_job_id)
            except KeyError as exc:
                raise FileNotFoundError(f"source job not found: {source_job_id}") from exc
            artifact_path = (source_job.get("artifacts") or {}).get(source_artifact)
            if not artifact_path:
                raise FileNotFoundError(f"source artifact {source_artifact!r} not found on job {source_job_id}")
            path = Path(artifact_path)
            if not path.exists():
                raise FileNotFoundError(f"source artifact file missing: {path}")
            return path
        source_uri = self.metadata_value(request, "source_mesh_uri", "source_mesh_path", "mesh_uri", "mesh_path")
        if not source_uri and request.get("mode") in {"edit_text", "edit_image"}:
            source_uri = request.get("input_uri")
        if not source_uri:
            raise ValueError("edit modes require input_uri or metadata.source_mesh_uri")
        return self.fetch_uri(job_id, source_uri, "source_mesh", "source_mesh.glb")

    def resolve_edit_image(self, job_id: str, request: dict) -> Path:
        edit_uri = self.metadata_value(request, "edit_image_uri", "edit_image_path", "target_image_uri", "target_image_path")
        if not edit_uri:
            raise ValueError("edit_image mode requires metadata.edit_image_uri")
        return self.fetch_uri(job_id, edit_uri, "edit_image", "edit_image.png")

    def easy3e_options_path(self, job_id: str, request: dict) -> Path:
        allowed = {
            "num_flow_steps",
            "gamma",
            "eta",
            "guidance_scale",
            "num_repaint_steps",
            "blend_boundary",
            "enable_texture",
            "texture_guidance_scale",
            "text_image_guidance",
            "text_guidance_scale",
            "text_num_steps",
            "grid_size",
            "enable_repair",
            "export_format",
        }
        raw = (request.get("pipeline") or {}).get("easy3e_options") or (request.get("metadata") or {}).get("easy3e_options") or {}
        options = {key: raw[key] for key in allowed if key in raw}
        if options.get("enable_texture") and not os.getenv("CLEARMESH_EASY3E_CTRL_ADAPTER_CHECKPOINT"):
            options["enable_texture"] = False
        options.setdefault("enable_repair", True)
        options.setdefault("export_format", "glb")
        path = self.job_dir(job_id) / "easy3e_options.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(options, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def run_easy3e_job(self, job_id: str, request: dict) -> Path:
        easy3e = self.easy3e_readiness()
        if not easy3e.get("enabled"):
            raise RuntimeError(f"Easy3E unavailable: {easy3e.get('reason')}")
        source_mesh = self.resolve_source_mesh(job_id, request)
        easy3e_dir = self.job_dir(job_id) / "easy3e"
        output_path = easy3e_dir / "edited_mesh.glb"
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts/product/run_easy3e_edit.py"),
            "--source-mesh",
            str(source_mesh),
            "--output-dir",
            str(easy3e_dir),
            "--output-name",
            output_path.name,
            "--trellis2-dir",
            os.getenv("CLEARMESH_TRELLIS2_DIR", "/ephemeral/TRELLIS.2"),
            "--model-dir",
            os.getenv("CLEARMESH_TRELLIS_MODEL", "microsoft/TRELLIS.2-4B"),
            "--options-json",
            str(self.easy3e_options_path(job_id, request)),
        ]
        source_image_uri = self.metadata_value(request, "source_image_uri", "source_image_path")
        if source_image_uri:
            command.extend(["--source-image", str(self.fetch_uri(job_id, source_image_uri, "source_image", "source_image.png"))])
        if request.get("mode") == "edit_image":
            command.extend(["--edit-image", str(self.resolve_edit_image(job_id, request))])
        else:
            instruction = request.get("prompt") or self.metadata_value(request, "instruction", "edit_instruction")
            if not instruction:
                raise ValueError("edit_text mode requires prompt or metadata.instruction")
            command.extend(["--instruction", instruction])
            view = self.metadata_value(request, "view") or "front"
            command.extend(["--view", view])
        device = os.getenv("CLEARMESH_EASY3E_DEVICE", os.getenv("CLEARMESH_FACEQ_DEVICE", "cuda"))
        if device:
            command.extend(["--device", device])
        self.run_command(job_id, "easy3e", command)
        if not output_path.exists():
            raise RuntimeError(f"Easy3E mesh missing: {output_path}")
        return output_path

    def texture_uv_enabled(self, request: dict) -> bool:
        if os.getenv("CLEARMESH_TEXTURE_UV_ENABLED", "1") == "0":
            return False
        pipeline = request.get("pipeline") or {}
        if pipeline.get("textures") is False or pipeline.get("uv_mapping") is False:
            return False
        return True

    def build_texture_uv_command(
        self,
        *,
        mesh_path: Path,
        trellis_mesh: Path | None,
        input_image: Path | None,
        request: dict,
        output_dir: Path,
        output_name: str,
    ) -> list[str]:
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts/product/run_texture_uv_postprocess.py"),
            "--input-mesh",
            str(mesh_path),
            "--output-dir",
            str(output_dir),
            "--output-name",
            output_name,
            "--mode",
            os.getenv("CLEARMESH_TEXTURE_UV_MODE", "auto"),
        ]
        if trellis_mesh is not None:
            command.extend(["--reference-mesh", str(trellis_mesh)])
        if input_image is not None:
            command.extend(["--reference-image", str(input_image)])
        prompt = str(request.get("prompt") or "")
        if prompt:
            command.extend(["--prompt", prompt])
        texture_command = os.getenv("CLEARMESH_TEXTURE_UV_COMMAND", "").strip()
        if texture_command:
            command.extend(["--command", texture_command])
        blender_bin = os.getenv("BLENDER_BIN", "").strip()
        if blender_bin:
            command.extend(["--blender-bin", blender_bin])
        return command

    def run_texture_uv_stage(
        self,
        job_id: str,
        *,
        final_mesh: Path,
        trellis_mesh: Path | None,
        input_image: Path | None,
        request: dict,
        artifacts: dict,
        quality_report: dict,
    ) -> tuple[Path, dict, dict]:
        if not self.texture_uv_enabled(request):
            return final_mesh, artifacts, quality_report
        texture_dir = self.job_dir(job_id) / "texture_uv"
        texture_output = texture_dir / "textured_mesh.glb"
        report_path = texture_dir / "texture_uv_report.json"
        timeout = _env_int("CLEARMESH_TEXTURE_UV_TIMEOUT_SECONDS", 600)
        try:
            self.run_command(
                job_id,
                "texture_uv",
                self.build_texture_uv_command(
                    mesh_path=final_mesh,
                    trellis_mesh=trellis_mesh,
                    input_image=input_image,
                    request=request,
                    output_dir=texture_dir,
                    output_name=texture_output.name,
                ),
                timeout_seconds=timeout,
            )
        except Exception as exc:  # noqa: BLE001 - texture/UV is additive, not job-fatal.
            quality_report["texture_uv"] = {
                "ok": False,
                "accepted": False,
                "error": f"{type(exc).__name__}: {exc}",
                "reasons": ["texture/UV postprocess failed; geometry artifact preserved"],
            }
            self.update_job(job_id, quality_report=quality_report)
            return final_mesh, artifacts, quality_report

        report: dict = {}
        if report_path.exists():
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                report = {"ok": False, "error": f"could not read texture report: {type(exc).__name__}: {exc}"}
        if texture_output.exists():
            artifacts["textured_mesh"] = str(texture_output)
            artifacts["uv_mesh"] = str(texture_output)
            if report_path.exists():
                artifacts["texture_uv_report"] = str(report_path)
            if os.getenv("CLEARMESH_TEXTURE_UV_REPLACE_FINAL", "0") == "1" and not report.get("fallback", False):
                final_mesh = texture_output
                artifacts["final_mesh"] = str(final_mesh)
        quality_report["texture_uv"] = report or {"ok": texture_output.exists(), "output_path": str(texture_output)}
        self.update_job(job_id, artifacts=artifacts, quality_report=quality_report)
        return final_mesh, artifacts, quality_report

    def render_preview_image_stage(
        self,
        job_id: str,
        *,
        mesh_path: Path,
        artifacts: dict,
        quality_report: dict,
        artifact_name: str = "preview_image",
        promote: bool = True,
    ) -> tuple[dict, dict]:
        if os.getenv("CLEARMESH_RENDER_PREVIEW_ENABLED", "1") == "0":
            return artifacts, quality_report
        preview_dir = self.job_dir(job_id) / "previews"
        preview_dir.mkdir(parents=True, exist_ok=True)
        output_path = preview_dir / f"{artifact_name}.png"
        logs = self.job_dir(job_id) / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        stdout_path = logs / f"{artifact_name}.stdout.log"
        stderr_path = logs / f"{artifact_name}.stderr.log"
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts/product/render_glb_preview.py"),
            str(mesh_path),
            str(output_path),
            "--size",
            os.getenv("CLEARMESH_RENDER_PREVIEW_SIZE", "900"),
        ]
        try:
            with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
                proc = subprocess.run(
                    command,
                    cwd=str(REPO_ROOT),
                    text=True,
                    stdout=stdout,
                    stderr=stderr,
                    timeout=_env_int("CLEARMESH_RENDER_PREVIEW_TIMEOUT_SECONDS", 60),
                    check=False,
                )
            if proc.returncode != 0:
                raise RuntimeError(f"render_glb_preview exited {proc.returncode}; stderr={stderr_path}")
            if not output_path.exists():
                raise RuntimeError(f"preview image missing: {output_path}")
            artifacts[artifact_name] = str(output_path)
            if promote:
                artifacts["preview_image"] = str(output_path)
            quality_report.setdefault("previews", {})[artifact_name] = {
                "ok": True,
                "output_path": str(output_path),
                "source_mesh": str(mesh_path),
            }
            self.update_job(job_id, artifacts=artifacts, quality_report=quality_report)
        except Exception as exc:  # noqa: BLE001 - preview image is non-fatal.
            quality_report.setdefault("previews", {})[artifact_name] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "source_mesh": str(mesh_path),
            }
            self.update_job(job_id, quality_report=quality_report)
        return artifacts, quality_report

    def build_faceq_command(
        self,
        *,
        mesh_path: Path,
        output_dir: Path,
        output_name: str,
        point_samples: str | None = None,
        generation_max_faces: str | None = None,
        constraint_top_k: str | None = None,
    ) -> list[str]:
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts/product/run_faceq_from_mesh.py"),
            "--checkpoint",
            str(self.checkpoint),
            "--mesh",
            str(mesh_path),
            "--output-dir",
            str(output_dir),
            "--output-name",
            output_name,
            "--device",
            os.getenv("CLEARMESH_FACEQ_DEVICE", "cuda"),
        ]
        faceq_point_samples = point_samples if point_samples is not None else os.getenv("CLEARMESH_FACEQ_POINT_SAMPLES", "").strip()
        if faceq_point_samples:
            command.extend(["--point-samples", str(faceq_point_samples)])
        faceq_generation_max_faces = (
            generation_max_faces if generation_max_faces is not None else os.getenv("CLEARMESH_FACEQ_GENERATION_MAX_FACES", "").strip()
        )
        if faceq_generation_max_faces:
            command.extend(["--generation-max-faces", str(faceq_generation_max_faces)])
        faceq_face_count_mode = os.getenv("CLEARMESH_FACEQ_FACE_COUNT_MODE", "").strip()
        if faceq_face_count_mode:
            command.extend(["--face-count-mode", faceq_face_count_mode])
        faceq_decode_mode = os.getenv("CLEARMESH_FACEQ_DECODE_MODE", "").strip()
        if faceq_decode_mode:
            command.extend(["--decode-mode", faceq_decode_mode])
        faceq_constraint_top_k = constraint_top_k if constraint_top_k is not None else os.getenv("CLEARMESH_FACEQ_CONSTRAINT_TOP_K", "").strip()
        if faceq_constraint_top_k:
            command.extend(["--constraint-top-k", str(faceq_constraint_top_k)])
        faceq_repair_mode = os.getenv("CLEARMESH_FACEQ_REPAIR_MODE", "").strip()
        if faceq_repair_mode:
            command.extend(["--repair-mode", faceq_repair_mode])
        faceq_boundary_fill = os.getenv("CLEARMESH_FACEQ_BOUNDARY_FILL", "").strip()
        if faceq_boundary_fill:
            command.extend(["--boundary-fill", faceq_boundary_fill])
        if os.getenv("CLEARMESH_FACEQ_NO_BOUNDARY_BUDGET", "0") == "1":
            command.append("--no-boundary-budget")
        if os.getenv("CLEARMESH_FACEQ_NO_VERTEX_LINK_CONSTRAINT", "0") == "1":
            command.append("--no-vertex-link-constraint")
        return command

    def build_trellis_command(
        self,
        *,
        image_path: Path,
        output_dir: Path,
        preview_name: str,
        face_proxy_name: str,
    ) -> list[str]:
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts/product/run_trellis2_proxy.py"),
            "--input",
            str(image_path),
            "--output-dir",
            str(output_dir),
            "--output-name",
            preview_name,
            "--face-proxy-output-name",
            face_proxy_name,
            "--model",
            os.getenv("CLEARMESH_TRELLIS_MODEL", "microsoft/TRELLIS.2-4B"),
            "--decimation-target",
            os.getenv("CLEARMESH_TRELLIS_DECIMATION_TARGET", "80000"),
            "--face-proxy-decimation-target",
            os.getenv("CLEARMESH_FACEQ_PROXY_DECIMATION_TARGET", "4096"),
            "--texture-size",
            os.getenv("CLEARMESH_TRELLIS_TEXTURE_SIZE", "1024"),
        ]
        if os.getenv("CLEARMESH_TRELLIS_REMESH", "0") == "1":
            command.append("--remesh")
        else:
            command.append("--no-remesh")
        return command

    def should_run_faceq(self, request: dict) -> bool:
        """Decide whether this request should run slow FACE-Q refinement."""

        pipeline_options = request.get("pipeline") or {}
        explicit_faceq = None
        if "faceq" in pipeline_options:
            explicit_faceq = bool(pipeline_options.get("faceq"))
        elif request.get("faceq") is not None:
            explicit_faceq = bool(request.get("faceq"))

        if os.getenv("CLEARMESH_ALLOW_TRELLIS_ONLY", "0") != "1":
            return True
        if explicit_faceq is not None:
            return explicit_faceq
        return request.get("quality_tier") != "draft"

    def run_job(self, job_id: str) -> None:
        root = self.job_dir(job_id)
        root.mkdir(parents=True, exist_ok=True)
        try:
            job = self.read_job(job_id)
            request = job["request"]
            self.event(job_id, "preflight", "running", "starting pipeline")
            if os.getenv("CLEARMESH_PIPELINE_DRY_RUN", "0") == "1":
                self.update_job(job_id, status="succeeded", stage="dry_run", artifacts={})
                return

            mode = request.get("mode", "text_to_3d")
            if mode in {"edit_text", "edit_image"}:
                edited_mesh = self.run_easy3e_job(job_id, request)
                artifacts = {
                    "source_mesh": str(self.resolve_source_mesh(job_id, request)),
                    "easy3e_mesh": str(edited_mesh),
                    "final_mesh": str(edited_mesh),
                }
                quality_report: dict = {}
                edited_mesh, artifacts, quality_report = self.run_texture_uv_stage(
                    job_id,
                    final_mesh=edited_mesh,
                    trellis_mesh=None,
                    input_image=None,
                    request=request,
                    artifacts=artifacts,
                    quality_report=quality_report,
                )
                artifacts["final_mesh"] = str(edited_mesh)
                artifacts, quality_report = self.render_preview_image_stage(
                    job_id,
                    mesh_path=edited_mesh,
                    artifacts=artifacts,
                    quality_report=quality_report,
                    artifact_name="final_preview_image",
                )
                self.update_job(job_id, status="succeeded", stage="complete", artifacts=artifacts, quality_report=quality_report)
                return

            image_path = self.resolve_input(job_id, request)
            self.update_job(job_id, artifacts={"input_image": str(image_path)})

            quality_report: dict = {}
            trellis_candidates: list[dict] = []
            max_trellis_attempts = max(1, _env_int("CLEARMESH_TRELLIS_MAX_ATTEMPTS", 2))
            if request.get("input_uri"):
                max_trellis_attempts = 1
            selected_trellis: dict | None = None
            for attempt in range(max_trellis_attempts):
                if attempt > 0:
                    next_image, image_quality = self.generate_text_reference_image(job_id, request, attempt=attempt)
                    if image_quality is not None and not image_quality.get("ok", True):
                        continue
                    image_path = next_image
                    self.update_job(job_id, artifacts={**self.read_job(job_id).get("artifacts", {}), "input_image": str(image_path)})
                trellis_dir = root / ("trellis" if attempt == 0 else f"trellis_retry_{attempt}")
                trellis_mesh = trellis_dir / "trellis_preview.glb"
                trellis_face_proxy_mesh = trellis_dir / "trellis_faceq_proxy.glb"
                self.run_command(
                    job_id,
                    "trellis" if attempt == 0 else f"trellis_retry_{attempt}",
                    self.build_trellis_command(
                        image_path=image_path,
                        output_dir=trellis_dir,
                        preview_name=trellis_mesh.name,
                        face_proxy_name=trellis_face_proxy_mesh.name,
                    ),
                    timeout_seconds=_env_int("CLEARMESH_TRELLIS_TIMEOUT_SECONDS", 1200),
                )
                if not trellis_mesh.exists():
                    raise RuntimeError(f"Trellis mesh missing: {trellis_mesh}")
                if not trellis_face_proxy_mesh.exists():
                    raise RuntimeError(f"Trellis FACE-Q proxy missing: {trellis_face_proxy_mesh}")
                trellis_qc = inspect_trellis_candidate(trellis_mesh)
                candidate = {
                    "attempt": attempt,
                    "input_image": str(image_path),
                    "trellis_mesh": str(trellis_mesh),
                    "trellis_faceq_proxy_mesh": str(trellis_face_proxy_mesh),
                    "qc": trellis_qc,
                }
                trellis_candidates.append(candidate)
                quality_report["trellis"] = {
                    "accepted": bool(trellis_qc.get("accepted", False)),
                    "selected_attempt": attempt,
                    "attempts": trellis_candidates,
                    "reasons": trellis_qc.get("reasons") or [],
                }
                self.update_job(job_id, quality_report=quality_report)
                if trellis_qc.get("accepted", False):
                    selected_trellis = candidate
                    self.event(job_id, "trellis_qc", "running", f"Trellis candidate accepted on attempt {attempt + 1}")
                    break
                self.event(
                    job_id,
                    "trellis_qc",
                    "running",
                    "Trellis candidate rejected: " + "; ".join(trellis_qc.get("reasons") or ["quality gate failed"]),
                )

            if selected_trellis is None:
                selected_trellis = max(
                    trellis_candidates,
                    key=lambda item: float((item.get("qc") or {}).get("score") or -1e9),
                )
                quality_report["trellis"] = {
                    "accepted": False,
                    "selected_attempt": selected_trellis["attempt"],
                    "attempts": trellis_candidates,
                    "reasons": (selected_trellis.get("qc") or {}).get("reasons") or ["all Trellis attempts failed QC"],
                }
                self.event(job_id, "trellis_qc", "running", "All Trellis attempts failed QC; using best fallback candidate")

            trellis_mesh = Path(str(selected_trellis["trellis_mesh"]))
            trellis_face_proxy_mesh = Path(str(selected_trellis["trellis_faceq_proxy_mesh"]))
            artifacts = {
                **self.read_job(job_id).get("artifacts", {}),
                "trellis_mesh": str(trellis_mesh),
                "trellis_faceq_proxy_mesh": str(trellis_face_proxy_mesh),
            }
            self.update_job(
                job_id,
                artifacts=artifacts,
            )
            artifacts, quality_report = self.render_preview_image_stage(
                job_id,
                mesh_path=trellis_mesh,
                artifacts=artifacts,
                quality_report=quality_report,
                artifact_name="trellis_preview_image",
            )

            run_faceq = self.should_run_faceq(request)
            if not run_faceq:
                quality_report["faceq"] = {
                    "accepted": False,
                    "skipped": True,
                    "reasons": ["FACE-Q skipped for Trellis-only preview"],
                }
                artifacts = {**self.read_job(job_id).get("artifacts", {}), "final_mesh": str(trellis_mesh)}
                final_mesh = trellis_mesh
                final_mesh, artifacts, quality_report = self.run_texture_uv_stage(
                    job_id,
                    final_mesh=final_mesh,
                    trellis_mesh=trellis_mesh,
                    input_image=image_path,
                    request=request,
                    artifacts=artifacts,
                    quality_report=quality_report,
                )
                artifacts["final_mesh"] = str(final_mesh)
                artifacts, quality_report = self.render_preview_image_stage(
                    job_id,
                    mesh_path=final_mesh,
                    artifacts=artifacts,
                    quality_report=quality_report,
                    artifact_name="final_preview_image",
                )
                self.update_job(job_id, status="succeeded", stage="complete", artifacts=artifacts, quality_report=quality_report)
                return

            if not (selected_trellis.get("qc") or {}).get("accepted", False) and os.getenv("CLEARMESH_FACEQ_ON_REJECTED_TRELLIS", "0") != "1":
                artifacts = {**self.read_job(job_id).get("artifacts", {}), "final_mesh": str(trellis_mesh)}
                final_mesh = trellis_mesh
                final_mesh, artifacts, quality_report = self.run_texture_uv_stage(
                    job_id,
                    final_mesh=final_mesh,
                    trellis_mesh=trellis_mesh,
                    input_image=image_path,
                    request=request,
                    artifacts=artifacts,
                    quality_report=quality_report,
                )
                artifacts["final_mesh"] = str(final_mesh)
                artifacts, quality_report = self.render_preview_image_stage(
                    job_id,
                    mesh_path=final_mesh,
                    artifacts=artifacts,
                    quality_report=quality_report,
                    artifact_name="final_preview_image",
                )
                self.update_job(job_id, status="succeeded", stage="complete", artifacts=artifacts, quality_report=quality_report)
                return

            if os.getenv("CLEARMESH_PUBLISH_TRELLIS_PREVIEW_BEFORE_FACEQ", "1") == "1":
                artifacts = {**self.read_job(job_id).get("artifacts", {}), "final_mesh": str(trellis_mesh)}
                preview_mesh = trellis_mesh
                _, artifacts, quality_report = self.run_texture_uv_stage(
                    job_id,
                    final_mesh=preview_mesh,
                    trellis_mesh=trellis_mesh,
                    input_image=image_path,
                    request=request,
                    artifacts=artifacts,
                    quality_report=quality_report,
                )
                artifacts["final_mesh"] = str(preview_mesh)
                self.update_job(job_id, artifacts=artifacts, quality_report=quality_report)
                self.event(job_id, "trellis_preview", "running", "Textured Trellis preview is available while FACE-Q runs")

            faceq_dir = root / "faceq"
            faceq_command = self.build_faceq_command(mesh_path=trellis_face_proxy_mesh, output_dir=faceq_dir, output_name="artist_mesh.glb")
            faceq_timeout = _env_int("CLEARMESH_FACEQ_TIMEOUT_SECONDS", 360)
            try:
                self.run_command(job_id, "faceq", faceq_command, timeout_seconds=faceq_timeout)
            except TimeoutError:
                quality_report = {
                    **quality_report,
                    "faceq": {
                        "accepted": False,
                        "timed_out": True,
                        "reasons": [f"FACE-Q exceeded serving timeout ({faceq_timeout}s)"],
                    }
                }
                artifacts = {**self.read_job(job_id).get("artifacts", {}), "final_mesh": str(trellis_mesh)}
                final_mesh = trellis_mesh
                final_mesh, artifacts, quality_report = self.run_texture_uv_stage(
                    job_id,
                    final_mesh=final_mesh,
                    trellis_mesh=trellis_mesh,
                    input_image=image_path,
                    request=request,
                    artifacts=artifacts,
                    quality_report=quality_report,
                )
                artifacts["final_mesh"] = str(final_mesh)
                artifacts, quality_report = self.render_preview_image_stage(
                    job_id,
                    mesh_path=final_mesh,
                    artifacts=artifacts,
                    quality_report=quality_report,
                    artifact_name="final_preview_image",
                )
                self.update_job(job_id, status="succeeded", stage="complete", artifacts=artifacts, quality_report=quality_report)
                return
            faceq_mesh = faceq_dir / "artist_mesh.glb"
            if not faceq_mesh.exists():
                raise RuntimeError(f"FACE-Q mesh missing: {faceq_mesh}")
            artifacts = {**self.read_job(job_id).get("artifacts", {}), "faceq_mesh": str(faceq_mesh)}

            quality_report = {**quality_report, "faceq": {"accepted": True, "reasons": []}}
            final_mesh = faceq_mesh
            if os.getenv("CLEARMESH_FACEQ_ACCEPTANCE_ENABLE", "1") != "0":
                faceq_qc = inspect_faceq_candidate(trellis_mesh, faceq_mesh)
                quality_report["faceq"] = faceq_qc
                self.update_job(job_id, quality_report=quality_report)
                if not faceq_qc.get("accepted", False):
                    artifacts["faceq_rejected_mesh"] = str(faceq_mesh)
                    final_mesh = trellis_mesh
                    self.event(
                        job_id,
                        "faceq_qc",
                        "running",
                        "FACE-Q candidate rejected; publishing Trellis fallback: "
                        + "; ".join(faceq_qc.get("reasons") or ["quality gate failed"]),
                    )
                    if os.getenv("CLEARMESH_FACEQ_RETRY_ON_REJECT", "1") == "1":
                        retry_dir = root / "faceq_retry"
                        retry_command = self.build_faceq_command(
                            mesh_path=trellis_face_proxy_mesh,
                            output_dir=retry_dir,
                            output_name="artist_mesh.glb",
                            point_samples=os.getenv("CLEARMESH_FACEQ_RETRY_POINT_SAMPLES", "16384"),
                            generation_max_faces=os.getenv("CLEARMESH_FACEQ_RETRY_GENERATION_MAX_FACES", "768"),
                            constraint_top_k=os.getenv("CLEARMESH_FACEQ_RETRY_CONSTRAINT_TOP_K", "24"),
                        )
                        try:
                            self.run_command(job_id, "faceq_retry", retry_command, timeout_seconds=faceq_timeout)
                        except TimeoutError:
                            quality_report["faceq_retry"] = {
                                "accepted": False,
                                "timed_out": True,
                                "reasons": [f"FACE-Q retry exceeded serving timeout ({faceq_timeout}s)"],
                            }
                            self.update_job(job_id, quality_report=quality_report)
                            retry_mesh = None
                        else:
                            retry_mesh = retry_dir / "artist_mesh.glb"
                        if retry_mesh is not None and retry_mesh.exists():
                            artifacts["faceq_retry_mesh"] = str(retry_mesh)
                            retry_qc = inspect_faceq_candidate(trellis_mesh, retry_mesh)
                            quality_report["faceq_retry"] = retry_qc
                            self.update_job(job_id, quality_report=quality_report)
                            if retry_qc.get("accepted", False):
                                final_mesh = retry_mesh
                                self.event(job_id, "faceq_qc", "running", "FACE-Q retry accepted")
                            else:
                                artifacts["faceq_retry_rejected_mesh"] = str(retry_mesh)
                                self.event(
                                    job_id,
                                    "faceq_qc",
                                    "running",
                                    "FACE-Q retry rejected; keeping Trellis fallback: "
                                    + "; ".join(retry_qc.get("reasons") or ["quality gate failed"]),
                                )
                else:
                    self.event(job_id, "faceq_qc", "running", "FACE-Q candidate accepted")

            easy3e_command = os.getenv("CLEARMESH_EASY3E_COMMAND", "").strip()
            if final_mesh != trellis_mesh and os.getenv("CLEARMESH_EASY3E_ENABLED", "0") == "1" and easy3e_command:
                easy3e_dir = root / "easy3e"
                easy3e_dir.mkdir(parents=True, exist_ok=True)
                easy3e_output = easy3e_dir / "edited_mesh.glb"
                command = [
                    part.format(input=str(final_mesh), output=str(easy3e_output), output_dir=str(easy3e_dir))
                    for part in shlex.split(easy3e_command)
                ]
                self.run_command(job_id, "easy3e", command)
                if easy3e_output.exists():
                    final_mesh = easy3e_output
                    artifacts["easy3e_mesh"] = str(easy3e_output)

            artifacts["final_mesh"] = str(final_mesh)
            final_mesh, artifacts, quality_report = self.run_texture_uv_stage(
                job_id,
                final_mesh=final_mesh,
                trellis_mesh=trellis_mesh,
                input_image=image_path,
                request=request,
                artifacts=artifacts,
                quality_report=quality_report,
            )
            artifacts["final_mesh"] = str(final_mesh)
            artifacts, quality_report = self.render_preview_image_stage(
                job_id,
                mesh_path=final_mesh,
                artifacts=artifacts,
                quality_report=quality_report,
                artifact_name="final_preview_image",
            )
            self.update_job(job_id, status="succeeded", stage="complete", artifacts=artifacts, quality_report=quality_report)
        except Exception as exc:  # noqa: BLE001 - bridge should persist errors.
            self.update_job(job_id, status="failed", stage="failed", error=f"{type(exc).__name__}: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-stats-json", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--mesh-chamfer-json", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--source-mesh", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--candidate-mesh", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--samples", type=int, default=2048, help=argparse.SUPPRESS)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--work-root", type=Path, default=Path("/workspace/clearmesh_pipeline_jobs"))
    parser.add_argument("--state-root", type=Path, default=Path("/workspace/clearmesh_pipeline_state"))
    parser.add_argument("--model-bundle", type=Path, default=Path(os.getenv("CLEARMESH_FACEQ_MODEL_BUNDLE", "/workspace/model_bundle_effective100k")))
    parser.add_argument("--checkpoint", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mesh_stats_json is not None:
        _apply_mesh_qc_resource_limits()
        print(json.dumps(_mesh_basic_stats_direct(args.mesh_stats_json), sort_keys=True))
        return 0
    if args.mesh_chamfer_json:
        if args.source_mesh is None or args.candidate_mesh is None:
            print(json.dumps({"ok": False, "load_error": "--source-mesh and --candidate-mesh are required"}, sort_keys=True))
            return 2
        _apply_mesh_qc_resource_limits()
        chamfer = _normalized_surface_chamfer_direct(args.source_mesh, args.candidate_mesh, args.samples)
        print(json.dumps({"ok": chamfer is not None, "chamfer_norm": chamfer}, sort_keys=True))
        return 0

    checkpoint = args.checkpoint or (args.model_bundle / "checkpoint.final.pt")
    server = PipelineServer(work_root=args.work_root, state_root=args.state_root, checkpoint=checkpoint, model_bundle=args.model_bundle)
    import uvicorn

    uvicorn.run(server.app, host=args.host, port=args.port, log_level=os.getenv("CLEARMESH_PIPELINE_LOG_LEVEL", "info"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
