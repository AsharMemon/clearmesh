#!/usr/bin/env python3
"""Generate coarse/fine training pairs for Stage 2 refinement.

v5 — Comprehensive improvements:
  1. Per-shard cache isolation (MESA_SHADER_CACHE_DIR, TMPDIR, XDG_CACHE_HOME)
  2. Granular failure tracking (10 categories instead of lumped "render_failures")
  3. Robust mesh rendering (largest connected component, fit-to-view camera)
  4. --retry-failed flag to re-attempt previously failed UIDs

Failure categories tracked:
  FILE_NOT_FOUND       — mesh file doesn't exist on disk
  MESH_LOAD_FAIL       — trimesh.load() threw an exception
  MESH_EMPTY           — loaded mesh has 0 vertices or faces
  MESH_DEGENERATE      — mesh has NaN/Inf vertices or zero extent
  RENDER_GL_ERROR      — OpenGL/pyglet error during scene.save_image()
  RENDER_ZERO_FG       — all views rendered but 0% foreground (invisible mesh)
  GATE_ALL_REJECTED    — views had foreground but all filtered by silhouette gate
  TRELLIS_EMPTY        — all TRELLIS.2 attempts produced empty voxels (numel==0)
  TRELLIS_OOM          — all TRELLIS.2 attempts hit OOM
  TRELLIS_ERROR        — other TRELLIS.2 runtime error
  QUALITY_REJECT       — TRELLIS.2 produced mesh but vertex count out of bounds

Multi-view + multi-seed strategy:
  - Single view, single seed: ~35% per model
  - Multi-seed (5 seeds), best view: ~60%
  - Multi-view (4) × multi-seed (5): ~95% (on good models)

Usage:
    # Standard 3-shard run
    for i in 0 1 2; do
      xvfb-run -a python generate_pairs.py \\
        --input_json /workspace/data/filtered/valid_models.json \\
        --output_dir /workspace/data/training_pairs \\
        --shard_id $i --num_shards 3 --gpu $i &
    done

    # Retry previously failed models
    xvfb-run -a python generate_pairs.py \\
        --input_json /workspace/data/filtered/valid_models.json \\
        --output_dir /workspace/data/training_pairs \\
        --retry-failed --shard_id 0 --num_shards 3 --gpu 0
"""

import argparse
import contextlib
import ctypes
import gc
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from collections import Counter
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image
from tqdm import tqdm

# ── Constants ───────────────────────────────────────────────────────────────

TRANSPARENT_BG = [0, 0, 0, 0]
DEFAULT_RENDER_BACKEND = "blender"
DEFAULT_BLENDER_BIN = "/workspace/tools/blender-3.0.1-linux-x64/blender"
DEFAULT_BLENDER_SAMPLES = 16
DEFAULT_MESH_PREP_MEMORY_LIMIT_GB = 96.0
DEFAULT_MESH_PREP_TIMEOUT_SEC = 240

MIN_COARSE_VERTS = 50
MAX_COARSE_VERTS = 1_500_000
DECIMATE_TARGET_FACES = 500_000  # Decimate over-vertex meshes instead of rejecting

GENERATION_SEEDS = [42, 123, 456]
MAX_VIEWS_TO_TRY = 4
INITIAL_RENDER_VIEWS = 4
STRONG_SECONDARY_VIEW_SCORE = 0.20
STRONG_SECONDARY_VIEW_MIN_FG = 0.12
SECONDARY_VIEW_EXTRA_SEEDS = [123]
MAX_RESCUE_CANDIDATES = 1
MAX_RESCUE_ATTEMPTS = 2
MAX_CONSECUTIVE_EMPTY = 3  # Skip remaining attempts + rescue if first N all produce empty voxels
RESCUE_TIGHT_ZOOM_SEEDS = [42, 123]

# Primary TRELLIS sampler params — boost guidance for synthetic conditioning images
DEFAULT_SPARSE_SAMPLER_PARAMS = {
    "steps": 12,
    "guidance_strength": 9.0,  # up from default 7.5
}
DEFAULT_SHAPE_SAMPLER_PARAMS = {
    "steps": 12,
    "guidance_strength": 4.5,  # up from default 3.0
}

RESCUE_SPARSE_SAMPLER_PARAMS = {
    "steps": 16,
    "guidance_strength": 9.0,
}
RESCUE_SHAPE_SAMPLER_PARAMS = {
    "steps": 14,
    "guidance_strength": 8.0,
}
RESCUE_PITCH_SHIFTS = [20.0, -20.0]

DEFAULT_MIN_FG_FRAC = 0.05   # Lowered from 0.08: thin/flat meshes need lower threshold
DEFAULT_MAX_FG_FRAC = 0.85
DEFAULT_EDGE_MARGIN = 0.02

CAMERA_VIEWS = [
    (0, 20), (45, 15), (-45, 15), (30, 35), (-30, 35), (180, 20),
    (90, 10), (-90, 10), (135, 25), (-135, 25), (0, 50), (180, 50),
]

CURRENT_MODEL_NONE = {"uid": None, "path": None, "phase": "idle"}


@contextlib.contextmanager
def suppress_output(enabled: bool = True):
    """Temporarily silence noisy inner progress bars/warnings."""
    if not enabled:
        yield
        return
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


def get_rss_gb() -> float:
    """Return process RSS in GiB (Linux only; returns 0 on non-Linux)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except Exception:
        return 0.0
    return 0.0


def get_cuda_memory_stats_mb() -> tuple[float, float, float]:
    """Return CUDA allocated/reserved/max_allocated memory in MiB."""
    if not torch.cuda.is_available():
        return 0.0, 0.0, 0.0
    try:
        return (
            float(torch.cuda.memory_allocated() / (1024 * 1024)),
            float(torch.cuda.memory_reserved() / (1024 * 1024)),
            float(torch.cuda.max_memory_allocated() / (1024 * 1024)),
        )
    except Exception:
        return 0.0, 0.0, 0.0


def malloc_trim() -> None:
    """Ask glibc to return free heap pages back to the OS (best effort)."""
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def cleanup_memory(use_malloc_trim: bool = True) -> None:
    """Force best-effort CPU/GPU allocator cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if use_malloc_trim:
        malloc_trim()


class RssWatchdog:
    """Poll RSS in the background and hard-exit before the pod OOMs."""

    def __init__(
        self,
        limit_gb: float,
        recycle_exit_code: int,
        poll_interval_sec: float,
        progress_callback,
        state_callback,
    ):
        self.limit_gb = limit_gb
        self.recycle_exit_code = recycle_exit_code
        self.poll_interval_sec = poll_interval_sec
        self.progress_callback = progress_callback
        self.state_callback = state_callback
        self._stop = threading.Event()
        self._triggered = threading.Event()
        self._thread = None

    def start(self) -> None:
        if self.limit_gb <= 0 or self.poll_interval_sec <= 0:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="rss-watchdog",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(self.poll_interval_sec * 2.0, 0.1))

    def _run(self) -> None:
        while not self._stop.wait(self.poll_interval_sec):
            rss_now = get_rss_gb()
            if rss_now < self.limit_gb:
                continue
            if self._triggered.is_set():
                continue
            self._triggered.set()
            state = self.state_callback()
            print(
                f"\n[watchdog] emergency recycle: RSS {rss_now:.1f} GiB >= "
                f"{self.limit_gb:.1f} GiB while uid={state['uid']} phase={state['phase']}",
                flush=True,
            )
            try:
                self.progress_callback(rss_now, state)
            finally:
                os._exit(self.recycle_exit_code)


def _required_model_keys(pipeline_type: str, geometry_only: bool) -> set[str]:
    """Return minimal model set needed for the selected pipeline mode."""
    required = {
        "sparse_structure_flow_model",
        "sparse_structure_decoder",
        "shape_slat_decoder",
    }

    if pipeline_type in {"512", "1024_cascade", "1536_cascade"}:
        required.add("shape_slat_flow_model_512")
    if pipeline_type in {"1024", "1024_cascade", "1536_cascade"}:
        required.add("shape_slat_flow_model_1024")

    if not geometry_only:
        required.add("tex_slat_decoder")
        if pipeline_type == "512":
            required.add("tex_slat_flow_model_512")
        else:
            required.add("tex_slat_flow_model_1024")

    return required


def prune_pipeline_models(pipeline, pipeline_type: str, geometry_only: bool) -> list[str]:
    """Delete unused model branches to reduce CPU RSS and fragmentation."""
    keep = _required_model_keys(pipeline_type, geometry_only)
    removed = []
    for key in list(getattr(pipeline, "models", {}).keys()):
        if key in keep:
            continue
        removed.append(key)
        del pipeline.models[key]

    if removed:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return removed

# ── Failure Categories ──────────────────────────────────────────────────────

FAIL_FILE_NOT_FOUND    = "FILE_NOT_FOUND"
FAIL_MESH_LOAD         = "MESH_LOAD_FAIL"
FAIL_MESH_EMPTY        = "MESH_EMPTY"
FAIL_MESH_DEGENERATE   = "MESH_DEGENERATE"
FAIL_RENDER_GL         = "RENDER_GL_ERROR"
FAIL_RENDER_ZERO_FG    = "RENDER_ZERO_FG"
FAIL_GATE_REJECTED     = "GATE_ALL_REJECTED"
FAIL_TRELLIS_EMPTY     = "TRELLIS_EMPTY"
FAIL_TRELLIS_OOM       = "TRELLIS_OOM"
FAIL_TRELLIS_ERROR     = "TRELLIS_ERROR"
FAIL_QUALITY_REJECT    = "QUALITY_REJECT"
FAIL_RSS_RECYCLE       = "RSS_EMERGENCY_RECYCLE"
FAIL_MESH_TOO_LARGE    = "MESH_FILE_TOO_LARGE"
FAIL_MESH_TOO_COMPLEX  = "MESH_TOO_COMPLEX"
FAIL_MESH_PREP_TIMEOUT = "MESH_PREP_TIMEOUT"

# Skip model files larger than this (bytes). Very large files cause huge RSS
# spikes during trimesh.load() and rarely produce useful TRELLIS results.
MAX_MESH_FILE_BYTES = 500 * 1024 * 1024  # 500 MB
MAX_MESH_FACES = 750_000


# ── Cache Isolation ─────────────────────────────────────────────────────────

def isolate_caches(shard_id: int):
    """Set per-shard cache directories to avoid GL/Mesa corruption between shards.

    Each shard gets its own:
      - MESA_SHADER_CACHE_DIR (Mesa OpenGL shader cache)
      - TMPDIR (temp files)
      - XDG_CACHE_HOME (generic cache)
      - MESA_GLSL_CACHE_DIR (older Mesa GLSL cache)
    """
    cache_base = f"/tmp/shard_{shard_id}_cache"
    os.makedirs(cache_base, exist_ok=True)

    mesa_dir = os.path.join(cache_base, "mesa_shader_cache")
    tmp_dir = os.path.join(cache_base, "tmp")
    xdg_dir = os.path.join(cache_base, "xdg_cache")
    os.makedirs(mesa_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(xdg_dir, exist_ok=True)

    os.environ["MESA_SHADER_CACHE_DIR"] = mesa_dir
    os.environ["MESA_GLSL_CACHE_DIR"] = mesa_dir
    os.environ["TMPDIR"] = tmp_dir
    os.environ["XDG_CACHE_HOME"] = xdg_dir
    # Also set HOME-level cache isolation for good measure
    os.environ["MESA_SHADER_CACHE_DISABLE"] = "false"

    print(f"  Cache isolation: MESA={mesa_dir}, TMP={tmp_dir}, XDG={xdg_dir}")


# ── Robust Mesh Loading ────────────────────────────────────────────────────

def load_mesh_robust(mesh_path: str):
    """Load mesh with robustness: largest connected component, degenerate checks.

    Returns:
        (trimesh.Trimesh, None) on success
        (None, failure_reason) on failure
    """
    if not os.path.exists(mesh_path):
        return None, FAIL_FILE_NOT_FOUND

    # Skip extremely large files that cause RSS spikes during loading
    try:
        file_size = os.path.getsize(mesh_path)
        if file_size > MAX_MESH_FILE_BYTES:
            return None, FAIL_MESH_TOO_LARGE
    except OSError:
        pass

    try:
        loaded = trimesh.load(mesh_path)
    except Exception as e:
        return None, FAIL_MESH_LOAD

    # Handle Scene objects (multi-part models, common in Objaverse)
    if isinstance(loaded, trimesh.Scene):
        meshes = [g for g in loaded.geometry.values()
                  if isinstance(g, trimesh.Trimesh) and len(g.faces) > 0]
        if not meshes:
            return None, FAIL_MESH_EMPTY
        if len(meshes) == 1:
            loaded = meshes[0]
        else:
            total_faces = sum(len(m.faces) for m in meshes)
            try:
                if total_faces > MAX_MESH_FACES:
                    loaded = max(meshes, key=lambda m: len(m.faces))
                else:
                    loaded = trimesh.util.concatenate(meshes)
            except Exception:
                # If concat fails, use the largest mesh
                loaded = max(meshes, key=lambda m: len(m.faces))

    if not isinstance(loaded, trimesh.Trimesh):
        # Last resort: try force="mesh"
        try:
            loaded = trimesh.load(mesh_path, force="mesh")
        except Exception:
            return None, FAIL_MESH_LOAD

    if len(loaded.vertices) == 0 or len(loaded.faces) == 0:
        return None, FAIL_MESH_EMPTY

    if len(loaded.faces) > MAX_MESH_FACES:
        return None, FAIL_MESH_TOO_COMPLEX

    # Check for degenerate geometry
    if np.any(~np.isfinite(loaded.vertices)):
        return None, FAIL_MESH_DEGENERATE

    # Keep largest connected component (removes floating debris).
    # Skip for large meshes — split() is O(n) and slow for >100K faces.
    if len(loaded.faces) < 100_000:
        try:
            components = loaded.split(only_watertight=False)
            if len(components) > 1:
                # Pick component with most faces
                largest = max(components, key=lambda c: len(c.faces))
                if len(largest.faces) >= 4:  # At least a tetrahedron
                    loaded = largest
        except Exception:
            pass  # If split fails, use the whole mesh

    if len(loaded.vertices) == 0 or len(loaded.faces) == 0:
        return None, FAIL_MESH_EMPTY

    # Center and normalize to unit cube
    loaded.vertices -= loaded.centroid
    extents = loaded.extents
    max_extent = extents.max()
    if max_extent < 1e-8:
        return None, FAIL_MESH_DEGENERATE
    loaded.vertices /= max_extent

    return loaded, None


def prepare_mesh_asset_subprocess(
    mesh_path: str,
    shard_id: int,
    memory_limit_gb: float,
    timeout_sec: int,
) -> tuple[str | None, str | None, Path | None, dict]:
    """Normalize and strip the source mesh in an isolated subprocess.

    This prevents pathological `trimesh.load()` calls from polluting the
    long-lived TRELLIS worker RSS. The child process writes a single prepared
    GLB, plus a small metadata JSON describing success or failure.
    """
    prep_root = Path(
        os.environ.get(
            "CLEARMESH_PREP_TMPDIR",
            f"/workspace/.cache/prepared_meshes/shard_{shard_id}",
        )
    )
    prep_root.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix="meshprep_", dir=str(prep_root)))
    asset_path = workdir / "prepared.glb"
    meta_path = workdir / "meta.json"
    helper = Path(__file__).with_name("prepare_mesh_asset.py")
    cmd = [
        sys.executable,
        str(helper),
        "--mesh",
        mesh_path,
        "--output",
        str(asset_path),
        "--meta",
        str(meta_path),
        "--max-file-bytes",
        str(MAX_MESH_FILE_BYTES),
        "--max-faces",
        str(MAX_MESH_FACES),
        "--memory-limit-gb",
        str(memory_limit_gb),
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=max(timeout_sec, 1),
        )
    except subprocess.TimeoutExpired:
        shutil.rmtree(workdir, ignore_errors=True)
        return None, FAIL_MESH_PREP_TIMEOUT, None, {}

    meta: dict = {}
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except Exception:
            meta = {}

    if result.returncode != 0 or not asset_path.exists():
        fail_reason = meta.get("reason") or FAIL_MESH_LOAD
        stderr_tail = "\n".join((result.stderr or "").splitlines()[-8:])
        stdout_tail = "\n".join((result.stdout or "").splitlines()[-8:])
        if stderr_tail or stdout_tail:
            print(
                "    [mesh_prep] failed: "
                + " | ".join(part for part in [stderr_tail, stdout_tail] if part)
            )
        shutil.rmtree(workdir, ignore_errors=True)
        return None, fail_reason, None, meta

    return str(asset_path), None, workdir, meta


# ── Fit-to-View Camera ─────────────────────────────────────────────────────

def _make_camera_transform(yaw_deg: float, pitch_deg: float, distance: float):
    """Create a 4x4 camera transform matrix (Z-up, OpenGL convention)."""
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)

    cx = distance * math.cos(pitch) * math.sin(yaw)
    cy = distance * math.cos(pitch) * math.cos(yaw)
    cz = distance * math.sin(pitch)
    cam_pos = np.array([cx, cy, cz])

    forward = cam_pos / np.linalg.norm(cam_pos)
    world_up = np.array([0, 0, 1.0])
    right = np.cross(world_up, forward)
    if np.linalg.norm(right) < 1e-6:
        world_up = np.array([0, 1.0, 0])
        right = np.cross(world_up, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    up /= np.linalg.norm(up)

    transform = np.eye(4)
    transform[:3, 0] = right
    transform[:3, 1] = up
    transform[:3, 2] = forward
    transform[:3, 3] = cam_pos
    return transform


def compute_fit_distance(mesh, fov_deg: float = 40.0, fill_fraction: float = 0.75):
    """Compute camera distance so mesh fills fill_fraction of the frame.

    Uses the mesh bounding sphere to ensure consistent framing regardless of
    mesh shape/size. This replaces the fixed distance=2.5 / scale=1.8 approach
    which could produce inconsistent fills for non-cuboid objects.

    Args:
        mesh: trimesh.Trimesh (already centered and normalized to unit cube)
        fov_deg: Camera field of view in degrees
        fill_fraction: Target fill (0.75 = 75% of frame)

    Returns:
        Camera distance from origin
    """
    # Bounding sphere radius
    r = np.linalg.norm(mesh.vertices, axis=1).max()
    if r < 1e-6:
        return 3.0  # Fallback for degenerate mesh

    # Distance so sphere subtends fill_fraction * FOV
    half_fov = math.radians(fov_deg / 2.0)
    # At distance d, the visible half-height is d * tan(half_fov)
    # We want r to fill fill_fraction of that: r = fill_fraction * d * tan(half_fov)
    # So: d = r / (fill_fraction * tan(half_fov))
    distance = r / (fill_fraction * math.tan(half_fov))

    # Clamp to reasonable range
    return max(1.5, min(distance, 8.0))


# ── Rendering ───────────────────────────────────────────────────────────────

def prepare_render_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Return a render-safe copy with meaningful vertex colors for DINOv2.

    Preserves existing vertex/face colors when they carry real variation.
    Falls back to normal-based shading (not flat gray) so DINOv2 gets
    geometric cues — concavities look dark, convexities look bright.
    TextureVisuals (UV-mapped materials) are stripped to avoid pyglet GL errors.
    """
    render_mesh = mesh.copy()
    try:
        visual = render_mesh.visual
        # If the mesh already carries varied vertex/face colors, keep them.
        if isinstance(visual, trimesh.visual.ColorVisuals):
            if hasattr(visual, "vertex_colors") and visual.vertex_colors is not None:
                vc = visual.vertex_colors
                if len(vc) > 0 and np.std(vc[:, :3].astype(float)) > 5.0:
                    return render_mesh

        # Fallback: normal-based shading gives DINOv2 geometric cues.
        normals = render_mesh.vertex_normals
        light_dir = np.array([0.5, 0.3, 0.8])
        light_dir /= np.linalg.norm(light_dir)
        diffuse = np.clip(normals @ light_dir, 0.15, 1.0)
        colors = (diffuse[:, None] * np.array([[210, 210, 215]])).clip(0, 255).astype(np.uint8)
        alpha = np.full((len(colors), 1), 255, dtype=np.uint8)
        render_mesh.visual = trimesh.visual.ColorVisuals(
            mesh=render_mesh, vertex_colors=np.hstack([colors, alpha]),
        )
    except Exception:
        pass
    return render_mesh


def render_multiview(
    mesh,
    num_views: int = 12,
    image_size: int = 1024,
    views: list[tuple[float, float]] | None = None,
    fill_fraction: float = 0.85,
):
    """Render a pre-loaded mesh from multiple camera angles.

    Returns:
        (list[Image.Image], None) on success
        ([], failure_reason) on failure

    The mesh should already be centered, normalized, and cleaned (via load_mesh_robust).
    """
    images = []
    views = views if views is not None else CAMERA_VIEWS[:num_views]
    render_mesh = prepare_render_mesh(mesh)

    # Compute fit-to-view distance
    distance = compute_fit_distance(render_mesh, fill_fraction=fill_fraction)

    gl_errors = 0

    for yaw_deg, pitch_deg in views:
        scene = trimesh.Scene(render_mesh)
        scene.camera.fov = (40, 40)
        transform = _make_camera_transform(yaw_deg, pitch_deg, distance)
        scene.camera_transform = transform

        try:
            with suppress_output(True):
                png = scene.save_image(
                    resolution=(image_size, image_size),
                    background=TRANSPARENT_BG,
                )
        except Exception as e:
            err = str(e).lower()
            gl_errors += 1
            if gl_errors >= 3:
                # If 3+ views fail on GL errors, the GL context is toast
                return [], FAIL_RENDER_GL
            continue  # Try other views

        if png is None or len(png) == 0:
            gl_errors += 1
            if gl_errors >= 3:
                return [], FAIL_RENDER_GL
            continue

        img = Image.open(BytesIO(png)).convert("RGBA")

        arr = np.array(img)
        transparent_mask = arr[:, :, 3] == 0
        arr[transparent_mask, :3] = 0
        images.append(Image.fromarray(arr))

    if not images:
        if gl_errors > 0:
            return [], FAIL_RENDER_GL
        return [], FAIL_RENDER_ZERO_FG

    return images, None


def render_single_view(
    mesh,
    view: tuple[float, float],
    image_size: int = 1024,
    fill_fraction: float = 0.9,
):
    """Render one rescue view with a tighter camera fit."""
    images, fail_reason = render_multiview(
        mesh,
        num_views=1,
        image_size=image_size,
        views=[view],
        fill_fraction=fill_fraction,
    )
    if images:
        return images[0], None
    return None, fail_reason


def render_multiview_blender_asset(
    asset_path: str,
    shard_id: int,
    blender_bin: str,
    blender_samples: int,
    num_views: int = 12,
    image_size: int = 1024,
    views: list[tuple[float, float]] | None = None,
    fill_fraction: float = 0.85,
):
    """Render a prepared asset through Blender Cycles in a short-lived subprocess."""
    views = views if views is not None else CAMERA_VIEWS[:num_views]
    render_root = Path(
        os.environ.get(
            "CLEARMESH_BLENDER_TMPDIR",
            f"/workspace/.cache/blender_renders/shard_{shard_id}",
        )
    )
    render_root.mkdir(parents=True, exist_ok=True)
    render_dir = Path(tempfile.mkdtemp(prefix="blender_", dir=str(render_root)))
    cmd = [
        blender_bin,
        "--background",
        "--factory-startup",
        "--python",
        str(Path(__file__).with_name("blender_render_views.py")),
        "--",
        "--mesh",
        asset_path,
        "--output_dir",
        str(render_dir),
        "--views_json",
        json.dumps(views),
        "--resolution",
        str(image_size),
        "--fill_fraction",
        str(fill_fraction),
        "--samples",
        str(blender_samples),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        shutil.rmtree(render_dir, ignore_errors=True)
        return [], FAIL_RENDER_GL

    if result.returncode != 0:
        tail = "\n".join(((result.stderr or "") + "\n" + (result.stdout or "")).splitlines()[-12:])
        if tail:
            print(f"    [blender] render failed: {tail}")
        shutil.rmtree(render_dir, ignore_errors=True)
        return [], FAIL_RENDER_GL

    images = []
    for idx in range(len(views)):
        path = render_dir / f"view_{idx:02d}.png"
        if not path.exists():
            continue
        img = Image.open(path).convert("RGBA")
        arr = np.array(img)
        transparent_mask = arr[:, :, 3] == 0
        arr[transparent_mask, :3] = 0
        images.append(Image.fromarray(arr))

    shutil.rmtree(render_dir, ignore_errors=True)
    if not images:
        return [], FAIL_RENDER_ZERO_FG
    return images, None


def render_single_view_blender_asset(
    asset_path: str,
    shard_id: int,
    blender_bin: str,
    blender_samples: int,
    view: tuple[float, float],
    image_size: int = 1024,
    fill_fraction: float = 0.9,
):
    images, fail_reason = render_multiview_blender_asset(
        asset_path=asset_path,
        shard_id=shard_id,
        blender_bin=blender_bin,
        blender_samples=blender_samples,
        num_views=1,
        image_size=image_size,
        views=[view],
        fill_fraction=fill_fraction,
    )
    if images:
        return images[0], None
    return None, fail_reason


def normalize_yaw_deg(yaw_deg: float) -> float:
    """Normalize yaw to [-180, 180)."""
    return ((yaw_deg + 180.0) % 360.0) - 180.0


def compute_silhouette_metrics(
    img: Image.Image,
    min_fg_frac: float = DEFAULT_MIN_FG_FRAC,
    max_fg_frac: float = DEFAULT_MAX_FG_FRAC,
    edge_margin: float = DEFAULT_EDGE_MARGIN,
) -> dict:
    """Summarize a rendered silhouette for ranking, logging, and rescue filtering."""
    arr = np.array(img)
    h, w = arr.shape[:2]
    total_pixels = max(h * w, 1)
    fg_mask = arr[:, :, 3] > 0
    fg_count = int(fg_mask.sum())
    fg_frac = fg_count / total_pixels
    metrics = {
        "fg_frac": float(fg_frac),
        "has_fg": bool(fg_frac > 0.001),
        "passes_gate": False,
        "bbox_fill": 0.0,
        "aspect_ratio": 0.0,
        "center_offset": 1.0,
        "edge_penalty": 1.0,
        "edge_density": 0.0,
        "edges_clipped": 0,
        "score": 999.0,
    }

    if fg_count <= 0:
        return metrics

    rows = np.any(fg_mask, axis=1)
    cols = np.any(fg_mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    bbox_h = int(rmax - rmin + 1)
    bbox_w = int(cmax - cmin + 1)
    bbox_area = max(bbox_h * bbox_w, 1)

    margin_h = int(h * edge_margin)
    margin_w = int(w * edge_margin)
    edges_clipped = 0
    if rmin < margin_h:
        edges_clipped += 1
    if rmax > h - margin_h - 1:
        edges_clipped += 1
    if cmin < margin_w:
        edges_clipped += 1
    if cmax > w - margin_w - 1:
        edges_clipped += 1

    row_center = (rmin + rmax) / 2.0
    col_center = (cmin + cmax) / 2.0
    center_offset = abs((row_center / h) - 0.5) + abs((col_center / w) - 0.5)
    edge_penalty = edges_clipped / 4.0
    bbox_fill = fg_count / bbox_area
    aspect_ratio = bbox_w / max(bbox_h, 1)

    padded = np.pad(fg_mask.astype(np.uint8), 1)
    edge_density = float(
        (
            np.abs(np.diff(padded, axis=0)).sum()
            + np.abs(np.diff(padded, axis=1)).sum()
        )
        / max(fg_count, 1)
    )
    ideal_fg = (min_fg_frac + max_fg_frac) / 2.0
    thinness_penalty = min(abs(math.log(max(aspect_ratio, 1e-6))), 2.0) / 2.0
    bbox_penalty = abs(bbox_fill - 0.58)
    score = (
        abs(fg_frac - ideal_fg)
        + 0.30 * center_offset
        + 0.20 * edge_penalty
        + 0.20 * bbox_penalty
        + 0.08 * thinness_penalty
        - 0.06 * min(edge_density, 2.0)
    )

    passes_gate = min_fg_frac <= fg_frac <= max_fg_frac
    if edges_clipped >= 2 and fg_frac > 0.5:
        passes_gate = False

    metrics.update({
        "passes_gate": bool(passes_gate),
        "bbox_fill": float(bbox_fill),
        "aspect_ratio": float(aspect_ratio),
        "center_offset": float(center_offset),
        "edge_penalty": float(edge_penalty),
        "edge_density": float(edge_density),
        "edges_clipped": int(edges_clipped),
        "score": float(score),
    })
    return metrics


def format_silhouette_metrics(metrics: dict) -> str:
    """Compact human-readable silhouette summary for logs."""
    return (
        f"fg={metrics['fg_frac']:.3f} "
        f"bbox_fill={metrics['bbox_fill']:.3f} "
        f"aspect={metrics['aspect_ratio']:.2f} "
        f"edge_density={metrics['edge_density']:.2f} "
        f"edges={metrics['edges_clipped']} "
        f"score={metrics['score']:.3f}"
    )


def select_diverse_gated_views(
    gated: list[tuple[Image.Image, float, str, int, float, dict]],
    max_views: int = MAX_VIEWS_TO_TRY,
) -> list[tuple[Image.Image, float, str, int, float, dict]]:
    """Pick a small set of high-quality but non-redundant gated views."""
    if not gated or max_views <= 0:
        return []

    selected = [gated[0]]
    remaining = gated[1:]
    while remaining and len(selected) < max_views:
        best_idx = 0
        best_score = None
        for idx, candidate in enumerate(remaining):
            min_distance = min(abs(candidate[3] - chosen[3]) for chosen in selected)
            adjusted = candidate[4] - 0.03 * min_distance
            if best_score is None or adjusted < best_score:
                best_score = adjusted
                best_idx = idx
        selected.append(remaining.pop(best_idx))
    return selected


def build_trellis_attempt_plan(
    selected_views: list[tuple[Image.Image, float, str, int, float, dict]],
) -> list[tuple[int, Image.Image, int]]:
    """Prioritize easy wins before paying for more views/seeds.

    Attempt order:
      1. Best gated view with the primary seed
      2. Best gated view with the remaining seeds
      3. Additional gated views, primary seed first
      4. Only strong secondary views get one extra seed
    """
    if not selected_views:
        return []

    attempt_plan = []
    primary_seed = GENERATION_SEEDS[0]
    attempt_plan.append((0, selected_views[0][0], primary_seed))

    for seed in GENERATION_SEEDS[1:]:
        attempt_plan.append((0, selected_views[0][0], seed))

    for view_idx, view_info in enumerate(selected_views[1:], start=1):
        image = view_info[0]
        attempt_plan.append((view_idx, image, primary_seed))
        if view_info[4] <= STRONG_SECONDARY_VIEW_SCORE and view_info[1] >= STRONG_SECONDARY_VIEW_MIN_FG:
            for seed in SECONDARY_VIEW_EXTRA_SEEDS:
                if seed != primary_seed:
                    attempt_plan.append((view_idx, image, seed))

    return attempt_plan


# ── Silhouette Gate ─────────────────────────────────────────────────────────

def silhouette_gate(
    images: list[Image.Image],
    min_fg_frac: float = DEFAULT_MIN_FG_FRAC,
    max_fg_frac: float = DEFAULT_MAX_FG_FRAC,
    edge_margin: float = DEFAULT_EDGE_MARGIN,
) -> tuple[list[tuple[Image.Image, float, str, int, float, dict]], bool]:
    """Filter rendered views by silhouette quality. Returns sorted passing views."""
    passed = []
    has_any_fg = False

    for image_idx, img in enumerate(images):
        metrics = compute_silhouette_metrics(
            img,
            min_fg_frac=min_fg_frac,
            max_fg_frac=max_fg_frac,
            edge_margin=edge_margin,
        )
        if metrics["has_fg"]:
            has_any_fg = True
        if not metrics["passes_gate"]:
            continue
        passed.append((img, metrics["fg_frac"], "pass", image_idx, metrics["score"], metrics))

    ideal_fg = (min_fg_frac + max_fg_frac) / 2.0
    passed.sort(key=lambda x: (x[4], abs(x[1] - ideal_fg), x[3]))
    return passed, has_any_fg


# ── TRELLIS.2 Generation ───────────────────────────────────────────────────

def generate_coarse_mesh_shape_only(
    pipeline,
    image: Image.Image,
    pipeline_type: str = "512",
    seed: int = 42,
    verbose_trellis_progress: bool = False,
    sparse_structure_sampler_params: dict | None = None,
    shape_slat_sampler_params: dict | None = None,
    return_intermediates: bool = False,
):
    """Run TRELLIS.2 shape-only path (skip texture generation)."""
    with torch.no_grad():
        processed_image = pipeline.preprocess_image(image)
        torch.manual_seed(seed)
        with suppress_output(not verbose_trellis_progress):
            cond_512 = pipeline.get_cond([processed_image], 512)
            cond_1024 = pipeline.get_cond([processed_image], 1024) if pipeline_type != "512" else None
            ss_res = {"512": 32, "1024": 64, "1024_cascade": 32, "1536_cascade": 32}[pipeline_type]
            coords = pipeline.sample_sparse_structure(
                cond_512,
                ss_res,
                1,
                sparse_structure_sampler_params or {},
            )

            if pipeline_type == "512":
                shape_slat = pipeline.sample_shape_slat(
                    cond_512,
                    pipeline.models["shape_slat_flow_model_512"],
                    coords,
                    shape_slat_sampler_params or {},
                )
                res = 512
            elif pipeline_type == "1024":
                shape_slat = pipeline.sample_shape_slat(
                    cond_1024,
                    pipeline.models["shape_slat_flow_model_1024"],
                    coords,
                    shape_slat_sampler_params or {},
                )
                res = 1024
            elif pipeline_type == "1024_cascade":
                shape_slat, res = pipeline.sample_shape_slat_cascade(
                    cond_512,
                    cond_1024,
                    pipeline.models["shape_slat_flow_model_512"],
                    pipeline.models["shape_slat_flow_model_1024"],
                    512,
                    1024,
                    coords,
                    shape_slat_sampler_params or {},
                    49152,
                )
            elif pipeline_type == "1536_cascade":
                shape_slat, res = pipeline.sample_shape_slat_cascade(
                    cond_512,
                    cond_1024,
                    pipeline.models["shape_slat_flow_model_512"],
                    pipeline.models["shape_slat_flow_model_1024"],
                    512,
                    1536,
                    coords,
                    shape_slat_sampler_params or {},
                    49152,
                )
            else:
                raise ValueError(f"Invalid pipeline type: {pipeline_type}")

            meshes, _ = pipeline.decode_shape_slat(shape_slat, res)

        coarse_mesh = meshes[0]

        # Capture intermediates for Stage 2 training before cleanup
        intermediates = None
        if return_intermediates:
            try:
                # coords: sparse structure voxel positions
                # May be (N, 4) with [batch, x, y, z] or (N, 3) [x, y, z]
                if isinstance(coords, torch.Tensor):
                    if coords.dim() == 2 and coords.shape[1] == 4:
                        coords_np = coords[:, 1:].cpu().numpy().astype(np.int32)
                    elif coords.dim() == 2 and coords.shape[1] == 3:
                        coords_np = coords.cpu().numpy().astype(np.int32)
                    else:
                        coords_np = coords.cpu().numpy().astype(np.int32)
                elif hasattr(coords, 'coords'):
                    # SparseTensor from spconv
                    c = coords.coords
                    coords_np = (c[:, 1:] if c.shape[1] == 4 else c).cpu().numpy().astype(np.int32)
                else:
                    coords_np = None

                # shape_slat: SLAT features (N, 32)
                if hasattr(shape_slat, 'feats'):
                    slat_feats = shape_slat.feats.cpu().numpy().astype(np.float16)
                elif hasattr(shape_slat, 'F'):
                    slat_feats = shape_slat.F.cpu().numpy().astype(np.float16)
                elif isinstance(shape_slat, torch.Tensor):
                    t = shape_slat.squeeze(0) if shape_slat.dim() == 3 else shape_slat
                    slat_feats = t.cpu().numpy().astype(np.float16)
                else:
                    slat_feats = None

                # cond_512: DINOv2 features (M, 1024)
                if isinstance(cond_512, dict):
                    cond_tensor = cond_512.get('cond', cond_512.get('image_cond'))
                    if cond_tensor is None:
                        # Try first value
                        cond_tensor = next(iter(cond_512.values()))
                elif isinstance(cond_512, (list, tuple)):
                    cond_tensor = cond_512[0]
                else:
                    cond_tensor = cond_512

                if isinstance(cond_tensor, torch.Tensor):
                    ct = cond_tensor.squeeze(0) if cond_tensor.dim() == 3 else cond_tensor
                    cond_feats = ct.cpu().numpy().astype(np.float16)
                else:
                    cond_feats = None

                # Shape assertions — catch extraction bugs early
                if coords_np is not None:
                    if coords_np.ndim != 2 or coords_np.shape[1] != 3:
                        print(f"    [intermediates] WARNING: coords shape {coords_np.shape}, expected (N, 3)")
                        coords_np = None

                if slat_feats is not None:
                    if slat_feats.ndim != 2 or slat_feats.shape[1] != 32:
                        print(f"    [intermediates] WARNING: SLAT shape {slat_feats.shape}, expected (N, 32)")
                        slat_feats = None

                if cond_feats is not None:
                    # DINOv2-ViT-L: 1024-dim, typically 257 tokens (256 patches + CLS)
                    # or 1025 for higher-res. Accept any reasonable M.
                    if cond_feats.ndim != 2 or cond_feats.shape[1] != 1024:
                        print(f"    [intermediates] WARNING: DINOv2 shape {cond_feats.shape}, expected (M, 1024)")
                        cond_feats = None
                    elif cond_feats.shape[0] not in (257, 1025):
                        print(f"    [intermediates] NOTE: DINOv2 token count {cond_feats.shape[0]}"
                              f" (expected 257 or 1025, continuing anyway)")

                if coords_np is not None and slat_feats is not None:
                    if coords_np.shape[0] != slat_feats.shape[0]:
                        print(f"    [intermediates] WARNING: coords/SLAT count mismatch:"
                              f" {coords_np.shape[0]} vs {slat_feats.shape[0]}")
                        coords_np, slat_feats = None, None

                if coords_np is not None and slat_feats is not None:
                    intermediates = {
                        'positions': coords_np,
                        'coarse_voxels': slat_feats,
                        'cond_features': cond_feats,
                        'ss_res': ss_res,
                    }
            except Exception as e:
                print(f"    [intermediates] failed to capture: {e}")
                intermediates = None

        # Eagerly release intermediates before returning — their CPU shadows
        # from low_vram model shuffling are the main source of RSS growth.
        del processed_image, cond_512, coords, shape_slat, meshes
        if cond_1024 is not None:
            del cond_1024
        cleanup_memory(use_malloc_trim=False)

        if return_intermediates:
            return coarse_mesh, intermediates
        return coarse_mesh


def generate_coarse_mesh(
    pipeline,
    image: Image.Image,
    pipeline_type: str = "512",
    seed: int = 42,
    geometry_only: bool = True,
    verbose_trellis_progress: bool = False,
    sparse_structure_sampler_params: dict | None = None,
    shape_slat_sampler_params: dict | None = None,
    return_intermediates: bool = False,
):
    """Run TRELLIS.2 to generate a coarse mesh from an image."""
    if geometry_only:
        return generate_coarse_mesh_shape_only(
            pipeline=pipeline,
            image=image,
            pipeline_type=pipeline_type,
            seed=seed,
            verbose_trellis_progress=verbose_trellis_progress,
            sparse_structure_sampler_params=sparse_structure_sampler_params,
            shape_slat_sampler_params=shape_slat_sampler_params,
            return_intermediates=return_intermediates,
        )

    with torch.no_grad():
        results = pipeline.run(
            image,
            pipeline_type=pipeline_type,
            preprocess_image=True,
            seed=seed,
            sparse_structure_sampler_params=sparse_structure_sampler_params or {},
            shape_slat_sampler_params=shape_slat_sampler_params or {},
        )
        coarse_mesh = results[0]
        del results
        cleanup_memory(use_malloc_trim=False)
        return coarse_mesh


def save_pair(coarse_mesh, fine_mesh_path: str, output_dir: str, uid: str,
              intermediates: dict | None = None):
    """Save a coarse/fine pair to disk.

    If *intermediates* is provided (from return_intermediates=True),
    also saves SLAT features, voxel positions, and DINOv2 conditioning
    as numpy arrays for Stage 2 training.
    """
    pair_dir = os.path.join(output_dir, uid)
    os.makedirs(pair_dir, exist_ok=True)

    coarse_path = os.path.join(pair_dir, "coarse.glb")
    verts = coarse_mesh.vertices
    faces = coarse_mesh.faces
    if isinstance(verts, torch.Tensor):
        verts = verts.cpu().numpy()
    if isinstance(faces, torch.Tensor):
        faces = faces.cpu().numpy()
    t = trimesh.Trimesh(vertices=verts, faces=faces)
    t.export(coarse_path)

    fine_ext = Path(fine_mesh_path).suffix
    fine_link = os.path.join(pair_dir, f"fine{fine_ext}")
    if not os.path.exists(fine_link):
        try:
            os.symlink(os.path.abspath(fine_mesh_path), fine_link)
        except OSError:
            import shutil
            shutil.copy2(fine_mesh_path, fine_link)

    # Save SLAT + DINOv2 intermediates for Stage 2 training
    if intermediates is not None:
        try:
            np.save(os.path.join(pair_dir, "coarse_voxels.npy"),
                    intermediates['coarse_voxels'])
            np.save(os.path.join(pair_dir, "positions.npy"),
                    intermediates['positions'])
            if intermediates.get('cond_features') is not None:
                np.save(os.path.join(pair_dir, "cond_features.npy"),
                        intermediates['cond_features'])
        except Exception as e:
            print(f"    [save_pair] warning: failed to save intermediates: {e}")

    return pair_dir


# ── Main Loop ───────────────────────────────────────────────────────────────

def generate_pairs(
    input_json: str,
    output_dir: str,
    pipeline_type: str = "512",
    render_size: int = 1024,
    num_views: int = 12,
    limit: int | None = None,
    trellis2_dir: str = "/workspace/TRELLIS.2",
    model_dir: str = "/workspace/models/trellis2-4b",
    min_fg_frac: float = DEFAULT_MIN_FG_FRAC,
    max_fg_frac: float = DEFAULT_MAX_FG_FRAC,
    edge_margin: float = DEFAULT_EDGE_MARGIN,
    shard_id: int = 0,
    num_shards: int = 1,
    gpu: int = 0,
    retry_failed: bool = False,
    geometry_only: bool = True,
    disable_rembg_model: bool = True,
    log_rss_every: int = 25,
    rss_hard_limit_gb: float = 0.0,
    rss_emergency_limit_gb: float = 0.0,
    rss_watch_interval_sec: float = 0.25,
    max_models_per_run: int = 0,
    max_emergency_recycles_per_uid: int = 3,
    low_vram: bool = False,
    verbose_trellis_progress: bool = False,
    use_malloc_trim: bool = True,
    recycle_exit_code: int = 75,
    render_backend: str = DEFAULT_RENDER_BACKEND,
    blender_bin: str = DEFAULT_BLENDER_BIN,
    blender_samples: int = DEFAULT_BLENDER_SAMPLES,
    mesh_prep_memory_limit_gb: float = DEFAULT_MESH_PREP_MEMORY_LIMIT_GB,
    mesh_prep_timeout_sec: int = DEFAULT_MESH_PREP_TIMEOUT_SEC,
):
    """Main pair generation loop with all v5 improvements."""

    # ── Step 0: Per-shard cache isolation ──
    isolate_caches(shard_id)

    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    if num_shards > 1:
        shard_output_dir = os.path.join(output_dir, f"shard_{shard_id}")
    else:
        shard_output_dir = output_dir

    output_path = Path(shard_output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load filtered model list
    with open(input_json) as f:
        models = json.load(f)

    if limit:
        models = models[:limit]

    # Shard: shuffle with fixed seed then interleave to balance quality across shards.
    # Without shuffling, interleaved modulo assignment groups Objaverse models by source
    # dataset (e.g., shard 0 gets all ShapeNet, shard 1 gets all Thingiverse), creating
    # quality imbalance.
    if num_shards > 1:
        import random as _random
        _rng = _random.Random(42)
        models = list(models)  # copy so we don't mutate the original
        _rng.shuffle(models)
        models = [m for i, m in enumerate(models) if i % num_shards == shard_id]

    print(f"Generating coarse/fine pairs for {len(models)} models (shard {shard_id}/{num_shards})")
    print(f"  Pipeline type: {pipeline_type}, GPU: {gpu}")
    print(f"  Render: {render_size}x{render_size}, {num_views} views/model")
    print(f"  Render backend: {render_backend}")
    if render_backend == "blender":
        print(f"  Blender bin: {blender_bin} (samples={blender_samples})")
    print(
        f"  Mesh prep subprocess: {mesh_prep_memory_limit_gb:.0f} GiB limit, "
        f"{mesh_prep_timeout_sec}s timeout"
    )
    print(
        f"  Adaptive render: first {min(INITIAL_RENDER_VIEWS, num_views)} views, "
        f"expand to {num_views} only if gate finds 0 valid"
    )
    print(f"  Silhouette gate: FG ∈ [{min_fg_frac:.0%}, {max_fg_frac:.0%}]")
    print(
        "  TRELLIS attempt budget: "
        f"best view all seeds, secondary strong views get one extra seed, "
        f"rescue capped at {MAX_RESCUE_ATTEMPTS} attempts across {MAX_RESCUE_CANDIDATES} candidates"
    )
    print("  TRELLIS retry ladder: best view first, then expand only promising views")
    print(f"  Retry-failed mode: {retry_failed}")
    print(f"  Geometry-only mode: {geometry_only}")
    print(f"  TRELLIS inner progress: {'verbose' if verbose_trellis_progress else 'suppressed'}")
    print(f"  Initial RSS: {get_rss_gb():.1f} GiB")
    if torch.cuda.is_available():
        a_mb, r_mb, m_mb = get_cuda_memory_stats_mb()
        print(f"  Initial CUDA mem: alloc={a_mb:.0f}MiB reserved={r_mb:.0f}MiB max={m_mb:.0f}MiB")
    effective_emergency_limit_gb = rss_emergency_limit_gb or rss_hard_limit_gb
    if rss_hard_limit_gb > 0:
        print(f"  RSS hard limit: {rss_hard_limit_gb:.1f} GiB (auto-recycle)")
    if effective_emergency_limit_gb > 0 and rss_watch_interval_sec > 0:
        print(
            f"  RSS emergency limit: {effective_emergency_limit_gb:.1f} GiB "
            f"(poll every {rss_watch_interval_sec:.2f}s)"
        )
    if max_models_per_run > 0:
        print(f"  Max models this run: {max_models_per_run} (auto-recycle)")
    if max_emergency_recycles_per_uid > 0:
        print(
            f"  Max consecutive emergency recycles per UID: "
            f"{max_emergency_recycles_per_uid}"
        )
        print("  Load-mesh emergency recycles quarantine after 1 hit")

    # ── Progress tracking ──
    progress_path = output_path / "progress.json"
    failed_path = output_path / "failed.json"
    failure_details_path = output_path / "failure_details.json"
    recycle_history_path = output_path / "recycle_history.json"

    # Load cross-shard completions
    all_completed = set()
    base_output = Path(output_dir)
    if (base_output / "progress.json").exists():
        with open(base_output / "progress.json") as f:
            all_completed.update(json.load(f))
    for shard_dir in sorted(base_output.glob("shard_*")):
        sp = shard_dir / "progress.json"
        if sp.exists():
            with open(sp) as f:
                all_completed.update(json.load(f))

    if progress_path.exists():
        with open(progress_path) as f:
            completed = set(json.load(f))
    else:
        completed = set()

    if failed_path.exists():
        with open(failed_path) as f:
            permanently_failed = set(json.load(f))
    else:
        permanently_failed = set()

    # Load existing failure details
    if failure_details_path.exists():
        with open(failure_details_path) as f:
            failure_details = json.load(f)
    else:
        failure_details = {}

    recycle_state_path = output_path / "recycle_state.json"
    progress_lock = threading.RLock()
    current_model = dict(CURRENT_MODEL_NONE)

    def _set_current_model(uid=None, path=None, phase="idle"):
        with progress_lock:
            current_model["uid"] = uid
            current_model["path"] = path
            current_model["phase"] = phase

    def _current_model_snapshot():
        with progress_lock:
            return dict(current_model)

    def _save_all_progress():
        with progress_lock:
            _save_progress(
                progress_path,
                failed_path,
                failure_details_path,
                completed,
                permanently_failed,
                failure_details,
            )

    def _save_recycle_state(rss_now: float, state: dict):
        payload = {
            "rss_gb": rss_now,
            "limit_gb": effective_emergency_limit_gb,
            "time": time.time(),
            "uid": state.get("uid"),
            "path": state.get("path"),
            "phase": state.get("phase"),
        }
        tmp = str(recycle_state_path) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, recycle_state_path)

    def _save_recycle_history(payload: dict):
        tmp = str(recycle_history_path) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, recycle_history_path)

    def _clear_recycle_artifacts():
        for path in (recycle_state_path, recycle_history_path):
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    def _handle_emergency_recycle(rss_now: float, state: dict):
        _save_all_progress()
        _save_recycle_state(rss_now, state)
        previous = {}
        if recycle_history_path.exists():
            try:
                with open(recycle_history_path) as f:
                    previous = json.load(f)
            except Exception:
                previous = {}

        same_target = (
            previous.get("uid") == state.get("uid")
            and previous.get("phase") == state.get("phase")
            and previous.get("path") == state.get("path")
        )
        count = int(previous.get("count", 0)) + 1 if same_target else 1
        _save_recycle_history(
            {
                "uid": state.get("uid"),
                "path": state.get("path"),
                "phase": state.get("phase"),
                "count": count,
                "rss_gb": rss_now,
                "limit_gb": effective_emergency_limit_gb,
                "time": time.time(),
            }
        )

    rss_watchdog = RssWatchdog(
        limit_gb=effective_emergency_limit_gb,
        recycle_exit_code=recycle_exit_code,
        poll_interval_sec=rss_watch_interval_sec,
        progress_callback=_handle_emergency_recycle,
        state_callback=_current_model_snapshot,
    )
    rss_watchdog.start()

    if max_emergency_recycles_per_uid > 0 and recycle_history_path.exists():
        try:
            with open(recycle_history_path) as f:
                recycle_history = json.load(f)
        except Exception:
            recycle_history = {}

        recycle_uid = recycle_history.get("uid")
        recycle_count = int(recycle_history.get("count", 0))
        recycle_path = recycle_history.get("path")
        recycle_phase = recycle_history.get("phase")
        recycle_threshold = 1 if recycle_phase == "load_mesh" else max_emergency_recycles_per_uid
        if (
            recycle_uid
            and recycle_count >= recycle_threshold
            and recycle_uid not in completed
            and recycle_uid not in permanently_failed
        ):
            permanently_failed.add(recycle_uid)
            failure_details[recycle_uid] = {
                "reason": FAIL_RSS_RECYCLE,
                "path": recycle_path,
                "phase": recycle_phase,
                "recycle_count": recycle_count,
                "rss_gb": recycle_history.get("rss_gb"),
                "limit_gb": recycle_history.get("limit_gb"),
                "skipped_on_startup": True,
            }
            _save_all_progress()
            _clear_recycle_artifacts()
            print(
                f"  Quarantined UID after {recycle_count} emergency recycles: "
                f"{recycle_uid} (phase={recycle_phase})"
            )

    all_failed = _collect_all_failed_uids(base_output)

    # ── Build work list ──
    if retry_failed:
        # Only retry failures that haven't been rescued
        retry_uids = all_failed - all_completed - completed
        uid_to_model = {m["uid"]: m for m in models}
        remaining = [uid_to_model[uid] for uid in retry_uids if uid in uid_to_model]

        # Clear this shard's failed list so retried models get a fresh chance
        permanently_failed = set()
        failure_details = {}

        print(f"  RETRY MODE: {len(remaining)} previously-failed models to retry")
    else:
        skip_uids = completed | all_failed | all_completed
        remaining = [m for m in models if m["uid"] not in skip_uids]
        print(f"  Already completed (this shard): {len(completed)}")
        print(f"  Already completed (all shards): {len(all_completed)}")
        print(f"  Previously failed (all runs): {len(all_failed)}")
        print(f"  Remaining: {len(remaining)}")

    if not remaining:
        print("  No remaining models for this shard; skipping TRELLIS load")
        rss_watchdog.stop()
        return {
            "recycle_requested": False,
            "recycle_reason": None,
            "processed": 0,
            "remaining": 0,
        }

    # Load TRELLIS.2 only after progress/failure/recycle preflight so
    # pathological load_mesh recycles don't force another full model reload.
    print("Loading TRELLIS.2...")
    _set_current_model(phase="pipeline_load")
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    sys.path.insert(0, trellis2_dir)
    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    t0 = time.time()
    if os.path.exists(model_dir):
        pipeline = Trellis2ImageTo3DPipeline.from_pretrained(model_dir)
    else:
        pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")

    print(f"  RSS after from_pretrained: {get_rss_gb():.1f} GiB")
    removed_models = prune_pipeline_models(pipeline, pipeline_type, geometry_only)
    if removed_models:
        print(f"  Pruned {len(removed_models)} unused model branches")
    if disable_rembg_model and getattr(pipeline, "rembg_model", None) is not None:
        pipeline.rembg_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if use_malloc_trim:
            malloc_trim()
        print("  Unloaded rembg model (input renders already include alpha)")
    print(f"  RSS after pruning/unload: {get_rss_gb():.1f} GiB")

    # Disable low_vram to keep all models on GPU (A100 80GB has plenty of VRAM).
    # With low_vram=True, each TRELLIS call shuffles models CPU↔GPU (3-5s overhead).
    # With low_vram=False, models stay on GPU and inference is ~3x faster.
    if not low_vram:
        pipeline.low_vram = False
        for m in pipeline.models.values():
            if hasattr(m, "low_vram"):
                m.low_vram = False
        if hasattr(pipeline, "image_cond_model") and pipeline.image_cond_model is not None:
            if hasattr(pipeline.image_cond_model, "low_vram"):
                pipeline.image_cond_model.low_vram = False
        print("  low_vram disabled: all models will stay on GPU")

    pipeline.cuda()
    print(f"  RSS after pipeline.cuda(): {get_rss_gb():.1f} GiB")
    if torch.cuda.is_available():
        a_mb, r_mb, m_mb = get_cuda_memory_stats_mb()
        print(f"  CUDA mem after pipeline.cuda(): alloc={a_mb:.0f}MiB reserved={r_mb:.0f}MiB max={m_mb:.0f}MiB")
    gc.collect()
    torch.cuda.empty_cache()
    _set_current_model()
    print(f"  Pipeline loaded in {time.time()-t0:.1f}s")

    # ── Counters ──
    pairs_created = 0
    failures = 0
    failure_counts = Counter()
    total_views_rendered = 0
    total_views_passed = 0
    total_time = 0.0
    start_time = time.time()
    recycle_reason = None

    try:
        for model in tqdm(remaining, desc=f"Shard {shard_id}"):
            processed = pairs_created + failures
            if max_models_per_run > 0 and processed >= max_models_per_run:
                recycle_reason = f"processed {processed} models (limit {max_models_per_run})"
                break
            if rss_hard_limit_gb > 0:
                rss_now = get_rss_gb()
                if rss_now >= rss_hard_limit_gb:
                    recycle_reason = f"RSS {rss_now:.1f} GiB reached hard limit {rss_hard_limit_gb:.1f} GiB"
                    break

            uid = model["uid"]
            mesh_path = model["path"]
            prepared_workdir = None
            prepared_asset_path = None
            mesh = None

            _set_current_model(uid=uid, path=mesh_path, phase="load_mesh")
            t_start = time.time()
            fail_reason = None

            # ── Step 1: Load + normalize mesh in a subprocess ──
            prepared_asset_path, fail_reason, prepared_workdir, prep_meta = prepare_mesh_asset_subprocess(
                mesh_path=mesh_path,
                shard_id=shard_id,
                memory_limit_gb=mesh_prep_memory_limit_gb,
                timeout_sec=mesh_prep_timeout_sec,
            )
            if fail_reason:
                with progress_lock:
                    failure_counts[fail_reason] += 1
                    failures += 1
                    permanently_failed.add(uid)
                    failure_details[uid] = {"reason": fail_reason, "path": mesh_path}
                _save_all_progress()
                _clear_recycle_artifacts()
                if use_malloc_trim:
                    malloc_trim()
                _set_current_model()
                continue

            if render_backend == "trimesh":
                try:
                    mesh = trimesh.load(prepared_asset_path, force="mesh", process=False)
                except Exception:
                    fail_reason = FAIL_MESH_LOAD
                    with progress_lock:
                        failure_counts[fail_reason] += 1
                        failures += 1
                        permanently_failed.add(uid)
                        failure_details[uid] = {
                            "reason": fail_reason,
                            "path": mesh_path,
                            "prepared_asset": prepared_asset_path,
                        }
                    _save_all_progress()
                    _clear_recycle_artifacts()
                    if prepared_workdir is not None:
                        shutil.rmtree(prepared_workdir, ignore_errors=True)
                    cleanup_memory(use_malloc_trim=use_malloc_trim)
                    _set_current_model()
                    continue

            # ── Step 2: Render multiple views ──
            _set_current_model(uid=uid, path=mesh_path, phase="render")
            initial_view_count = min(num_views, INITIAL_RENDER_VIEWS)
            initial_views = CAMERA_VIEWS[:initial_view_count]
            if render_backend == "blender":
                images, fail_reason = render_multiview_blender_asset(
                    asset_path=prepared_asset_path,
                    shard_id=shard_id,
                    blender_bin=blender_bin,
                    blender_samples=blender_samples,
                    num_views=initial_view_count,
                    image_size=render_size,
                    views=initial_views,
                )
            else:
                images, fail_reason = render_multiview(
                    mesh, initial_view_count, render_size, views=initial_views
                )
            total_views_rendered += initial_view_count

            if fail_reason:
                with progress_lock:
                    failure_counts[fail_reason] += 1
                    failures += 1
                    permanently_failed.add(uid)
                    failure_details[uid] = {"reason": fail_reason, "path": mesh_path}
                _save_all_progress()
                _clear_recycle_artifacts()
                if mesh is not None:
                    del mesh
                if prepared_workdir is not None:
                    shutil.rmtree(prepared_workdir, ignore_errors=True)
                gc.collect()
                if use_malloc_trim:
                    malloc_trim()
                _set_current_model()
                continue

            # ── Step 3: Silhouette gate ──
            _set_current_model(uid=uid, path=mesh_path, phase="gate")
            gated, has_any_fg = silhouette_gate(images, min_fg_frac, max_fg_frac, edge_margin)
            if not gated and num_views > initial_view_count:
                _set_current_model(uid=uid, path=mesh_path, phase="render_expand")
                extra_views = CAMERA_VIEWS[initial_view_count:num_views]
                if render_backend == "blender":
                    expanded_images, expanded_fail_reason = render_multiview_blender_asset(
                        asset_path=prepared_asset_path,
                        shard_id=shard_id,
                        blender_bin=blender_bin,
                        blender_samples=blender_samples,
                        num_views=len(extra_views),
                        image_size=render_size,
                        views=extra_views,
                    )
                else:
                    expanded_images, expanded_fail_reason = render_multiview(
                        mesh, len(extra_views), render_size, views=extra_views
                    )
                total_views_rendered += len(extra_views)
                if expanded_images:
                    images.extend(expanded_images)
                if expanded_fail_reason and not expanded_images:
                    fail_reason = expanded_fail_reason
                _set_current_model(uid=uid, path=mesh_path, phase="gate")
                gated, has_any_fg = silhouette_gate(images, min_fg_frac, max_fg_frac, edge_margin)
            valid_images = [g[0] for g in gated]
            total_views_passed += len(valid_images)

            if not valid_images:
                if fail_reason is None and not has_any_fg:
                    fail_reason = FAIL_RENDER_ZERO_FG
                elif fail_reason is None:
                    fail_reason = FAIL_GATE_REJECTED
                fg_fracs = []
                for img in images:
                    arr = np.array(img)
                    fg_fracs.append(float((arr[:,:,3] > 0).sum() / (arr.shape[0]*arr.shape[1])))
                with progress_lock:
                    failure_counts[fail_reason] += 1
                    failures += 1
                    permanently_failed.add(uid)
                    failure_details[uid] = {
                        "reason": fail_reason,
                        "path": mesh_path,
                        "fg_fracs": fg_fracs,
                    }
                _save_all_progress()
                _clear_recycle_artifacts()
                if mesh is not None:
                    del mesh
                if prepared_workdir is not None:
                    shutil.rmtree(prepared_workdir, ignore_errors=True)
                del images
                del valid_images
                del gated
                cleanup_memory(use_malloc_trim=use_malloc_trim)
                _set_current_model()
                continue

            selected_views = select_diverse_gated_views(gated, MAX_VIEWS_TO_TRY)
            valid_images = [view[0] for view in selected_views]

            # ── Step 4: Multi-seed + multi-view TRELLIS.2 generation ──
            _set_current_model(uid=uid, path=mesh_path, phase="trellis")
            success = False
            total_trellis_attempts = 0
            trellis_empty_count = 0
            trellis_oom_count = 0
            trellis_errors = []
            rss_before_trellis = get_rss_gb()

            consecutive_empty = 0
            attempt_plan = build_trellis_attempt_plan(selected_views)
            for view_idx, image, seed in attempt_plan:
                total_trellis_attempts += 1
                rss_pre = get_rss_gb()
                _set_current_model(uid=uid, path=mesh_path, phase=f"trellis:view{view_idx}:seed{seed}")
                try:
                    trellis_result = generate_coarse_mesh(
                        pipeline, image, pipeline_type, seed=seed,
                        geometry_only=geometry_only,
                        verbose_trellis_progress=verbose_trellis_progress,
                        sparse_structure_sampler_params=DEFAULT_SPARSE_SAMPLER_PARAMS,
                        shape_slat_sampler_params=DEFAULT_SHAPE_SAMPLER_PARAMS,
                        return_intermediates=True,
                    )
                    if isinstance(trellis_result, tuple):
                        coarse, slat_intermediates = trellis_result
                    else:
                        coarse, slat_intermediates = trellis_result, None

                    n_verts = int(coarse.vertices.shape[0])
                    if n_verts < MIN_COARSE_VERTS:
                        del coarse, slat_intermediates
                        cleanup_memory(use_malloc_trim=use_malloc_trim)
                        consecutive_empty = 0  # Non-empty result resets counter
                        rss_post = get_rss_gb()
                        print(f"    [trellis] v{view_idx}/s{seed} quality_reject verts={n_verts} (too sparse) RSS={rss_post:.1f}GiB (Δ={rss_post-rss_pre:+.1f})")
                        continue

                    if n_verts > MAX_COARSE_VERTS:
                        # Decimate instead of rejecting — these are usable pairs
                        verts = coarse.vertices.cpu().numpy() if torch.is_tensor(coarse.vertices) else coarse.vertices
                        faces = coarse.faces.cpu().numpy() if torch.is_tensor(coarse.faces) else coarse.faces
                        coarse_tri = trimesh.Trimesh(vertices=verts, faces=faces)
                        coarse_tri = coarse_tri.simplify_quadric_decimation(DECIMATE_TARGET_FACES)
                        coarse = coarse_tri
                        n_verts = int(coarse.vertices.shape[0])
                        consecutive_empty = 0
                        print(f"    [trellis] v{view_idx}/s{seed} decimated to verts={n_verts}")

                    # ── Save pair ──
                    pair_dir = save_pair(coarse, mesh_path, shard_output_dir, uid,
                                        intermediates=slat_intermediates)
                    image.save(os.path.join(pair_dir, "rendered.png"))

                    meta = {
                        "uid": uid,
                        "fine_path": mesh_path,
                        "coarse_vertices": n_verts,
                        "coarse_faces": int(coarse.faces.shape[0]),
                        "seed": seed,
                        "view_index": view_idx,
                        "trellis_attempts": total_trellis_attempts,
                        "views_rendered": len(images),
                        "views_passed_gate": len(valid_images),
                        "winning_fg_fraction": float(selected_views[view_idx][1]) if view_idx < len(selected_views) else -1,
                        "winning_view_metrics": selected_views[view_idx][5] if view_idx < len(selected_views) else None,
                        "render_size": render_size,
                        "pipeline_type": pipeline_type,
                        "ss_res": slat_intermediates.get("ss_res") if slat_intermediates else None,
                        "has_slat": slat_intermediates is not None,
                    }
                    with open(os.path.join(pair_dir, "meta.json"), "w") as f:
                        json.dump(meta, f, indent=2)

                    with progress_lock:
                        completed.add(uid)
                        pairs_created += 1
                        elapsed = time.time() - t_start
                        total_time += elapsed
                    success = True
                    _clear_recycle_artifacts()

                    del coarse, slat_intermediates
                    cleanup_memory(use_malloc_trim=use_malloc_trim)
                    rss_post = get_rss_gb()
                    print(f"    [trellis] v{view_idx}/s{seed} SUCCESS verts={n_verts} RSS={rss_post:.1f}GiB (Δ={rss_post-rss_pre:+.1f})")
                    break

                except RuntimeError as e:
                    err_msg = str(e)
                    if "numel() == 0" in err_msg:
                        trellis_empty_count += 1
                        consecutive_empty += 1
                        cleanup_memory(use_malloc_trim=use_malloc_trim)
                        rss_post = get_rss_gb()
                        print(f"    [trellis] v{view_idx}/s{seed} empty ({consecutive_empty}/{MAX_CONSECUTIVE_EMPTY}) RSS={rss_post:.1f}GiB (Δ={rss_post-rss_pre:+.1f})")
                        if consecutive_empty >= MAX_CONSECUTIVE_EMPTY:
                            print(f"    [trellis] {consecutive_empty} consecutive empties — skipping remaining attempts")
                            break
                        continue
                    if "out of memory" in err_msg.lower():
                        trellis_oom_count += 1
                        cleanup_memory(use_malloc_trim=use_malloc_trim)
                        rss_post = get_rss_gb()
                        print(f"    [trellis] v{view_idx}/s{seed} OOM RSS={rss_post:.1f}GiB (Δ={rss_post-rss_pre:+.1f})")
                        continue
                    trellis_errors.append(err_msg[:200])
                    cleanup_memory(use_malloc_trim=use_malloc_trim)
                    rss_post = get_rss_gb()
                    print(f"    [trellis] v{view_idx}/s{seed} error RSS={rss_post:.1f}GiB (Δ={rss_post-rss_pre:+.1f})")
                    break
                except Exception as e:
                    trellis_errors.append(f"{type(e).__name__}: {str(e)[:200]}")
                    cleanup_memory(use_malloc_trim=use_malloc_trim)
                    rss_post = get_rss_gb()
                    print(f"    [trellis] v{view_idx}/s{seed} exception RSS={rss_post:.1f}GiB (Δ={rss_post-rss_pre:+.1f})")
                    break

            if (
                not success
                and selected_views
                and trellis_empty_count == total_trellis_attempts
                and total_trellis_attempts > 0
                and consecutive_empty < MAX_CONSECUTIVE_EMPTY
            ):
                rescue_camera_idx = selected_views[0][3]
                primary_view = CAMERA_VIEWS[rescue_camera_idx]
                selected_indices = {view[3] for view in selected_views}
                rescue_candidates = []
                rescue_attempts_used = 0

                extra_gated = [view for view in gated if view[3] not in selected_indices]
                extra_gated.sort(
                    key=lambda view: (
                        -min(abs(view[3] - chosen[3]) for chosen in selected_views),
                        view[4],
                    )
                )
                for view in extra_gated[:2]:
                    rescue_candidates.append(
                        (f"alt_view_{view[3]}", view[0], float(view[1]), float(view[4]), view[5], [42, 123])
                    )

                seen_rotations = set()
                for yaw_shift in (45.0, -45.0, 90.0, -90.0):
                    rotated_view = (
                        normalize_yaw_deg(primary_view[0] + yaw_shift),
                        primary_view[1],
                    )
                    rounded_view = (round(rotated_view[0], 3), round(rotated_view[1], 3))
                    if rounded_view in seen_rotations:
                        continue
                    seen_rotations.add(rounded_view)
                    if render_backend == "blender":
                        rotated_image, rotated_fail = render_single_view_blender_asset(
                            asset_path=prepared_asset_path,
                            shard_id=shard_id,
                            blender_bin=blender_bin,
                            blender_samples=blender_samples,
                            view=rotated_view,
                            image_size=render_size,
                            fill_fraction=0.9,
                        )
                    else:
                        rotated_image, rotated_fail = render_single_view(
                            mesh,
                            rotated_view,
                            image_size=render_size,
                            fill_fraction=0.9,
                        )
                    if rotated_image is not None:
                        metrics = compute_silhouette_metrics(
                            rotated_image,
                            min_fg_frac=min_fg_frac,
                            max_fg_frac=max_fg_frac,
                            edge_margin=edge_margin,
                        )
                        if metrics["has_fg"] and metrics["score"] < 0.60:
                            rescue_candidates.append(
                                (f"rot_{int(yaw_shift):+d}", rotated_image, metrics["fg_frac"], metrics["score"], metrics, [42])
                            )
                    elif rotated_fail:
                        print(f"    [trellis] rescue render skipped ({yaw_shift:+.0f}deg): {rotated_fail}")

                for pitch_shift in RESCUE_PITCH_SHIFTS:
                    rotated_view = (
                        primary_view[0],
                        max(-75.0, min(75.0, primary_view[1] + pitch_shift)),
                    )
                    rounded_view = (round(rotated_view[0], 3), round(rotated_view[1], 3))
                    if rounded_view in seen_rotations:
                        continue
                    seen_rotations.add(rounded_view)
                    if render_backend == "blender":
                        rotated_image, rotated_fail = render_single_view_blender_asset(
                            asset_path=prepared_asset_path,
                            shard_id=shard_id,
                            blender_bin=blender_bin,
                            blender_samples=blender_samples,
                            view=rotated_view,
                            image_size=render_size,
                            fill_fraction=0.9,
                        )
                    else:
                        rotated_image, rotated_fail = render_single_view(
                            mesh,
                            rotated_view,
                            image_size=render_size,
                            fill_fraction=0.9,
                        )
                    if rotated_image is not None:
                        metrics = compute_silhouette_metrics(
                            rotated_image,
                            min_fg_frac=min_fg_frac,
                            max_fg_frac=max_fg_frac,
                            edge_margin=edge_margin,
                        )
                        if metrics["has_fg"] and metrics["score"] < 0.60:
                            rescue_candidates.append(
                                (f"pitch_{int(pitch_shift):+d}", rotated_image, metrics["fg_frac"], metrics["score"], metrics, [42])
                            )
                    elif rotated_fail:
                        print(f"    [trellis] rescue render skipped (pitch {pitch_shift:+.0f}deg): {rotated_fail}")

                if render_backend == "blender":
                    same_view_image, same_view_fail = render_single_view_blender_asset(
                        asset_path=prepared_asset_path,
                        shard_id=shard_id,
                        blender_bin=blender_bin,
                        blender_samples=blender_samples,
                        view=primary_view,
                        image_size=render_size,
                        fill_fraction=0.9,
                    )
                else:
                    same_view_image, same_view_fail = render_single_view(
                        mesh,
                        primary_view,
                        image_size=render_size,
                        fill_fraction=0.9,
                    )
                if same_view_image is not None:
                    metrics = compute_silhouette_metrics(
                        same_view_image,
                        min_fg_frac=min_fg_frac,
                        max_fg_frac=max_fg_frac,
                        edge_margin=edge_margin,
                    )
                    if metrics["has_fg"]:
                        rescue_candidates.append(
                            ("tight_zoom", same_view_image, metrics["fg_frac"], metrics["score"], metrics, RESCUE_TIGHT_ZOOM_SEEDS)
                        )
                elif same_view_fail:
                    print(f"    [trellis] rescue render skipped (tight_zoom): {same_view_fail}")

                print(
                    "    [trellis] rescue sampler: "
                    f"sparse={RESCUE_SPARSE_SAMPLER_PARAMS}, "
                    f"shape={RESCUE_SHAPE_SAMPLER_PARAMS}"
                )
                if rescue_candidates:
                    rescue_candidates.sort(key=lambda item: (item[3], -item[2], item[0]))
                    rescue_candidates = rescue_candidates[:MAX_RESCUE_CANDIDATES]
                    print(
                        "    [trellis] rescue candidates: "
                        + ", ".join(
                            f"{label}({format_silhouette_metrics(metrics)})"
                            for label, _, _, _, metrics, _ in rescue_candidates
                        )
                    )
                    for rescue_label, rescue_image, rescue_fg, _rescue_score, rescue_metrics, rescue_seeds in rescue_candidates:
                        if success:
                            break
                        if rescue_attempts_used >= MAX_RESCUE_ATTEMPTS:
                            print(f"    [trellis] rescue budget exhausted ({MAX_RESCUE_ATTEMPTS} attempts)")
                            break
                        print(f"    [trellis] rescue view: {rescue_label}")
                        if rescue_image is None:
                            continue
                        rescue_succeeded = False
                        for seed in rescue_seeds:
                            if rescue_attempts_used >= MAX_RESCUE_ATTEMPTS:
                                print(f"    [trellis] rescue budget exhausted ({MAX_RESCUE_ATTEMPTS} attempts)")
                                break
                            total_trellis_attempts += 1
                            rescue_attempts_used += 1
                            rss_pre = get_rss_gb()
                            _set_current_model(uid=uid, path=mesh_path, phase=f"trellis:rescue:{rescue_label}:seed{seed}")
                            try:
                                trellis_result = generate_coarse_mesh(
                                    pipeline, rescue_image, pipeline_type, seed=seed,
                                    geometry_only=geometry_only,
                                    verbose_trellis_progress=verbose_trellis_progress,
                                    sparse_structure_sampler_params=RESCUE_SPARSE_SAMPLER_PARAMS,
                                    shape_slat_sampler_params=RESCUE_SHAPE_SAMPLER_PARAMS,
                                    return_intermediates=True,
                                )
                                if isinstance(trellis_result, tuple):
                                    coarse, slat_intermediates = trellis_result
                                else:
                                    coarse, slat_intermediates = trellis_result, None

                                n_verts = int(coarse.vertices.shape[0])
                                if n_verts < MIN_COARSE_VERTS or n_verts > MAX_COARSE_VERTS:
                                    del coarse, slat_intermediates
                                    cleanup_memory(use_malloc_trim=use_malloc_trim)
                                    rss_post = get_rss_gb()
                                    print(f"    [trellis] rescue/{rescue_label}/s{seed} quality_reject verts={n_verts} RSS={rss_post:.1f}GiB (Δ={rss_post-rss_pre:+.1f})")
                                    continue

                                pair_dir = save_pair(coarse, mesh_path, shard_output_dir, uid,
                                                    intermediates=slat_intermediates)
                                rescue_image.save(os.path.join(pair_dir, "rendered.png"))
                                meta = {
                                    "uid": uid,
                                    "fine_path": mesh_path,
                                    "coarse_vertices": n_verts,
                                    "coarse_faces": int(coarse.faces.shape[0]),
                                    "seed": seed,
                                    "view_index": f"rescue_{rescue_label}",
                                    "trellis_attempts": total_trellis_attempts,
                                    "views_rendered": len(images) + 1,
                                    "views_passed_gate": len(valid_images),
                                    "winning_fg_fraction": rescue_fg,
                                    "winning_view_metrics": rescue_metrics,
                                    "render_size": render_size,
                                    "pipeline_type": pipeline_type,
                                    "ss_res": slat_intermediates.get("ss_res") if slat_intermediates else None,
                                    "has_slat": slat_intermediates is not None,
                                }
                                with open(os.path.join(pair_dir, "meta.json"), "w") as f:
                                    json.dump(meta, f, indent=2)

                                with progress_lock:
                                    completed.add(uid)
                                    pairs_created += 1
                                    elapsed = time.time() - t_start
                                    total_time += elapsed
                                success = True
                                rescue_succeeded = True
                                _clear_recycle_artifacts()

                                del coarse, slat_intermediates
                                cleanup_memory(use_malloc_trim=use_malloc_trim)
                                rss_post = get_rss_gb()
                                print(f"    [trellis] rescue/{rescue_label}/s{seed} SUCCESS verts={n_verts} RSS={rss_post:.1f}GiB (Δ={rss_post-rss_pre:+.1f})")
                                break
                            except RuntimeError as e:
                                err_msg = str(e)
                                if "numel() == 0" in err_msg:
                                    trellis_empty_count += 1
                                    cleanup_memory(use_malloc_trim=use_malloc_trim)
                                    rss_post = get_rss_gb()
                                    print(f"    [trellis] rescue/{rescue_label}/s{seed} empty RSS={rss_post:.1f}GiB (Δ={rss_post-rss_pre:+.1f})")
                                    continue
                                if "out of memory" in err_msg.lower():
                                    trellis_oom_count += 1
                                    cleanup_memory(use_malloc_trim=use_malloc_trim)
                                    rss_post = get_rss_gb()
                                    print(f"    [trellis] rescue/{rescue_label}/s{seed} OOM RSS={rss_post:.1f}GiB (Δ={rss_post-rss_pre:+.1f})")
                                    continue
                                trellis_errors.append(err_msg[:200])
                                cleanup_memory(use_malloc_trim=use_malloc_trim)
                                rss_post = get_rss_gb()
                                print(f"    [trellis] rescue/{rescue_label}/s{seed} error RSS={rss_post:.1f}GiB (Δ={rss_post-rss_pre:+.1f})")
                                break
                            except Exception as e:
                                trellis_errors.append(f"{type(e).__name__}: {str(e)[:200]}")
                                cleanup_memory(use_malloc_trim=use_malloc_trim)
                                rss_post = get_rss_gb()
                                print(f"    [trellis] rescue/{rescue_label}/s{seed} exception RSS={rss_post:.1f}GiB (Δ={rss_post-rss_pre:+.1f})")
                                break
                        if rescue_succeeded:
                            break

            rss_after_trellis = get_rss_gb()
            rss_delta_model = rss_after_trellis - rss_before_trellis
            if total_trellis_attempts > 1 or abs(rss_delta_model) > 0.5:
                print(f"    [trellis] model RSS: {rss_before_trellis:.1f}→{rss_after_trellis:.1f}GiB (Δ={rss_delta_model:+.1f}) attempts={total_trellis_attempts}")

            if not success:
                if trellis_errors:
                    fail_reason = FAIL_TRELLIS_ERROR
                elif trellis_oom_count > 0 and trellis_empty_count == 0:
                    fail_reason = FAIL_TRELLIS_OOM
                elif trellis_empty_count > 0:
                    fail_reason = FAIL_TRELLIS_EMPTY
                else:
                    fail_reason = FAIL_QUALITY_REJECT

                with progress_lock:
                    failure_counts[fail_reason] += 1
                    failures += 1
                    permanently_failed.add(uid)
                    failure_details[uid] = {
                        "reason": fail_reason,
                        "path": mesh_path,
                        "trellis_attempts": total_trellis_attempts,
                        "empty_count": trellis_empty_count,
                        "oom_count": trellis_oom_count,
                        "errors": trellis_errors[:3],
                        "selected_view_metrics": [view[5] for view in selected_views],
                    }
                if fail_reason == FAIL_TRELLIS_EMPTY and selected_views:
                    summary = " | ".join(
                        f"v{view[3]} {format_silhouette_metrics(view[5])}"
                        for view in selected_views
                    )
                    print(f"    [trellis] empty view stats: {summary}")
                _save_all_progress()
                _clear_recycle_artifacts()

            _set_current_model(uid=uid, path=mesh_path, phase="cleanup")

            # Drop per-model buffers aggressively to avoid long-run RSS creep.
            if mesh is not None:
                del mesh
            if prepared_workdir is not None:
                shutil.rmtree(prepared_workdir, ignore_errors=True)
            del images
            del valid_images
            del gated
            cleanup_memory(use_malloc_trim=use_malloc_trim)
            _set_current_model()

            # ── Periodic save + status ──
            processed = pairs_created + failures
            if processed % 25 == 0 and processed > 0:
                _save_all_progress()

                rate = pairs_created / max(processed, 1) * 100
                wall_hrs = (time.time() - start_time) / 3600
                avg_time = total_time / max(pairs_created, 1)
                gate_rate = total_views_passed / max(total_views_rendered, 1) * 100

                fc_parts = []
                for reason, count in failure_counts.most_common():
                    fc_parts.append(f"{reason}:{count}")
                fc_str = ", ".join(fc_parts) if fc_parts else "none"

                tqdm.write(
                    f"  [{processed}/{len(remaining)}] "
                    f"{pairs_created} ok ({rate:.0f}%), "
                    f"gate:{gate_rate:.0f}%, "
                    f"wall:{wall_hrs:.1f}h, "
                    f"avg:{avg_time:.1f}s/ok | "
                    f"FAIL: {fc_str}"
                    f"{f', RSS:{get_rss_gb():.1f}GiB' if log_rss_every and processed % log_rss_every == 0 else ''}"
                    f"{f', CUDA:{get_cuda_memory_stats_mb()[0]:.0f}/{get_cuda_memory_stats_mb()[1]:.0f}/{get_cuda_memory_stats_mb()[2]:.0f}MiB' if torch.cuda.is_available() and log_rss_every and processed % log_rss_every == 0 else ''}"
                )
    finally:
        rss_watchdog.stop()

    # ── Final save ──
    _save_all_progress()

    # ── Summary ──
    processed = pairs_created + failures
    rate = pairs_created / max(processed, 1) * 100
    gate_rate = total_views_passed / max(total_views_rendered, 1) * 100
    wall_hrs = (time.time() - start_time) / 3600

    print(f"\n{'='*70}")
    if recycle_reason:
        print(f"SHARD {shard_id} RECYCLE REQUESTED")
    else:
        print(f"SHARD {shard_id} COMPLETE")
    print(f"{'='*70}")
    print(f"  Pairs created:     {pairs_created}")
    print(f"  Failures:          {failures}")
    print(f"  Success rate:      {rate:.1f}%")
    print(f"  Gate pass rate:    {gate_rate:.0f}%")
    print(f"  Wall time:         {wall_hrs:.1f}h")
    print(f"  Final RSS:         {get_rss_gb():.1f} GiB")
    if torch.cuda.is_available():
        a_mb, r_mb, m_mb = get_cuda_memory_stats_mb()
        print(f"  Final CUDA mem:    alloc={a_mb:.0f}MiB reserved={r_mb:.0f}MiB max={m_mb:.0f}MiB")
    if pairs_created > 0:
        print(f"  Avg time per ok:   {total_time/pairs_created:.1f}s")
    print(f"\n  Failure breakdown:")
    for reason, count in failure_counts.most_common():
        pct = count / max(failures, 1) * 100
        print(f"    {reason:25s} {count:5d} ({pct:.0f}%)")
    print(f"\n  Total completed:   {len(completed)}")
    print(f"  Output:            {shard_output_dir}")
    if recycle_reason:
        print(f"  Recycle reason:    {recycle_reason}")
        print(f"  Remaining queued:  {max(len(remaining) - processed, 0)}")
    print(f"{'='*70}")

    return {
        "recycle_requested": recycle_reason is not None,
        "recycle_reason": recycle_reason,
        "processed": processed,
        "remaining": max(len(remaining) - processed, 0),
    }


def _save_progress(progress_path, failed_path, failure_details_path,
                   completed, permanently_failed, failure_details):
    """Atomically save all progress files."""
    for path, data in [
        (progress_path, list(completed)),
        (failed_path, list(permanently_failed)),
        (failure_details_path, failure_details),
    ]:
        tmp = str(path) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, path)


def _load_uid_list(path: Path) -> set[str]:
    """Best-effort JSON UID list loader."""
    if not path.exists():
        return set()
    try:
        with open(path) as f:
            return set(json.load(f))
    except Exception:
        return set()


def _collect_all_failed_uids(base_output: Path) -> set[str]:
    """Collect historical failures from the global and per-shard outputs."""
    failed = _load_uid_list(base_output / "failed.json")
    for shard_dir in sorted(base_output.glob("shard_*")):
        failed.update(_load_uid_list(shard_dir / "failed.json"))
    return failed


def main():
    parser = argparse.ArgumentParser(description="Generate coarse/fine training pairs (v5)")
    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="/workspace/data/training_pairs")
    parser.add_argument("--pipeline_type", type=str, default="512",
                        choices=["512", "1024", "1024_cascade", "1536_cascade"])
    parser.add_argument("--render_size", type=int, default=1024)
    parser.add_argument("--num_views", type=int, default=12)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--trellis2_dir", type=str, default="/workspace/TRELLIS.2")
    parser.add_argument("--model_dir", type=str, default="/workspace/models/trellis2-4b")
    parser.add_argument("--min_fg_frac", type=float, default=DEFAULT_MIN_FG_FRAC)
    parser.add_argument("--max_fg_frac", type=float, default=DEFAULT_MAX_FG_FRAC)
    parser.add_argument("--edge_margin", type=float, default=DEFAULT_EDGE_MARGIN)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU device index (default 0)")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Re-attempt previously failed UIDs instead of remaining ones")
    parser.add_argument("--geometry_only", action=argparse.BooleanOptionalAction, default=True,
                        help="Skip texture synthesis and decode shape-only meshes (default: true)")
    parser.add_argument("--disable_rembg_model", action=argparse.BooleanOptionalAction, default=True,
                        help="Unload rembg model after load (input renders already include alpha)")
    parser.add_argument("--log_rss_every", type=int, default=25,
                        help="Print RSS every N processed models (0 disables)")
    parser.add_argument("--rss_hard_limit_gb", type=float, default=0.0,
                        help="If RSS reaches this GiB value, save progress and exit for safe recycle (0 disables)")
    parser.add_argument("--rss_emergency_limit_gb", type=float, default=0.0,
                        help="Background-polled RSS limit; defaults to --rss_hard_limit_gb when unset")
    parser.add_argument("--rss_watch_interval_sec", type=float, default=0.25,
                        help="RSS watchdog polling interval in seconds (0 disables watchdog)")
    parser.add_argument("--max_models_per_run", type=int, default=0,
                        help="Process at most N models then exit for safe recycle (0 disables)")
    parser.add_argument("--max_emergency_recycles_per_uid", type=int, default=3,
                        help="Mark a UID failed after this many consecutive watchdog recycles (0 disables)")
    parser.add_argument("--low_vram", action=argparse.BooleanOptionalAction, default=False,
                        help="Keep TRELLIS.2 models on CPU between inference steps (default: False, "
                             "models stay on GPU for faster inference on A100 80GB)")
    parser.add_argument("--verbose_trellis_progress", action="store_true",
                        help="Keep TRELLIS inner tqdm/progress output enabled")
    parser.add_argument("--no_malloc_trim", action="store_true",
                        help="Disable glibc malloc_trim calls after cleanup points")
    parser.add_argument("--recycle_exit_code", type=int, default=75,
                        help="Exit code used when recycle is requested")
    parser.add_argument("--render_backend", type=str, default=DEFAULT_RENDER_BACKEND,
                        choices=["trimesh", "blender"],
                        help="Renderer used for synthetic views")
    parser.add_argument("--blender_bin", type=str, default=DEFAULT_BLENDER_BIN,
                        help="Path to Blender binary for --render_backend blender")
    parser.add_argument("--blender_samples", type=int, default=DEFAULT_BLENDER_SAMPLES,
                        help="Cycles samples per rendered view")
    parser.add_argument("--mesh_prep_memory_limit_gb", type=float, default=DEFAULT_MESH_PREP_MEMORY_LIMIT_GB,
                        help="Address-space cap for isolated mesh prep subprocess")
    parser.add_argument("--mesh_prep_timeout_sec", type=int, default=DEFAULT_MESH_PREP_TIMEOUT_SEC,
                        help="Timeout for isolated mesh prep subprocess")
    args = parser.parse_args()

    result = generate_pairs(
        args.input_json,
        args.output_dir,
        args.pipeline_type,
        args.render_size,
        args.num_views,
        args.limit,
        args.trellis2_dir,
        args.model_dir,
        args.min_fg_frac,
        args.max_fg_frac,
        args.edge_margin,
        args.shard_id,
        args.num_shards,
        args.gpu,
        args.retry_failed,
        args.geometry_only,
        args.disable_rembg_model,
        args.log_rss_every,
        args.rss_hard_limit_gb,
        args.rss_emergency_limit_gb,
        args.rss_watch_interval_sec,
        args.max_models_per_run,
        args.max_emergency_recycles_per_uid,
        args.low_vram,
        args.verbose_trellis_progress,
        not args.no_malloc_trim,
        args.recycle_exit_code,
        args.render_backend,
        args.blender_bin,
        args.blender_samples,
        args.mesh_prep_memory_limit_gb,
        args.mesh_prep_timeout_sec,
    )
    if result.get("recycle_requested", False):
        raise SystemExit(args.recycle_exit_code)


if __name__ == "__main__":
    main()
