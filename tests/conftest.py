"""Shared pytest fixtures for the ClearMesh test suite.

Fixture scopes are chosen so that expensive setup (loading a 4B-param
TRELLIS.2 pipeline, spinning up CUDA contexts) happens at most once per
session. Tests should request the narrowest fixture that works.

Markers (registered in pytest.ini):
  gpu       — requires torch.cuda.is_available()
  slow      — long-running e2e test
  trellis2  — requires the TRELLIS.2 pipeline to be importable + weights downloaded
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Environment auto-detection
# ---------------------------------------------------------------------------

def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _trellis2_importable() -> bool:
    """Cheap check: is trellis2 importable (or its dir on sys.path)?"""
    trellis2_dir = os.environ.get("TRELLIS2_DIR", "/workspace/TRELLIS.2")
    if Path(trellis2_dir).exists() and trellis2_dir not in sys.path:
        sys.path.insert(0, trellis2_dir)
    # Default to flash_attn_3 attention backend (matches what we install on
    # Vast.ai pods); TRELLIS.2 tries to `import flash_attn` for the 2.x
    # backend and will crash if only flash_attn_3 is installed.
    os.environ.setdefault("ATTN_BACKEND", "flash_attn_3")
    os.environ.setdefault("SPCONV_ALGO", "native")
    try:
        import trellis2  # noqa: F401
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Auto-skip based on markers
# ---------------------------------------------------------------------------

def pytest_collection_modifyitems(config, items):
    """Skip gpu/trellis2 tests when their prerequisites aren't met.

    Does NOT skip `slow` — that marker is only for -m filtering, not skipping.
    """
    cuda_ok = _cuda_available()
    trellis2_ok = _trellis2_importable()

    skip_gpu = pytest.mark.skip(reason="CUDA not available")
    skip_trellis2 = pytest.mark.skip(reason="TRELLIS.2 not importable (set TRELLIS2_DIR)")

    for item in items:
        if "gpu" in item.keywords and not cuda_ok:
            item.add_marker(skip_gpu)
        if "trellis2" in item.keywords and not trellis2_ok:
            item.add_marker(skip_trellis2)


# ---------------------------------------------------------------------------
# Path fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def assets_dir(repo_root: Path) -> Path:
    """Test assets dir. Created on first use; tests that need fixtures
    (sample meshes, reference images) should place them here.
    """
    d = repo_root / "tests" / "assets"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """Per-test temp dir for output meshes/images. Auto-cleaned by pytest."""
    d = tmp_path / "out"
    d.mkdir(exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Mesh fixtures (CPU-only; useful for unit tests that don't need the pipeline)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def sample_cube_path(assets_dir: Path) -> Path:
    """A unit cube GLB at tests/assets/cube.glb. Generated once per session."""
    out = assets_dir / "cube.glb"
    if not out.exists():
        import trimesh
        mesh = trimesh.creation.box(extents=(1, 1, 1))
        mesh.export(out)
    return out


@pytest.fixture(scope="session")
def sample_sphere_path(assets_dir: Path) -> Path:
    """A unit sphere GLB at tests/assets/sphere.glb. Generated once per session."""
    out = assets_dir / "sphere.glb"
    if not out.exists():
        import trimesh
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
        mesh.export(out)
    return out


# ---------------------------------------------------------------------------
# TRELLIS.2 pipeline (session-scoped; loaded at most once per test run)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def trellis2_pipeline():
    """Load Trellis2ImageTo3DPipeline once per session.

    Tests that request this fixture should be marked @pytest.mark.trellis2
    and @pytest.mark.gpu so they skip cleanly on CPU-only machines.
    """
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    model_dir = os.environ.get("TRELLIS2_MODEL_DIR", "/workspace/models/trellis2-4b")
    if Path(model_dir).exists():
        pipeline = Trellis2ImageTo3DPipeline.from_pretrained(model_dir)
    else:
        pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    import torch
    pipeline.to("cuda" if torch.cuda.is_available() else "cpu")
    return pipeline
