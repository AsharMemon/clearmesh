"""Small, dependency-light abstractions for external mesh-head repos.

The public mesh-head projects move quickly and each has its own environment. These
adapters intentionally keep ClearMesh core state and external repo state separate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os
import subprocess
from typing import Mapping, Sequence

MESH_EXTENSIONS = (".glb", ".gltf", ".obj", ".ply", ".stl")


class MeshHeadError(RuntimeError):
    """Raised when an external mesh-head command cannot produce an output mesh."""


@dataclass(frozen=True)
class MeshHeadInput:
    """Input bundle passed to an external artist-mesh head."""

    case_id: str
    point_cloud_path: Path
    output_dir: Path
    proxy_mesh_path: Path | None = None
    part_id: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class MeshHeadResult:
    """Result returned by a mesh-head adapter."""

    mesh_path: Path
    adapter_name: str
    command: list[str]
    stdout_path: Path
    stderr_path: Path
    metadata: dict[str, str] = field(default_factory=dict)


def merged_env(extra_env: Mapping[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ)
    if extra_env:
        env.update({str(k): str(v) for k, v in extra_env.items()})
    return env


def run_external_command(
    command: Sequence[str],
    *,
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
    env: Mapping[str, str] | None = None,
    timeout_seconds: int | None = None,
) -> None:
    """Run a command while streaming logs to files for product traceability."""

    cwd = Path(cwd)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        proc = subprocess.run(
            list(command),
            cwd=str(cwd),
            env=merged_env(env),
            stdout=stdout,
            stderr=stderr,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    if proc.returncode != 0:
        raise MeshHeadError(
            f"External mesh head failed with exit code {proc.returncode}. "
            f"See {stdout_path} and {stderr_path}."
        )


def discover_mesh_outputs(output_dir: Path, *, since_mtime: float | None = None) -> list[Path]:
    """Return generated mesh-like files, newest first."""

    output_dir = Path(output_dir)
    if not output_dir.exists():
        return []
    meshes: list[Path] = []
    for path in output_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in MESH_EXTENSIONS:
            continue
        if since_mtime is not None and path.stat().st_mtime < since_mtime:
            continue
        meshes.append(path)
    return sorted(meshes, key=lambda p: p.stat().st_mtime, reverse=True)
