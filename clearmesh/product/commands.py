"""Command helpers for GPU worker hooks.

These keep external repo execution declarative: job metadata can provide command
arrays with placeholders instead of requiring Python edits for every new repo CLI.
"""

from __future__ import annotations

from pathlib import Path
import shlex
import subprocess
from typing import Mapping, Sequence

from clearmesh.mesh_heads.base import discover_mesh_outputs, merged_env


def render_command(command: str | Sequence[str], values: Mapping[str, str]) -> list[str]:
    parts = shlex.split(command) if isinstance(command, str) else [str(part) for part in command]
    return [part.format(**values) for part in parts]


def run_logged_command(
    command: str | Sequence[str],
    *,
    cwd: str | Path | None,
    values: Mapping[str, str],
    stdout_path: Path,
    stderr_path: Path,
    env: Mapping[str, str] | None = None,
    timeout_seconds: int | None = None,
) -> list[str]:
    rendered = render_command(command, values)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        proc = subprocess.run(
            rendered,
            cwd=str(cwd) if cwd else None,
            env=merged_env(env),
            stdout=stdout,
            stderr=stderr,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {' '.join(rendered)}")
    return rendered


def discover_new_mesh(search_dir: Path, *, since_mtime: float) -> Path | None:
    candidates = discover_mesh_outputs(search_dir, since_mtime=since_mtime)
    return candidates[0] if candidates else None
