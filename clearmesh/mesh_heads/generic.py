"""Generic command adapter for public mesh-head repos during bake-offs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Mapping

from .base import MeshHeadError, MeshHeadInput, MeshHeadResult, discover_mesh_outputs, run_external_command


@dataclass(frozen=True)
class GenericCommandConfig:
    name: str
    repo_dir: Path
    command_template: tuple[str, ...]
    timeout_seconds: int | None = 60 * 60
    env: Mapping[str, str] = field(default_factory=dict)


class GenericCommandMeshHead:
    """Adapter for repos whose inference command is known at run time."""

    def __init__(self, config: GenericCommandConfig) -> None:
        self.config = config
        self.name = config.name

    def _render_command(self, mesh_input: MeshHeadInput) -> list[str]:
        values = {
            "case_id": mesh_input.case_id,
            "point_cloud": str(mesh_input.point_cloud_path),
            "proxy_mesh": str(mesh_input.proxy_mesh_path or ""),
            "output_dir": str(mesh_input.output_dir),
            "part_id": str(mesh_input.part_id or ""),
        }
        return [part.format(**values) for part in self.config.command_template]

    def run(self, mesh_input: MeshHeadInput) -> MeshHeadResult:
        repo_dir = Path(self.config.repo_dir)
        if not repo_dir.exists():
            raise MeshHeadError(f"Repo for {self.name} not found at {repo_dir}.")
        if not Path(mesh_input.point_cloud_path).exists():
            raise MeshHeadError(f"Point cloud not found: {mesh_input.point_cloud_path}")

        output_dir = Path(mesh_input.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = output_dir / "logs" / f"{self.name}.stdout.log"
        stderr_path = output_dir / "logs" / f"{self.name}.stderr.log"
        command = self._render_command(mesh_input)
        before = time.time()
        run_external_command(
            command,
            cwd=repo_dir,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            env=self.config.env,
            timeout_seconds=self.config.timeout_seconds,
        )
        candidates = discover_mesh_outputs(output_dir, since_mtime=before)
        if not candidates:
            candidates = discover_mesh_outputs(repo_dir, since_mtime=before)
        if not candidates:
            raise MeshHeadError(f"{self.name} completed but no mesh output was discovered.")
        return MeshHeadResult(candidates[0], self.name, command, stdout_path, stderr_path)
