"""Adapter for ClearMesh FACE-level learned topology checkpoints."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Mapping

from .base import MeshHeadError, MeshHeadInput, MeshHeadResult, run_external_command


@dataclass(frozen=True)
class FaceLevelConfig:
    repo_dir: Path = Path("/workspace/clearmesh")
    python: Path | str = "python"
    checkpoint: Path = Path("/workspace/checkpoints/face_level_conditioned.pt")
    point_samples: int = 8192
    face_count: int | None = None
    timeout_seconds: int | None = 60 * 30
    env: Mapping[str, str] = field(default_factory=dict)


class FaceLevelMeshHead:
    """Run the in-repo FACE-level sampler against a reference/proxy mesh."""

    name = "face-level"

    def __init__(self, config: FaceLevelConfig | None = None) -> None:
        self.config = config or FaceLevelConfig()

    def build_command(self, mesh_input: MeshHeadInput, output_mesh: Path) -> list[str]:
        if mesh_input.proxy_mesh_path is None:
            raise MeshHeadError("FACE-level mesh head requires MeshHeadInput.proxy_mesh_path")
        command = [
            str(self.config.python),
            "scripts/research/sample_face_level_from_mesh.py",
            "--checkpoint",
            str(self.config.checkpoint),
            "--mesh",
            str(mesh_input.proxy_mesh_path),
            "--output",
            str(output_mesh),
            "--point-samples",
            str(self.config.point_samples),
        ]
        if self.config.face_count is not None:
            command.extend(["--face-count", str(self.config.face_count)])
        return command

    def run(self, mesh_input: MeshHeadInput) -> MeshHeadResult:
        repo_dir = Path(self.config.repo_dir)
        if not repo_dir.exists():
            raise MeshHeadError(f"ClearMesh repo for FACE-level head not found at {repo_dir}.")
        checkpoint = Path(self.config.checkpoint)
        if not checkpoint.exists():
            raise MeshHeadError(f"FACE-level checkpoint not found: {checkpoint}")
        if mesh_input.proxy_mesh_path is None or not Path(mesh_input.proxy_mesh_path).exists():
            raise MeshHeadError(f"FACE-level proxy/reference mesh not found: {mesh_input.proxy_mesh_path}")

        output_dir = Path(mesh_input.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = output_dir / "logs" / "face_level.stdout.log"
        stderr_path = output_dir / "logs" / "face_level.stderr.log"
        output_mesh = output_dir / f"{mesh_input.case_id}_face_level.glb"
        command = self.build_command(mesh_input, output_mesh)
        before = time.time()
        run_external_command(
            command,
            cwd=repo_dir,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            env=self.config.env,
            timeout_seconds=self.config.timeout_seconds,
        )
        if not output_mesh.exists() or output_mesh.stat().st_mtime < before:
            raise MeshHeadError(f"FACE-level command completed but did not create {output_mesh}")
        return MeshHeadResult(
            mesh_path=output_mesh,
            adapter_name=self.name,
            command=command,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            metadata={
                "checkpoint": str(checkpoint),
                "point_samples": str(self.config.point_samples),
                "face_count": "" if self.config.face_count is None else str(self.config.face_count),
            },
        )
