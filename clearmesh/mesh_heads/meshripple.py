"""Adapter for the public MeshRipple inference repo.

MeshRipple's public README currently exposes demo inference commands around config
files. This adapter supports that repo-native command path first, while preserving
the point-cloud input and output bookkeeping ClearMesh needs for bake-offs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import shutil
import time
from typing import Any, Mapping

from .base import MeshHeadError, MeshHeadInput, MeshHeadResult, discover_mesh_outputs, run_external_command


@dataclass(frozen=True)
class MeshRippleConfig:
    repo_dir: Path = Path("/workspace/mesh-heads/MeshRipple")
    python: Path | str = Path("/workspace/miniconda/envs/meshripple/bin/python")
    config_path: str = "config_loader/config_20k_nsa.yaml"
    checkpoint_dir: Path | None = Path("/workspace/mesh-heads/MeshRipple/ckpt")
    timeout_seconds: int | None = 60 * 60
    extra_args: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)
    config_overrides: Mapping[str, Any] = field(default_factory=dict)


class MeshRippleAdapter:
    name = "meshripple"

    def __init__(self, config: MeshRippleConfig | None = None) -> None:
        self.config = config or MeshRippleConfig()

    def build_command(self, mesh_input: MeshHeadInput, config_path: Path | str | None = None) -> list[str]:
        command = [
            str(self.config.python),
            "main.py",
            "--config",
            str(config_path or self.config.config_path),
        ]
        command.extend(self.config.extra_args)
        return command

    def run(self, mesh_input: MeshHeadInput) -> MeshHeadResult:
        repo_dir = Path(self.config.repo_dir)
        if not repo_dir.exists():
            raise MeshHeadError(f"MeshRipple repo not found at {repo_dir}. Run scripts/setup/install_mesh_heads.sh first.")
        if not Path(mesh_input.point_cloud_path).exists():
            raise MeshHeadError(f"Point cloud not found: {mesh_input.point_cloud_path}")

        output_dir = Path(mesh_input.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logs_dir = output_dir / "logs"
        stdout_path = logs_dir / "meshripple.stdout.log"
        stderr_path = logs_dir / "meshripple.stderr.log"
        manifest_path = output_dir / "clearmesh_meshripple_input.json"
        clearmesh_config_path = self._write_job_config(mesh_input, output_dir)
        manifest_path.write_text(
            json.dumps(
                {
                    "case_id": mesh_input.case_id,
                    "point_cloud_path": str(mesh_input.point_cloud_path),
                    "proxy_mesh_path": str(mesh_input.proxy_mesh_path) if mesh_input.proxy_mesh_path else None,
                    "part_id": mesh_input.part_id,
                    "checkpoint_dir": str(self.config.checkpoint_dir) if self.config.checkpoint_dir else None,
                    "meshripple_config_path": str(clearmesh_config_path),
                    "note": "MeshRipple public inference consumes meshes from eval_dataset_path and samples point clouds internally.",
                    "metadata": mesh_input.metadata,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        env = dict(self.config.env)
        env["CLEARMESH_MESH_HEAD_OUTPUT_DIR"] = str(output_dir)
        env["CLEARMESH_POINT_CLOUD"] = str(mesh_input.point_cloud_path)
        if self.config.checkpoint_dir:
            env["MESHRIPPLE_CKPT_DIR"] = str(self.config.checkpoint_dir)

        before = time.time()
        command = self.build_command(mesh_input, config_path=clearmesh_config_path)
        run_external_command(
            command,
            cwd=repo_dir,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            env=env,
            timeout_seconds=self.config.timeout_seconds,
        )

        candidates = discover_mesh_outputs(output_dir, since_mtime=before)
        if not candidates:
            repo_candidates = discover_mesh_outputs(repo_dir / "sample_results", since_mtime=before)
            candidates = repo_candidates
        if not candidates:
            raise MeshHeadError(
                "MeshRipple command completed but no mesh output was found. "
                f"Check {stdout_path}, {stderr_path}, and the repo sample_results directory."
            )

        return MeshHeadResult(
            mesh_path=candidates[0],
            adapter_name=self.name,
            command=command,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            metadata={"manifest_path": str(manifest_path)},
        )

    def _write_job_config(self, mesh_input: MeshHeadInput, output_dir: Path) -> Path:
        """Create a per-job MeshRipple config that points at ClearMesh assets."""

        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - only happens in misconfigured GPU envs.
            raise MeshHeadError("PyYAML is required for MeshRipple config generation") from exc

        repo_dir = Path(self.config.repo_dir)
        base_config_path = repo_dir / self.config.config_path
        if not base_config_path.exists():
            raise MeshHeadError(f"MeshRipple config not found: {base_config_path}")
        base_config = yaml.safe_load(base_config_path.read_text(encoding="utf-8")) or {}
        _deep_update(base_config, self.config.config_overrides)

        input_mesh = mesh_input.proxy_mesh_path
        if input_mesh is None:
            input_mesh = mesh_input.metadata.get("mesh_path") if mesh_input.metadata else None
        if input_mesh is None:
            raise MeshHeadError(
                "MeshRipple public inference requires a proxy mesh path. "
                "Provide MeshHeadInput.proxy_mesh_path; point-cloud-only inference is not exposed by the public repo."
            )
        input_mesh = Path(input_mesh)
        if not input_mesh.exists():
            raise MeshHeadError(f"MeshRipple input mesh not found: {input_mesh}")

        eval_dir = output_dir / "meshripple_eval_input"
        eval_dir.mkdir(parents=True, exist_ok=True)
        eval_mesh = eval_dir / input_mesh.name
        if not eval_mesh.exists():
            shutil.copy2(input_mesh, eval_mesh)

        base_config.setdefault("data", {})["eval_dataset_path"] = str(eval_dir)
        base_config["output_folder_base"] = str(output_dir)
        base_config["project_name"] = "meshripple"
        base_config.setdefault("generate", {})["batch_size"] = 1

        model = base_config.setdefault("model", {})
        model_path = model.get("model_path")
        if model_path and not Path(str(model_path)).is_absolute():
            candidate = repo_dir / str(model_path)
            model["model_path"] = str(candidate)

        if self.config.checkpoint_dir:
            ckpt_dir = Path(self.config.checkpoint_dir)
            default_name = "meshRipple_nsa.pth" if base_config.get("model", {}).get("model_version") == "v1-nsa" else "meshRipple_10k.pth"
            ckpt_candidate = ckpt_dir / default_name
            if ckpt_candidate.exists():
                model["model_path"] = str(ckpt_candidate)

        config_path = output_dir / "clearmesh_meshripple_config.yaml"
        config_path.write_text(yaml.safe_dump(base_config, sort_keys=False), encoding="utf-8")
        return config_path


def _deep_update(target: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge JSON/YAML-style config overrides."""

    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
    return target
