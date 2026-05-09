"""Mesh-head adapter registry."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .face_level import FaceLevelConfig, FaceLevelMeshHead
from .generic import GenericCommandConfig, GenericCommandMeshHead
from .meshripple import MeshRippleAdapter, MeshRippleConfig


def build_mesh_head(name: str, config: Mapping[str, Any] | None = None):
    config = dict(config or {})
    normalized = name.lower().replace("_", "-")
    if normalized == "meshripple":
        if "repo_dir" in config:
            config["repo_dir"] = Path(config["repo_dir"])
        if "checkpoint_dir" in config and config["checkpoint_dir"] is not None:
            config["checkpoint_dir"] = Path(config["checkpoint_dir"])
        if "extra_args" in config and not isinstance(config["extra_args"], tuple):
            config["extra_args"] = tuple(config["extra_args"])
        return MeshRippleAdapter(MeshRippleConfig(**config))
    if normalized in {"face-level", "facelevel", "face"}:
        for key in ("repo_dir", "checkpoint"):
            if key in config:
                config[key] = Path(config[key])
        if "python" in config and isinstance(config["python"], str):
            config["python"] = config["python"]
        return FaceLevelMeshHead(FaceLevelConfig(**config))
    if "command_template" in config:
        config["repo_dir"] = Path(config["repo_dir"])
        config["command_template"] = tuple(config["command_template"])
        return GenericCommandMeshHead(GenericCommandConfig(name=normalized, **config))
    raise ValueError(f"Unknown mesh head '{name}'. Provide command_template for a generic adapter.")
