#!/usr/bin/env python3
"""Audit what MeshRipple will see after its own preprocessing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meshripple-repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, action="append", required=True)
    parser.add_argument("--dec-to-facenum", type=int, action="append", default=[])
    args = parser.parse_args()

    repo = args.meshripple_repo.resolve()
    sys.path.insert(0, str(repo))

    import trimesh
    import yaml
    from config_loader.load_config import DictToObject
    from data_load.mesh_dataset_more_aug import dec, discrete_and_clean, normalize
    from ripple_tokenizer.tokenizer import undiscretize

    conf = DictToObject(_load_meshripple_config(args.config, repo, yaml))
    dec_targets = args.dec_to_facenum or [int(conf.data_processing.dec_to_facenum)]
    rows = []
    for mesh_path in args.mesh:
        for dec_target in dec_targets:
            row = {"mesh": str(mesh_path), "dec_to_facenum": dec_target}
            try:
                mesh = trimesh.load(mesh_path, force="mesh")
                row["raw_faces"] = int(len(mesh.faces))
                row["raw_vertices"] = int(len(mesh.vertices))
                if getattr(conf.data_processing, "y_up", False):
                    mesh.vertices = mesh.vertices[:, [2, 0, 1]]
                normalize(mesh)
                mesh = discrete_and_clean(mesh, conf.data_processing.n_discrete_size)
                row["faces_after_discrete_clean"] = int(len(mesh.faces))
                row["vertices_after_discrete_clean"] = int(len(mesh.vertices))
                if dec_target != -1 and mesh.faces.shape[0] > dec_target:
                    mesh = dec(mesh=mesh, target_face=dec_target)
                undiscrete_mesh = trimesh.Trimesh(vertices=undiscretize(mesh.vertices), faces=mesh.faces, process=False)
                row.update(
                    {
                        "faces_after": int(len(undiscrete_mesh.faces)),
                        "vertices_after": int(len(undiscrete_mesh.vertices)),
                        "components_after": int(len(undiscrete_mesh.split(only_watertight=False))),
                        "watertight_after": bool(undiscrete_mesh.is_watertight),
                        "bounds_after": undiscrete_mesh.bounds.tolist(),
                    }
                )
            except Exception as exc:  # noqa: BLE001 - audit should report all failures.
                row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
    print(json.dumps({"rows": rows}, indent=2, sort_keys=True))
    return 0


def _load_meshripple_config(config_path: Path, repo: Path, yaml_module) -> dict[str, Any]:
    raw_text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        wrapper = json.loads(raw_text)
        inner_path = Path(wrapper.get("config_path", ""))
        if not inner_path.is_absolute():
            inner_path = repo / inner_path
        data = yaml_module.safe_load(inner_path.read_text(encoding="utf-8"))
        _deep_update(data, wrapper.get("config_overrides", {}))
        return data
    return yaml_module.safe_load(raw_text)


def _deep_update(target: dict[str, Any], overrides: dict[str, Any]) -> None:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value


if __name__ == "__main__":
    raise SystemExit(main())
