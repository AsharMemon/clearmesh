#!/usr/bin/env python3
"""Run OmniPart inference and normalize outputs into parts_manifest.json.

OmniPart's official CLI requires an input image plus a 2D part-id mask. This
wrapper keeps ClearMesh's worker contract stable while we decide how masks are
created in production.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

MESH_SUFFIXES = {".obj", ".glb", ".gltf", ".ply", ".stl"}


def discover_outputs(output_dir: Path, since: float) -> list[Path]:
    return sorted(path for path in output_dir.rglob("*") if path.is_file() and path.stat().st_mtime >= since)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config-json", type=Path, required=True)
    args, passthrough = parser.parse_known_args()

    config = json.loads(args.config_json.read_text(encoding="utf-8"))
    repo_dir = Path(config["repo_dir"]).expanduser()
    python = str(Path(config.get("python", "python")).expanduser())
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        python,
        "-m",
        "scripts.inference_omnipart",
        "--image_input",
        str(Path(args.image).expanduser()),
        "--mask_input",
        str(Path(args.mask).expanduser()),
    ]
    extra_args = [str(item) for item in config.get("extra_args", [])]
    if "--output_dir" not in extra_args and "--output-dir" not in extra_args:
        extra_args.extend(["--output_dir", str(output_dir)])
    command.extend(extra_args)
    command.extend(passthrough)

    before = time.time()
    stdout_path = output_dir / "omnipart.stdout.log"
    stderr_path = output_dir / "omnipart.stderr.log"
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        proc = subprocess.run(command, cwd=repo_dir, stdout=stdout, stderr=stderr, text=True, timeout=int(config.get("timeout_seconds", 7200)), check=False)
    if proc.returncode != 0:
        raise SystemExit(f"OmniPart failed with exit code {proc.returncode}; see {stderr_path}")

    outputs = discover_outputs(output_dir, before)
    manifest = {
        "adapter": "omnipart",
        "image": str(Path(args.image).expanduser()),
        "mask": str(Path(args.mask).expanduser()),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "parts": [
            {"id": f"part_{index:03d}", "uri": str(path), "kind": "mesh" if path.suffix.lower() in MESH_SUFFIXES else "artifact"}
            for index, path in enumerate(outputs)
            if path.name != "parts_manifest.json"
        ],
    }
    manifest_path = output_dir / "parts_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
