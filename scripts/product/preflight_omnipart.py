#!/usr/bin/env python3
"""Validate that an OmniPart checkout can be used by ClearMesh hooks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-json", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config_json.read_text(encoding="utf-8"))
    repo_dir = Path(config["repo_dir"]).expanduser()
    python = Path(config.get("python", "python")).expanduser()
    checks = {
        "repo_dir": repo_dir.exists(),
        "python": python.exists() if python.is_absolute() else True,
        "inference_script": (repo_dir / "scripts" / "inference_omnipart.py").exists(),
        "app": (repo_dir / "app.py").exists(),
        "requirements": (repo_dir / "requirements.txt").exists(),
    }
    result = subprocess.run([str(python), "-c", "import torch; print(torch.__version__)"], cwd=repo_dir if repo_dir.exists() else None, text=True, capture_output=True, check=False)
    checks["torch_import"] = result.returncode == 0
    print(json.dumps({"ok": all(checks.values()), "checks": checks, "torch": result.stdout.strip(), "stderr": result.stderr.strip()}, indent=2, sort_keys=True))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
