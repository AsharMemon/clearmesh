#!/usr/bin/env python3
"""Build a source-ID skip registry from existing FACE manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from source_skip_registry import load_source_skip_ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Manifest files/directories: txt, jsonl, or json.")
    parser.add_argument("--output", type=Path, required=True, help="One source key per line.")
    parser.add_argument("--summary-output", type=Path, default=None)
    args = parser.parse_args()

    keys = load_source_skip_ids(args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(sorted(keys)) + ("\n" if keys else ""), encoding="utf-8")
    summary = {
        "inputs": [str(path) for path in args.inputs],
        "output": str(args.output),
        "source_skip_key_count": len(keys),
    }
    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
