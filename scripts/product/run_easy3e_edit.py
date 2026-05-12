#!/usr/bin/env python3
"""Run an Easy3E edit as a ClearMesh worker command hook.

This is intentionally a thin wrapper around ``clearmesh.editing.easy3e`` so the
product worker can launch Easy3E with trusted metadata.easy3e_command while the
public API keeps executable metadata disabled.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-mesh", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-name", default="edited_mesh.glb")
    parser.add_argument("--edit-image")
    parser.add_argument("--instruction")
    parser.add_argument("--view", default="front", choices=["front", "back", "left", "right", "top", "bottom"])
    parser.add_argument("--trellis2-dir", default="/home/ubuntu/TRELLIS.2")
    parser.add_argument("--model-dir", default="microsoft/TRELLIS.2-4B")
    parser.add_argument("--device")
    parser.add_argument("--options-json", type=Path)
    args = parser.parse_args()

    if not args.edit_image and not args.instruction:
        raise SystemExit("provide --edit-image or --instruction")

    source_mesh = Path(args.source_mesh).expanduser().resolve()
    if not source_mesh.exists():
        raise SystemExit(f"source mesh not found: {source_mesh}")
    if args.edit_image and not Path(args.edit_image).expanduser().exists():
        raise SystemExit(f"edit image not found: {args.edit_image}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_name
    options = json.loads(args.options_json.read_text(encoding="utf-8")) if args.options_json else {}

    from clearmesh.editing.easy3e import Easy3EEditor

    started = time.time()
    editor = Easy3EEditor(trellis2_dir=args.trellis2_dir, model_dir=args.model_dir, device=args.device)
    if args.edit_image:
        mode = "edit_image"
        result = editor.edit(source_mesh=source_mesh, edit_image=args.edit_image, output_path=str(output_path), options=options)
    else:
        mode = "edit_text"
        result = editor.edit_from_text(source_mesh=source_mesh, instruction=args.instruction or "", view=args.view, output_path=str(output_path), options=options)

    report = {
        "adapter": "easy3e",
        "mode": mode,
        "source_mesh": str(source_mesh),
        "edit_image": str(Path(args.edit_image).expanduser()) if args.edit_image else None,
        "instruction": args.instruction,
        "view": args.view,
        "output_path": str(output_path),
        "timings": getattr(result, "timings", {}),
        "elapsed_seconds": time.time() - started,
    }
    report_path = output_dir / "easy3e_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
