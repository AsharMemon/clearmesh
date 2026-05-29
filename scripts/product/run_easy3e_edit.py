#!/usr/bin/env python3
"""Run an Easy3E edit as a ClearMesh worker command hook.

This is intentionally a thin wrapper around ``clearmesh.editing.easy3e`` so the
product worker can launch Easy3E with trusted metadata.easy3e_command while the
public API keeps executable metadata disabled.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time


class Easy3EPreflightError(RuntimeError):
    """Raised when the configured TRELLIS snapshot cannot support Easy3E."""


def _looks_like_hf_repo_id(value: str) -> bool:
    """Return true for IDs like ``microsoft/TRELLIS.2-4B``, not paths."""
    if value.startswith(("/", "./", "../", "~")):
        return False
    return "/" in value and not Path(value).expanduser().exists()


def _cached_snapshot_dirs(repo_id: str) -> list[Path]:
    """Return existing local HF snapshot dirs without calling the network."""
    encoded = "models--" + repo_id.replace("/", "--")
    roots: list[Path] = []
    if os.getenv("HF_HUB_CACHE"):
        roots.append(Path(os.environ["HF_HUB_CACHE"]))
    if os.getenv("HF_HOME"):
        roots.append(Path(os.environ["HF_HOME"]) / "hub")
    roots.extend(
        [
            Path("/ephemeral/hf-cache/hub"),
            Path.home() / ".cache" / "huggingface" / "hub",
        ]
    )
    seen: set[Path] = set()
    snapshots: list[Path] = []
    for root in roots:
        root = root.expanduser()
        if root in seen:
            continue
        seen.add(root)
        snap_root = root / encoded / "snapshots"
        if not snap_root.exists():
            continue
        snapshots.extend(path for path in snap_root.iterdir() if path.is_dir())
    return sorted(snapshots, key=lambda path: path.stat().st_mtime, reverse=True)


def resolve_model_dir(model_dir: str) -> Path:
    """Resolve a local model path or an already-cached Hugging Face repo ID.

    Easy3E needs direct filesystem access to ``pipeline.json`` and several
    checkpoint files. The public product bridge passes the model by ID because
    TRELLIS.2 generation can load that lazily, but editing should not trigger
    huge downloads or hang on a missing snapshot. We therefore only resolve
    Hugging Face IDs from the local cache.
    """
    path = Path(model_dir).expanduser()
    if path.exists():
        return path.resolve()
    if not _looks_like_hf_repo_id(model_dir):
        return path.resolve()
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - depends on runtime image
        for snapshot in _cached_snapshot_dirs(model_dir):
            if (snapshot / "pipeline.json").exists():
                return snapshot.resolve()
        raise Easy3EPreflightError(
            "huggingface_hub is required to resolve a Hugging Face model ID for Easy3E"
        ) from exc
    try:
        return Path(snapshot_download(repo_id=model_dir, local_files_only=True)).resolve()
    except Exception as exc:  # noqa: BLE001 - surface actionable runtime detail.
        for snapshot in _cached_snapshot_dirs(model_dir):
            if (snapshot / "pipeline.json").exists():
                return snapshot.resolve()
        raise Easy3EPreflightError(
            f"Easy3E model snapshot {model_dir!r} is not available in the local HF cache"
        ) from exc


def validate_easy3e_model_dir(model_dir: Path) -> None:
    """Validate the files the current Easy3E implementation requires."""
    missing: list[str] = []
    if not (model_dir / "pipeline.json").exists():
        missing.append("pipeline.json")
    shape_encoder = model_dir / "ckpts" / "shape_enc_next_dc_f16c32_fp16"
    if not shape_encoder.with_suffix(".json").exists():
        missing.append("ckpts/shape_enc_next_dc_f16c32_fp16.json")
    if not shape_encoder.with_suffix(".safetensors").exists():
        missing.append("ckpts/shape_enc_next_dc_f16c32_fp16.safetensors")
    if missing:
        raise Easy3EPreflightError(
            "Easy3E editing is not available for this TRELLIS snapshot; "
            "missing " + ", ".join(missing) + ". "
            "Generation-only TRELLIS.2 snapshots often ship decoders/flows but "
            "not the mesh-to-SLAT shape encoder needed for editing. Keep "
            "CLEARMESH_EASY3E_ENABLED=0 or provide CLEARMESH_TRELLIS_MODEL "
            "pointing at a snapshot that includes the encoder checkpoint."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-mesh", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-name", default="edited_mesh.glb")
    parser.add_argument("--edit-image")
    parser.add_argument("--source-image")
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

    try:
        model_dir = resolve_model_dir(args.model_dir)
        validate_easy3e_model_dir(model_dir)
    except Easy3EPreflightError as exc:
        report_path = output_dir / "easy3e_report.json"
        report_path.write_text(
            json.dumps(
                {
                    "adapter": "easy3e",
                    "ok": False,
                    "error": str(exc),
                    "model_dir": args.model_dir,
                    "output_path": str(output_path),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        raise SystemExit(str(exc)) from exc

    from clearmesh.editing.easy3e import Easy3EEditor

    started = time.time()
    editor = Easy3EEditor(trellis2_dir=args.trellis2_dir, model_dir=str(model_dir), device=args.device)
    if args.edit_image:
        mode = "edit_image"
        result = editor.edit(
            source_mesh=source_mesh,
            edit_image=args.edit_image,
            source_image=args.source_image,
            output_path=str(output_path),
            options=options,
        )
    else:
        mode = "edit_text"
        result = editor.edit_from_text(source_mesh=source_mesh, instruction=args.instruction or "", view=args.view, output_path=str(output_path), options=options)

    report = {
        "adapter": "easy3e",
        "mode": mode,
        "source_mesh": str(source_mesh),
        "edit_image": str(Path(args.edit_image).expanduser()) if args.edit_image else None,
        "source_image": str(Path(args.source_image).expanduser()) if args.source_image else None,
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
