#!/usr/bin/env python3
"""Prepare a frozen FACE paper-augmentation corpus split.

This helper is intentionally prep-only: it exports deterministic offline mesh
augmentations, tokenizes them once, and creates a grouped train/test split. It
does not launch training or Thunder jobs. Train frozen rungs with online
augmentation disabled so this finite corpus stays reproducible.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


BUILD_SCRIPT = REPO_ROOT / "scripts" / "research" / "build_face_token_dataset.py"
SPLIT_SCRIPT = REPO_ROOT / "scripts" / "research" / "split_face_token_dataset.py"
DEFAULT_SPLIT_GROUP_FIELD = "source_name"
DEFAULT_SPLIT_GROUP_REGEX = r"^(?P<group>.*)_(?:base|aug[0-9]+)$"


def _json_line(row: dict[str, Any]) -> str:
    return json.dumps(row, sort_keys=True) + "\n"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _run(cmd: list[str], *, cwd: Path) -> None:
    printable = " ".join(shlex.quote(part) for part in cmd)
    print(f"+ {printable}", flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _ensure_fresh_output(output_dir: Path) -> None:
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        return
    if any(output_dir.iterdir()):
        raise SystemExit(
            f"{output_dir} is not empty; choose a fresh --output-dir so stale meshes "
            "cannot leak into the frozen split"
        )


def _validate_args(args: argparse.Namespace) -> None:
    if args.mesh_dir is None and args.manifest is None and args.synthetic_count <= 0:
        raise SystemExit("one of --mesh-dir, --manifest, or --synthetic-count is required")
    if args.variants < 0:
        raise SystemExit("--variants must be non-negative")
    if args.scale_min <= 0.0 or args.scale_max <= 0.0:
        raise SystemExit("augmentation scale bounds must be positive")
    if args.scale_min > args.scale_max:
        raise SystemExit("--scale-min cannot exceed --scale-max")
    if not 0.0 <= args.flip_prob <= 1.0:
        raise SystemExit("--flip-prob must be in [0, 1]")
    if args.num_bins <= 1:
        raise SystemExit("--num-bins must be greater than 1")
    if args.max_faces <= 0:
        raise SystemExit("--max-faces must be positive")
    if args.point_samples < 0:
        raise SystemExit("--point-samples must be non-negative")
    if args.test_count < 0:
        raise SystemExit("--test-count must be non-negative")
    if not 0.0 < args.test_ratio < 1.0:
        raise SystemExit("--test-ratio must be in (0, 1)")
    if not args.split_group_field:
        raise SystemExit("--split-group-field is required for frozen base/aug leakage prevention")
    for script in (BUILD_SCRIPT, SPLIT_SCRIPT):
        if not script.exists():
            raise SystemExit(f"missing required helper: {script}")


def _export_frozen_augmented_meshes(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    # Keep mesh dependencies lazy so `--help` works before trimesh is installed.
    try:
        import numpy as np

        from scripts.research.build_face_token_dataset import _iter_meshes
        from scripts.research.export_augmented_mesh_corpus import (
            _augment_mesh,
            _augmentation_matrix,
            _export_mesh,
            _slug,
        )
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "missing mesh prep dependency; run with the project Python environment "
            "(for example python3) so numpy/trimesh are available"
        ) from exc

    mesh_dir = output_dir / "augmented_meshes"
    mesh_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = output_dir / "augmented_mesh_manifest.jsonl"
    rng = np.random.default_rng(args.seed)
    source_count = 0
    exported_count = 0

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for source_name, mesh in _iter_meshes(
            args.mesh_dir,
            args.manifest,
            args.synthetic_count,
            args.seed,
            args.synthetic_kind,
        ):
            if args.limit and source_count >= args.limit:
                break
            source_group = f"{source_count:06d}_{_slug(source_name)}"
            base_path = mesh_dir / f"{source_group}_base.glb"
            _export_mesh(mesh, base_path)
            manifest.write(
                _json_line(
                    {
                        "path": str(base_path),
                        "source_index": source_count,
                        "source_name": source_name,
                        "variant": "base",
                        "variant_index": -1,
                        "group": source_group,
                    }
                )
            )
            exported_count += 1

            for variant_index in range(args.variants):
                affine = _augmentation_matrix(
                    rng,
                    rotation=args.rotation,
                    scale_min=args.scale_min,
                    scale_max=args.scale_max,
                    flip_prob=args.flip_prob,
                )
                augmented = _augment_mesh(mesh, affine)
                variant_path = mesh_dir / f"{source_group}_aug{variant_index:03d}.glb"
                _export_mesh(augmented, variant_path)
                manifest.write(
                    _json_line(
                        {
                            "path": str(variant_path),
                            "source_index": source_count,
                            "source_name": source_name,
                            "variant": f"aug{variant_index:03d}",
                            "variant_index": variant_index,
                            "group": source_group,
                            "rotation": args.rotation,
                            "scale_min": args.scale_min,
                            "scale_max": args.scale_max,
                            "flip_prob": args.flip_prob,
                        }
                    )
                )
                exported_count += 1
            source_count += 1

    if source_count == 0:
        raise SystemExit("no source meshes were exported")
    return {
        "mesh_dir": str(mesh_dir),
        "manifest": str(manifest_path),
        "source_count": source_count,
        "exported_mesh_count": exported_count,
        "variants_per_source": args.variants,
    }


def _variant_group(row: dict[str, Any]) -> str:
    raw = str(row.get("source_name") or Path(str(row["path"])).stem)
    match = re.match(DEFAULT_SPLIT_GROUP_REGEX, raw)
    if match:
        return str(match.group("group"))
    return raw


def _validate_no_variant_leakage(split_dir: Path) -> dict[str, Any]:
    train_manifest = split_dir / "train" / "manifest.jsonl"
    test_manifest = split_dir / "test" / "manifest.jsonl"
    split_summary_path = split_dir / "split_summary.json"
    split_summary = json.loads(split_summary_path.read_text(encoding="utf-8"))
    if split_summary.get("split_mode") != "group":
        raise SystemExit("split did not run in grouped mode")
    train_groups = {_variant_group(row) for row in _read_jsonl(train_manifest)}
    test_groups = {_variant_group(row) for row in _read_jsonl(test_manifest)}
    overlap = sorted(train_groups & test_groups)
    if overlap:
        preview = ", ".join(overlap[:10])
        raise SystemExit(f"base/augmentation leakage across train/test groups: {preview}")
    return {
        "split_mode": split_summary.get("split_mode"),
        "group_field": split_summary.get("group_field"),
        "group_regex": split_summary.get("group_regex"),
        "group_count": split_summary.get("group_count"),
        "test_group_count": split_summary.get("test_group_count"),
        "train_count": split_summary.get("train_count"),
        "test_count": split_summary.get("test_count"),
        "train_variant_group_count": len(train_groups),
        "test_variant_group_count": len(test_groups),
        "leakage_checked": True,
    }


def _build_token_command(args: argparse.Namespace, mesh_dir: Path, dataset_dir: Path) -> list[str]:
    return [
        args.python,
        str(BUILD_SCRIPT),
        "--mesh-dir",
        str(mesh_dir),
        "--output-dir",
        str(dataset_dir),
        "--num-bins",
        str(args.num_bins),
        "--max-faces",
        str(args.max_faces),
        "--point-samples",
        str(args.point_samples),
        "--paper-within-face-order",
        args.paper_within_face_order,
        "--indexed-face-order",
        args.indexed_face_order,
        "--seed",
        str(args.seed),
    ]


def _split_command(args: argparse.Namespace, dataset_dir: Path, split_dir: Path) -> list[str]:
    cmd = [
        args.python,
        str(SPLIT_SCRIPT),
        "--dataset-dir",
        str(dataset_dir),
        "--output-dir",
        str(split_dir),
        "--test-count",
        str(args.test_count),
        "--test-ratio",
        str(args.test_ratio),
        "--seed",
        str(args.split_seed if args.split_seed is not None else args.seed),
        "--group-field",
        args.split_group_field,
        "--group-regex",
        args.split_group_regex,
    ]
    if args.shuffle:
        cmd.append("--shuffle")
    return cmd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_argument_group("source meshes")
    source.add_argument("--mesh-dir", type=Path, default=None, help="Directory of source meshes.")
    source.add_argument("--manifest", type=Path, default=None, help="JSON/JSONL manifest accepted by build_face_token_dataset.py.")
    source.add_argument("--synthetic-count", type=int, default=0, help="Optional synthetic source meshes for local rungs.")
    source.add_argument(
        "--synthetic-kind",
        choices=["boxes", "mixed", "mixed_cycle"],
        default="mixed_cycle",
        help="Synthetic source shape family when --synthetic-count is used.",
    )
    source.add_argument("--limit", type=int, default=0, help="Limit source meshes before augmentation.")

    output = parser.add_argument_group("outputs")
    output.add_argument("--output-dir", type=Path, required=True, help="Fresh directory for augmented meshes, tokens, and split.")
    output.add_argument("--python", default=sys.executable, help="Python executable used for build/split helper subprocesses.")

    aug = parser.add_argument_group("frozen FACE-style augmentation")
    aug.add_argument("--variants", type=int, default=3, help="Frozen augmented variants per source; base is always included.")
    aug.add_argument("--rotation", choices=["none", "z", "so3"], default="so3")
    aug.add_argument("--scale-min", type=float, default=0.75)
    aug.add_argument("--scale-max", type=float, default=1.25)
    aug.add_argument("--flip-prob", type=float, default=0.5)
    aug.add_argument("--seed", type=int, default=0)

    tokens = parser.add_argument_group("FACE tokenization")
    tokens.add_argument("--num-bins", "--quantization-bins", dest="num_bins", type=int, default=128)
    tokens.add_argument("--max-faces", type=int, default=512)
    tokens.add_argument("--point-samples", type=int, default=8192)
    tokens.add_argument(
        "--paper-within-face-order",
        choices=["preserve", "rotate_min_zyx", "sort_zyx"],
        default="preserve",
    )
    tokens.add_argument("--indexed-face-order", choices=["lex", "boundary_growth"], default="lex")

    split = parser.add_argument_group("grouped split")
    split.add_argument("--test-count", type=int, default=0)
    split.add_argument("--test-ratio", type=float, default=0.2)
    split.add_argument("--split-seed", type=int, default=None)
    split.add_argument("--split-group-field", default=DEFAULT_SPLIT_GROUP_FIELD)
    split.add_argument("--split-group-regex", default=DEFAULT_SPLIT_GROUP_REGEX)
    shuffle = split.add_mutually_exclusive_group()
    shuffle.add_argument("--shuffle", dest="shuffle", action="store_true", default=True)
    shuffle.add_argument("--no-shuffle", dest="shuffle", action="store_false")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    _validate_args(args)
    _ensure_fresh_output(args.output_dir)

    export_summary = _export_frozen_augmented_meshes(args, args.output_dir)
    mesh_dir = Path(export_summary["mesh_dir"])
    dataset_dir = args.output_dir / "dataset_all"
    split_dir = args.output_dir / "split"

    build_cmd = _build_token_command(args, mesh_dir, dataset_dir)
    _run(build_cmd, cwd=REPO_ROOT)
    split_cmd = _split_command(args, dataset_dir, split_dir)
    _run(split_cmd, cwd=REPO_ROOT)
    split_validation = _validate_no_variant_leakage(split_dir)

    summary = {
        "prep_only": True,
        "paper_close_defaults": {
            "rotation": args.rotation,
            "scale_min": args.scale_min,
            "scale_max": args.scale_max,
            "flip_prob": args.flip_prob,
            "num_bins": args.num_bins,
            "max_faces": args.max_faces,
            "point_samples": args.point_samples,
            "paper_within_face_order": args.paper_within_face_order,
        },
        "paths": {
            "augmented_mesh_dir": str(mesh_dir),
            "dataset_all": str(dataset_dir),
            "split_dir": str(split_dir),
            "train_dataset": str(split_dir / "train"),
            "test_dataset": str(split_dir / "test"),
        },
        "export": export_summary,
        "split_validation": split_validation,
        "commands": {
            "build_face_token_dataset": build_cmd,
            "split_face_token_dataset": split_cmd,
        },
        "training_note": "For this frozen rung, train on split/train with online augmentation disabled.",
    }
    summary_path = args.output_dir / "prep_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
