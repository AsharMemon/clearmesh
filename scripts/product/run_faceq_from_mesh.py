#!/usr/bin/env python3
"""Run the indexed FACE-Q checkpoint against a proxy/reference mesh.

The FACE-Q model predicts indexed face connectivity over a quantized vertex table.
For product inference we derive that table from the Trellis proxy mesh, sample its
surface for conditioning, decode with topology-aware constraints, and optionally
repair/fill for delivery.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import trimesh

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research.build_face_token_dataset import _encode_mesh_to_npz  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--output-name", default="faceq_artist_mesh.glb")
    parser.add_argument("--num-bins", type=int, default=1024)
    parser.add_argument("--max-faces", type=int, default=4096)
    parser.add_argument("--point-samples", type=int, default=65536)
    parser.add_argument(
        "--generation-max-faces",
        type=int,
        default=0,
        help="Optional inference-time generated-face cap. Use 0 for the checkpoint max.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--face-count-mode", choices=["gt", "predicted", "max"], default="predicted")
    parser.add_argument("--decode-mode", choices=["edge_constrained", "boundary_edge", "unconstrained"], default="boundary_edge")
    parser.add_argument("--constraint-top-k", type=int, default=24)
    parser.add_argument("--repair-mode", choices=["none", "dedupe", "manifold"], default="manifold")
    parser.add_argument("--boundary-fill", choices=["none", "fan", "centroid"], default="centroid")
    parser.add_argument("--no-boundary-budget", action="store_true")
    parser.add_argument("--no-vertex-link-constraint", action="store_true")
    return parser.parse_args()


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        pieces = [geom for geom in loaded.geometry.values() if isinstance(geom, trimesh.Trimesh)]
        if not pieces:
            raise ValueError(f"{path} does not contain mesh geometry")
        loaded = trimesh.util.concatenate(pieces)
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
        raise ValueError(f"{path} does not contain a non-empty triangular mesh")
    if not loaded.is_watertight:
        # Do not mutate topology aggressively here; final cleanup/repair happens after decode.
        loaded.remove_unreferenced_vertices()
    if len(loaded.faces) > 4096:
        raise ValueError(
            f"FACE-Q proxy mesh has {len(loaded.faces)} faces; decimate Trellis output to <=4096 before FACE-Q."
        )
    return loaded


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = args.output_dir / "faceq_work"
    sample_dir = work_dir / "sample"
    export_dir = work_dir / "exports"
    cleanup_dir = work_dir / "cleanup"
    sample_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)
    cleanup_dir.mkdir(parents=True, exist_ok=True)

    mesh = load_mesh(args.mesh)
    record = _encode_mesh_to_npz(
        index=0,
        name=args.mesh.stem,
        mesh=mesh,
        output_dir=sample_dir,
        stem="proxy_sample",
        num_bins=int(args.num_bins),
        max_faces=int(args.max_faces),
        point_samples=int(args.point_samples),
        paper_within_face_order="rotate_min_zyx",
        indexed_face_order="boundary_growth",
        seed=int(args.seed),
    )

    report_path = work_dir / "faceq_decode_report.json"
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts/research/eval_face_indexed_conditioned_tiny.py"),
        "--checkpoint",
        str(args.checkpoint),
        "--dataset-dir",
        str(sample_dir),
        "--output",
        str(report_path),
        "--export-dir",
        str(export_dir),
        "--cleanup-export-dir",
        str(cleanup_dir),
        "--limit",
        "1",
        "--decode-strategy",
        "free_run",
        "--face-count-mode",
        args.face_count_mode,
        "--generation-max-faces",
        str(args.generation_max_faces),
        "--point-samples",
        str(args.point_samples),
        "--device",
        args.device,
        "--decode-mode",
        args.decode_mode,
        "--corner-decode",
        "causal",
        "--constraint-top-k",
        str(args.constraint_top_k),
        "--token-repair-mode",
        args.repair_mode,
        "--boundary-fill",
        args.boundary_fill,
        "--split-pinched-vertices",
    ]
    if not args.no_boundary_budget:
        command.append("--boundary-budget-constraint")
    if not args.no_vertex_link_constraint:
        command.append("--vertex-link-constraint")

    proc = subprocess.run(command, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    (work_dir / "faceq_decode.stdout.log").write_text(proc.stdout, encoding="utf-8")
    (work_dir / "faceq_decode.stderr.log").write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise SystemExit(f"FACE-Q decode failed with exit={proc.returncode}; see {work_dir}")

    candidates = sorted(cleanup_dir.glob("*_cleaned.glb"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        candidates = sorted(export_dir.glob("*_generated.glb"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise SystemExit(f"FACE-Q decode produced no mesh under {export_dir} or {cleanup_dir}")
    output_path = args.output_dir / args.output_name
    shutil.copy2(candidates[0], output_path)

    summary = {
        "output": str(output_path),
        "proxy_mesh": str(args.mesh),
        "checkpoint": str(args.checkpoint),
        "sample_record": record,
        "decode_report": str(report_path),
        "selected_decode_mesh": str(candidates[0]),
        "command": command,
    }
    (args.output_dir / "faceq_inference_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
