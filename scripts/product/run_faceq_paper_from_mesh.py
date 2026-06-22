#!/usr/bin/env python3
"""Run the pure-coordinate (paper) FACE checkpoint against a proxy/reference mesh.

Drop-in analog of ``run_faceq_from_mesh.py`` for the corrected pure-coordinate
``face_paper`` model (the one that matches/beats FACE Table 2). It keeps the
*same* CLI contract as the indexed runner so ``clearmesh_pipeline_server`` can
invoke either path by an env flag.

Flow: sample a point cloud (+ normals) from the proxy mesh, derive the AR
generation length from the proxy face count (the faithful model has no EOS
head), run greedy generation via the proven ``face_paper_head`` inference, and
write the result as ``<output_dir>/<output_name>``.

The indexed-only decode flags (``--decode-mode``, ``--constraint-top-k``, ...)
are accepted and ignored so the server's existing command builder stays
compatible.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import trimesh  # noqa: E402

from clearmesh.mesh_heads.face_paper_head import run_inference  # noqa: E402


def _derive_face_count(mesh_path: Path, generation_max_faces: int) -> int:
    """Target AR length: explicit cap, else the proxy mesh face count, clamped."""
    if generation_max_faces and generation_max_faces > 0:
        return max(32, min(int(generation_max_faces), 4096))
    faces = 0
    try:
        proxy = trimesh.load(mesh_path, process=False, force="mesh")
        faces = int(len(proxy.faces))
    except Exception:
        faces = 0
    return max(32, min(faces or 800, 4096))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--point-samples", type=int, default=16384)
    parser.add_argument("--generation-max-faces", type=int, default=0)
    # Accepted-and-ignored indexed-only flags (keeps the server command builder happy):
    parser.add_argument("--face-count-mode", default=None)
    parser.add_argument("--decode-mode", default=None)
    parser.add_argument("--constraint-top-k", default=None)
    parser.add_argument("--repair-mode", default=None)
    parser.add_argument("--boundary-fill", default=None)
    parser.add_argument("--no-boundary-budget", action="store_true")
    parser.add_argument("--no-vertex-link-constraint", action="store_true")
    args = parser.parse_args()

    face_count = _derive_face_count(args.mesh, args.generation_max_faces)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    produced = run_inference(
        checkpoint=args.checkpoint,
        point_cloud=args.mesh,            # _load_point_cloud samples surface points + normals from a mesh
        output_dir=output_dir,
        case_id=Path(args.output_name).stem,
        point_samples=int(args.point_samples),
        face_count=face_count,
        device_name=args.device,
    )
    target = output_dir / args.output_name
    if Path(produced) != target:
        Path(produced).replace(target)
    print(f"[run_faceq_paper] wrote {target} (target_faces={face_count})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
