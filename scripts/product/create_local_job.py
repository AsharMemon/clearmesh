#!/usr/bin/env python3
"""Create a local ClearMesh job without running the API server."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clearmesh.product.billing import CreditLedger
from clearmesh.product.jobs import JobService
from clearmesh.product.models import GenerationRequest
from clearmesh.product.store import JsonJobStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a local ClearMesh scaffold job")
    parser.add_argument("--input-uri", required=True)
    parser.add_argument("--team-id", default="team_dev")
    parser.add_argument("--user-id", default="user_dev")
    parser.add_argument("--state-root", default=".clearmesh_state")
    parser.add_argument("--grant-credits", type=int, default=100)
    parser.add_argument("--mode", default="image_to_3d", choices=["image_to_3d", "text_to_3d", "edit_image", "edit_text"])
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--enable-rigging", action="store_true")
    parser.add_argument("--disable-parts", action="store_true")
    parser.add_argument("--quality-tier", default="standard", choices=["draft", "standard", "high"])
    parser.add_argument("--project-id", default="dev")
    parser.add_argument("--proxy-mesh-path", default=None, help="Precomputed TRELLIS.2 proxy mesh for GPU bake-off jobs")
    parser.add_argument("--artist-mesh-path", default=None, help="Precomputed artist mesh for repair/eval/export testing")
    parser.add_argument("--metadata-json", type=Path, help="Optional JSON metadata merged into the job request")
    parser.add_argument("--trellis-command", default=None, help="Optional command string for GPU TRELLIS.2 proxy generation")
    parser.add_argument("--trellis-cwd", default=None)
    parser.add_argument("--part-structure-command", default=None, help="Optional command string for OmniPart-style part manifest generation")
    parser.add_argument("--part-structure-cwd", default=None)
    args = parser.parse_args()

    store = JsonJobStore(args.state_root)
    ledger = CreditLedger(Path(args.state_root) / "credits.json")
    if args.grant_credits:
        ledger.grant(args.team_id, args.grant_credits, reason="local_dev_grant")
    service = JobService(store=store, ledger=ledger)
    metadata = {"project_id": args.project_id}
    if args.proxy_mesh_path:
        metadata["proxy_mesh_path"] = args.proxy_mesh_path
    if args.artist_mesh_path:
        metadata["artist_mesh_path"] = args.artist_mesh_path
    if args.trellis_command:
        metadata["trellis_command"] = args.trellis_command
    if args.trellis_cwd:
        metadata["trellis_cwd"] = args.trellis_cwd
    if args.part_structure_command:
        metadata["part_structure_command"] = args.part_structure_command
    if args.part_structure_cwd:
        metadata["part_structure_cwd"] = args.part_structure_cwd
    if args.metadata_json:
        metadata.update(json.loads(args.metadata_json.read_text(encoding="utf-8")))
    job = service.create_job(
        team_id=args.team_id,
        user_id=args.user_id,
        request=GenerationRequest(
            input_uri=args.input_uri,
            mode=args.mode,
            prompt=args.prompt,
            enable_parts=not args.disable_parts,
            enable_rigging=args.enable_rigging,
            quality_tier=args.quality_tier,
            metadata=metadata,
        ),
    )
    print(job.to_dict()["id"])


if __name__ == "__main__":
    main()
