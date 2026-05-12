#!/usr/bin/env python3
"""Create a batch of local ClearMesh jobs from a CSV manifest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clearmesh.product.artifacts import ArtifactStore
from clearmesh.product.billing import CreditLedger
from clearmesh.product.jobs import JobService
from clearmesh.product.models import GenerationRequest
from clearmesh.product.store import JsonJobStore


def parse_bool(value: str | bool | None, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_point_budgets(value: str | None) -> list[int]:
    if not value:
        return [16_384, 40_960, 100_000]
    return [int(part.strip()) for part in value.split(";") if part.strip()] if ";" in value else [int(part.strip()) for part in value.split(",") if part.strip()]


def copy_input_if_needed(row: dict[str, str], artifacts: ArtifactStore) -> str:
    input_uri = row.get("input_uri", "").strip()
    input_path = row.get("input_path", "").strip()
    if input_uri:
        return input_uri
    if not input_path:
        raise ValueError("each row needs input_uri or input_path")
    source = Path(input_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    case_id = row["case_id"].strip()
    destination = artifacts.root / "uploads" / "benchmark" / f"{case_id}{source.suffix or '.bin'}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return f"local://uploads/benchmark/{destination.name}"


def metadata_from_row(row: dict[str, str], args: argparse.Namespace) -> dict:
    metadata = {"project_id": row.get("project_id") or args.project_id, "case_id": row["case_id"].strip()}
    command_keys = [
        "trellis_command",
        "trellis_cwd",
        "trellis_timeout_seconds",
        "part_structure_command",
        "part_structure_cwd",
        "part_structure_timeout_seconds",
        "part_mask_uri",
        "part_mask_path",
        "easy3e_command",
        "easy3e_cwd",
        "autorigging_command",
        "autorigging_cwd",
        "source_mesh_path",
        "edit_image_uri",
        "edit_image_path",
    ]
    for key in command_keys:
        value = row.get(key, "").strip()
        if value:
            metadata[key] = value
    if args.metadata_json:
        metadata.update(json.loads(args.metadata_json.read_text(encoding="utf-8")))
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--state-root", default=".clearmesh_state")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--team-id", default="team_dev")
    parser.add_argument("--user-id", default="user_dev")
    parser.add_argument("--grant-credits", type=int, default=0)
    parser.add_argument("--project-id", default="benchmark_v0")
    parser.add_argument("--metadata-json", type=Path)
    args = parser.parse_args()

    store = JsonJobStore(args.state_root)
    ledger = CreditLedger(Path(args.state_root) / "credits.json")
    if args.grant_credits:
        ledger.grant(args.team_id, args.grant_credits, reason="batch_dev_grant")
    service = JobService(store=store, ledger=ledger)
    artifacts = ArtifactStore(args.artifact_root)

    job_ids: list[str] = []
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if not row.get("case_id") or row["case_id"].strip().startswith("#"):
                continue
            input_uri = copy_input_if_needed(row, artifacts)
            request = GenerationRequest(
                input_uri=input_uri,
                mode=row.get("mode") or "image_to_3d",
                prompt=row.get("prompt") or None,
                output_formats=[part.strip() for part in (row.get("output_formats") or "glb,obj").split(",") if part.strip()],
                point_budgets=parse_point_budgets(row.get("point_budgets")),
                enable_parts=parse_bool(row.get("enable_parts"), default=True),
                enable_rigging=parse_bool(row.get("enable_rigging"), default=False),
                quality_tier=row.get("quality_tier") or "standard",
                metadata=metadata_from_row(row, args),
            )
            job = service.create_job(args.team_id, args.user_id, request)
            job_ids.append(job.id)
            print(job.id)
    summary = Path(args.state_root) / "last_batch_jobs.json"
    summary.write_text(json.dumps({"jobs": job_ids}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
