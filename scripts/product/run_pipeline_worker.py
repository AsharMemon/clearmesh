#!/usr/bin/env python3
"""Run the staged ClearMesh product worker.

Use this on GPU hosts with --execute-heavy once mesh deps and public mesh-head
repos are installed. Without --execute-heavy it behaves like a safe scaffold.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clearmesh.product.artifacts import ArtifactStore  # noqa: E402
from clearmesh.product.billing import CreditLedger  # noqa: E402
from clearmesh.product.jobs import JobService  # noqa: E402
from clearmesh.product.pipeline_worker import PipelineWorker  # noqa: E402
from clearmesh.product.store import JsonJobStore  # noqa: E402


def build_job_store(state_root: str):
    postgres_dsn = os.getenv("CLEARMESH_POSTGRES_DSN")
    if postgres_dsn:
        from clearmesh.product.postgres_store import PostgresJobStore

        return PostgresJobStore(postgres_dsn)
    return JsonJobStore(state_root)


def build_artifact_store(artifact_root: str):
    bucket = os.getenv("CLEARMESH_S3_BUCKET")
    if bucket:
        from clearmesh.product.s3_artifacts import S3ArtifactStore

        return S3ArtifactStore(
            artifact_root,
            bucket=bucket,
            prefix=os.getenv("CLEARMESH_S3_PREFIX", ""),
        )
    return ArtifactStore(artifact_root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", default=".clearmesh_state")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--execute-heavy", action="store_true")
    parser.add_argument("--mesh-head", default="meshripple")
    parser.add_argument("--mesh-head-config-json", type=Path)
    parser.add_argument("--preferred-point-budget", type=int, default=40960)
    args = parser.parse_args()

    mesh_head_config = {}
    if args.mesh_head_config_json:
        mesh_head_config = json.loads(args.mesh_head_config_json.read_text(encoding="utf-8"))

    store = build_job_store(args.state_root)
    service = JobService(store=store, ledger=CreditLedger(Path(args.state_root) / "credits.json"))
    worker = PipelineWorker(
        store=store,
        service=service,
        artifacts=build_artifact_store(args.artifact_root),
        execute_heavy=args.execute_heavy,
        mesh_head=args.mesh_head,
        mesh_head_config=mesh_head_config,
        preferred_point_budget=args.preferred_point_budget,
    )

    while True:
        job_id = worker.run_once()
        if job_id is None:
            print("no queued jobs")
            return 0
        print(f"processed {job_id}")
        if args.once:
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
