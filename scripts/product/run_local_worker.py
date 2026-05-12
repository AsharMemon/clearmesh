#!/usr/bin/env python3
"""Local worker skeleton for advancing queued ClearMesh jobs.

This worker does not run TRELLIS.2 by default. It marks deterministic scaffold
steps and creates the directories that a GPU worker will fill in production.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clearmesh.product.artifacts import ArtifactStore
from clearmesh.product.billing import CreditLedger
from clearmesh.product.jobs import JobService
from clearmesh.product.models import JobStatus, JobStepStatus
from clearmesh.product.store import JsonJobStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local ClearMesh scaffold worker")
    parser.add_argument("--state-root", default=".clearmesh_state")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    store = JsonJobStore(args.state_root)
    service = JobService(store=store, ledger=CreditLedger(Path(args.state_root) / "credits.json"))
    artifacts = ArtifactStore(args.artifact_root)

    for job in store.list_jobs():
        if job.status != JobStatus.QUEUED:
            continue
        store.set_status(job.id, JobStatus.RUNNING)
        project_id = job.request.metadata.get("project_id", "default")
        try:
            for step in job.steps:
                store.update_step(job.id, step.name, JobStepStatus.RUNNING)
                placeholder = artifacts.path(project_id, job.id, "reports", f"{step.name}.txt")
                placeholder.write_text(f"{step.name} placeholder for {job.id}\n", encoding="utf-8")
                store.update_step(
                    job.id,
                    step.name,
                    JobStepStatus.SUCCEEDED if step.name not in {"trellis_proxy", "mesh_head"} else JobStepStatus.SKIPPED,
                    artifacts={"placeholder": str(placeholder)},
                )
            service.mark_succeeded(job.id)
            print(f"advanced scaffold job {job.id}")
        except Exception as exc:  # noqa: BLE001 - worker should record failures.
            service.mark_failed(job.id, error=f"{type(exc).__name__}: {exc}")
            print(f"failed job {job.id}: {exc}")
        if args.once:
            break


if __name__ == "__main__":
    main()
