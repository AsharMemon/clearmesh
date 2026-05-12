"""Tiny JSON-backed store for local API/product development.

This is intentionally simple. Production should replace it with Postgres, but
keeping the interface explicit makes that migration boring.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from threading import Lock
from typing import Any

from .models import GenerationRequest, JobRecord, JobStatus, JobStep, JobStepStatus, AssetRecord, utc_now


class JsonJobStore:
    def __init__(self, root: str | Path = ".clearmesh_state"):
        self.root = Path(root)
        self.jobs_dir = self.root / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def create_job(self, job: JobRecord) -> JobRecord:
        with self._lock:
            self._write_job(job)
        return job

    def get_job(self, job_id: str) -> JobRecord:
        path = self.jobs_dir / f"{job_id}.json"
        if not path.exists():
            raise KeyError(job_id)
        return self._decode_job(json.loads(path.read_text(encoding="utf-8")))

    def list_jobs(self, team_id: str | None = None) -> list[JobRecord]:
        jobs = [self._decode_job(json.loads(path.read_text(encoding="utf-8"))) for path in self.jobs_dir.glob("job_*.json")]
        if team_id is not None:
            jobs = [job for job in jobs if job.team_id == team_id]
        return sorted(jobs, key=lambda job: job.created_at, reverse=True)

    def claim_next_queued(self) -> JobRecord | None:
        """Claim the oldest queued job for a worker process."""

        with self._lock:
            jobs = [
                self._decode_job(json.loads(path.read_text(encoding="utf-8")))
                for path in self.jobs_dir.glob("job_*.json")
            ]
            queued = sorted(
                (job for job in jobs if job.status == JobStatus.QUEUED),
                key=lambda job: job.created_at,
            )
            if not queued:
                return None
            job = queued[0]
            job.status = JobStatus.RUNNING
            job.updated_at = utc_now()
            self._write_job(job)
            return job

    def update_job(self, job: JobRecord) -> JobRecord:
        job.updated_at = utc_now()
        with self._lock:
            self._write_job(job)
        return job

    def set_status(self, job_id: str, status: JobStatus, error: str | None = None) -> JobRecord:
        job = self.get_job(job_id)
        job.status = status
        job.error = error
        return self.update_job(job)

    def update_step(
        self,
        job_id: str,
        step_name: str,
        status: JobStepStatus,
        error: str | None = None,
        artifacts: dict[str, str] | None = None,
    ) -> JobRecord:
        job = self.get_job(job_id)
        for step in job.steps:
            if step.name == step_name:
                step.status = status
                step.error = error
                if status == JobStepStatus.RUNNING and step.started_at is None:
                    step.started_at = utc_now()
                if status in {JobStepStatus.SUCCEEDED, JobStepStatus.FAILED, JobStepStatus.SKIPPED}:
                    step.finished_at = utc_now()
                if artifacts:
                    step.artifacts.update(artifacts)
                break
        else:
            raise KeyError(f"step {step_name!r} not found for {job_id}")
        return self.update_job(job)

    def add_asset(self, job_id: str, asset: AssetRecord) -> JobRecord:
        job = self.get_job(job_id)
        job.assets.append(asset)
        return self.update_job(job)

    def _write_job(self, job: JobRecord) -> None:
        (self.jobs_dir / f"{job.id}.json").write_text(json.dumps(job.to_dict(), indent=2), encoding="utf-8")

    def _decode_job(self, data: dict[str, Any]) -> JobRecord:
        request = GenerationRequest(**data["request"])
        steps = [JobStep(name=s["name"], status=JobStepStatus(s["status"]), started_at=s.get("started_at"), finished_at=s.get("finished_at"), error=s.get("error"), artifacts=s.get("artifacts", {})) for s in data.get("steps", [])]
        assets = [AssetRecord(**asset) for asset in data.get("assets", [])]
        return JobRecord(
            id=data["id"],
            team_id=data["team_id"],
            user_id=data["user_id"],
            request=request,
            status=JobStatus(data["status"]),
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            error=data.get("error"),
            steps=steps,
            assets=assets,
            usage_credits=int(data.get("usage_credits", 0)),
        )
