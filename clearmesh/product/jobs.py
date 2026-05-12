"""Job service tying auth, billing, store, and future worker enqueue together."""

from __future__ import annotations

from .billing import CreditLedger
from .models import GenerationRequest, JobRecord, JobStatus
from .store import JsonJobStore


class JobService:
    def __init__(self, store: JsonJobStore | None = None, ledger: CreditLedger | None = None):
        self.store = store or JsonJobStore()
        self.ledger = ledger or CreditLedger()

    def create_job(self, team_id: str, user_id: str, request: GenerationRequest) -> JobRecord:
        job = JobRecord.create(team_id=team_id, user_id=user_id, request=request)
        credits = self.ledger.estimate_job_credits(
            quality_tier=request.quality_tier,
            enable_rigging=request.enable_rigging,
            mode=request.mode,
        )
        job.usage_credits = credits
        self.ledger.reserve(team_id=team_id, job_id=job.id, credits=credits)
        return self.store.create_job(job)

    def estimate_cost(self, request: GenerationRequest) -> int:
        return self.ledger.estimate_job_credits(
            quality_tier=request.quality_tier,
            enable_rigging=request.enable_rigging,
            mode=request.mode,
        )

    def cancel_job(self, job_id: str) -> JobRecord:
        job = self.store.get_job(job_id)
        if job.status not in {JobStatus.QUEUED, JobStatus.RUNNING}:
            return job
        self.ledger.release(job.team_id, job.id, job.usage_credits, reason="canceled_release")
        return self.store.set_status(job_id, JobStatus.CANCELED)

    def mark_succeeded(self, job_id: str) -> JobRecord:
        job = self.store.get_job(job_id)
        self.ledger.consume(job.team_id, job.id, job.usage_credits)
        return self.store.set_status(job_id, JobStatus.SUCCEEDED)

    def mark_failed(self, job_id: str, error: str, charge: bool = False) -> JobRecord:
        job = self.store.get_job(job_id)
        if charge:
            self.ledger.consume(job.team_id, job.id, job.usage_credits)
        else:
            self.ledger.release(job.team_id, job.id, job.usage_credits, reason="failed_release")
        return self.store.set_status(job_id, JobStatus.FAILED, error=error)
