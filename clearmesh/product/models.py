"""Production-facing data models for jobs, assets, and usage."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"


class JobStepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class GenerationRequest:
    input_uri: str
    mode: str = "image_to_3d"
    prompt: str | None = None
    output_formats: list[str] = field(default_factory=lambda: ["glb", "obj"])
    point_budgets: list[int] = field(default_factory=lambda: [16_384, 40_960, 100_000])
    enable_parts: bool = True
    enable_rigging: bool = False
    quality_tier: str = "standard"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class JobStep:
    name: str
    status: JobStepStatus = JobStepStatus.PENDING
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    artifacts: dict[str, str] = field(default_factory=dict)


@dataclass
class AssetRecord:
    id: str
    job_id: str
    kind: str
    uri: str
    content_type: str | None = None
    created_at: str = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class JobRecord:
    id: str
    team_id: str
    user_id: str
    request: GenerationRequest
    status: JobStatus = JobStatus.QUEUED
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    error: str | None = None
    steps: list[JobStep] = field(default_factory=list)
    assets: list[AssetRecord] = field(default_factory=list)
    usage_credits: int = 0

    @classmethod
    def create(cls, team_id: str, user_id: str, request: GenerationRequest) -> "JobRecord":
        steps = [
            JobStep("input_validation"),
        ]
        if request.mode.startswith("edit"):
            steps.append(JobStep("easy3e_edit"))
        steps.extend(
            [
                JobStep("trellis_proxy"),
                JobStep("coarse_adapter"),
                JobStep("reference_refinement"),
                JobStep("mesh_passport"),
                JobStep("surface_normalization"),
                JobStep("shrinkwrap_projection"),
                JobStep("retopology_planning"),
                JobStep("chart_remesh"),
                JobStep("chart_stitch"),
                JobStep("quad_remesh"),
                JobStep("feature_projection"),
                JobStep("preview_publish"),
                JobStep("point_cloud_bridge"),
                JobStep("part_structure"),
                JobStep("mesh_head"),
                JobStep("mesh_cleanup"),
                JobStep("repair_validation"),
                JobStep("production_gate"),
                JobStep("export_package"),
            ]
        )
        if request.enable_rigging:
            steps.append(JobStep("autorigging"))
        return cls(id=f"job_{uuid4().hex}", team_id=team_id, user_id=user_id, request=request, steps=steps)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        for step in data["steps"]:
            step["status"] = step["status"].value if hasattr(step["status"], "value") else step["status"]
        return data
