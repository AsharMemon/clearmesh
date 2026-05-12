"""Runtime and product tier estimates for mesh generation profiles."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeProfile:
    name: str
    label: str
    target_faces: int | None
    preview_seconds_low: int
    preview_seconds_high: int
    mesh_head_seconds_low: int
    mesh_head_seconds_high: int
    user_visible: bool
    description: str

    @property
    def mesh_head_minutes(self) -> tuple[float, float]:
        return (self.mesh_head_seconds_low / 60, self.mesh_head_seconds_high / 60)


RUNTIME_PROFILES: dict[str, RuntimeProfile] = {
    "draft": RuntimeProfile(
        name="draft",
        label="Preview + Control Mesh",
        target_faces=30_000,
        preview_seconds_low=60,
        preview_seconds_high=240,
        mesh_head_seconds_low=300,
        mesh_head_seconds_high=900,
        user_visible=True,
        description="Fast TRELLIS/control-surface result shown first, with refinement continuing when eligible.",
    ),
    "standard": RuntimeProfile(
        name="standard",
        label="Standard",
        target_faces=80_000,
        preview_seconds_low=60,
        preview_seconds_high=240,
        mesh_head_seconds_low=600,
        mesh_head_seconds_high=1800,
        user_visible=True,
        description="Default production profile: preview first, normalized high-resolution mesh next, generative refinement only when it passes the mesh passport.",
    ),
    "high": RuntimeProfile(
        name="high",
        label="High Resolution",
        target_faces=200_000,
        preview_seconds_low=60,
        preview_seconds_high=240,
        mesh_head_seconds_low=1200,
        mesh_head_seconds_high=3600,
        user_visible=True,
        description="Higher-detail control surface and optional targeted artist-mesh refinement. The first preview still appears early.",
    ),
}


def profile_for_quality_tier(quality_tier: str) -> RuntimeProfile:
    return RUNTIME_PROFILES.get(quality_tier, RUNTIME_PROFILES["standard"])


def runtime_quote(quality_tier: str, *, includes_trellis: bool = True, enable_parts: bool = False, part_count: int | None = None) -> dict:
    """Return a conservative user-facing runtime quote.

    Part-aware generation can improve quality, but it multiplies mesh-head work
    unless parts are distributed across multiple GPUs. We show both the serial
    and ideal parallel implication so product copy does not overpromise.
    """

    profile = profile_for_quality_tier(quality_tier)
    trellis_low = 8 * 60 if includes_trellis else 0
    trellis_high = 25 * 60 if includes_trellis else 0
    low = profile.mesh_head_seconds_low + trellis_low
    high = profile.mesh_head_seconds_high + trellis_high
    preview_low = profile.preview_seconds_low + min(trellis_low, 8 * 60)
    preview_high = profile.preview_seconds_high + min(trellis_high, 15 * 60)
    quote = {
        "profile": profile.name,
        "label": profile.label,
        "target_faces": profile.target_faces,
        "preview_seconds": {
            "low": preview_low,
            "high": preview_high,
        },
        "mesh_head_seconds": {
            "low": profile.mesh_head_seconds_low,
            "high": profile.mesh_head_seconds_high,
        },
        "end_to_end_seconds": {"low": low, "high": high},
        "user_visible": profile.user_visible,
        "description": profile.description,
    }
    if enable_parts:
        count = max(1, int(part_count or 4))
        quote["part_aware"] = {
            "estimated_part_count": count,
            "serial_mesh_head_seconds": {
                "low": profile.mesh_head_seconds_low * count,
                "high": profile.mesh_head_seconds_high * count,
            },
            "ideal_parallel_mesh_head_seconds": {
                "low": profile.mesh_head_seconds_low,
                "high": profile.mesh_head_seconds_high,
            },
            "note": "Per-part generation improves locality but needs parallel GPUs to avoid multiplying latency.",
        }
    return quote
