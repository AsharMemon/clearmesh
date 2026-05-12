"""Helpers for OmniPart-style part manifests."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PartRecord:
    id: str
    label: str | None = None
    point_cloud_path: str | None = None
    proxy_mesh_path: str | None = None
    bbox: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def load_parts_manifest(path: str | Path) -> list[PartRecord]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_parts = data.get("parts", data if isinstance(data, list) else [])
    parts: list[PartRecord] = []
    for index, raw in enumerate(raw_parts):
        if not isinstance(raw, dict):
            continue
        part_id = str(raw.get("id") or raw.get("part_id") or f"part_{index:03d}")
        parts.append(
            PartRecord(
                id=part_id,
                label=raw.get("label") or raw.get("name"),
                point_cloud_path=raw.get("point_cloud_path") or raw.get("point_cloud"),
                proxy_mesh_path=raw.get("proxy_mesh_path") or raw.get("proxy_mesh"),
                bbox=raw.get("bbox"),
                metadata={key: value for key, value in raw.items() if key not in {"id", "part_id", "label", "name", "point_cloud_path", "point_cloud", "proxy_mesh_path", "proxy_mesh", "bbox"}},
            )
        )
    return parts


def write_parts_manifest(path: str | Path, parts: list[PartRecord], *, source: str | None = None) -> Path:
    payload = {
        "source": source,
        "parts": [
            {
                "id": part.id,
                "label": part.label,
                "point_cloud_path": part.point_cloud_path,
                "proxy_mesh_path": part.proxy_mesh_path,
                "bbox": part.bbox,
                **part.metadata,
            }
            for part in parts
        ],
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output
