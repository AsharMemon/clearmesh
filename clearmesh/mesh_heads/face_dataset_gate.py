"""Quality gates for FACE-token training targets.

FACE-like heads are useful only when the target sequence represents an editable
surface. Raw generator fragments are valuable diagnostics, but they should not
quietly become artist-mesh supervision.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FaceDatasetGateThresholds:
    require_decoded_watertight: bool = True
    require_token_watertight: bool = True
    max_boundary_edges: int | None = 0
    max_nonmanifold_edges: int | None = 0
    min_edge_pairing_ratio: float | None = 0.999


def thresholds_for_profile(profile: str) -> FaceDatasetGateThresholds:
    if profile == "strict":
        return FaceDatasetGateThresholds()
    if profile == "proxy":
        return FaceDatasetGateThresholds(
            require_decoded_watertight=False,
            require_token_watertight=False,
            max_boundary_edges=None,
            max_nonmanifold_edges=None,
            min_edge_pairing_ratio=None,
        )
    if profile == "tolerant":
        return FaceDatasetGateThresholds(
            require_decoded_watertight=False,
            require_token_watertight=False,
            max_boundary_edges=64,
            max_nonmanifold_edges=16,
            min_edge_pairing_ratio=0.95,
        )
    raise ValueError(f"unknown FACE dataset gate profile: {profile}")


def load_face_dataset_manifest(path: str | Path) -> list[dict[str, Any]]:
    manifest_path = Path(path)
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            records.append(json.loads(stripped))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{manifest_path}:{line_number} is not valid JSON") from exc
    return records


def evaluate_face_dataset_records(
    records: list[dict[str, Any]],
    thresholds: FaceDatasetGateThresholds,
    *,
    token_family: str = "coordinate",
) -> dict[str, Any]:
    results = []
    for record in records:
        normalized = _normalize_record(record, token_family)
        violations = _record_violations(normalized, thresholds)
        results.append(
            {
                "path": normalized.get("path"),
                "source_name": normalized.get("source_name"),
                "faces": normalized.get("faces"),
                "source_faces": normalized.get("source_faces"),
                "token_family": token_family,
                "decoded_watertight": normalized.get("decoded_watertight"),
                "token_watertight_edge_graph": normalized.get("token_watertight_edge_graph"),
                "token_boundary_edge_count": normalized.get("token_boundary_edge_count"),
                "token_nonmanifold_edge_count": normalized.get("token_nonmanifold_edge_count"),
                "token_edge_pairing_ratio": normalized.get("token_edge_pairing_ratio"),
                "passes": not violations,
                "violations": violations,
            }
        )
    passing = [result for result in results if result["passes"]]
    has_samples = len(results) > 0
    return {
        "thresholds": asdict(thresholds),
        "token_family": token_family,
        "sample_count": len(results),
        "passing": len(passing),
        "failing": len(results) - len(passing),
        "pass_rate": len(passing) / len(results) if has_samples else 0.0,
        "passes": has_samples and len(passing) == len(results),
        "results": results,
    }


def evaluate_face_dataset_manifest(
    path: str | Path,
    thresholds: FaceDatasetGateThresholds,
    *,
    token_family: str = "coordinate",
) -> dict[str, Any]:
    return evaluate_face_dataset_records(load_face_dataset_manifest(path), thresholds, token_family=token_family)


def _normalize_record(record: dict[str, Any], token_family: str) -> dict[str, Any]:
    token_family = token_family.strip().lower()
    if token_family in {"coordinate", "coordinates", "face", "legacy"}:
        prefix = ""
    elif token_family == "paper":
        prefix = "paper_"
    elif token_family == "indexed":
        prefix = "indexed_"
    else:
        raise ValueError(f"unknown FACE token family: {token_family}")
    if not prefix:
        return dict(record)
    normalized = dict(record)
    for source_key, target_key in [
        (f"{prefix}decoded_watertight", "decoded_watertight"),
        (f"{prefix}token_watertight_edge_graph", "token_watertight_edge_graph"),
        (f"{prefix}token_boundary_edge_count", "token_boundary_edge_count"),
        (f"{prefix}token_nonmanifold_edge_count", "token_nonmanifold_edge_count"),
        (f"{prefix}token_edge_pairing_ratio", "token_edge_pairing_ratio"),
    ]:
        if source_key in record:
            normalized[target_key] = record[source_key]
    return normalized


def _record_violations(record: dict[str, Any], thresholds: FaceDatasetGateThresholds) -> list[str]:
    violations: list[str] = []
    if thresholds.require_decoded_watertight and not bool(record.get("decoded_watertight")):
        violations.append("decoded mesh is not watertight")
    if thresholds.require_token_watertight and not bool(record.get("token_watertight_edge_graph")):
        violations.append("FACE token edge graph is not watertight")

    boundary = _optional_int(record.get("token_boundary_edge_count"))
    if thresholds.max_boundary_edges is not None:
        if boundary is None:
            violations.append("missing token boundary edge count")
        elif boundary > thresholds.max_boundary_edges:
            violations.append(f"boundary edges {boundary} > {thresholds.max_boundary_edges}")

    nonmanifold = _optional_int(record.get("token_nonmanifold_edge_count"))
    if thresholds.max_nonmanifold_edges is not None:
        if nonmanifold is None:
            violations.append("missing token nonmanifold edge count")
        elif nonmanifold > thresholds.max_nonmanifold_edges:
            violations.append(f"nonmanifold edges {nonmanifold} > {thresholds.max_nonmanifold_edges}")

    pairing = _optional_float(record.get("token_edge_pairing_ratio"))
    if thresholds.min_edge_pairing_ratio is not None:
        if pairing is None:
            violations.append("missing token edge pairing ratio")
        elif pairing < thresholds.min_edge_pairing_ratio:
            violations.append(f"edge pairing ratio {pairing:.6f} < {thresholds.min_edge_pairing_ratio:.6f}")
    return violations


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
