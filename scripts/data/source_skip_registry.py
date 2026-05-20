"""Helpers for skipping source assets that are already represented elsewhere."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable


_DIRECT_ID_KEYS = (
    "uid",
    "UID",
    "objaverse_uid",
    "Objaverse UID",
    "object_uid",
    "sha256",
    "sha",
    "file_identifier",
    "fileIdentifier",
    "hf_path",
    "path",
    "source_name",
    "source_uid",
    "source_id",
)


def _add_key(out: set[str], value: Any, *, prefix: str | None = None) -> None:
    if value is None:
        return
    text = str(value).strip()
    if not text:
        return
    variants = {text, text.lower()}
    for item in variants:
        out.add(item)
        if prefix:
            out.add(f"{prefix}:{item}")


def _looks_like_local_path(value: Any) -> bool:
    text = str(value or "").strip()
    return text.startswith(("/", "./", "../")) or "/tmp/" in text or "/ephemeral/" in text


def source_keys_from_row(row: dict[str, Any]) -> set[str]:
    """Return stable source IDs for Objaverse++, ObjaverseXL, and TexVerse rows."""

    keys: set[str] = set()
    for key in _DIRECT_ID_KEYS:
        if key in row:
            value = row.get(key)
            prefix = None
            if key in {"uid", "UID", "objaverse_uid", "Objaverse UID", "object_uid"}:
                prefix = "uid"
            elif key in {"sha256", "sha"}:
                prefix = "sha256"
            elif key in {"file_identifier", "fileIdentifier"}:
                prefix = "file"
            elif key in {"hf_path", "path"}:
                if key == "path" and _looks_like_local_path(value):
                    continue
                prefix = "path"
            elif key in {"source_name", "source_uid", "source_id"}:
                prefix = "source"
            _add_key(keys, value, prefix=prefix)
            if key == "source_name":
                for part in re.split(r"[^A-Za-z0-9]+", str(value)):
                    if len(part) >= 12:
                        _add_key(keys, part, prefix="uid")
                        if re.fullmatch(r"[0-9a-fA-F]{32,64}", part):
                            _add_key(keys, part, prefix="sha256")

    repo_id = row.get("repo_id") or row.get("hf_repo")
    hf_path = row.get("hf_path") or row.get("path")
    if repo_id and hf_path:
        _add_key(keys, f"{repo_id}/{hf_path}", prefix="hf")

    annotation = row.get("annotation")
    if isinstance(annotation, dict):
        keys.update(source_keys_from_row(annotation))
        if repo_id and (annotation.get("hf_path") or annotation.get("path")):
            _add_key(keys, f"{repo_id}/{annotation.get('hf_path') or annotation.get('path')}", prefix="hf")

    return keys


def row_is_excluded(row: dict[str, Any], skip_ids: set[str]) -> bool:
    if not skip_ids:
        return False
    return bool(source_keys_from_row(row) & skip_ids)


def filter_excluded_rows(rows: list[dict[str, Any]], skip_ids: set[str]) -> tuple[list[dict[str, Any]], int]:
    if not skip_ids:
        return rows, 0
    kept: list[dict[str, Any]] = []
    excluded = 0
    for row in rows:
        if row_is_excluded(row, skip_ids):
            excluded += 1
        else:
            kept.append(row)
    return kept, excluded


def _iter_jsonl(path: Path) -> Iterable[Any]:
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        yield json.loads(stripped)


def _keys_from_payload(payload: Any) -> set[str]:
    out: set[str] = set()
    if isinstance(payload, dict):
        if any(key in payload for key in _DIRECT_ID_KEYS) or isinstance(payload.get("annotation"), dict):
            out.update(source_keys_from_row(payload))
        else:
            for key, value in payload.items():
                _add_key(out, key, prefix="uid")
                if isinstance(value, dict):
                    row = dict(value)
                    row.setdefault("uid", key)
                    out.update(source_keys_from_row(row))
                elif isinstance(value, str):
                    if not _looks_like_local_path(value):
                        _add_key(out, value)
    elif isinstance(payload, list):
        for item in payload:
            out.update(_keys_from_payload(item))
    elif isinstance(payload, str):
        _add_key(out, payload)
    return out


def _iter_input_files(path: Path) -> Iterable[Path]:
    if path.is_dir():
        for suffix in ("*.txt", "*.jsonl", "*.json"):
            yield from sorted(path.rglob(suffix))
    else:
        yield path


def load_source_skip_ids(paths: Iterable[Path]) -> set[str]:
    skip_ids: set[str] = set()
    for input_path in paths:
        if not input_path:
            continue
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(path)
        for file_path in _iter_input_files(path):
            suffix = file_path.suffix.lower()
            if suffix == ".jsonl":
                for row in _iter_jsonl(file_path):
                    skip_ids.update(_keys_from_payload(row))
            elif suffix == ".json":
                skip_ids.update(_keys_from_payload(json.loads(file_path.read_text(encoding="utf-8"))))
            else:
                for line in file_path.read_text(encoding="utf-8").splitlines():
                    _add_key(skip_ids, line.strip())
    return skip_ids
