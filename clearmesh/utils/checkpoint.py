"""Checkpoint serialization helpers."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def json_safe(value: Any) -> Any:
    """Return a checkpoint metadata value that is safe for weights-only loads."""

    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def args_to_json_safe(args: Any) -> dict[str, Any]:
    """Convert an argparse namespace to simple metadata for torch checkpoints."""

    return {str(key): json_safe(value) for key, value in vars(args).items()}
