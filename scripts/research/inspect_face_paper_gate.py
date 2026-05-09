#!/usr/bin/env python3
"""Summarize a completed or in-progress FACE paper gate.

This is a thin decision helper for overnight monitoring. It does not replace
visual inspection, but it makes the first-pass scale decision reproducible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _find_run_dir(lab_root: Path) -> Path | None:
    runs = lab_root / "runs"
    if not runs.is_dir():
        return None
    candidates = [path for path in runs.iterdir() if path.is_dir()]
    if not candidates:
        return None
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def _train_rows(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "logs" / "train.log"
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "step" in payload:
            rows.append(payload)
    return rows


def _train_tail(rows: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    return rows[-limit:]


def _number(payload: dict[str, Any], key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _progress_summary(rows: list[dict[str, Any]], readiness: dict[str, Any] | None) -> dict[str, Any]:
    if not rows:
        return {
            "state": "no_train_log_rows",
            "latest_step": None,
            "selection_count": 0,
        }

    latest = rows[-1]
    selection_rows = [row for row in rows if row.get("selection_loss") is not None]
    best_selection = None
    if selection_rows:
        best_selection = min(selection_rows, key=lambda row: float(row["selection_loss"]))
    first_selection_loss = (
        float(selection_rows[0]["selection_loss"])
        if selection_rows and selection_rows[0].get("selection_loss") is not None
        else None
    )
    latest_selection_loss = (
        float(selection_rows[-1]["selection_loss"])
        if selection_rows and selection_rows[-1].get("selection_loss") is not None
        else None
    )
    best_selection_loss = (
        float(best_selection["selection_loss"])
        if best_selection is not None and best_selection.get("selection_loss") is not None
        else None
    )

    latest_step = int(latest["step"]) if str(latest.get("step", "")).isdigit() else latest.get("step")
    eta_sec = _number(latest, "eta_sec")
    steps_per_sec = _number(latest, "steps_per_sec")
    inferred_target_step = None
    progress_percent = None
    if isinstance(latest_step, int) and eta_sec is not None and steps_per_sec is not None and steps_per_sec > 0:
        inferred_target_step = int(round(latest_step + eta_sec * steps_per_sec))
        if inferred_target_step > 0:
            progress_percent = 100.0 * latest_step / inferred_target_step

    improvement = None
    improvement_percent = None
    if first_selection_loss is not None and best_selection_loss is not None:
        improvement = first_selection_loss - best_selection_loss
        if first_selection_loss != 0:
            improvement_percent = 100.0 * improvement / first_selection_loss

    return {
        "state": "metrics_complete" if readiness is not None else "training_or_eval_in_progress",
        "latest_step": latest_step,
        "latest_loss": _number(latest, "loss"),
        "latest_best_loss": _number(latest, "best_loss"),
        "latest_best_step": latest.get("best_step"),
        "latest_eta_sec": eta_sec,
        "latest_eta_sec_ex_selection": _number(latest, "eta_sec_ex_selection"),
        "latest_steps_per_sec": steps_per_sec,
        "inferred_target_step": inferred_target_step,
        "progress_percent": progress_percent,
        "selection_count": len(selection_rows),
        "first_selection_loss": first_selection_loss,
        "latest_selection_loss": latest_selection_loss,
        "best_selection_loss": best_selection_loss,
        "best_selection_step": best_selection.get("step") if best_selection else None,
        "selection_improvement": improvement,
        "selection_improvement_percent": improvement_percent,
        "recent_selection_losses": [
            {"step": row.get("step"), "loss": row.get("selection_loss")}
            for row in selection_rows[-8:]
        ],
    }


def _eval_summary(run_dir: Path, name: str) -> dict[str, Any] | None:
    payload = _load_json(run_dir / "eval" / f"{name}.json")
    if not payload:
        return None
    return payload.get("summary") or {}


def inspect(lab_root: Path, run_dir: Path | None = None) -> dict[str, Any]:
    run_dir = run_dir or _find_run_dir(lab_root)
    if run_dir is None:
        return {
            "lab_root": str(lab_root),
            "run_dir": None,
            "state": "missing_run",
            "next_action": "hold_missing_run_directory",
            "scale_ready": False,
        }

    readiness = _load_json(run_dir / "scale_readiness.json")
    run_summary = _load_json(run_dir / "summary.json")
    corpus_summary = (
        _load_json(lab_root / "paper_corpus_gate_summary.json")
        or _load_json(lab_root / "paper_existing_split_gate_summary.json")
    )
    train_rows = _train_rows(run_dir)
    train_tail = _train_tail(train_rows)
    galleries = {
        "train_ar": lab_root / "train_ar_contact_sheet.png",
        "test_ar": lab_root / "test_ar_contact_sheet.png",
    }
    missing_artifacts = [
        str(path)
        for path in [
            run_dir / "summary.json",
            run_dir / "scale_readiness.json",
            galleries["train_ar"],
            galleries["test_ar"],
        ]
        if not path.exists()
    ]

    if readiness is None:
        state = "running_or_incomplete"
        next_action = "wait_for_training_eval_archive"
        scale_ready = False
        recommendation = "wait"
        blockers: list[str] = []
        warnings: list[str] = []
    else:
        scale_ready = bool(readiness.get("scale_ready"))
        recommendation = str(readiness.get("recommendation") or "hold")
        blockers = list(readiness.get("blockers") or [])
        warnings = list(readiness.get("warnings") or [])
        if scale_ready and not any(path in missing_artifacts for path in map(str, galleries.values())):
            state = "ready_for_visual_review"
            next_action = "promote_next_corpus_rung_after_visual_review"
        elif scale_ready:
            state = "metrics_ready_missing_visuals"
            next_action = "hold_until_contact_sheets_exist"
        else:
            state = "not_ready"
            next_action = recommendation

    return {
        "lab_root": str(lab_root),
        "run_dir": str(run_dir),
        "state": state,
        "scale_ready": scale_ready,
        "recommendation": recommendation,
        "next_action": next_action,
        "blockers": blockers[:12],
        "warnings": warnings[:12],
        "missing_artifacts": missing_artifacts,
        "progress": _progress_summary(train_rows, readiness),
        "latest_train_rows": train_tail,
        "eval": {
            "train_teacher_forced": _eval_summary(run_dir, "train_teacher_forced"),
            "train_autoregressive": _eval_summary(run_dir, "train_autoregressive"),
            "test_teacher_forced": _eval_summary(run_dir, "test_teacher_forced"),
            "test_autoregressive": _eval_summary(run_dir, "test_autoregressive"),
        },
        "dataset": (corpus_summary or {}).get("dataset"),
        "settings": ((run_summary or {}).get("settings") or ((run_summary or {}).get("run_summary") or {}).get("settings")),
        "galleries": {key: str(path) for key, path in galleries.items() if path.exists()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lab_root", type=Path)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--fail-on-not-ready", action="store_true")
    args = parser.parse_args()

    payload = inspect(args.lab_root, args.run_dir)
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    if args.fail_on_not_ready and not payload.get("scale_ready"):
        return 30
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
