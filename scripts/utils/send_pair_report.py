#!/usr/bin/env python3
"""Send an hourly pair-generation report via SMTP.

The script:
  1. Reads current shard counters from the shared /workspace volume via SSH.
  2. Compares them against the previous saved snapshot.
  3. Emails deltas, success rate, and average seconds per processed model.
  4. Persists the new snapshot for the next hourly run.
"""

from __future__ import annotations

import argparse
import json
import smtplib
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / ".codex" / "pair_report_config.json"
DEFAULT_STATE = ROOT / ".codex" / "pair_report_state.json"


@dataclass
class PodConfig:
    name: str
    host: str
    port: int
    shard_id: int


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def run_ssh(host: str, port: int, remote_cmd: str) -> str:
    cmd = [
        "ssh",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "StrictHostKeyChecking=no",
        "-i",
        str(Path.home() / ".ssh" / "id_ed25519"),
        "-p",
        str(port),
        f"root@{host}",
        remote_cmd,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or f"ssh failed ({proc.returncode})")
    return proc.stdout


def load_counts_from_shared_volume(pods: list[PodConfig]) -> dict:
    remote = """python3 - <<'PY'
import json, os
base='/workspace/data/training_pairs'
out={}
for shard in ['shard_0','shard_1','shard_2']:
    p=os.path.join(base, shard, 'progress.json')
    f=os.path.join(base, shard, 'failed.json')
    progress=json.load(open(p)) if os.path.exists(p) else []
    failed=json.load(open(f)) if os.path.exists(f) else []
    out[shard]={'completed': len(progress), 'failed': len(failed)}
print(json.dumps(out))
PY"""
    errors = []
    for pod in pods:
        try:
            raw = run_ssh(pod.host, pod.port, remote)
            return json.loads(raw.strip().splitlines()[-1])
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{pod.name}: {exc}")
    raise RuntimeError("unable to read shared counters from any pod: " + "; ".join(errors))


def load_pod_status(pod: PodConfig) -> dict:
    remote = f"""python3 - <<'PY'
import json, os, subprocess
shard='shard_{pod.shard_id}'
log_path=f'/workspace/logs/shard{pod.shard_id}.log'
proc = subprocess.run(
    "ps -eo pid,etimes,cmd | grep -E 'run_pairs_watchdog|generate_pairs.py' | grep -v grep || true",
    shell=True, capture_output=True, text=True
)
gpu = subprocess.run(
    "nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits",
    shell=True, capture_output=True, text=True
)
tail = subprocess.run(
    f"tail -n 5 {{log_path}} 2>/dev/null || true",
    shell=True, capture_output=True, text=True
)
print(json.dumps({{
    'processes': [line for line in proc.stdout.splitlines() if line.strip()],
    'gpu': gpu.stdout.strip(),
    'log_tail': [line for line in tail.stdout.splitlines() if line.strip()],
}}))
PY"""
    try:
        raw = run_ssh(pod.host, pod.port, remote)
        data = json.loads(raw.strip().splitlines()[-1])
        data["reachable"] = True
        return data
    except Exception as exc:  # noqa: BLE001
        return {"reachable": False, "error": str(exc)}


def build_report(now: datetime, counts: dict, pod_status: dict, prior: dict | None) -> tuple[str, str, dict]:
    now_iso = now.isoformat()
    current = {
        "timestamp": now_iso,
        "counts": counts,
    }

    lines = [f"Clearmesh pair-generation report", f"Generated: {now_iso}", ""]

    total_completed = sum(v["completed"] for v in counts.values())
    total_failed = sum(v["failed"] for v in counts.values())
    lines.append(f"Current totals: completed={total_completed}, failed={total_failed}")
    lines.append("")

    if prior is None:
        subject = "Clearmesh hourly report: baseline established"
        lines.append("Baseline established. The next report will include hourly deltas.")
        lines.append("")
        lines.append("Per-shard totals:")
        for shard in sorted(counts):
            lines.append(
                f"  {shard}: completed={counts[shard]['completed']}, failed={counts[shard]['failed']}"
            )
        delta_payload = current
    else:
        prev_counts = prior["counts"]
        prev_time = datetime.fromisoformat(prior["timestamp"])
        elapsed_sec = max((now - prev_time).total_seconds(), 1.0)
        total_delta_completed = 0
        total_delta_failed = 0
        lines.append(f"Window: {prior['timestamp']} -> {now_iso} ({elapsed_sec/3600:.2f}h)")
        lines.append("")
        lines.append("Per-shard deltas:")
        for shard in sorted(counts):
            curr = counts[shard]
            prev = prev_counts.get(shard, {"completed": 0, "failed": 0})
            d_completed = curr["completed"] - prev.get("completed", 0)
            d_failed = curr["failed"] - prev.get("failed", 0)
            d_processed = d_completed + d_failed
            success_rate = (d_completed / d_processed * 100.0) if d_processed else 0.0
            avg_sec = (elapsed_sec / d_processed) if d_processed else None
            total_delta_completed += d_completed
            total_delta_failed += d_failed
            avg_text = f"{avg_sec:.1f}s/model" if avg_sec is not None else "n/a"
            lines.append(
                f"  {shard}: +{d_completed} completed, +{d_failed} failed, "
                f"success={success_rate:.1f}%, avg={avg_text}"
            )

        total_processed = total_delta_completed + total_delta_failed
        total_success = (total_delta_completed / total_processed * 100.0) if total_processed else 0.0
        total_avg_sec = (elapsed_sec / total_processed) if total_processed else None
        total_avg_text = f"{total_avg_sec:.1f}s/model" if total_avg_sec is not None else "n/a"
        lines.append("")
        lines.append(
            f"Total delta: +{total_delta_completed} completed, +{total_delta_failed} failed, "
            f"success={total_success:.1f}%, avg={total_avg_text}"
        )
        subject = (
            f"Clearmesh hourly report: +{total_delta_completed} completed / "
            f"+{total_delta_failed} failed ({total_success:.1f}% success)"
        )
        delta_payload = current

    lines.append("")
    lines.append("Pod status:")
    for pod_name in sorted(pod_status):
        status = pod_status[pod_name]
        if not status.get("reachable"):
            lines.append(f"  {pod_name}: UNREACHABLE ({status.get('error', 'unknown error')})")
            continue
        gpu = status.get("gpu") or "unknown"
        proc_count = len(status.get("processes", []))
        lines.append(f"  {pod_name}: gpu={gpu}, processes={proc_count}")
        for tail_line in status.get("log_tail", [])[-2:]:
            lines.append(f"    {tail_line}")

    return subject, "\n".join(lines) + "\n", delta_payload


def send_email(config: dict, subject: str, body: str) -> None:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = config["from_email"]
    msg["To"] = ", ".join(config["to_emails"])
    msg["X-PM-Message-Stream"] = config["message_stream"]
    msg.set_content(body)

    with smtplib.SMTP(config["smtp_host"], config["smtp_port"], timeout=30) as server:
        server.starttls()
        server.login(config["smtp_username"], config["smtp_password"])
        server.send_message(msg)


def main() -> int:
    parser = argparse.ArgumentParser(description="Send a shard delta report via SMTP")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--dry-run", action="store_true", help="Print the report instead of sending")
    parser.add_argument("--reset-state", action="store_true", help="Ignore the saved prior snapshot")
    args = parser.parse_args()

    config = load_json(args.config)
    pods = [PodConfig(**pod) for pod in config["pods"]]

    prior = None
    if not args.reset_state and args.state.exists():
        prior = load_json(args.state)

    now = datetime.now(timezone.utc)
    counts = load_counts_from_shared_volume(pods)
    pod_status = {pod.name: load_pod_status(pod) for pod in pods}
    subject, body, snapshot = build_report(now, counts, pod_status, prior)

    if args.dry_run:
        print(f"Subject: {subject}")
        print()
        print(body, end="")
        return 0

    send_email(config, subject, body)
    save_json(args.state, snapshot)
    print(f"sent: {subject}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
