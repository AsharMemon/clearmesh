"""Credit-based billing/entitlement scaffold.

This local ledger is intentionally conservative: usage is reserved before a job
runs, consumed on success, and released on infrastructure failure. Stripe should
own payment state in production; this module owns product entitlements.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import Lock


@dataclass
class CreditAccount:
    team_id: str
    balance: int = 0
    reserved: int = 0


@dataclass
class UsageEvent:
    team_id: str
    job_id: str
    credits: int
    kind: str
    metadata: dict = field(default_factory=dict)


class CreditLedger:
    def __init__(self, path: str | Path = ".clearmesh_state/credits.json"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        if not self.path.exists():
            self.path.write_text(json.dumps({"accounts": {}, "events": []}, indent=2), encoding="utf-8")

    def estimate_job_credits(self, quality_tier: str, enable_rigging: bool, mode: str) -> int:
        base = {"draft": 1, "standard": 3, "high": 8}.get(quality_tier, 3)
        if enable_rigging:
            base += 2
        if mode.startswith("edit"):
            base += 1
        return base

    def grant(self, team_id: str, credits: int, reason: str = "manual_grant") -> CreditAccount:
        with self._lock:
            data = self._read()
            account = self._account(data, team_id)
            account.balance += credits
            data["accounts"][team_id] = asdict(account)
            data["events"].append(asdict(UsageEvent(team_id=team_id, job_id="", credits=credits, kind=reason)))
            self._write(data)
            return account

    def grant_once(self, team_id: str, credits: int, reason: str, idempotency_key: str) -> CreditAccount:
        """Grant credits once for an external event such as a Stripe webhook."""

        with self._lock:
            data = self._read()
            if idempotency_key and any(
                event.get("team_id") == team_id
                and event.get("kind") == reason
                and event.get("metadata", {}).get("idempotency_key") == idempotency_key
                for event in data.get("events", [])
            ):
                return self._account(data, team_id)
            account = self._account(data, team_id)
            account.balance += credits
            data["accounts"][team_id] = asdict(account)
            data["events"].append(
                asdict(
                    UsageEvent(
                        team_id=team_id,
                        job_id="",
                        credits=credits,
                        kind=reason,
                        metadata={"idempotency_key": idempotency_key},
                    )
                )
            )
            self._write(data)
            return account

    def reserve(self, team_id: str, job_id: str, credits: int) -> CreditAccount:
        with self._lock:
            data = self._read()
            account = self._account(data, team_id)
            if account.balance - account.reserved < credits:
                raise ValueError("insufficient credits")
            account.reserved += credits
            data["accounts"][team_id] = asdict(account)
            data["events"].append(asdict(UsageEvent(team_id=team_id, job_id=job_id, credits=credits, kind="reserve")))
            self._write(data)
            return account

    def consume(self, team_id: str, job_id: str, credits: int) -> CreditAccount:
        with self._lock:
            data = self._read()
            account = self._account(data, team_id)
            account.reserved = max(0, account.reserved - credits)
            account.balance -= credits
            data["accounts"][team_id] = asdict(account)
            data["events"].append(asdict(UsageEvent(team_id=team_id, job_id=job_id, credits=credits, kind="consume")))
            self._write(data)
            return account

    def release(self, team_id: str, job_id: str, credits: int, reason: str = "release") -> CreditAccount:
        with self._lock:
            data = self._read()
            account = self._account(data, team_id)
            account.reserved = max(0, account.reserved - credits)
            data["accounts"][team_id] = asdict(account)
            data["events"].append(asdict(UsageEvent(team_id=team_id, job_id=job_id, credits=credits, kind=reason)))
            self._write(data)
            return account

    def get_account(self, team_id: str) -> CreditAccount:
        return self._account(self._read(), team_id)

    def list_events(self, team_id: str, limit: int = 100) -> list[UsageEvent]:
        data = self._read()
        events = [UsageEvent(**event) for event in data.get("events", []) if event.get("team_id") == team_id]
        return events[-limit:]

    def _account(self, data: dict, team_id: str) -> CreditAccount:
        raw = data["accounts"].get(team_id)
        return CreditAccount(**raw) if raw else CreditAccount(team_id=team_id, balance=0, reserved=0)

    def _read(self) -> dict:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _write(self, data: dict) -> None:
        self.path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
