"""User, team, session, and API-key store for the product web app.

The store is JSON-backed so local development stays dependency-light. The data
model mirrors the production shape closely enough to migrate to Postgres later:
users own identities, teams own billing/settings, and sessions/API keys resolve
to the same `Principal` used by the generation API.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from .auth import Principal, hash_api_key

PBKDF2_ITERATIONS = 210_000
DEFAULT_USER_SETTINGS = {
    "default_mode": "image_to_3d",
    "default_quality_tier": "standard",
    "auto_parts": True,
    "auto_rigging": False,
    "email_updates": True,
}
DEFAULT_TEAM_SETTINGS = {
    "workspace_name": "ClearMesh Studio",
    "model_endpoint": "",
    "webhook_url": "",
    "retention_days": 30,
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().isoformat()


def normalize_email(email: str) -> str:
    return email.strip().lower()


def public_user(user: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": user["id"],
        "email": user["email"],
        "name": user.get("name") or user["email"].split("@", 1)[0],
        "avatar_url": user.get("avatar_url"),
        "settings": {**DEFAULT_USER_SETTINGS, **user.get("settings", {})},
        "identities": sorted(user.get("identities", {}).keys()),
        "created_at": user.get("created_at"),
    }


def public_team(team: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": team["id"],
        "name": team["name"],
        "plan": team.get("plan", "free"),
        "settings": {**DEFAULT_TEAM_SETTINGS, **team.get("settings", {})},
        "subscription": team.get("subscription", {}),
        "created_at": team.get("created_at"),
    }


@dataclass(frozen=True)
class AccountContext:
    user: dict[str, Any]
    team: dict[str, Any]
    session_id: str | None = None

    @property
    def principal(self) -> Principal:
        return Principal(user_id=self.user["id"], team_id=self.team["id"], api_key_id=self.session_id or "session")

    def to_public_dict(self) -> dict[str, Any]:
        return {"user": public_user(self.user), "team": public_team(self.team)}


class ProductAccountStore:
    def __init__(self, path: str | Path = ".clearmesh_state/accounts.json"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        if not self.path.exists():
            self.path.write_text(
                json.dumps(
                    {
                        "users": {},
                        "teams": {},
                        "email_index": {},
                        "identity_index": {},
                        "sessions": {},
                        "oauth_states": {},
                        "api_keys": {},
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

    def create_email_user(
        self,
        *,
        email: str,
        password: str,
        name: str | None = None,
        team_name: str | None = None,
    ) -> tuple[AccountContext, bool]:
        email = normalize_email(email)
        self._validate_email(email)
        self._validate_password(password)
        with self._lock:
            data = self._read()
            if email in data["email_index"]:
                raise ValueError("email already registered")
            user, team = self._create_user_and_team(
                data,
                email=email,
                name=name or email.split("@", 1)[0],
                team_name=team_name or f"{(name or email.split('@', 1)[0]).strip()}'s workspace",
                avatar_url=None,
            )
            user["password_hash"] = self._password_hash(password)
            self._write(data)
            return AccountContext(user=user, team=team), True

    def authenticate_password(self, email: str, password: str) -> AccountContext | None:
        email = normalize_email(email)
        with self._lock:
            data = self._read()
            user_id = data["email_index"].get(email)
            if not user_id:
                return None
            user = data["users"][user_id]
            encoded = user.get("password_hash")
            if not encoded or not self._verify_password(password, encoded):
                return None
            team = data["teams"][user["default_team_id"]]
            return AccountContext(user=user, team=team)

    def upsert_oauth_user(
        self,
        *,
        provider: str,
        provider_user_id: str,
        email: str,
        name: str | None,
        avatar_url: str | None,
    ) -> tuple[AccountContext, bool]:
        email = normalize_email(email)
        self._validate_email(email)
        identity_key = f"{provider}:{provider_user_id}"
        with self._lock:
            data = self._read()
            created = False
            user_id = data["identity_index"].get(identity_key) or data["email_index"].get(email)
            if user_id:
                user = data["users"][user_id]
                user.setdefault("identities", {})[provider] = provider_user_id
                user["name"] = user.get("name") or name or email.split("@", 1)[0]
                if avatar_url:
                    user["avatar_url"] = avatar_url
                data["identity_index"][identity_key] = user_id
                team = data["teams"][user["default_team_id"]]
            else:
                user, team = self._create_user_and_team(
                    data,
                    email=email,
                    name=name or email.split("@", 1)[0],
                    team_name=f"{(name or email.split('@', 1)[0]).strip()}'s workspace",
                    avatar_url=avatar_url,
                )
                user["identities"] = {provider: provider_user_id}
                data["identity_index"][identity_key] = user["id"]
                created = True
            self._write(data)
            return AccountContext(user=user, team=team), created

    def create_session(self, user_id: str, team_id: str, ttl_days: int | None = None) -> tuple[str, str]:
        ttl_days = ttl_days or int(os.getenv("CLEARMESH_SESSION_DAYS", "30"))
        raw = "cms_" + secrets.token_urlsafe(32)
        session_id = "sess_" + uuid4().hex
        with self._lock:
            data = self._read()
            if user_id not in data["users"] or team_id not in data["teams"]:
                raise KeyError("unknown user or team")
            data["sessions"][self._hash_secret(raw)] = {
                "id": session_id,
                "user_id": user_id,
                "team_id": team_id,
                "created_at": utc_now_iso(),
                "expires_at": (utc_now() + timedelta(days=ttl_days)).isoformat(),
            }
            self._write(data)
        return raw, session_id

    def authenticate_session(self, raw_session: str | None) -> AccountContext | None:
        if not raw_session:
            return None
        with self._lock:
            data = self._read()
            session = data["sessions"].get(self._hash_secret(raw_session))
            if not session:
                return None
            if self._is_expired(session.get("expires_at")):
                data["sessions"].pop(self._hash_secret(raw_session), None)
                self._write(data)
                return None
            user = data["users"].get(session["user_id"])
            team = data["teams"].get(session["team_id"])
            if not user or not team:
                return None
            return AccountContext(user=user, team=team, session_id=session["id"])

    def delete_session(self, raw_session: str | None) -> None:
        if not raw_session:
            return
        with self._lock:
            data = self._read()
            data["sessions"].pop(self._hash_secret(raw_session), None)
            self._write(data)

    def create_oauth_state(self, provider: str, next_path: str = "/") -> str:
        state = "state_" + secrets.token_urlsafe(24)
        with self._lock:
            data = self._read()
            data["oauth_states"][self._hash_secret(state)] = {
                "provider": provider,
                "next_path": next_path if next_path.startswith("/") else "/",
                "created_at": utc_now_iso(),
                "expires_at": (utc_now() + timedelta(minutes=10)).isoformat(),
            }
            self._write(data)
        return state

    def consume_oauth_state(self, state: str, provider: str) -> str:
        with self._lock:
            data = self._read()
            key = self._hash_secret(state)
            record = data["oauth_states"].pop(key, None)
            self._write(data)
        if not record or record.get("provider") != provider or self._is_expired(record.get("expires_at")):
            raise ValueError("invalid oauth state")
        return record.get("next_path") or "/"

    def create_api_key(self, user_id: str, team_id: str, label: str) -> dict[str, Any]:
        raw = "cmk_" + secrets.token_urlsafe(32)
        key_id = "key_" + uuid4().hex
        with self._lock:
            data = self._read()
            if user_id not in data["users"] or team_id not in data["teams"]:
                raise KeyError("unknown user or team")
            data["api_keys"][hash_api_key(raw)] = {
                "id": key_id,
                "team_id": team_id,
                "user_id": user_id,
                "label": label.strip()[:80] or "API key",
                "created_at": utc_now_iso(),
                "last_used_at": None,
            }
            self._write(data)
        return {"id": key_id, "token": raw, "label": label.strip()[:80] or "API key"}

    def authenticate_api_key(self, raw_key: str) -> Principal | None:
        candidate = hash_api_key(raw_key)
        with self._lock:
            data = self._read()
            for stored_hash, record in data["api_keys"].items():
                if hmac.compare_digest(candidate, stored_hash):
                    record["last_used_at"] = utc_now_iso()
                    self._write(data)
                    return Principal(
                        user_id=record["user_id"],
                        team_id=record["team_id"],
                        api_key_id=record["id"],
                    )
        return None

    def list_api_keys(self, team_id: str) -> list[dict[str, Any]]:
        with self._lock:
            data = self._read()
            keys = [
                {
                    "id": record["id"],
                    "label": record["label"],
                    "created_at": record["created_at"],
                    "last_used_at": record.get("last_used_at"),
                }
                for record in data["api_keys"].values()
                if record["team_id"] == team_id
            ]
        return sorted(keys, key=lambda item: item["created_at"], reverse=True)

    def delete_api_key(self, team_id: str, key_id: str) -> bool:
        with self._lock:
            data = self._read()
            for stored_hash, record in list(data["api_keys"].items()):
                if record["team_id"] == team_id and record["id"] == key_id:
                    del data["api_keys"][stored_hash]
                    self._write(data)
                    return True
        return False

    def get_context(self, user_id: str, team_id: str) -> AccountContext:
        with self._lock:
            data = self._read()
            return AccountContext(user=data["users"][user_id], team=data["teams"][team_id])

    def update_settings(
        self,
        *,
        user_id: str,
        team_id: str,
        profile: dict[str, Any] | None = None,
        user_settings: dict[str, Any] | None = None,
        team_settings: dict[str, Any] | None = None,
    ) -> AccountContext:
        with self._lock:
            data = self._read()
            user = data["users"][user_id]
            team = data["teams"][team_id]
            if profile:
                name = str(profile.get("name", "")).strip()
                if name:
                    user["name"] = name[:120]
            if user_settings:
                user["settings"] = self._merge_allowed(user.get("settings", {}), user_settings, DEFAULT_USER_SETTINGS)
            if team_settings:
                team["settings"] = self._merge_allowed(team.get("settings", {}), team_settings, DEFAULT_TEAM_SETTINGS)
                workspace_name = str(team["settings"].get("workspace_name") or "").strip()
                if workspace_name:
                    team["name"] = workspace_name[:120]
            self._write(data)
            return AccountContext(user=user, team=team)

    def set_team_subscription(
        self,
        *,
        team_id: str,
        tier: str,
        status: str,
        stripe_customer_id: str | None = None,
        stripe_subscription_id: str | None = None,
        current_period_end: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            data = self._read()
            team = data["teams"][team_id]
            subscription = {
                **team.get("subscription", {}),
                "tier": tier,
                "status": status,
                "stripe_customer_id": stripe_customer_id or team.get("subscription", {}).get("stripe_customer_id"),
                "stripe_subscription_id": stripe_subscription_id
                or team.get("subscription", {}).get("stripe_subscription_id"),
                "current_period_end": current_period_end
                or team.get("subscription", {}).get("current_period_end"),
                "updated_at": utc_now_iso(),
            }
            team["plan"] = tier
            team["subscription"] = subscription
            self._write(data)
            return public_team(team)

    def _create_user_and_team(
        self,
        data: dict[str, Any],
        *,
        email: str,
        name: str,
        team_name: str,
        avatar_url: str | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        user_id = "user_" + uuid4().hex
        team_id = "team_" + uuid4().hex
        team = {
            "id": team_id,
            "name": team_name.strip()[:120] or "ClearMesh Studio",
            "owner_user_id": user_id,
            "plan": "free",
            "settings": {**DEFAULT_TEAM_SETTINGS, "workspace_name": team_name.strip()[:120] or "ClearMesh Studio"},
            "subscription": {"tier": "free", "status": "trialing", "updated_at": utc_now_iso()},
            "created_at": utc_now_iso(),
        }
        user = {
            "id": user_id,
            "email": email,
            "name": name.strip()[:120] or email.split("@", 1)[0],
            "avatar_url": avatar_url,
            "default_team_id": team_id,
            "settings": dict(DEFAULT_USER_SETTINGS),
            "identities": {},
            "created_at": utc_now_iso(),
        }
        data["users"][user_id] = user
        data["teams"][team_id] = team
        data["email_index"][email] = user_id
        return user, team

    def _read(self) -> dict[str, Any]:
        data = json.loads(self.path.read_text(encoding="utf-8"))
        data.setdefault("users", {})
        data.setdefault("teams", {})
        data.setdefault("email_index", {})
        data.setdefault("identity_index", {})
        data.setdefault("sessions", {})
        data.setdefault("oauth_states", {})
        data.setdefault("api_keys", {})
        return data

    def _write(self, data: dict[str, Any]) -> None:
        self.path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")

    def _password_hash(self, password: str) -> str:
        salt = secrets.token_bytes(16)
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
        return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt.hex()}${digest.hex()}"

    def _verify_password(self, password: str, encoded: str) -> bool:
        try:
            algorithm, iterations, salt_hex, digest_hex = encoded.split("$", 3)
            if algorithm != "pbkdf2_sha256":
                return False
            digest = hashlib.pbkdf2_hmac(
                "sha256",
                password.encode("utf-8"),
                bytes.fromhex(salt_hex),
                int(iterations),
            )
        except (ValueError, TypeError):
            return False
        return hmac.compare_digest(digest.hex(), digest_hex)

    def _hash_secret(self, value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def _is_expired(self, value: str | None) -> bool:
        if not value:
            return True
        try:
            expires_at = datetime.fromisoformat(value)
        except ValueError:
            return True
        return expires_at < utc_now()

    def _validate_email(self, email: str) -> None:
        if "@" not in email or "." not in email.rsplit("@", 1)[-1] or len(email) > 254:
            raise ValueError("valid email required")

    def _validate_password(self, password: str) -> None:
        if len(password) < 8:
            raise ValueError("password must be at least 8 characters")

    def _merge_allowed(self, current: dict[str, Any], updates: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
        merged = {**defaults, **current}
        for key, value in updates.items():
            if key not in defaults:
                continue
            if isinstance(defaults[key], bool):
                merged[key] = bool(value)
            elif isinstance(defaults[key], int):
                try:
                    merged[key] = max(1, int(value))
                except (TypeError, ValueError):
                    merged[key] = defaults[key]
            else:
                merged[key] = str(value).strip()[:500]
        return merged
