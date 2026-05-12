"""API-key helpers for local production scaffolding."""

from __future__ import annotations

import hashlib
import hmac
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Principal:
    user_id: str
    team_id: str
    api_key_id: str


def hash_api_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def load_api_keys(env_value: str | None = None) -> dict[str, Principal]:
    """Load API keys from CLEAR_MESH_API_KEYS.

    Format:
      key_id:raw_key:user_id:team_id,key_id2:raw_key2:user2:team2
    """
    env_value = env_value if env_value is not None else os.getenv("CLEAR_MESH_API_KEYS", "")
    keys: dict[str, Principal] = {}
    for item in [part.strip() for part in env_value.split(",") if part.strip()]:
        key_id, raw_key, user_id, team_id = item.split(":", 3)
        keys[hash_api_key(raw_key)] = Principal(user_id=user_id, team_id=team_id, api_key_id=key_id)
    return keys


def authenticate_api_key(raw_key: str, key_map: dict[str, Principal] | None = None) -> Principal | None:
    key_map = key_map if key_map is not None else load_api_keys()
    candidate = hash_api_key(raw_key)
    for stored_hash, principal in key_map.items():
        if hmac.compare_digest(candidate, stored_hash):
            return principal
    return None
