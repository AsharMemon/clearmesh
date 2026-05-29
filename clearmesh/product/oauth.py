"""OAuth provider helpers for browser sign-in.

The implementation is intentionally small and provider-specific. It supports
GitHub and Google without adding another runtime dependency; production deploys
only need to provide the usual client id/secret environment variables.
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OAuthProfile:
    provider: str
    provider_user_id: str
    email: str
    name: str | None = None
    avatar_url: str | None = None


PROVIDERS = {
    "github": {
        "label": "GitHub",
        "client_id_env": "CLEARMESH_GITHUB_CLIENT_ID",
        "client_secret_env": "CLEARMESH_GITHUB_CLIENT_SECRET",
        "authorize_url": "https://github.com/login/oauth/authorize",
        "token_url": "https://github.com/login/oauth/access_token",
        "scope": "read:user user:email",
    },
    "google": {
        "label": "Google",
        "client_id_env": "CLEARMESH_GOOGLE_CLIENT_ID",
        "client_secret_env": "CLEARMESH_GOOGLE_CLIENT_SECRET",
        "authorize_url": "https://accounts.google.com/o/oauth2/v2/auth",
        "token_url": "https://oauth2.googleapis.com/token",
        "scope": "openid email profile",
    },
}


def provider_status() -> list[dict[str, Any]]:
    return [
        {
            "id": provider,
            "label": config["label"],
            "configured": bool(os.getenv(config["client_id_env"]) and os.getenv(config["client_secret_env"])),
        }
        for provider, config in PROVIDERS.items()
    ]


def build_authorization_url(provider: str, *, redirect_uri: str, state: str) -> str:
    config = _provider_config(provider)
    client_id = os.getenv(config["client_id_env"])
    client_secret = os.getenv(config["client_secret_env"])
    if not client_id or not client_secret:
        raise RuntimeError(f"{config['label']} OAuth is not configured")
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": config["scope"],
        "state": state,
        "response_type": "code",
    }
    if provider == "google":
        params["access_type"] = "offline"
        params["prompt"] = "select_account"
    return f"{config['authorize_url']}?{urllib.parse.urlencode(params)}"


def exchange_code_for_profile(provider: str, *, code: str, redirect_uri: str) -> OAuthProfile:
    if provider == "github":
        return _exchange_github(code=code, redirect_uri=redirect_uri)
    if provider == "google":
        return _exchange_google(code=code, redirect_uri=redirect_uri)
    raise ValueError("unsupported oauth provider")


def _exchange_github(*, code: str, redirect_uri: str) -> OAuthProfile:
    config = _provider_config("github")
    token = _token_request(
        config["token_url"],
        {
            "client_id": os.environ[config["client_id_env"]],
            "client_secret": os.environ[config["client_secret_env"]],
            "code": code,
            "redirect_uri": redirect_uri,
        },
    )["access_token"]
    user = _json_get("https://api.github.com/user", token)
    emails = _json_get("https://api.github.com/user/emails", token)
    primary_email = next((item["email"] for item in emails if item.get("primary") and item.get("verified")), None)
    fallback_email = user.get("email") or f"{user['id']}+{user.get('login', 'github')}@users.noreply.github.com"
    return OAuthProfile(
        provider="github",
        provider_user_id=str(user["id"]),
        email=primary_email or fallback_email,
        name=user.get("name") or user.get("login"),
        avatar_url=user.get("avatar_url"),
    )


def _exchange_google(*, code: str, redirect_uri: str) -> OAuthProfile:
    config = _provider_config("google")
    token = _token_request(
        config["token_url"],
        {
            "client_id": os.environ[config["client_id_env"]],
            "client_secret": os.environ[config["client_secret_env"]],
            "code": code,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        },
    )["access_token"]
    user = _json_get("https://openidconnect.googleapis.com/v1/userinfo", token)
    if not user.get("email"):
        raise RuntimeError("Google profile did not include an email address")
    return OAuthProfile(
        provider="google",
        provider_user_id=str(user["sub"]),
        email=user["email"],
        name=user.get("name"),
        avatar_url=user.get("picture"),
    )


def _token_request(url: str, payload: dict[str, str]) -> dict[str, Any]:
    body = urllib.parse.urlencode(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Accept": "application/json", "Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=15) as response:  # noqa: S310 - provider URL is fixed.
        data = json.loads(response.read().decode("utf-8"))
    if "error" in data:
        raise RuntimeError(str(data.get("error_description") or data["error"]))
    return data


def _json_get(url: str, token: str) -> Any:
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}", "Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=15) as response:  # noqa: S310 - provider URL is fixed.
        return json.loads(response.read().decode("utf-8"))


def _provider_config(provider: str) -> dict[str, str]:
    try:
        return PROVIDERS[provider]
    except KeyError as exc:
        raise ValueError("unsupported oauth provider") from exc
