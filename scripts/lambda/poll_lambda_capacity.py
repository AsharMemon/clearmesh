#!/usr/bin/env python3
"""Poll Lambda Cloud for a suitable FACE-Q scale-training node.

This helper intentionally avoids printing API keys or private material. It can
optionally register a local public SSH key and launch exactly one matching
instance when capacity appears.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_TARGETS = (
    "gpu_8x_h100_sxm5",
    "gpu_8x_a100_80gb_sxm4",
)


def _auth_header(api_key: str) -> str:
    return "Basic " + base64.b64encode(f"{api_key}:".encode()).decode()


def _request(
    base_url: str,
    api_key: str,
    method: str,
    path: str,
    body: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    data = None
    headers = {
        "Accept": "application/json",
        "Authorization": _auth_header(api_key),
        # Lambda's Cloudflare config rejects Python's default urllib signature.
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125 Safari/537.36"
        ),
    }
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=data,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            text = resp.read().decode("utf-8", "replace")
            return resp.status, json.loads(text or "{}")
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", "replace")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = {"error": text[:2000]}
        return exc.code, payload


def _redact(payload: Any) -> Any:
    if isinstance(payload, dict):
        out: dict[str, Any] = {}
        for key, value in payload.items():
            lower = key.lower()
            if "key" in lower or "token" in lower:
                out[key] = "[redacted]"
            else:
                out[key] = _redact(value)
        return out
    if isinstance(payload, list):
        return [_redact(item) for item in payload]
    if isinstance(payload, str) and "PRIVATE KEY" in payload:
        return "[redacted]"
    return payload


def _ensure_ssh_key(
    base_url: str,
    api_key: str,
    key_name: str,
    public_key_file: Path,
    out_dir: Path,
) -> None:
    status, payload = _request(base_url, api_key, "GET", "/ssh-keys")
    (out_dir / "ssh_keys.latest.json").write_text(
        json.dumps(_redact(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if status >= 400:
        raise RuntimeError(f"failed to list Lambda SSH keys: HTTP {status}")
    keys = payload.get("data", [])
    if any(item.get("name") == key_name for item in keys if isinstance(item, dict)):
        return
    public_key = public_key_file.expanduser().read_text(encoding="utf-8").strip()
    status, payload = _request(
        base_url,
        api_key,
        "POST",
        "/ssh-keys",
        {"name": key_name, "public_key": public_key},
    )
    (out_dir / "ssh_key_add.latest.json").write_text(
        json.dumps(_redact(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if status >= 400:
        raise RuntimeError(f"failed to add Lambda SSH key: HTTP {status}")


def _candidate_rows(instance_types: dict[str, Any], targets: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    data = instance_types.get("data", {})
    for target in targets:
        item = data.get(target)
        if not isinstance(item, dict):
            rows.append({"name": target, "present": False, "regions": []})
            continue
        instance_type = item.get("instance_type", {})
        regions = item.get("regions_with_capacity_available") or []
        rows.append(
            {
                "name": target,
                "present": True,
                "description": instance_type.get("description"),
                "gpu": instance_type.get("gpu_description"),
                "price_per_hour": (
                    None
                    if instance_type.get("price_cents_per_hour") is None
                    else instance_type.get("price_cents_per_hour") / 100
                ),
                "specs": instance_type.get("specs", {}),
                "regions": regions,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="https://cloud.lambdalabs.com/api/v1")
    parser.add_argument("--out-dir", default=".codex_outputs/lambda_scale_20260517")
    parser.add_argument("--target", action="append", dest="targets", default=[])
    parser.add_argument("--ssh-key-name", default="clearmesh-codex-mac-id-ed25519")
    parser.add_argument("--public-key-file", default="~/.ssh/id_ed25519.pub")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--name-prefix", default="clearmesh-faceq-scale-smoke")
    args = parser.parse_args()

    api_key = os.environ.get("LAMBDA_KEY")
    if not api_key:
        print(json.dumps({"ok": False, "error": "LAMBDA_KEY is not set"}))
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = args.targets or list(DEFAULT_TARGETS)
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    _ensure_ssh_key(
        args.base_url,
        api_key,
        args.ssh_key_name,
        Path(args.public_key_file),
        out_dir,
    )

    status, instance_types = _request(args.base_url, api_key, "GET", "/instance-types")
    (out_dir / "instance_types.latest.json").write_text(
        json.dumps(instance_types, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if status >= 400:
        raise RuntimeError(f"failed to list Lambda instance types: HTTP {status}")

    rows = _candidate_rows(instance_types, targets)
    summary: dict[str, Any] = {
        "ok": True,
        "time": timestamp,
        "targets": rows,
        "launched": False,
    }

    launched_flag = out_dir / "lambda_launched.flag"
    if launched_flag.exists():
        summary["skipped"] = "already_launched"
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    for row in rows:
        regions = row.get("regions") or []
        if not regions:
            continue
        region_name = regions[0]["name"] if isinstance(regions[0], dict) else str(regions[0])
        summary["selected"] = {"instance_type_name": row["name"], "region_name": region_name}
        if not args.launch:
            summary["launch_mode"] = "dry_run"
            break
        request = {
            "region_name": region_name,
            "instance_type_name": row["name"],
            "ssh_key_names": [args.ssh_key_name],
            "file_system_names": [],
            "name": f"{args.name_prefix}-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}",
        }
        launch_status, launch_payload = _request(
            args.base_url,
            api_key,
            "POST",
            "/instance-operations/launch",
            request,
        )
        redacted_payload = _redact(launch_payload)
        (out_dir / "launch.latest.json").write_text(
            json.dumps(redacted_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        summary["launch_http_status"] = launch_status
        summary["launch_response"] = redacted_payload
        if launch_status >= 400:
            summary["launch_error"] = True
            break
        summary["launched"] = True
        launched_flag.write_text(
            json.dumps(
                {
                    "time": timestamp,
                    "instance_type_name": row["name"],
                    "region_name": region_name,
                    "response": redacted_payload,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        break

    (out_dir / "poll_summary.latest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
