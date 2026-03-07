#!/usr/bin/env python3
"""Execute shell commands on RunPod pods via Jupyter nbconvert API.

Usage:
    python scripts/runpod/pod_exec.py "tail -20 /workspace/pairs_shard_0.log" fml2
    python scripts/runpod/pod_exec.py "nvidia-smi" all
    python scripts/runpod/pod_exec.py "hostname" all
"""

import json
import os
import re
import sys
import requests

RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "")

# Pod definitions: name -> (pod_id, jupyter_password, shard_id)
# Set via environment: POD_CONFIG='{"fml2":["pod_id","jupyter_pass",0],...}'
_DEFAULT_PODS = {}
_pod_config_env = os.environ.get("POD_CONFIG", "")
if _pod_config_env:
    _raw = json.loads(_pod_config_env)
    _DEFAULT_PODS = {k: tuple(v) for k, v in _raw.items()}

PODS = _DEFAULT_PODS


def _get_session(pod_id: str, jupyter_pass: str) -> requests.Session:
    """Authenticate with Jupyter and return a session."""
    base = f"https://{pod_id}-8888.proxy.runpod.net"
    s = requests.Session()
    s.get(f"{base}/login")
    xsrf = s.cookies.get("_xsrf", "")
    s.post(
        f"{base}/login",
        data={"password": jupyter_pass, "_xsrf": xsrf},
        allow_redirects=False,
    )
    return s


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", text)


def exec_on_pod(pod_name: str, command: str, timeout: int = 120) -> str:
    """Execute a shell command on a RunPod pod via Jupyter nbconvert.

    Returns the stdout+stderr output as a string.
    """
    pod_id, jupyter_pass, shard_id = PODS[pod_name]
    base = f"https://{pod_id}-8888.proxy.runpod.net"
    s = _get_session(pod_id, jupyter_pass)
    xsrf = s.cookies.get("_xsrf", "")

    # Create a notebook that runs the command via subprocess
    nb = {
        "cells": [
            {
                "cell_type": "code",
                "source": (
                    "import subprocess, sys\n"
                    f"r = subprocess.run({json.dumps(command)}, shell=True,\n"
                    "    capture_output=True, text=True, timeout=300)\n"
                    "sys.stdout.write(r.stdout)\n"
                    "sys.stderr.write(r.stderr)\n"
                ),
                "metadata": {},
                "outputs": [],
                "execution_count": None,
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }

    # Upload notebook
    s.put(
        f"{base}/api/contents/workspace/_exec.ipynb",
        headers={"X-XSRFToken": xsrf},
        json={"type": "notebook", "content": nb, "format": "json"},
    )

    # Execute via nbconvert (synchronous HTTP request)
    resp = s.get(
        f"{base}/nbconvert/notebook/workspace/_exec.ipynb?execute=1",
        timeout=timeout,
    )

    if resp.status_code != 200:
        return f"ERROR: nbconvert returned {resp.status_code}"

    # Parse the executed notebook to extract outputs
    try:
        executed_nb = json.loads(resp.text)
        outputs = []
        for cell in executed_nb.get("cells", []):
            for out in cell.get("outputs", []):
                if out.get("output_type") == "stream":
                    outputs.append("".join(out.get("text", [])))
                elif out.get("output_type") == "error":
                    outputs.append("\n".join(out.get("traceback", [])))
        return _strip_ansi("\n".join(outputs)).strip()
    except json.JSONDecodeError:
        # Might be HTML format
        return resp.text[:2000]


def exec_all(command: str, timeout: int = 120) -> dict:
    """Execute a command on all pods."""
    results = {}
    for name in PODS:
        print(f"\n{'='*60}")
        print(f"  {name} (shard {PODS[name][2]})")
        print(f"{'='*60}")
        try:
            output = exec_on_pod(name, command, timeout)
            print(output)
            results[name] = output
        except Exception as e:
            print(f"ERROR: {e}")
            results[name] = f"ERROR: {e}"
    return results


def pod_status() -> dict:
    """Get the status of pair generation on all pods."""
    return exec_all(
        "tail -30 /workspace/pairs_shard_*.log 2>/dev/null || echo 'No log files found'"
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: pod_exec.py <command> [pod_name|all|status]")
        print("  pod_exec.py 'hostname' all          # Run on all pods")
        print("  pod_exec.py 'nvidia-smi' fml2       # Run on specific pod")
        print("  pod_exec.py status                   # Show shard status")
        sys.exit(1)

    if sys.argv[1] == "status":
        pod_status()
    else:
        command = sys.argv[1]
        target = sys.argv[2] if len(sys.argv) > 2 else "all"

        if target == "all":
            exec_all(command)
        elif target in PODS:
            output = exec_on_pod(target, command)
            print(output)
        else:
            print(f"Unknown pod: {target}. Available: {', '.join(PODS.keys())}, all")
            sys.exit(1)
