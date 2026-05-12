#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="${OMNIPART_REPO_DIR:-/home/ubuntu/mesh-heads/OmniPart}"
VENV_DIR="${OMNIPART_VENV_DIR:-/home/ubuntu/omnipart-venv}"
REPO_URL="${OMNIPART_REPO_URL:-https://github.com/HKU-MMLab/OmniPart.git}"
mkdir -p "$(dirname "$REPO_DIR")"
if [ ! -d "$REPO_DIR/.git" ]; then
  git clone "$REPO_URL" "$REPO_DIR"
else
  git -C "$REPO_DIR" pull --ff-only
fi
python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
"$VENV_DIR/bin/python" -m pip install -r "$REPO_DIR/requirements.txt"
echo "OmniPart installed at $REPO_DIR with venv $VENV_DIR"
