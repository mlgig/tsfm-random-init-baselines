#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
uv venv "${HERE}/.venv" --python 3.11
uv pip install --python "${HERE}/.venv/bin/python" -r "${HERE}/requirements.txt"
