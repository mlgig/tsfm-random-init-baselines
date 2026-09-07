#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
exec "${HERE}/.venv/bin/python" "${HERE}/evaluate_aeon.py" --model hydra "$@"
