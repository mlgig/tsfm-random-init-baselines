#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${HERE}/.venv/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
  echo "Run: bash setup.sh" >&2
  exit 1
fi
exec "${PYTHON}" "${HERE}/evaluate_aeon.py" --model rocket "$@"
