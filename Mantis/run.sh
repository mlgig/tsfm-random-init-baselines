#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"
PYTHON="${HERE}/.venv/bin/python"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Run: bash setup.sh" >&2
  exit 1
fi
PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON}" "${HERE}/mantis_random.py" "$@"
