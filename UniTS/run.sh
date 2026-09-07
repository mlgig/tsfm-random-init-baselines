#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"
PYTHON="${HERE}/.venv/bin/python"
UNITS_REPO="${UNITS_REPO:-${ROOT}/external/units}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Run: bash setup.sh" >&2
  exit 1
fi
if [[ ! -d "${UNITS_REPO}" ]]; then
  echo "Clone the official source into ${ROOT}/external/units, or set UNITS_REPO." >&2
  exit 1
fi

"${PYTHON}" "${HERE}/launch.py" --units-root "${UNITS_REPO}" -- "$@"
