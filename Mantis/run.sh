#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"
PYTHON="${HERE}/.venv/bin/python"
MANTIS_REPO="${MANTIS_REPO:-${ROOT}/external/mantis}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Run: bash setup.sh" >&2
  exit 1
fi
if [[ ! -d "${MANTIS_REPO}" ]]; then
  echo "Clone the official source into ${ROOT}/external/mantis, or set MANTIS_REPO." >&2
  exit 1
fi

RUN_DIR="${HERE}/runs/$(date +%Y%m%d-%H%M%S)"
mkdir -p "${RUN_DIR}"
PORTABLE="${RUN_DIR}/mantis_random.py"
"${PYTHON}" "${HERE}/make_portable.py" --output "${PORTABLE}"

MANTIS_REPO="${MANTIS_REPO}" "${PYTHON}" "${PORTABLE}" "$@"
