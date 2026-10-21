#!/usr/bin/env bash
# Create a Python 3.11 environment and install one runner's pinned requirements.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: bash scripts/setup_venv.sh /path/to/.venv /path/to/requirements.txt" >&2
  exit 2
fi

VENV_PATH="$1"
REQUIREMENTS_PATH="$2"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

if [[ ! -f "${REQUIREMENTS_PATH}" ]]; then
  echo "Requirements file does not exist: ${REQUIREMENTS_PATH}" >&2
  exit 2
fi

if command -v uv >/dev/null 2>&1; then
  # Preserve a previously created environment so a failed dependency download can
  # be retried without requiring a manual cleanup.
  uv venv "${VENV_PATH}" --python "${PYTHON_VERSION}" --allow-existing
  uv pip install --python "${VENV_PATH}/bin/python" -r "${REQUIREMENTS_PATH}"
  exit 0
fi

PYTHON_BIN="${PYTHON_BIN:-python${PYTHON_VERSION}}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  printf '%s\n' \
    "Neither uv nor ${PYTHON_BIN} is available. Install uv (recommended) or Python" \
    "${PYTHON_VERSION}, then rerun this setup command. Set PYTHON_BIN to use a" \
    "non-default Python executable." >&2
  exit 127
fi

ACTUAL_VERSION="$("${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "${ACTUAL_VERSION}" != "${PYTHON_VERSION}" ]]; then
  echo "${PYTHON_BIN} is Python ${ACTUAL_VERSION}; this runner requires Python ${PYTHON_VERSION}." >&2
  exit 2
fi

"${PYTHON_BIN}" -m venv "${VENV_PATH}"
"${VENV_PATH}/bin/python" -m pip install --upgrade pip
"${VENV_PATH}/bin/python" -m pip install -r "${REQUIREMENTS_PATH}"
