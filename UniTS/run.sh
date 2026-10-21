#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"
PYTHON="${HERE}/.venv/bin/python"
UNITS_REPO="${UNITS_REPO:-${ROOT}/external/units}"
UNITS_REVISION="f8e5dfc71f6799164f30b090240a74ee70da01ca"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Run: bash setup.sh" >&2
  exit 1
fi
if [[ ! -d "${UNITS_REPO}" ]]; then
  echo "Clone the official source into ${ROOT}/external/units, or set UNITS_REPO." >&2
  exit 1
fi
if [[ ! -f "${UNITS_REPO}/models/UniTS.py" ]]; then
  echo "UniTS checkout is missing models/UniTS.py: ${UNITS_REPO}" >&2
  exit 1
fi
ACTUAL_REVISION="$(git -C "${UNITS_REPO}" rev-parse HEAD 2>/dev/null || true)"
if [[ "${ACTUAL_REVISION}" != "${UNITS_REVISION}" ]]; then
  echo "UniTS checkout must be ${UNITS_REVISION}; got ${ACTUAL_REVISION:-not a git checkout}." >&2
  exit 1
fi
if ! git -C "${UNITS_REPO}" diff --quiet || ! git -C "${UNITS_REPO}" diff --cached --quiet; then
  echo "UniTS checkout has modified tracked files; restore a clean ${UNITS_REVISION} checkout." >&2
  exit 1
fi

"${PYTHON}" "${HERE}/launch.py" --units-root "${UNITS_REPO}" -- "$@"
