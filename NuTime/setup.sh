#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"
bash "${ROOT}/scripts/setup_venv.sh" "${HERE}/.venv" "${HERE}/requirements.txt"
