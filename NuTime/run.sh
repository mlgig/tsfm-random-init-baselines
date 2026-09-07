#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"
PYTHON="${HERE}/.venv/bin/python"
NUTIME_REPO="${NUTIME_REPO:-${ROOT}/external/nutime}"
DATA_ROOT=""
CHECKPOINT=""
INIT="random"
SEED="0"
OUTPUT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --init) INIT="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ ! -x "${PYTHON}" ]]; then
  echo "Run: bash setup.sh" >&2
  exit 1
fi
if [[ -z "${DATA_ROOT}" || -z "${CHECKPOINT}" ]]; then
  echo "Usage: bash run.sh --data-root /path/to/UEA --checkpoint /path/to/checkpoint.pth" >&2
  exit 2
fi
if [[ ! -d "${NUTIME_REPO}" ]]; then
  echo "Clone the official source into ${ROOT}/external/nutime, or set NUTIME_REPO." >&2
  exit 1
fi

RUN_DIR="${HERE}/runs/$(date +%Y%m%d-%H%M%S)"
mkdir -p "${RUN_DIR}"
CONFIGURED="${RUN_DIR}/nutime_configured.ipynb"
"${PYTHON}" "${HERE}/configure_notebook.py" \
  --output "${CONFIGURED}" \
  --data-root "${DATA_ROOT}" \
  --checkpoint "${CHECKPOINT}" \
  --init "${INIT}" \
  --seed "${SEED}"

if [[ -z "${OUTPUT}" ]]; then
  OUTPUT="${RUN_DIR}/nutime_executed.ipynb"
fi
mkdir -p "$(dirname "${OUTPUT}")"
(
  cd "${NUTIME_REPO}"
  "${PYTHON}" -m jupyter nbconvert --to notebook --execute "${CONFIGURED}" \
    --output "$(basename "${OUTPUT}" .ipynb)" \
    --output-dir "$(dirname "${OUTPUT}")" \
    --ExecutePreprocessor.timeout=-1
)
