#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"
PYTHON="${HERE}/.venv/bin/python"
NUTIME_REPO="${NUTIME_REPO:-${ROOT}/external/nutime}"
NUTIME_REVISION="9acdd3d0e82df0914217ef752fa3b95ec82cd960"
DATA_ROOT=""
CHECKPOINT=""
INIT="random"
SEED="0"
OUTPUT=""
RESULTS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --init) INIT="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --results) RESULTS="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ ! -x "${PYTHON}" ]]; then
  echo "Run: bash setup.sh" >&2
  exit 1
fi
if [[ -z "${DATA_ROOT}" ]]; then
  echo "Usage: bash run.sh --data-root /path/to/UEA [--checkpoint /path/to/checkpoint.pth]" >&2
  exit 2
fi
if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "UEA data root does not exist: ${DATA_ROOT}" >&2
  exit 2
fi
if [[ "${INIT}" == "pretrained" && -z "${CHECKPOINT}" ]]; then
  echo "--checkpoint is required when --init pretrained." >&2
  exit 2
fi
if [[ "${INIT}" == "pretrained" && ! -f "${CHECKPOINT}" ]]; then
  echo "NuTime checkpoint does not exist: ${CHECKPOINT}" >&2
  exit 2
fi
if [[ "${INIT}" != "pretrained" && "${INIT}" != "random" ]]; then
  echo "--init must be pretrained or random." >&2
  exit 2
fi
if [[ ! -f "${NUTIME_REPO}/src/config.py" || \
      ! -f "${NUTIME_REPO}/src/models/encoders/build.py" || \
      ! -f "${NUTIME_REPO}/configs/demo_ft_epilepsy.json" ]]; then
  echo "Clone the compatible official NuTime source into ${ROOT}/external/nutime, or set NUTIME_REPO." >&2
  exit 1
fi
ACTUAL_REVISION="$(git -C "${NUTIME_REPO}" rev-parse HEAD 2>/dev/null || true)"
if [[ "${ACTUAL_REVISION}" != "${NUTIME_REVISION}" ]]; then
  echo "NuTime checkout must be ${NUTIME_REVISION}; got ${ACTUAL_REVISION:-not a git checkout}." >&2
  exit 1
fi
if ! git -C "${NUTIME_REPO}" diff --quiet || ! git -C "${NUTIME_REPO}" diff --cached --quiet; then
  echo "NuTime checkout has modified tracked files; restore a clean ${NUTIME_REVISION} checkout." >&2
  exit 1
fi

RUN_DIR="${HERE}/runs/$(date +%Y%m%d-%H%M%S)"
mkdir -p "${RUN_DIR}"
CONFIGURED="${RUN_DIR}/nutime_configured.ipynb"
KERNEL_PREFIX="${RUN_DIR}/kernel"
KERNEL_NAME="nutime-run"

if [[ -z "${OUTPUT}" ]]; then
  OUTPUT="${RUN_DIR}/nutime_executed.ipynb"
elif [[ "${OUTPUT}" != /* ]]; then
  OUTPUT="$(pwd)/${OUTPUT}"
fi
mkdir -p "$(dirname "${OUTPUT}")"
if [[ -z "${RESULTS}" ]]; then
  RESULTS="$(dirname "${OUTPUT}")/nutime_results.csv"
elif [[ "${RESULTS}" != /* ]]; then
  RESULTS="$(pwd)/${RESULTS}"
fi
mkdir -p "$(dirname "${RESULTS}")"

CONFIGURE_ARGS=(
  --output "${CONFIGURED}"
  --data-root "${DATA_ROOT}"
  --nutime-root "${NUTIME_REPO}"
  --results "${RESULTS}"
  --init "${INIT}"
  --seed "${SEED}"
)
if [[ -n "${CHECKPOINT}" ]]; then
  CONFIGURE_ARGS+=(--checkpoint "${CHECKPOINT}")
fi
"${PYTHON}" "${HERE}/configure_notebook.py" "${CONFIGURE_ARGS[@]}"
# nbconvert otherwise may select a globally installed `python3` kernel rather
# than this runner's virtual environment.  Register a throwaway local kernel
# and expose only it to this execution.
"${PYTHON}" -m ipykernel install --prefix "${KERNEL_PREFIX}" \
  --name "${KERNEL_NAME}" --display-name "NuTime runner" >/dev/null

(
  cd "${NUTIME_REPO}"
  JUPYTER_PATH="${KERNEL_PREFIX}/share/jupyter${JUPYTER_PATH:+:${JUPYTER_PATH}}" \
    "${PYTHON}" -m jupyter nbconvert --to notebook --execute "${CONFIGURED}" \
    --output "$(basename "${OUTPUT}" .ipynb)" \
    --output-dir "$(dirname "${OUTPUT}")" \
    --ExecutePreprocessor.kernel_name="${KERNEL_NAME}" \
    --ExecutePreprocessor.timeout=-1
)
