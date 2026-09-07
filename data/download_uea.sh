#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DESTINATION="${1:-${ROOT}/data/UEA}"
ARCHIVE_DIR="${ROOT}/data/raw"
ARCHIVE="${ARCHIVE_DIR}/Multivariate2018_ts.zip"
URL="https://www.timeseriesclassification.com/aeon-toolkit/Archives/Multivariate2018_ts.zip"

mkdir -p "${ARCHIVE_DIR}" "${DESTINATION}"
curl -L --fail --retry 3 -o "${ARCHIVE}" "${URL}"
unzip -q "${ARCHIVE}" -d "${DESTINATION}"

echo "UEA .ts archive extracted to: ${DESTINATION}"
echo "Use: export UEA_ROOT=${DESTINATION}"

