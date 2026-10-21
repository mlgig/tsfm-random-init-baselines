#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DESTINATION="${1:-${ROOT}/data/UEA}"
ARCHIVE_DIR="${ROOT}/data/raw"
ARCHIVE="${ARCHIVE_DIR}/Multivariate2018_ts.zip"
PARTIAL_ARCHIVE="${ARCHIVE}.part"
URL="https://www.timeseriesclassification.com/aeon-toolkit/Archives/Multivariate2018_ts.zip"
EXPECTED_DATASET_COUNT=30
EXPECTED_MANIFEST="${ROOT}/data/paper_all_30_datasets.txt"

dataset_names() {
  local split="$1"
  find "${DESTINATION}" -type f -name "*_${split}.ts" -exec basename {} \; \
    | sed "s/_${split}\\.ts$//" \
    | sort
}

validate_destination() {
  local train_names test_names expected_names train_count test_count
  train_names="$(dataset_names TRAIN)"
  test_names="$(dataset_names TEST)"
  expected_names="$(awk '!/^[[:space:]]*#/ && NF { print $1 }' "${EXPECTED_MANIFEST}" | sort)"
  train_count="$(printf '%s\n' "${train_names}" | sed '/^$/d' | wc -l | tr -d '[:space:]')"
  test_count="$(printf '%s\n' "${test_names}" | sed '/^$/d' | wc -l | tr -d '[:space:]')"

  if [[ "${train_count}" -ne "${EXPECTED_DATASET_COUNT}" || "${test_count}" -ne "${EXPECTED_DATASET_COUNT}" ]]; then
    echo "Archive extraction validation failed: found ${train_count} TRAIN and ${test_count} TEST files; expected ${EXPECTED_DATASET_COUNT} each." >&2
    return 1
  fi
  if [[ "${train_names}" != "${test_names}" ]]; then
    echo "Archive extraction validation failed: TRAIN and TEST dataset names do not match." >&2
    return 1
  fi
  if [[ "${train_names}" != "${expected_names}" ]]; then
    echo "Archive extraction validation failed: extracted datasets do not match data/paper_all_30_datasets.txt." >&2
    return 1
  fi
  return 0
}

if [[ -d "${DESTINATION}" ]] && find "${DESTINATION}" -type f \( -name '*_TRAIN.ts' -o -name '*_TEST.ts' \) -print -quit | grep -q .; then
  if validate_destination; then
    echo "A validated UEA archive already exists at: ${DESTINATION}"
    exit 0
  fi
  echo "Destination already contains incomplete or invalid UEA data: ${DESTINATION}" >&2
  echo "Use an empty destination rather than overwriting an existing archive." >&2
  exit 1
fi

mkdir -p "${ARCHIVE_DIR}" "${DESTINATION}"
if [[ -f "${ARCHIVE}" ]]; then
  echo "Validating cached UEA archive: ${ARCHIVE}"
  unzip -tqq "${ARCHIVE}"
else
  echo "Downloading the approximately 846 MiB UEA archive; interrupted downloads resume automatically."
  curl -L --fail --retry 3 --continue-at - -o "${PARTIAL_ARCHIVE}" "${URL}"
  unzip -tqq "${PARTIAL_ARCHIVE}"
  mv -f "${PARTIAL_ARCHIVE}" "${ARCHIVE}"
fi
unzip -q "${ARCHIVE}" -d "${DESTINATION}"

if ! validate_destination; then
  exit 1
fi

echo "Validated ${EXPECTED_DATASET_COUNT} UEA train/test dataset pairs at: ${DESTINATION}"
echo "Use: export UEA_ROOT=${DESTINATION}"
