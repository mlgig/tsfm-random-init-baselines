#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required for the locked benchmark environment. Install it from https://docs.astral.sh/uv/ and rerun." >&2
  exit 127
fi
uv python install 3.11

if [[ "${1:-}" == "--with-models" ]]; then
  uv sync --extra cpu --extra models --extra legacy
else
  uv sync --extra cpu
fi
