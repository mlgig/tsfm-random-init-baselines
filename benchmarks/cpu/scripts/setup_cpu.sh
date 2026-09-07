#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
uv python install 3.11

if [[ "${1:-}" == "--with-models" ]]; then
  uv sync --extra cpu --extra models --extra legacy
else
  uv sync --extra cpu
fi

