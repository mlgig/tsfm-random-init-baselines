#!/usr/bin/env bash
set -euo pipefail

# Requires an NVIDIA driver new enough for CUDA 12.4.  Change the index and
# package versions together if your driver requires cu121 or a newer CUDA wheel.
cd "$(dirname "$0")/../gpu"
uv python install 3.11
uv sync
uv run python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; verify nvidia-smi and the driver before benchmarking.")
print("GPU:", torch.cuda.get_device_name(0))
PY
