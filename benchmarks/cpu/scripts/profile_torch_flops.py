"""Count Torch-dispatched FLOPs for the pinned neural encoder paths.

This is deliberately restricted to the MOMENT and Mantis adapters in
``tslimits_bench.adapters``.  Those adapters reproduce the multivariate paths
used by the CPU microbenchmark: MOMENT receives [B,C,T], while Mantis is
applied once per channel and channel features are concatenated.  The result is
an operator-coverage-limited diagnostic, not an architecture-paper FLOP claim.
It must not be compared numerically with aeon's ROCKET/Hydra/QUANT, whose
NumPy/Numba/scikit-learn kernels are outside Torch's FLOP counter.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.flop_counter import FlopCounterMode


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tslimits_bench.adapters import load_builtin, load_manifest  # noqa: E402


def _serialise_counts(counts: dict[Any, dict[Any, int]]) -> dict[str, dict[str, int]]:
    global_counts = counts.get("Global", {})
    return {"Global": {str(operator): int(value) for operator, value in global_counts.items()}}


def _tensor_shape(value: Any) -> list[int] | None:
    if isinstance(value, torch.Tensor):
        return list(value.shape)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "configs" / "model_manifest.json")
    parser.add_argument("--model", required=True, help="A MOMENT or Mantis model from the manifest")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.batch_size < 1 or args.channels < 1 or args.length < 1 or args.threads < 1:
        raise ValueError("batch size, channels, length, and threads must all be positive")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    device = torch.device("cpu")
    models = load_manifest(args.manifest)
    if args.model not in models:
        raise KeyError(f"Unknown model {args.model!r}; choose one of: {', '.join(models)}")
    spec = models[args.model]
    if spec["kind"] not in {"moment", "mantis-v1", "mantis-v2"}:
        raise ValueError("This script supports only the built-in Torch MOMENT/Mantis adapters")
    if str(spec["kind"]).startswith("mantis") and args.length % 32:
        raise ValueError("Mantis requires --length divisible by 32")

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    batch = torch.randn((args.batch_size, args.channels, args.length), generator=generator, dtype=torch.float32)
    forward = load_builtin(spec, device)
    # FlopCounterMode is a Torch dispatch mode and therefore needs no_grad(),
    # rather than inference_mode(), to observe the operations it supports.
    with torch.no_grad():
        for _ in range(args.warmup):
            _ = forward(batch)
        counter = FlopCounterMode(depth=0, display=False)
        with counter:
            output = forward(batch)

    total_flops = int(counter.get_total_flops())
    payload = {
        "model": {key: value for key, value in spec.items() if key != "benchmark_note"},
        "input_shape": list(batch.shape),
        "output_shape": _tensor_shape(output),
        "input_source": "synthetic-standard-normal",
        "torch_threads": args.threads,
        "warmup": args.warmup,
        "registered_flops_per_forward": total_flops,
        "registered_gflops_per_forward": total_flops / 1e9,
        "operator_flops": _serialise_counts(counter.get_flop_counts()),
        "scope": (
            "Loaded encoder forward only. Mantis applies the official single-channel backbone once per input "
            "channel before concatenation; MOMENT uses its released embedding reduction. Excludes input I/O, "
            "model loading, probe fitting, and probe prediction."
        ),
        "caveat": (
            "PyTorch FlopCounterMode counts registered matrix-multiply, convolution, and related Torch operations, "
            "but omits operations such as normalization, softmax, activations, interpolation, indexing, and non-Torch code. "
            "Treat this as a reproducible partial diagnostic, not a complete architecture FLOP total, and do not compare it "
            "numerically with aeon ROCKET/Hydra/QUANT. Use measured CPU latency/RSS for cross-framework comparisons."
        ),
        "torch_version": torch.__version__,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
