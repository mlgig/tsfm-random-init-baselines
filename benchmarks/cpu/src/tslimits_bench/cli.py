from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import torch

from .adapters import load_builtin, load_manifest
from .benchmark import benchmark_forward


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark loaded TSFM forward-pass resources.")
    parser.add_argument("--manifest", type=Path, default=_project_root() / "configs" / "model_manifest.json")
    parser.add_argument("--model", required=True, help="Name from configs/model_manifest.json")
    parser.add_argument("--device", default="cpu", help="cpu or cuda[:index]")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--channels", type=int, default=None, help="Override manifest channel count")
    parser.add_argument("--length", type=int, default=None, help="Override manifest sequence length")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--threads", type=int, default=1, help="CPU intra-op thread count; report this value")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--input-npy", type=Path, help="Optional preprocessed float32 input of shape [B,C,T] or [C,T]")
    parser.add_argument("--output", type=Path, help="JSON output path")
    return parser


def _load_input(args: argparse.Namespace, spec: dict[str, object], device: torch.device) -> torch.Tensor:
    if args.input_npy:
        import numpy as np

        array = np.load(args.input_npy)
        if array.ndim == 2:
            array = array[None, ...]
        if array.ndim != 3:
            raise ValueError("--input-npy must have shape [C,T] or [B,C,T]")
        return torch.as_tensor(array, dtype=torch.float32, device=device)

    channels = args.channels or int(spec.get("channels", 1))
    length = args.length or int(spec.get("length", 512))
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    return torch.randn((args.batch_size, channels, length), generator=generator, dtype=torch.float32, device=device)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable. Install the GPU project on an NVIDIA CUDA machine.")
    if args.threads < 1:
        raise ValueError("--threads must be at least 1")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    models = load_manifest(args.manifest)
    if args.model not in models:
        raise KeyError(f"Unknown model {args.model!r}; choose one of: {', '.join(models)}")
    spec = models[args.model]
    batch = _load_input(args, spec, device)
    forward = load_builtin(spec, device)
    result = benchmark_forward(
        forward,
        batch,
        device=device,
        warmup=args.warmup,
        repetitions=args.repetitions,
    ).to_dict()
    result["model"] = {
        key: value
        for key, value in spec.items()
        if key not in {"benchmark_note"}
    }
    result["input_source"] = str(args.input_npy) if args.input_npy else "synthetic-standard-normal"
    result["torch_threads"] = args.threads
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
