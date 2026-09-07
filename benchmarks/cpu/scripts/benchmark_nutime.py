"""Benchmark the official legacy NuTime BYOL checkpoint without CUDA-only runner code.

This is intentionally separate from the built-in Hugging Face adapters. NuTime's
official release is a GitHub repository/checkpoint rather than a Hub AutoModel,
and the official ClassificationExp forces CUDA. The loader below follows the
released demo architecture, strips the BYOL projector, and returns its 128-D CLS
representation.

For a faithful multivariate result, replace this script with the exact adapter
used by the paper. The public checkpoint was pretrained on univariate samples;
with the default official configuration, a C>1 target introduces random channel
embedding parameters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from tslimits_bench.benchmark import benchmark_forward


OFFICIAL_SOURCE_REVISION = "9acdd3d0e82df0914217ef752fa3b95ec82cd960"
OFFICIAL_CHECKPOINT_SHA256 = "65543d55c6a9d30837b46b3809b7358a6ecb3e9702f4eaf01bfd68a96707f7a2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_nutime(source_dir: Path, checkpoint: Path, channels: int, device: torch.device) -> nn.Module:
    source_dir = source_dir.resolve()
    if not (source_dir / "src" / "config.py").is_file():
        raise FileNotFoundError(f"--source-dir must be the checked-out NuTime repository, got {source_dir}")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    sys.path.insert(0, str(source_dir))

    from src.config import Config
    from src.models.build import get_model
    from src.models.encoders.build import get_encoder

    config = Config()
    with (source_dir / "configs" / "demo_ft_epilepsy.json").open(encoding="utf-8") as handle:
        config.update_by_dict(json.load(handle))
    # Released demo inference: deterministic linear resize to 176 points, then
    # WindowNormEncoder (11 windows) and the WinT backbone.
    config.task = "cls"
    config.transform_size = 176
    config.model_series_size = 176
    config.num_channels = channels
    config.num_classes = 1  # disposable classification head
    # The official factory accesses dataset.samples/targets even though the
    # released `mbe` encoder does not use them. Supply a no-data placeholder.
    dataset_stub = type("DatasetStub", (), {"samples": None, "targets": None})()
    encoder = get_encoder(config, dataset=dataset_stub)
    network = get_model(config)
    model = nn.Sequential(encoder, network)

    # The official checkpoint is a BYOL training checkpoint. Keep only the
    # online backbone and remove its projection/classification head.
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    state_dict = {
        key[len("backbone.") :]: value
        for key, value in raw.items()
        if key.startswith("backbone.") and not key.startswith("backbone.1.fc.")
    }
    info = model.load_state_dict(state_dict, strict=False)
    if info.unexpected_keys:
        raise RuntimeError(f"Unexpected checkpoint keys: {info.unexpected_keys}")
    # For C=1 only the disposable network.fc weights should be absent.
    if channels == 1 and set(info.missing_keys) != {"1.fc.weight", "1.fc.bias"}:
        raise RuntimeError(f"Unexpected missing C=1 checkpoint keys: {info.missing_keys}")
    network.fc = nn.Identity()
    return model.to(device).eval()


def load_input(path: Path | None, batch_size: int, channels: int, length: int, seed: int, device: torch.device) -> torch.Tensor:
    if path is not None:
        import numpy as np

        values = np.load(path)
        if values.ndim == 2:
            values = values[None, ...]
        if values.ndim != 3:
            raise ValueError("--input-npy must be [C,T] or [B,C,T]")
        return torch.as_tensor(values, dtype=torch.float32, device=device)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn((batch_size, channels, length), generator=generator, dtype=torch.float32, device=device)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--input-npy", type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--length", type=int, default=176)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.channels != 1:
        print("WARNING: C>1 introduces a non-pretrained NuTime channel embedding under the official demo config.", file=sys.stderr)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    observed_sha256 = sha256(args.checkpoint)
    if observed_sha256 != OFFICIAL_CHECKPOINT_SHA256:
        raise RuntimeError(
            "Checkpoint SHA-256 does not match the pinned official release. "
            f"Expected {OFFICIAL_CHECKPOINT_SHA256}, got {observed_sha256}."
        )
    model = load_nutime(args.source_dir, args.checkpoint, args.channels, device)
    batch = load_input(args.input_npy, args.batch_size, args.channels, args.length, args.seed, device)

    @torch.inference_mode()
    def forward(value: torch.Tensor) -> torch.Tensor:
        value = F.interpolate(value.to(device=device, dtype=torch.float32), size=176, mode="linear", align_corners=False)
        return model(value)

    result = benchmark_forward(forward, batch, device=device, warmup=args.warmup, repetitions=args.repetitions).to_dict()
    result["model"] = {
        "name": "nutime-legacy-byol",
        "source": "https://github.com/chenguolin/NuTime",
        "source_revision": OFFICIAL_SOURCE_REVISION,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": observed_sha256,
        "representation": "128-D normalized CLS feature; BYOL projector removed",
    }
    result["input_source"] = str(args.input_npy) if args.input_npy else "synthetic-standard-normal"
    result["torch_threads"] = args.threads
    result["warning"] = (
        "Faithful only for C=1 released NuTime representation extraction. "
        "Use the paper's documented multivariate adapter before reporting C>1 results."
    )
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
