"""Benchmark a released UniTS source-task core representation path on synthetic data.

The standard x128 UniTS release is not a generic arbitrary-target classifier:
it contains source-dataset-specific prompt and CLS-token tensors. This script
therefore benchmarks a named *released source task* (UWaveGestureLibrary by
default), using synthetic [B,T,V] input matching that task's channel count. It
returns the [B,V,128] feature tensor before any paper-specific target pooling
or external classifier. It is a core-encoder diagnostic, not a faithful UEA
end-to-end result for a newly created target task.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import yaml

from tslimits_bench.benchmark import benchmark_forward


OFFICIAL_SOURCE_REVISION = "f8e5dfc71f6799164f30b090240a74ee70da01ca"
OFFICIAL_CHECKPOINT_SHA256 = "35d17336c8857d2bacefe403fe7f21ec33ffae9f67d497d569e6af0cb38b5474"
DEFAULT_TASK_ID = "CLS_UWaveGestureLibrary"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_configs(source_dir: Path) -> list[list[object]]:
    config_path = source_dir / "data_provider" / "multi_task_pretrain.yaml"
    with config_path.open(encoding="utf-8") as handle:
        task_dataset = yaml.safe_load(handle)["task_dataset"]
    return [[name, {**settings, "max_batch": None}] for name, settings in task_dataset.items()]


def load_units(source_dir: Path, checkpoint: Path, device: torch.device) -> tuple[torch.nn.Module, list[list[object]]]:
    source_dir = source_dir.resolve()
    if not (source_dir / "models" / "UniTS.py").is_file():
        raise FileNotFoundError(f"--source-dir must be the checked-out UniTS repository, got {source_dir}")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    sys.path.insert(0, str(source_dir))
    from models.UniTS import Model

    configs = read_configs(source_dir)
    args = argparse.Namespace(
        d_model=128,
        n_heads=8,
        e_layers=3,
        prompt_num=10,
        patch_len=16,
        stride=16,
        dropout=0.1,
        right_prob=0.5,
        min_mask_ratio=0.7,
        max_mask_ratio=0.8,
    )
    model = Model(args, configs, pretrain=True)
    # The SHA-256 has already been verified before this call, so this trusted
    # official checkpoint is safe to load using its required metadata wrapper.
    raw_state = torch.load(checkpoint, map_location="cpu", weights_only=False)["student"]
    # The released pretraining artifact was saved under DataParallel.
    state = {key.removeprefix("module."): value for key, value in raw_state.items()}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), configs


def task_spec(configs: list[list[object]], task_id: str) -> dict[str, object]:
    for name, settings in configs:
        if name == task_id:
            return settings  # type: ignore[return-value]
    choices = ", ".join(str(name) for name, _ in configs if str(name).startswith("CLS_"))
    raise KeyError(f"Unknown UniTS source task {task_id!r}; choose a classification task such as: {choices}")


def load_input(path: Path | None, batch_size: int, length: int, channels: int, seed: int, device: torch.device) -> torch.Tensor:
    if path is not None:
        import numpy as np

        values = np.load(path)
        if values.ndim == 2:
            values = values[None, ...]
        if values.ndim != 3:
            raise ValueError("--input-npy must be [T,V] or [B,T,V]")
        if values.shape[-1] != channels:
            raise ValueError(f"Input has {values.shape[-1]} channels; selected source prompt requires {channels}")
        return torch.as_tensor(values, dtype=torch.float32, device=device)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn((batch_size, length, channels), generator=generator, dtype=torch.float32, device=device)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task-id", default=DEFAULT_TASK_ID)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--input-npy", type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--length", type=int, help="Synthetic length; defaults to the released source task length")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.threads < 1:
        raise ValueError("--threads must be at least 1")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    observed_sha256 = sha256(args.checkpoint)
    if observed_sha256 != OFFICIAL_CHECKPOINT_SHA256:
        raise RuntimeError(
            "Checkpoint SHA-256 does not match the pinned official release. "
            f"Expected {OFFICIAL_CHECKPOINT_SHA256}, got {observed_sha256}."
        )
    model, configs = load_units(args.source_dir, args.checkpoint, device)
    spec = task_spec(configs, args.task_id)
    channels = int(spec["enc_in"])
    length = args.length or int(spec["seq_len"])
    dataset_name = str(spec["dataset"])
    batch = load_input(args.input_npy, args.batch_size, length, channels, args.seed, device)
    prefix_prompt = model.prompt_tokens[dataset_name]
    cls_prompt = model.cls_tokens[args.task_id]

    @torch.inference_mode()
    def forward(value: torch.Tensor) -> torch.Tensor:
        tokens, _, _, n_vars, _ = model.tokenize(value.to(device=device, dtype=torch.float32))
        sequence_tokens = tokens.shape[-2]
        tokens = model.prepare_prompt(tokens, n_vars, prefix_prompt, cls_prompt, 1, task_name="classification")
        tokens = model.backbone(tokens, prefix_prompt.shape[2], sequence_tokens)
        return model.cls_head(tokens, return_feature=True).squeeze(2)

    result = benchmark_forward(forward, batch, device=device, warmup=args.warmup, repetitions=args.repetitions).to_dict()
    result["model"] = {
        "name": "units-x128-source-task-core",
        "source": "https://github.com/mims-harvard/UniTS/releases/tag/ckpt",
        "source_revision": OFFICIAL_SOURCE_REVISION,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": observed_sha256,
        "source_task_id": args.task_id,
        "source_dataset": dataset_name,
        "representation": "[B,V,128] CLS feature before target pooling or external probe",
    }
    result["input_source"] = str(args.input_npy) if args.input_npy else "synthetic-standard-normal"
    result["torch_threads"] = args.threads
    result["warning"] = (
        "Released-source-task core encoder diagnostic only. It uses a pretrained source prompt/CLS path; "
        "it is not a generic arbitrary-UEA target adapter or an end-to-end paper result."
    )
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
