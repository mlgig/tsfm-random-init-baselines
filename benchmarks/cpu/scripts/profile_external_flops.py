"""Count registered FMA/convolution FLOPs for the pinned native external paths.

This script intentionally keeps NuTime and UniTS separate from the common
``[1,3,512]`` diagnostic.  The official NuTime checkpoint is faithful only for
the univariate ``[B,1,176]`` released path.  The standard UniTS x128 checkpoint
is source-task-specific, so this profiles its released UWave source prompt at
``[B,T,V]=[1,315,3]``.  Neither is a substitute for the paper's exact target
multivariate adapter/pooling pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

import benchmark_nutime as nutime  # noqa: E402
import benchmark_units as units  # noqa: E402


def count(forward: Callable[[torch.Tensor], torch.Tensor], batch: torch.Tensor, warmup: int) -> tuple[torch.Tensor, int, dict[str, int]]:
    # Warm-up does not need autograd. PyTorch's ModuleTracker hook used by
    # FlopCounterMode asserts on one UniTS positional-embedding path under
    # no_grad(), so the one counted eval forward deliberately enables grad.
    # This changes neither the executed arithmetic nor model eval mode.
    with torch.no_grad():
        for _ in range(warmup):
            _ = forward(batch)
    with torch.enable_grad():
        counter = FlopCounterMode(depth=0, display=False)
        with counter:
            output = forward(batch)
    global_counts = counter.get_flop_counts().get("Global", {})
    return output, int(counter.get_total_flops()), {str(key): int(value) for key, value in global_counts.items()}


def units_path(args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor], dict[str, Any], str]:
    model, configs = units.load_units(args.source_dir, args.checkpoint, device)
    spec = units.task_spec(configs, args.task_id)
    channels = int(spec["enc_in"])
    length = args.length or int(spec["seq_len"])
    dataset_name = str(spec["dataset"])
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    batch = torch.randn((args.batch_size, length, channels), generator=generator, dtype=torch.float32, device=device)
    prefix_prompt = model.prompt_tokens[dataset_name]
    cls_prompt = model.cls_tokens[args.task_id]

    def forward(value: torch.Tensor) -> torch.Tensor:
        tokens, _, _, n_vars, _ = model.tokenize(value.to(device=device, dtype=torch.float32))
        sequence_tokens = tokens.shape[-2]
        tokens = model.prepare_prompt(tokens, n_vars, prefix_prompt, cls_prompt, 1, task_name="classification")
        tokens = model.backbone(tokens, prefix_prompt.shape[2], sequence_tokens)
        return model.cls_head(tokens, return_feature=True).squeeze(2)

    metadata = {
        "name": "units-x128-source-task-core",
        "source": "https://github.com/mims-harvard/UniTS/releases/tag/ckpt",
        "source_revision": units.OFFICIAL_SOURCE_REVISION,
        "checkpoint_sha256": units.sha256(args.checkpoint),
        "source_task_id": args.task_id,
        "source_dataset": dataset_name,
        "representation": "[B,V,128] CLS feature before target pooling or external probe",
    }
    return batch, forward, metadata, "[B,T,V]"


def nutime_path(args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor], dict[str, Any], str]:
    if args.channels != 1:
        raise ValueError("The released NuTime checkpoint is faithful only for --channels 1; use the paper adapter for C>1.")
    model = nutime.load_nutime(args.source_dir, args.checkpoint, args.channels, device)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    batch = torch.randn((args.batch_size, args.channels, args.length or 176), generator=generator, dtype=torch.float32, device=device)

    def forward(value: torch.Tensor) -> torch.Tensor:
        value = F.interpolate(value.to(device=device, dtype=torch.float32), size=176, mode="linear", align_corners=False)
        return model(value)

    metadata = {
        "name": "nutime-legacy-byol",
        "source": "https://github.com/chenguolin/NuTime",
        "source_revision": nutime.OFFICIAL_SOURCE_REVISION,
        "checkpoint_sha256": nutime.sha256(args.checkpoint),
        "representation": "128-D normalized CLS feature; BYOL projector removed",
    }
    return batch, forward, metadata, "[B,C,T]"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("units", "nutime"), required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task-id", default=units.DEFAULT_TASK_ID, help="Used only for UniTS")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--channels", type=int, default=1, help="Used only for NuTime")
    parser.add_argument("--length", type=int, help="Synthetic source length; NuTime default is 176")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1 or args.warmup < 0:
        raise ValueError("batch size and threads must be positive, and warmup must be non-negative")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    device = torch.device("cpu")
    if args.model == "units":
        batch, forward, model, layout = units_path(args, device)
        caveat = "Native released UWave source-task core only; it is not a generic arbitrary-UEA target adapter or end-to-end paper result."
    else:
        batch, forward, model, layout = nutime_path(args, device)
        caveat = "Native released univariate path only; the public C>1 path adds non-pretrained channel embeddings."
    output, total, operator_flops = count(forward, batch, args.warmup)
    payload = {
        "model": model,
        "input_shape": list(batch.shape),
        "input_layout": layout,
        "input_source": "synthetic-standard-normal",
        "torch_threads": args.threads,
        "warmup": args.warmup,
        "registered_flops_per_forward": total,
        "registered_gflops_per_forward": total / 1e9,
        "operator_flops": {"Global": operator_flops},
        "output_shape": list(output.shape),
        "scope": "Loaded native core forward only; excludes model loading, input I/O, target pooling, probe fitting, and probe prediction.",
        "caveat": caveat + " FlopCounterMode counts registered matrix-multiply/convolution operations only (two FLOPs per MAC).",
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
