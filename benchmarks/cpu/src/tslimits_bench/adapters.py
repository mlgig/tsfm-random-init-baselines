from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import torch


def load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {item["name"]: item for item in data["models"]}


def _first_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    for attribute in ("embeddings", "last_hidden_state", "logits"):
        candidate = getattr(value, attribute, None)
        if isinstance(candidate, torch.Tensor):
            return candidate
    if isinstance(value, (list, tuple)):
        for candidate in value:
            try:
                return _first_tensor(candidate)
            except TypeError:
                pass
    raise TypeError(f"Could not extract a tensor from output of type {type(value)!r}")


def _load_moment(spec: dict[str, Any], device: torch.device) -> Callable[[torch.Tensor], torch.Tensor]:
    from momentfm import MOMENTPipeline

    model = MOMENTPipeline.from_pretrained(
        spec["repo_id"],
        revision=spec["revision"],
        model_kwargs={"task_name": "embedding"},
    )
    model.init()
    model = model.to(device).eval()

    def forward(batch: torch.Tensor) -> torch.Tensor:
        mask = torch.ones((batch.shape[0], batch.shape[-1]), dtype=torch.float32, device=batch.device)
        return model.embed(x_enc=batch, input_mask=mask, reduction="mean").embeddings

    return forward


def _load_mantis(spec: dict[str, Any], device: torch.device, version: int) -> Callable[[torch.Tensor], torch.Tensor]:
    from mantis.architecture import MantisV1, MantisV2

    model_type = MantisV1 if version == 1 else MantisV2
    model = model_type(
        device=str(device),
        return_transf_layer=2,
        output_token="combined",
    ).from_pretrained(spec["repo_id"], revision=spec["revision"])
    if hasattr(model, "to"):
        model = model.to(device)
    model = model.eval()

    def forward(batch: torch.Tensor) -> torch.Tensor:
        if batch.shape[-1] % 32:
            raise ValueError("Mantis inputs require a sequence length divisible by 32")
        # MantisTrainer.transform applies the single-channel backbone per channel
        # before concatenating channel representations. Preserve that behavior here.
        outputs = [_first_tensor(model(batch[:, channel : channel + 1, :])) for channel in range(batch.shape[1])]
        return torch.cat(outputs, dim=-1)

    return forward


def load_builtin(spec: dict[str, Any], device: torch.device) -> Callable[[torch.Tensor], torch.Tensor]:
    kind = spec["kind"]
    if kind == "moment":
        return _load_moment(spec, device)
    if kind == "mantis-v1":
        return _load_mantis(spec, device, version=1)
    if kind == "mantis-v2":
        return _load_mantis(spec, device, version=2)
    if kind == "external-adapter-required":
        raise RuntimeError(
            f"{spec['name']} is not officially hosted as a Hugging Face checkpoint. "
            "Use the exact UniTS/NuTime repository and experiment preprocessing, then add an adapter "
            "with the same callable contract as this module."
        )
    raise KeyError(f"Unsupported model kind: {kind}")
