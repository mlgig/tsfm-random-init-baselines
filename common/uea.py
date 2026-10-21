"""UEA ``.ts`` collection helpers used by the reproducible runners.

The public Multivariate2018 archive contains a few unequal-length datasets.
The original local experiment paths referred to an untracked
``Multivariate_ts_length_fixed`` derivative, so raw archive collections must
be made rectangular before they reach PyTorch or aeon estimators that require
equal lengths.  Right zero-padding is deliberately explicit here: it is
deterministic, preserves observed values and order, and is applied identically
to pretrained and random conditions within a runner.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def load_uea_collection(path: str) -> tuple[Any, np.ndarray]:
    """Load a UEA ``.ts`` split, retaining a ragged collection when present."""
    from aeon.datasets import load_from_ts_file

    values, labels = load_from_ts_file(path)
    return values, np.asarray(labels)


def _cases(values: Any) -> list[np.ndarray]:
    """Normalize aeon's equal-length or ragged output into ``[channels, time]`` cases."""
    if isinstance(values, np.ndarray):
        if values.ndim != 3:
            raise ValueError(f"expected a 3D UEA collection, got {values.shape}")
        cases = [values[index] for index in range(values.shape[0])]
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        cases = list(values)
    else:
        raise ValueError(f"unsupported UEA collection type: {type(values).__name__}")
    if not cases:
        raise ValueError("UEA collection contains no cases")

    normalized: list[np.ndarray] = []
    channels: int | None = None
    for index, case in enumerate(cases):
        array = np.asarray(case, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError(
                f"case {index} must have shape [channels, time], got {array.shape}"
            )
        if array.shape[1] < 1:
            raise ValueError(f"case {index} has no time points")
        if channels is None:
            channels = array.shape[0]
        elif array.shape[0] != channels:
            raise ValueError(
                f"case {index} has {array.shape[0]} channels; expected {channels}"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"case {index} contains non-finite values")
        normalized.append(array)
    return normalized


def collection_lengths(values: Any) -> list[int]:
    """Return the per-case time lengths without changing any values."""
    return [case.shape[1] for case in _cases(values)]


def right_pad_collection(values: Any, target_length: int | None = None) -> np.ndarray:
    """Convert a UEA collection to contiguous ``float32 [cases, channels, time]``.

    If ``target_length`` is omitted, the longest case in this collection is
    used. Short cases are padded on the right with zeros. Passing a target
    length is useful for a model's fixed context window, where longer cases
    are deterministically truncated on the right.
    """
    cases = _cases(values)
    longest = max(case.shape[1] for case in cases)
    if target_length is None:
        target_length = longest
    if target_length < 1:
        raise ValueError("target_length must be positive")
    output = np.zeros((len(cases), cases[0].shape[0], target_length), dtype=np.float32)
    for index, case in enumerate(cases):
        copied = min(case.shape[1], target_length)
        output[index, :, :copied] = case[:, :copied]
    return np.ascontiguousarray(output)


def pad_train_test_to_common_length(train_values: Any, test_values: Any) -> tuple[np.ndarray, np.ndarray]:
    """Right-pad train and test collections to one dataset-level maximum length."""
    train_cases = _cases(train_values)
    test_cases = _cases(test_values)
    if train_cases[0].shape[0] != test_cases[0].shape[0]:
        raise ValueError(
            "train/test channel counts differ: "
            f"{train_cases[0].shape[0]} vs {test_cases[0].shape[0]}"
        )
    target_length = max(
        max(case.shape[1] for case in train_cases),
        max(case.shape[1] for case in test_cases),
    )
    return (
        right_pad_collection(train_cases, target_length),
        right_pad_collection(test_cases, target_length),
    )
