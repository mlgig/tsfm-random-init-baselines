"""Generate an explicit UniTS classification-task YAML from raw UEA splits.

The paper names the UEA collections but does not preserve the
historical UniTS task recipe.  This helper derives only structural metadata
from the public ``.ts`` files: channel count, the maximum sequence length
across the official TRAIN/TEST pair, and the number of training labels.  It
does not inspect feature values or select a downstream classifier.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
UNITS_UNAVAILABLE_DATASETS = frozenset(
    {"DuckDuckGeese", "InsectWingbeat", "PEMS-SF"}
)


def _paper_datasets(*, include_unavailable: bool) -> list[str]:
    """Return the canonical paper list, optionally including UniTS exclusions."""
    names = [
        line.strip()
        for line in (REPOSITORY_ROOT / "data" / "paper_all_30_datasets.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if include_unavailable:
        return names

    return [name for name in names if name not in UNITS_UNAVAILABLE_DATASETS]


def _read_split(path: Path) -> tuple[int, int, Any]:
    """Return ``(channels, maximum_length, labels)`` using UniTS's parser."""
    try:
        from sktime.datasets import load_from_tsfile_to_dataframe
    except ModuleNotFoundError as exc:  # provide a setup-specific error
        raise RuntimeError("Install the UniTS environment first: bash UniTS/setup.sh") from exc

    frame, labels = load_from_tsfile_to_dataframe(str(path), return_separate_X_and_y=True)
    if frame.empty or frame.shape[1] == 0:
        raise ValueError(f"{path} contains no time-series cases or channels")

    lengths = [len(series) for column in frame.columns for series in frame[column]]
    if not lengths or min(lengths) < 1:
        raise ValueError(f"{path} contains an empty time series")
    return frame.shape[1], max(lengths), labels


def _task(name: str, data_root: Path) -> dict[str, Any]:
    dataset_dir = data_root / name
    train_path = dataset_dir / f"{name}_TRAIN.ts"
    test_path = dataset_dir / f"{name}_TEST.ts"
    missing = [str(path) for path in (train_path, test_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing UEA split(s): " + ", ".join(missing))

    train_channels, train_length, train_labels = _read_split(train_path)
    test_channels, test_length, _ = _read_split(test_path)
    if train_channels != test_channels:
        raise ValueError(
            f"{name} has {train_channels} TRAIN channels but {test_channels} TEST channels"
        )
    num_classes = len({str(label) for label in train_labels})
    if num_classes < 2:
        raise ValueError(f"{name} has fewer than two training classes")

    # UniTS's collator pads/truncates to ``seq_len``.  This dataset-level
    # maximum therefore preserves every observed value from either split.
    return {
        "task_name": "classification",
        "dataset": name,
        "data": "UEA",
        "embed": "timeF",
        "root_path": str(dataset_dir.resolve()),
        "seq_len": max(train_length, test_length),
        "label_len": 0,
        "pred_len": 0,
        "enc_in": train_channels,
        "num_class": num_classes,
        "c_out": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="UEA archive root")
    parser.add_argument("--output", type=Path, required=True, help="task YAML to create")
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="explicit dataset names; defaults to UniTS's 27 paper-eligible collections",
    )
    parser.add_argument(
        "--include-paper-unavailable",
        action="store_true",
        help="when using the default list, include the three collections marked unavailable for UniTS",
    )
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    if not data_root.is_dir():
        parser.error(f"UEA data root does not exist: {data_root}")
    datasets = args.datasets or _paper_datasets(include_unavailable=args.include_paper_unavailable)
    if len(set(datasets)) != len(datasets):
        parser.error("--datasets contains duplicate names")

    tasks = {name: _task(name, data_root) for name in datasets}
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump({"task_dataset": tasks}, sort_keys=False), encoding="utf-8")
    print(f"Wrote {len(tasks)} UniTS task definitions to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
