#!/usr/bin/env python3
"""Evaluate ROCKET, Hydra, or QUANT on UEA .ts train/test files.

The classifiers receive the archive's native multivariate arrays directly:
no resampling, channel pooling, or external standardisation is applied.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import random
import sys
import time
import traceback

import numpy as np
import pandas as pd
from aeon.classification.convolution_based import HydraClassifier, RocketClassifier
from aeon.classification.interval_based import QUANTClassifier
from aeon.datasets import load_from_ts_file
from sklearn.metrics import accuracy_score
from threadpoolctl import threadpool_limits
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one aeon time-series classifier on UEA .ts files."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model", choices=("rocket", "hydra", "quant"), required=True)
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Dataset names. Defaults to every dataset found below --data-root.",
    )
    parser.add_argument(
        "--datasets-file",
        type=Path,
        help="One dataset name per line; blank lines and # comments are ignored.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def find_datasets(data_root: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for train_path in sorted(data_root.rglob("*_TRAIN.ts")):
        suffix = "_TRAIN.ts"
        name = train_path.name[: -len(suffix)]
        test_path = train_path.with_name(f"{name}_TEST.ts")
        if test_path.is_file():
            if name in found and found[name] != train_path.parent:
                raise RuntimeError(f"Dataset {name!r} appears in more than one directory.")
            found[name] = train_path.parent
    return found


def requested_datasets(args: argparse.Namespace, available: dict[str, Path]) -> list[str]:
    names: list[str] = []
    if args.datasets_file:
        for raw in args.datasets_file.read_text(encoding="utf-8").splitlines():
            value = raw.split("#", 1)[0].strip()
            if value:
                names.append(value)
    if args.datasets:
        names.extend(args.datasets)
    if not names:
        names = sorted(available)
    seen: set[str] = set()
    return [name for name in names if not (name in seen or seen.add(name))]


def make_classifier(model: str, seed: int, n_jobs: int):
    if model == "rocket":
        return RocketClassifier(n_kernels=10_000, n_jobs=n_jobs, random_state=seed), {
            "n_kernels": 10_000,
            "n_jobs": n_jobs,
            "random_state": seed,
        }
    if model == "hydra":
        return HydraClassifier(n_kernels=8, n_groups=64, n_jobs=n_jobs, random_state=seed), {
            "n_kernels": 8,
            "n_groups": 64,
            "n_jobs": n_jobs,
            "random_state": seed,
        }
    return QUANTClassifier(interval_depth=6, quantile_divisor=4, random_state=seed), {
        "interval_depth": 6,
        "quantile_divisor": 4,
        "random_state": seed,
    }


def load_dataset(directory: Path, dataset: str):
    train_path = directory / f"{dataset}_TRAIN.ts"
    test_path = directory / f"{dataset}_TEST.ts"
    x_train, y_train, meta_train = load_from_ts_file(train_path, return_meta_data=True)
    x_test, y_test, meta_test = load_from_ts_file(test_path, return_meta_data=True)
    if not meta_train.get("equallength") or not meta_test.get("equallength"):
        raise ValueError("unequal-length data are not supported by this runner")
    if meta_train.get("missing") or meta_test.get("missing"):
        raise ValueError("missing values are not supported by this runner")
    x_train = np.ascontiguousarray(np.asarray(x_train, dtype=np.float32))
    x_test = np.ascontiguousarray(np.asarray(x_test, dtype=np.float32))
    if x_train.ndim != 3 or x_test.ndim != 3:
        raise ValueError(f"expected 3D [cases, channels, timepoints] arrays; got {x_train.shape}, {x_test.shape}")
    if x_train.shape[1:] != x_test.shape[1:]:
        raise ValueError(f"train/test shape mismatch: {x_train.shape[1:]} vs {x_test.shape[1:]}")
    if not np.isfinite(x_train).all() or not np.isfinite(x_test).all():
        raise ValueError("non-finite values are not supported by this runner")
    if not set(np.unique(y_test)).issubset(set(np.unique(y_train))):
        raise ValueError("test labels are not a subset of training labels")
    return x_train, y_train, x_test, y_test


def version(name: str) -> str:
    return importlib.metadata.version(name)


def main() -> int:
    args = parse_args()
    if args.n_jobs < 1:
        raise ValueError("--n-jobs must be positive")
    if not args.data_root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {args.data_root}")

    available = find_datasets(args.data_root)
    if not available:
        raise FileNotFoundError(f"No *_TRAIN.ts / *_TEST.ts pairs found below {args.data_root}")
    names = requested_datasets(args, available)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.n_jobs)
    torch.set_num_interop_threads(1)

    result_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, str]] = []
    for dataset in names:
        if dataset not in available:
            failure_rows.append({"dataset": dataset, "reason": "dataset not found"})
            continue
        try:
            x_train, y_train, x_test, y_test = load_dataset(available[dataset], dataset)
            classifier, parameters = make_classifier(args.model, args.seed, args.n_jobs)
            with threadpool_limits(limits=args.n_jobs):
                started = time.perf_counter()
                classifier.fit(x_train, y_train)
                fit_seconds = time.perf_counter() - started
                started = time.perf_counter()
                prediction = classifier.predict(x_test)
                predict_seconds = time.perf_counter() - started
            result_rows.append(
                {
                    "dataset": dataset,
                    "model": args.model,
                    "seed": args.seed,
                    "n_train": x_train.shape[0],
                    "n_test": x_test.shape[0],
                    "channels": x_train.shape[1],
                    "length": x_train.shape[2],
                    "n_classes": len(np.unique(y_train)),
                    "accuracy": accuracy_score(y_test, prediction),
                    "fit_seconds": fit_seconds,
                    "predict_seconds": predict_seconds,
                    "aeon_version": version("aeon"),
                    "sklearn_version": version("scikit-learn"),
                    "numpy_version": version("numpy"),
                    "parameters": json.dumps(parameters, sort_keys=True),
                }
            )
            print(f"{dataset}: {result_rows[-1]['accuracy']:.4f}", flush=True)
        except Exception as exc:  # write every failure instead of silently changing coverage
            failure_rows.append({"dataset": dataset, "reason": f"{type(exc).__name__}: {exc}"})
            print(f"{dataset}: FAILED — {exc}", file=sys.stderr, flush=True)

    results = pd.DataFrame(result_rows)
    results.to_csv(args.output_dir / "results.csv", index=False)
    failures = pd.DataFrame(failure_rows, columns=("dataset", "reason"))
    failures.to_csv(args.output_dir / "failures.csv", index=False)
    if not results.empty:
        print(f"Mean accuracy over {len(results)} completed datasets: {results['accuracy'].mean():.4f}")
    if failure_rows:
        print(f"{len(failure_rows)} dataset(s) failed; see {args.output_dir / 'failures.csv'}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        traceback.print_exception(exc)
        raise SystemExit(2)
