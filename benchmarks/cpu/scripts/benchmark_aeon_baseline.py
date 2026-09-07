"""CPU resource benchmark for aeon's ROCKET, Hydra, and QUANT classifiers.

The script separates fit time from prediction latency.  Fit covers feature
transformation plus classifier training; prediction covers feature
transformation plus `predict` on an already fitted classifier.  Inputs are
deterministic synthetic 3D NumPy arrays so the default `C=1, T=512` protocol
aligns with the existing MOMENT/Mantis CPU microbenchmark shape.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import psutil
import torch
from threadpoolctl import threadpool_limits

from tslimits_bench.benchmark import RSSMonitor, benchmark_forward


def classifier_factory(name: str, threads: int, seed: int) -> tuple[Callable[[], Any], dict[str, Any]]:
    if name == "rocket":
        from aeon.classification.convolution_based import RocketClassifier

        parameters = {"n_kernels": 10000, "n_jobs": threads, "random_state": seed}
        return lambda: RocketClassifier(**parameters), parameters
    if name == "hydra":
        from aeon.classification.convolution_based import HydraClassifier

        parameters = {"n_kernels": 8, "n_groups": 64, "n_jobs": threads, "random_state": seed}
        return lambda: HydraClassifier(**parameters), parameters
    if name == "quant":
        from aeon.classification.interval_based import QUANTClassifier

        parameters = {"interval_depth": 6, "quantile_divisor": 4, "random_state": seed}
        return lambda: QUANTClassifier(**parameters), parameters
    raise ValueError(f"Unsupported baseline {name!r}")


def fit_with_metrics(factory: Callable[[], Any], x_train: np.ndarray, y_train: np.ndarray, threads: int) -> tuple[Any, dict[str, float]]:
    process = psutil.Process()
    rss_before = process.memory_info().rss
    monitor = RSSMonitor()
    monitor.start()
    try:
        start = time.perf_counter()
        with threadpool_limits(limits=threads):
            classifier = factory()
            classifier.fit(x_train, y_train)
        elapsed_s = time.perf_counter() - start
    finally:
        monitor.stop()
    rss_after = process.memory_info().rss
    return classifier, {
        "fit_time_s": elapsed_s,
        "process_rss_before_mib": rss_before / (1024**2),
        "process_rss_after_mib": rss_after / (1024**2),
        "process_rss_peak_mib": monitor.peak / (1024**2),
    }


def feature_transform_callable(name: str, classifier: Any) -> Callable[[np.ndarray], Any]:
    """Return the fitted transform/scaling path, excluding final prediction."""
    if name == "rocket":
        return classifier.pipeline_[:-1].transform
    if name == "hydra":
        transformer = classifier._clf.named_steps["hydratransformer"]
        scaler = classifier._clf.named_steps["_sparsescaler"]
        return lambda values: scaler.transform(transformer.transform(values))
    if name == "quant":
        return classifier._transformer.transform
    raise ValueError(f"Unsupported baseline {name!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("rocket", "hydra", "quant"), required=True)
    parser.add_argument("--train-cases", type=int, default=100)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--classes", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--compile-warmup", action=argparse.BooleanOptionalAction, default=True, help="Exclude ROCKET's one-time Numba compilation from fit timing")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.train_cases < args.classes * 2:
        raise ValueError("--train-cases must include at least two examples per class")
    if args.channels < 1 or args.length < 8 or args.classes < 2 or args.threads < 1:
        raise ValueError("channels, length, classes, and threads must be valid positive values")

    import aeon
    import sklearn

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    rng = np.random.default_rng(args.seed)
    x_train = rng.standard_normal((args.train_cases, args.channels, args.length), dtype=np.float32)
    y_train = np.arange(args.train_cases, dtype=np.int64) % args.classes
    x_predict = rng.standard_normal((1, args.channels, args.length), dtype=np.float32)
    factory, parameters = classifier_factory(args.model, args.threads, args.seed)
    if args.compile_warmup and args.model == "rocket":
        # ROCKET's first feature transform JIT-compiles Numba kernels. It is a
        # process-start cost, not a per-dataset fit cost, so warm it separately.
        warm_cases = min(args.train_cases, max(4, args.classes * 2))
        with threadpool_limits(limits=args.threads):
            disposable = factory()
            disposable.fit(x_train[:warm_cases], y_train[:warm_cases]).predict(x_predict)
    classifier, fit = fit_with_metrics(factory, x_train, y_train, args.threads)

    transform = feature_transform_callable(args.model, classifier)
    with threadpool_limits(limits=args.threads):
        feature_transform = benchmark_forward(
            transform,
            x_predict,
            device=torch.device("cpu"),
            warmup=args.warmup,
            repetitions=args.repetitions,
        ).to_dict()
    feature_transform["metadata"]["measurement_scope"] = (
        "already-fitted aeon feature transformation only; includes the fitted preprocessing/scaling stages "
        "before the final estimator, and excludes classifier prediction, fit, imports, synthetic-data generation, and disk I/O"
    )

    with threadpool_limits(limits=args.threads):
        prediction = benchmark_forward(
            classifier.predict,
            x_predict,
            device=torch.device("cpu"),
            warmup=args.warmup,
            repetitions=args.repetitions,
        ).to_dict()
    prediction["metadata"]["measurement_scope"] = (
        "already-fitted aeon classifier.predict; includes feature transformation and final estimator prediction; "
        "excludes classifier fit, imports, synthetic-data generation, and disk I/O"
    )
    payload = {
        "schema_version": 1,
        "baseline": {
            "name": args.model,
            "parameters": parameters,
            "rocket_jit_compile_warmup_excluded": bool(args.compile_warmup and args.model == "rocket"),
            "aeon_version": aeon.__version__,
            "sklearn_version": sklearn.__version__,
        },
        "fit": fit,
        "feature_transform": feature_transform,
        "prediction": prediction,
        "synthetic_training_shape": list(x_train.shape),
        "synthetic_prediction_shape": list(x_predict.shape),
        "synthetic_labels": {"classes": args.classes, "balanced": True, "seed": args.seed},
        "protocol": {
            "cpu_threads": args.threads,
            "warmup": args.warmup,
            "repetitions": args.repetitions,
            "dtype": "float32",
            "platform": platform.platform(),
            "cpu": platform.processor() or "unknown",
            "fit_scope": "transformer fit plus classifier fit on synthetic training data",
        },
        "warning": (
            "Synthetic CPU baseline diagnostic. Fit time is strongly dependent on training-cases, channels, "
            "length, class count, and library version; use actual UEA splits for the paper's final table."
        ),
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
