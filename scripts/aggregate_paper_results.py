#!/usr/bin/env python3
"""Aggregate standardized frozen-representation results for the paper.

Expected layout:

    INPUT_ROOT/
      MOMENT-Small/
        pretrained.csv
        random_seed0.csv
        random_seed7.csv
        random_seed42.csv
        random_seed123.csv
        random_seed2026.csv
      Mantis/
        ...

Each input CSV may use either direct probe columns (Ridge, LogReg, RBF_SVM,
RF) or runner-style accuracy columns (Ridge_acc, LogReg_acc, SVM_acc,
RF_acc). A long form with dataset, probe/classifier, and accuracy/acc columns
is also accepted. Only datasets with all four valid probes in the pretrained
file and every random-seed file enter the paired analysis.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


PROBES = ("Ridge", "LogReg", "RBF_SVM", "RF")
DEFAULT_SEEDS = (0, 7, 42, 123, 2026)
DATASET_COLUMN_NAMES = {"dataset", "datasetname", "name", "unnamed0"}
LONG_PROBE_COLUMN_NAMES = {"probe", "classifier"}
LONG_ACCURACY_COLUMN_NAMES = {"accuracy", "acc", "score"}
SUMMARY_DATASET_NAMES = {"mean", "average", "all"}


def _token(value: object) -> str:
    """Normalize a header or probe label for tolerant but unambiguous matching."""
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def canonical_probe(value: object) -> str | None:
    """Map supported wide-column and long-form names to paper probe names."""
    token = _token(value)
    aliases = {
        "ridge": "Ridge",
        "ridgeacc": "Ridge",
        "ridgeaccuracy": "Ridge",
        "ridgeclassifier": "Ridge",
        "logreg": "LogReg",
        "logregacc": "LogReg",
        "logregaccuracy": "LogReg",
        "logisticregression": "LogReg",
        "logisticregressionacc": "LogReg",
        "logisticregressionaccuracy": "LogReg",
        "rbfsvm": "RBF_SVM",
        "rbfsvmacc": "RBF_SVM",
        "rbfsvmaccuracy": "RBF_SVM",
        "svm": "RBF_SVM",
        "svmacc": "RBF_SVM",
        "svmaccuracy": "RBF_SVM",
        "rbf": "RBF_SVM",
        "rbfacc": "RBF_SVM",
        "rf": "RF",
        "rfacc": "RF",
        "rfaccuracy": "RF",
        "randomforest": "RF",
        "randomforestacc": "RF",
        "randomforestaccuracy": "RF",
    }
    return aliases.get(token)


def _find_named_column(columns: Iterable[object], candidates: set[str]) -> object | None:
    matches = [column for column in columns if _token(column) in candidates]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous columns for {sorted(candidates)}: {matches}")
    return matches[0] if matches else None


def _dataset_column(frame: pd.DataFrame, source: Path) -> object:
    column = _find_named_column(frame.columns, DATASET_COLUMN_NAMES)
    if column is None:
        raise ValueError(
            f"{source}: expected a dataset column named Dataset/dataset "
            "(or a CSV index column such as Unnamed: 0)."
        )
    return column


def _clean_dataset_values(values: pd.Series, source: Path) -> pd.Series:
    cleaned = values.astype("string").str.strip()
    if cleaned.isna().any() or (cleaned == "").any():
        raise ValueError(f"{source}: dataset names must be non-empty.")
    return cleaned


def _validate_wide_frame(frame: pd.DataFrame, source: Path) -> pd.DataFrame:
    if frame.index.duplicated().any():
        duplicates = sorted(frame.index[frame.index.duplicated()].unique().tolist())
        raise ValueError(f"{source}: duplicate dataset rows: {duplicates}")
    selected: dict[str, object] = {}
    for column in frame.columns:
        probe = canonical_probe(column)
        if probe is None:
            continue
        if probe in selected:
            raise ValueError(
                f"{source}: multiple columns map to {probe}: "
                f"{selected[probe]!r} and {column!r}"
            )
        selected[probe] = column
    missing = [probe for probe in PROBES if probe not in selected]
    if missing:
        raise ValueError(
            f"{source}: missing required probe columns {missing}. "
            "Accepted names include Ridge, LogReg, RBF_SVM/SVM, RF and *_acc."
        )
    normalized = frame.loc[:, [selected[probe] for probe in PROBES]].copy()
    normalized.columns = list(PROBES)
    for probe in PROBES:
        normalized[probe] = pd.to_numeric(normalized[probe], errors="coerce")
    normalized.index.name = "dataset"
    return normalized.astype(float)


def normalize_results_csv(source: Path) -> pd.DataFrame:
    """Read one accepted raw-runner CSV into Dataset x paper-probe accuracy."""
    try:
        raw = pd.read_csv(source)
    except Exception as exc:
        raise ValueError(f"Could not read {source}: {type(exc).__name__}: {exc}") from exc
    if raw.empty:
        raise ValueError(f"{source}: contains no result rows.")

    dataset_column = _dataset_column(raw, source)
    raw = raw.copy()
    raw[dataset_column] = _clean_dataset_values(raw[dataset_column], source)
    raw = raw[~raw[dataset_column].str.casefold().isin(SUMMARY_DATASET_NAMES)].copy()
    if raw.empty:
        raise ValueError(f"{source}: contains only summary rows, not datasets.")

    probe_column = _find_named_column(raw.columns, LONG_PROBE_COLUMN_NAMES)
    accuracy_column = _find_named_column(raw.columns, LONG_ACCURACY_COLUMN_NAMES)
    if (probe_column is None) != (accuracy_column is None):
        raise ValueError(
            f"{source}: long-form results require both a probe/classifier column "
            "and an accuracy/acc column."
        )
    if probe_column is not None and accuracy_column is not None:
        long = raw.loc[:, [dataset_column, probe_column, accuracy_column]].copy()
        long.columns = ("dataset", "probe", "accuracy")
        long["probe"] = long["probe"].map(canonical_probe)
        unsupported = long["probe"].isna()
        if unsupported.any():
            values = sorted(raw.loc[unsupported, probe_column].astype(str).unique().tolist())
            raise ValueError(f"{source}: unsupported probe labels in long form: {values}")
        if long.duplicated(["dataset", "probe"]).any():
            raise ValueError(f"{source}: duplicate dataset/probe rows in long form.")
        normalized = long.pivot(index="dataset", columns="probe", values="accuracy")
        normalized = normalized.reindex(columns=PROBES)
        if normalized.shape[1] != len(PROBES) or normalized.isna().all(axis=0).any():
            missing = [
                probe
                for probe in PROBES
                if probe not in normalized.columns or normalized[probe].isna().all()
            ]
            raise ValueError(f"{source}: long form is missing required probes: {missing}")
        normalized.index.name = "dataset"
        for probe in PROBES:
            normalized[probe] = pd.to_numeric(normalized[probe], errors="coerce")
        return normalized.astype(float)

    wide = raw.set_index(dataset_column)
    return _validate_wide_frame(wide, source)


def _row_reason(frame: pd.DataFrame, dataset: str, label: str) -> str | None:
    if dataset not in frame.index:
        return f"{label}: missing dataset"
    values = frame.loc[dataset, list(PROBES)]
    invalid = [
        probe
        for probe, value in values.items()
        if not isinstance(value, (int, float, np.floating, np.integer))
        or not math.isfinite(float(value))
    ]
    if invalid:
        return f"{label}: non-finite accuracy for {', '.join(invalid)}"
    return None


def _bootstrap_ci(
    deltas_pp: np.ndarray, repetitions: int, rng: np.random.Generator
) -> tuple[float, float]:
    if repetitions < 1:
        raise ValueError("--bootstrap-repetitions must be positive.")
    indices = rng.integers(0, len(deltas_pp), size=(repetitions, len(deltas_pp)))
    means = deltas_pp[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def _wilcoxon(deltas_pp: np.ndarray) -> tuple[float, float]:
    """Compute the two-sided signed-rank test, including the all-ties case."""
    try:
        from scipy.stats import wilcoxon
    except ImportError as exc:
        raise RuntimeError(
            "SciPy is required for Table-1 Wilcoxon/Holm/bootstrap inference. "
            "Install scipy or use --skip-inference for aggregate-only output."
        ) from exc
    if np.all(np.isclose(deltas_pp, 0.0, rtol=0.0, atol=1e-12)):
        return 0.0, 1.0
    try:
        result = wilcoxon(deltas_pp, alternative="two-sided", method="auto")
    except TypeError:  # Compatibility with older SciPy releases.
        result = wilcoxon(deltas_pp, alternative="two-sided")
    return float(result.statistic), float(result.pvalue)


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Apply Holm's step-down correction to finite p-values."""
    finite = sorted(
        ((name, value) for name, value in p_values.items() if math.isfinite(value)),
        key=lambda item: item[1],
    )
    adjusted: dict[str, float] = {name: float("nan") for name in p_values}
    previous = 0.0
    total = len(finite)
    for index, (name, value) in enumerate(finite):
        candidate = min(1.0, value * (total - index))
        previous = max(previous, candidate)
        adjusted[name] = previous
    return adjusted


def _source_paths(instance_dir: Path, seeds: tuple[int, ...]) -> dict[str, Path]:
    paths = {"pretrained": instance_dir / "pretrained.csv"}
    paths.update(
        {f"random_seed{seed}": instance_dir / f"random_seed{seed}.csv" for seed in seeds}
    )
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        rendered = "\n  ".join(str(path) for path in missing)
        raise FileNotFoundError(f"{instance_dir.name}: missing required CSV file(s):\n  {rendered}")
    return paths


def aggregate_instance(
    instance_dir: Path,
    seeds: tuple[int, ...],
    *,
    bootstrap_repetitions: int,
    rng: np.random.Generator,
    skip_inference: bool,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Aggregate one model instance and retain all inclusion/exclusion evidence."""
    paths = _source_paths(instance_dir, seeds)
    frames = {name: normalize_results_csv(path) for name, path in paths.items()}
    random_names = [f"random_seed{seed}" for seed in seeds]
    datasets = sorted(set().union(*(set(frame.index) for frame in frames.values())))

    availability: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    pretrained_scores: list[float] = []
    random_scores: list[float] = []
    random_seed_scores: dict[int, list[float]] = {seed: [] for seed in seeds}
    pretrained_probe_scores: dict[str, list[float]] = {probe: [] for probe in PROBES}
    random_probe_scores: dict[str, list[float]] = {probe: [] for probe in PROBES}

    for dataset in datasets:
        reasons: list[str] = []
        pretrained_reason = _row_reason(frames["pretrained"], dataset, "pretrained")
        if pretrained_reason:
            reasons.append(pretrained_reason)
        seed_reasons: dict[int, str | None] = {}
        for seed, name in zip(seeds, random_names):
            seed_reasons[seed] = _row_reason(frames[name], dataset, name)
            if seed_reasons[seed]:
                reasons.append(seed_reasons[seed])
        included = not reasons
        availability_row: dict[str, Any] = {
            "instance": instance_dir.name,
            "dataset": dataset,
            "included": included,
            "reason": " | ".join(reasons) if reasons else "",
            "pretrained_available": pretrained_reason is None,
        }
        for seed in seeds:
            availability_row[f"random_seed{seed}_available"] = seed_reasons[seed] is None
        availability.append(availability_row)
        if not included:
            continue

        pretrained = frames["pretrained"].loc[dataset, list(PROBES)].to_numpy(dtype=float)
        random_by_seed = np.stack(
            [
                frames[name].loc[dataset, list(PROBES)].to_numpy(dtype=float)
                for name in random_names
            ]
        )
        random_average = random_by_seed.mean(axis=0)
        pretrained_score = float(pretrained.mean())
        random_score = float(random_average.mean())
        pretrained_scores.append(pretrained_score)
        random_scores.append(random_score)
        for seed, row in zip(seeds, random_by_seed):
            random_seed_scores[seed].append(float(row.mean()))

        paired_row: dict[str, Any] = {
            "instance": instance_dir.name,
            "dataset": dataset,
            "pretrained_aggregate": pretrained_score,
            "random_aggregate": random_score,
            "delta_pp": (pretrained_score - random_score) * 100.0,
        }
        for index, probe in enumerate(PROBES):
            pretrained_probe_scores[probe].append(float(pretrained[index]))
            random_probe_scores[probe].append(float(random_average[index]))
            paired_row[f"pretrained_{probe}"] = float(pretrained[index])
            paired_row[f"random_{probe}"] = float(random_average[index])
            paired_row[f"delta_{probe}_pp"] = float(
                (pretrained[index] - random_average[index]) * 100.0
            )
        paired_rows.append(paired_row)

    if not paired_rows:
        raise ValueError(
            f"{instance_dir.name}: no datasets have all four valid probes in pretrained "
            "and every random-seed result."
        )

    deltas_pp = (np.asarray(pretrained_scores) - np.asarray(random_scores)) * 100.0
    if skip_inference:
        statistic = raw_p = ci_low = ci_high = float("nan")
        inference_status = "skipped"
    else:
        statistic, raw_p = _wilcoxon(deltas_pp)
        ci_low, ci_high = _bootstrap_ci(deltas_pp, bootstrap_repetitions, rng)
        inference_status = "computed"
    seed_means = np.asarray([np.mean(random_seed_scores[seed]) for seed in seeds], dtype=float)
    random_seed_sd = float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else 0.0
    table_row: dict[str, Any] = {
        "instance": instance_dir.name,
        "n_paired": len(paired_rows),
        "pretrained_mean": float(np.mean(pretrained_scores)),
        "random_mean": float(np.mean(random_scores)),
        "random_seed_sd": random_seed_sd,
        "mean_delta_pp": float(np.mean(deltas_pp)),
        "ci_low_pp": ci_low,
        "ci_high_pp": ci_high,
        "median_delta_pp": float(np.median(deltas_pp)),
        "wins": int(np.sum(deltas_pp > 1e-12)),
        "ties": int(np.sum(np.isclose(deltas_pp, 0.0, rtol=0.0, atol=1e-12))),
        "losses": int(np.sum(deltas_pp < -1e-12)),
        "wilcoxon_statistic": statistic,
        "wilcoxon_pvalue": raw_p,
        "holm_pvalue": float("nan"),
        "bootstrap_repetitions": bootstrap_repetitions if not skip_inference else 0,
        "inference_status": inference_status,
    }
    classifier_rows = [
        {
            "instance": instance_dir.name,
            "probe": probe,
            "n_paired": len(paired_rows),
            "pretrained_mean": float(np.mean(pretrained_probe_scores[probe])),
            "random_mean": float(np.mean(random_probe_scores[probe])),
            "mean_delta_pp": float(
                (np.mean(pretrained_probe_scores[probe]) - np.mean(random_probe_scores[probe]))
                * 100.0
            ),
            "median_delta_pp": float(
                np.median(
                    (np.asarray(pretrained_probe_scores[probe])
                    - np.asarray(random_probe_scores[probe]))
                    * 100.0
                )
            ),
        }
        for probe in PROBES
    ]
    return table_row, classifier_rows, paired_rows, availability


def _instance_directories(input_root: Path, requested: list[str] | None) -> list[Path]:
    if requested:
        directories = [input_root / name for name in requested]
    else:
        directories = sorted(path.parent for path in input_root.glob("*/pretrained.csv"))
    missing = [directory for directory in directories if not directory.is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing instance directory/directories: {missing}")
    if not directories:
        raise FileNotFoundError(
            f"No instance directories found below {input_root}; expected <instance>/pretrained.csv."
        )
    names = [directory.name for directory in directories]
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate instance names requested: {names}")
    return directories


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--instances",
        nargs="*",
        help="Optional instance directory names. Defaults to all immediate directories with pretrained.csv.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="Random encoder seeds; defaults to the five paper seeds.",
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument(
        "--fail-on-incomplete",
        action="store_true",
        help="Fail if any dataset lacks a complete pretrained/random paired result.",
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Write paired aggregates without Wilcoxon/Holm/bootstrap statistics.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input_root.is_dir():
        raise FileNotFoundError(f"--input-root does not exist: {args.input_root}")
    seeds = tuple(args.seeds)
    if len(seeds) != len(set(seeds)):
        raise ValueError("--seeds must not contain duplicates.")
    if not seeds:
        raise ValueError("--seeds must not be empty.")
    if args.bootstrap_repetitions < 1:
        raise ValueError("--bootstrap-repetitions must be positive.")
    if not args.skip_inference:
        try:
            import scipy  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "SciPy is required for Table-1 Wilcoxon/Holm/bootstrap inference. "
                "Install scipy or pass --skip-inference for aggregate-only output."
            ) from exc

    directories = _instance_directories(args.input_root, args.instances)
    rng = np.random.default_rng(args.bootstrap_seed)
    table_rows: list[dict[str, Any]] = []
    classifier_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    availability_rows: list[dict[str, Any]] = []
    for directory in directories:
        table, classifier, paired, availability = aggregate_instance(
            directory,
            seeds,
            bootstrap_repetitions=args.bootstrap_repetitions,
            rng=rng,
            skip_inference=args.skip_inference,
        )
        table_rows.append(table)
        classifier_rows.extend(classifier)
        paired_rows.extend(paired)
        availability_rows.extend(availability)

    if args.fail_on_incomplete:
        incomplete = [row for row in availability_rows if not row["included"]]
        if incomplete:
            preview = ", ".join(
                f"{row['instance']}/{row['dataset']}" for row in incomplete[:10]
            )
            raise ValueError(
                f"{len(incomplete)} dataset result(s) are incomplete; see availability.csv after "
                f"running without --fail-on-incomplete. First entries: {preview}"
            )

    adjusted = holm_adjust({row["instance"]: float(row["wilcoxon_pvalue"]) for row in table_rows})
    for row in table_rows:
        row["holm_pvalue"] = adjusted[row["instance"]]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(table_rows).to_csv(
        args.output_dir / "table1.csv", index=False, float_format="%.10g"
    )
    pd.DataFrame(classifier_rows).to_csv(
        args.output_dir / "classifier_deltas.csv", index=False, float_format="%.10g"
    )
    pd.DataFrame(paired_rows).to_csv(
        args.output_dir / "paired_dataset_scores.csv", index=False, float_format="%.10g"
    )
    pd.DataFrame(availability_rows).to_csv(args.output_dir / "availability.csv", index=False)
    metadata = {
        "probes": list(PROBES),
        "random_encoder_seeds": list(seeds),
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "bootstrap_seed": args.bootstrap_seed,
        "inference": (
            "skipped"
            if args.skip_inference
            else "two-sided Wilcoxon plus Holm and percentile bootstrap"
        ),
        "paired_rule": "all four probes must be finite in pretrained and every random-seed CSV",
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote aggregates for {len(table_rows)} instance(s) to {args.output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
