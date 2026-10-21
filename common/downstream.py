"""The frozen-representation probe protocol reported in the camera-ready paper.

Every foundation-model runner uses this module so that classifier selection is
identical across model families, conditions, datasets, and random encoder
seeds.  Hyperparameters are selected exclusively on the UEA training split
with the three-fold stratified procedure documented in Appendix B (Table 5).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


PROBE_ORDER = ("Ridge", "LogReg", "RBF_SVM", "RF")
CV_RANDOM_STATE = 42
N_SPLITS = 3


def _cv() -> StratifiedKFold:
    """Return the one CV splitter used by every reported probe."""
    return StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=CV_RANDOM_STATE,
    )


def probe_configurations() -> Mapping[str, tuple[Pipeline, Mapping[str, list[Any]]]]:
    """Build the exact Appendix B / Table 5 classifier grids.

    Pipelines deliberately contain ``StandardScaler`` only where stated in
    the paper.  In particular, random forest receives the frozen features
    directly.
    """
    alpha_or_c = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 1e2, 1e3]
    return {
        "Ridge": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("ridge", RidgeClassifier()),
                ]
            ),
            {"ridge__alpha": alpha_or_c},
        ),
        "LogReg": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "logreg",
                        LogisticRegression(
                            max_iter=5000,
                            solver="lbfgs",
                            random_state=CV_RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            {"logreg__C": alpha_or_c},
        ),
        "RBF_SVM": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("svm", SVC(kernel="rbf", random_state=CV_RANDOM_STATE)),
                ]
            ),
            {
                "svm__C": [1e-2, 1e-1, 1.0, 10.0, 1e2],
                "svm__gamma": ["scale", "auto"],
            },
        ),
        "RF": (
            Pipeline(
                [
                    (
                        "rf",
                        RandomForestClassifier(
                            n_estimators=200,
                            max_features="sqrt",
                            random_state=CV_RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            {
                "rf__max_depth": [None, 10],
                "rf__min_samples_leaf": [1, 2],
            },
        ),
    }


def _as_2d_features(features: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(features)
    if array.ndim < 2:
        raise ValueError(f"{name} must have a sample dimension and at least one feature dimension; got {array.shape}")
    if array.ndim > 2:
        array = array.reshape(array.shape[0], -1)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _validate_inputs(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_x = _as_2d_features(train_features, name="train_features")
    test_x = _as_2d_features(test_features, name="test_features")
    train_y = np.asarray(train_labels).reshape(-1)
    test_y = np.asarray(test_labels).reshape(-1)
    if train_x.shape[0] != train_y.shape[0]:
        raise ValueError("train feature and label counts differ")
    if test_x.shape[0] != test_y.shape[0]:
        raise ValueError("test feature and label counts differ")
    if train_x.shape[1] != test_x.shape[1]:
        raise ValueError("train and test feature dimensions differ")
    classes, counts = np.unique(train_y, return_counts=True)
    if len(classes) < 2:
        raise ValueError("at least two training classes are required")
    if counts.min() < N_SPLITS:
        raise ValueError(
            f"the reported {N_SPLITS}-fold stratified CV requires at least "
            f"{N_SPLITS} examples in every training class; smallest class has {counts.min()}"
        )
    return train_x, train_y, test_x, test_y


def evaluate_probes(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    *,
    n_jobs: int | None = 1,
) -> dict[str, dict[str, Any]]:
    """Select and evaluate every downstream probe with training data only.

    The return value contains unrounded accuracy, selected parameters, and a
    readable error for each probe.  Serial CV is the default because the paper
    does not prescribe worker parallelism, and process workers can be fragile
    around PyTorch/OpenMP runtimes.  A failed probe is represented explicitly
    rather than silently fitted without the documented cross-validation step.
    """
    results: dict[str, dict[str, Any]] = {}
    try:
        train_x, train_y, test_x, test_y = _validate_inputs(
            train_features, train_labels, test_features, test_labels
        )
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        return {
            name: {
                "accuracy": float("nan"),
                "macro_f1": float("nan"),
                "best_params": None,
                "error": reason,
            }
            for name in PROBE_ORDER
        }

    for name, (pipeline, parameter_grid) in probe_configurations().items():
        try:
            search = GridSearchCV(
                estimator=pipeline,
                param_grid=parameter_grid,
                cv=_cv(),
                scoring="accuracy",
                n_jobs=n_jobs,
                refit=True,
                error_score="raise",
            )
            search.fit(train_x, train_y)
            prediction = search.predict(test_x)
            results[name] = {
                "accuracy": float(accuracy_score(test_y, prediction)),
                "macro_f1": float(f1_score(test_y, prediction, average="macro")),
                "best_params": dict(search.best_params_),
                "error": None,
            }
        except Exception as exc:  # retain explicit per-probe availability information
            results[name] = {
                "accuracy": float("nan"),
                "macro_f1": float("nan"),
                "best_params": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
    return results
