"""
Mantis Zero-Shot Evaluation on UEA Multivariate Time Series Datasets
- Loads datasets using load_from_tsfile
- Extracts features using Mantis foundation model (pretrained OR random init)
- Evaluates with SVM (GridSearchCV), LogisticRegression, RidgeClassifier, RandomForest
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import warnings
import argparse
from pathlib import Path

from aeon.datasets import load_from_ts_file
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ─── Add mantis source to path ───
MANTIS_REPO = os.path.expanduser("~/workspace_pinar/mantis")
sys.path.insert(0, MANTIS_REPO)

from src.mantis.architecture import MantisV2, Mantis8M
from src.mantis.trainer.trainer import MantisTrainer


def set_seed(seed: int):
    """Set all relevant seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resize_to_multiple_of_32(X: np.ndarray, target_len: int = 512) -> np.ndarray:
    """Resize time series to a length that is a multiple of 32."""
    X_tensor = torch.tensor(X, dtype=torch.float)
    X_resized = F.interpolate(X_tensor, size=target_len, mode='linear', align_corners=False)
    return X_resized.numpy()


def get_classifiers():
    """Return dict of classifier name -> (pipeline, param_grid) for GridSearchCV."""
    classifiers = {}

    # SVM with GridSearchCV
    classifiers["SVM"] = {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", random_state=42))
        ]),
        "param_grid": {
            "svm__C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "svm__gamma": ["scale", "auto"],
        },
        "use_gridsearch": False,
    }

    # Logistic Regression with GridSearchCV
    classifiers["LogReg"] = {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(max_iter=2000, random_state=42))
        ]),
        "param_grid": {
            "logreg__C": [0.01, 0.1, 1.0, 10.0, 100.0],
        },
        "use_gridsearch": True,
    }

    # RidgeClassifierCV (built-in CV, no GridSearch needed)
    classifiers["Ridge"] = {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)))
        ]),
        "param_grid": None,
        "use_gridsearch": False,
    }

    # Random Forest
    classifiers["RF"] = {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(n_estimators=200, random_state=42))
         ]),
        "param_grid": None,
        "use_gridsearch": False,
    }

    return classifiers


def evaluate_dataset(dataset_name, data_path, model, device, target_len=512):
    """Evaluate Mantis on a single dataset. Returns dict of classifier -> accuracy."""
    train_file = os.path.join(data_path, dataset_name, f"{dataset_name}_TRAIN.ts")
    test_file = os.path.join(data_path, dataset_name, f"{dataset_name}_TEST.ts")

    if not os.path.exists(train_file) or not os.path.exists(test_file):
        print(f"  [SKIP] Missing .ts files for {dataset_name}")
        return None

    # Load data
    X_train, y_train = load_from_ts_file(train_file)
    X_test, y_test = load_from_ts_file(test_file)

    # Resize to target_len (must be multiple of 32)
    X_train = resize_to_multiple_of_32(X_train, target_len)
    X_test = resize_to_multiple_of_32(X_test, target_len)

    # Extract features with Mantis
    Z_train = model.transform(X_train)
    Z_test = model.transform(X_test)

    # Evaluate classifiers
    classifiers = get_classifiers()
    results = {}

    for clf_name, clf_config in classifiers.items():
        try:
            if clf_config["use_gridsearch"]:
                gs = GridSearchCV(
                    clf_config["pipeline"],
                    clf_config["param_grid"],
                    cv=5,
                    scoring="accuracy",
                    n_jobs=-1,
                    refit=True,
                )
                gs.fit(Z_train, y_train)
                y_pred = gs.predict(Z_test)
                acc = np.mean(y_test == y_pred)
                print(f"  {clf_name}: {acc:.4f} (best params: {gs.best_params_})")
            else:
                pipe = clf_config["pipeline"]
                pipe.fit(Z_train, y_train)
                y_pred = pipe.predict(Z_test)
                acc = np.mean(y_test == y_pred)
                print(f"  {clf_name}: {acc:.4f}")

            results[clf_name] = acc
        except Exception as e:
            print(f"  {clf_name}: FAILED — {e}")
            results[clf_name] = np.nan

    return results


def main():
    parser = argparse.ArgumentParser(description="Mantis zero-shot eval on UEA datasets")
    parser.add_argument(
        "--data_path",
        type=str,
        default=os.path.expanduser(
            "/home/pinar/workspace_pinar/tsfm/cross_corr/Multivariate_ts_length_fixed"
        ),
        help="Path to UEA dataset root (each subdirectory = one dataset)",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--target_len", type=int, default=512)
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        default=None,
        help="Specific dataset names to evaluate (default: all found in data_path)",
    )
    parser.add_argument(
        "--init",
        type=str,
        default="random",
        choices=["pretrained", "random"],
        help="Use pretrained weights from HuggingFace or random initialization",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (only relevant for --init random)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Mantis8M",
        choices=["MantisV2", "Mantis8M"],
        help="Which Mantis architecture to use",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file for results (default: auto-named based on init/seed/model)",
    )
    args = parser.parse_args()

    # ─── Set seed BEFORE building the network so random init is reproducible ───
    set_seed(args.seed)

    # ─── Load Mantis ───
    print(f"Loading {args.model} ({args.init} init, seed={args.seed})...")

    if args.model == "MantisV2":
        network = MantisV2(device=args.device)
        hf_id = "paris-noah/MantisV2"
    else:
        network = Mantis8M(device=args.device)
        hf_id = "paris-noah/Mantis-8M"

    if args.init == "pretrained":
        network = network.from_pretrained(hf_id)
        print(f"  Loaded pretrained weights from {hf_id}")
    else:
        # Random init: keep the freshly instantiated weights, just move to device + eval mode
        network = network.to(args.device)
        network.eval()
        print(f"  Using random initialization (no pretrained weights loaded)")

    model = MantisTrainer(device=args.device, network=network)
    print("Model ready.\n")

    # ─── Discover datasets ───
    if args.datasets:
        dataset_names = args.datasets
    else:
        dataset_names = sorted([
            d for d in os.listdir(args.data_path)
            if os.path.isdir(os.path.join(args.data_path, d))
        ])

    print(f"Found {len(dataset_names)} datasets to evaluate.\n")

    # ─── Run evaluation ───
    all_results = {}
    for i, ds_name in enumerate(dataset_names, 1):
        print(f"[{i}/{len(dataset_names)}] {ds_name}")
        try:
            results = evaluate_dataset(
                ds_name, args.data_path, model, args.device, args.target_len
            )
            if results is not None:
                all_results[ds_name] = results
        except Exception as e:
            print(f"  [ERROR] {e}")
        print()

    # ─── Save results ───
    if all_results:
        import pandas as pd

        # Auto-generate output filename if not provided
        if args.output is None:
            if args.init == "random":
                args.output = f"{args.model}_random_seed{args.seed}_results.csv"
            else:
                args.output = f"{args.model}_pretrained_results.csv"

        df = pd.DataFrame(all_results).T
        df.index.name = "Dataset"
        df.loc["MEAN"] = df.mean()
        df.to_csv(args.output, float_format="%.4f")
        print(f"\nResults saved to {args.output}")
        print(f"\n{'='*60}")
        print(df.to_string(float_format=lambda x: f"{x:.4f}"))
        print(f"{'='*60}")
    else:
        print("No results to save.")


if __name__ == "__main__":
    main()