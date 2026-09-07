"""
MOMENT Zero-Shot Evaluation on UEA Multivariate Time Series Datasets
- Loads datasets using load_from_tsfile
- Extracts features using MOMENT foundation model (pretrained OR random init via Xavier reset)
- Evaluates with SVM (GridSearchCV), LogisticRegression, RidgeClassifier, RandomForest

Random init uses reset_weights (Xavier uniform for Linear/Conv, ones+zeros for LayerNorm)
— same strategy used for the Mantis random-init baseline, for fair comparison.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import argparse
from pathlib import Path
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from aeon.datasets import load_from_ts_file as aeon_load
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from momentfm import MOMENTPipeline

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════
# Seed + Random Init helpers
# ═══════════════════════════════════════════════════════════

def set_seed(seed: int):
    """Set all relevant seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def reset_weights(m):
    """
    Re-initialize a module's weights randomly.
    - Linear / Conv1d / Conv2d: Xavier uniform, bias zeroed
    - LayerNorm: weight=1, bias=0
    - BatchNorm: weight=1, bias=0 (running stats reset)
    - Embedding: Xavier uniform
    """
    if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.LayerNorm):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
        m.running_mean.zero_()
        m.running_var.fill_(1.0)
    elif isinstance(m, nn.Embedding):
        nn.init.xavier_uniform_(m.weight.data)


# ═══════════════════════════════════════════════════════════
# Data utilities
# ═══════════════════════════════════════════════════════════

def load_from_tsfile(file_path):
    """Load .ts file -> (X: [N, C, T] float32, y: [N] int64)."""
    try:
        X, y = aeon_load(str(file_path), return_type="numpy3d")
        X = np.array(X, dtype=np.float32)
        unique = np.unique(y)
        lmap = {l: i for i, l in enumerate(unique)}
        return X, np.array([lmap[l] for l in y], dtype=np.int64)
    except Exception:
        from sktime.datasets import load_from_ts_file as sk_load
        X_df, y = sk_load(str(file_path))
        N, C = X_df.shape
        T = len(X_df.iloc[0, 0])
        X_np = np.zeros((N, C, T), dtype=np.float32)
        for i in range(N):
            for c in range(C):
                s = X_df.iloc[i, c]
                v = s.values if hasattr(s, "values") else np.asarray(s, dtype=np.float32)
                L = min(len(v), T)
                X_np[i, c, :L] = v[:L]
        unique = np.unique(y)
        lmap = {l: i for i, l in enumerate(unique)}
        return X_np, np.array([lmap[l] for l in y], dtype=np.int64)


def pad_or_truncate(X, target_len=512):
    """Pad with zeros or truncate to target_len along time axis."""
    N, C, T = X.shape
    if T == target_len:
        return X
    elif T < target_len:
        pad = np.zeros((N, C, target_len - T), dtype=X.dtype)
        return np.concatenate([X, pad], axis=2)
    else:
        return X[:, :, :target_len]


class TSDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        self.masks = torch.ones((self.X.shape[0], self.X.shape[2]))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.masks[idx], self.y[idx]


# ═══════════════════════════════════════════════════════════
# Embedding extraction
# ═══════════════════════════════════════════════════════════

def get_embeddings(model, dataloader, device):
    """Run MOMENT in embedding mode, mean-pool over patches."""
    model.eval()
    embeddings, labels = [], []
    with torch.no_grad():
        for batch_x, batch_masks, batch_labels in tqdm(dataloader, leave=False):
            batch_x = batch_x.to(device).float()
            batch_masks = batch_masks.to(device)

            output = model(x_enc=batch_x, input_mask=batch_masks)
            # Mean over patches dimension -> [B, d_model]
            embedding = output.embeddings.mean(dim=1)

            embeddings.append(embedding.detach().cpu().numpy())
            labels.append(batch_labels.numpy())

    return np.concatenate(embeddings), np.concatenate(labels)


# ═══════════════════════════════════════════════════════════
# Classifiers
# ═══════════════════════════════════════════════════════════

def get_classifiers():
    classifiers = {}

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

    # RidgeClassifierCV
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


# ═══════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════

def build_moment(hf_id, n_channels, num_class, init, seed, device):
    """Build a MOMENT model with either pretrained or random-init weights."""
    model = MOMENTPipeline.from_pretrained(
        hf_id,
        model_kwargs={
            "task_name": "classification",
            "n_channels": n_channels,
            "num_class": num_class,
        },
    )
    model.init()  # initialize classification head

    if init == "random":
        # Seed right before reset so the same seed -> same random weights
        set_seed(seed)
        model.apply(reset_weights)

    model = model.to(device).float().eval()
    return model


def evaluate_dataset(
    dataset_name, data_path, hf_id, init, seed, device,
    target_len=512, batch_size=8
):
    """Evaluate MOMENT on a single dataset. Returns dict of classifier -> accuracy."""
    ds_dir = Path(data_path) / dataset_name
    train_file = ds_dir / f"{dataset_name}_TRAIN.ts"
    test_file = ds_dir / f"{dataset_name}_TEST.ts"

    if not train_file.exists() or not test_file.exists():
        print(f"  [SKIP] Missing .ts files for {dataset_name}")
        return None

    # Load data
    X_train, y_train = load_from_tsfile(train_file)
    X_test, y_test = load_from_tsfile(test_file)

    n_channels = X_train.shape[1]
    num_class = len(np.unique(y_train))

    # Build a fresh model per dataset (because n_channels / num_class vary)
    model = build_moment(hf_id, n_channels, num_class, init, seed, device)

    # Pad/truncate to context length
    X_train = pad_or_truncate(X_train, target_len)
    X_test = pad_or_truncate(X_test, target_len)

    train_loader = DataLoader(TSDataset(X_train, y_train), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TSDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    # Extract features
    Z_train, y_train_emb = get_embeddings(model, train_loader, device)
    Z_test, y_test_emb = get_embeddings(model, test_loader, device)

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
                gs.fit(Z_train, y_train_emb)
                y_pred = gs.predict(Z_test)
                acc = np.mean(y_test_emb == y_pred)
                print(f"  {clf_name}: {acc:.4f} (best params: {gs.best_params_})")
            else:
                pipe = clf_config["pipeline"]
                pipe.fit(Z_train, y_train_emb)
                y_pred = pipe.predict(Z_test)
                acc = np.mean(y_test_emb == y_pred)
                print(f"  {clf_name}: {acc:.4f}")

            results[clf_name] = acc
        except Exception as e:
            print(f"  {clf_name}: FAILED — {e}")
            results[clf_name] = np.nan

    # Cleanup
    del model
    torch.cuda.empty_cache()

    return results


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="MOMENT zero-shot eval on UEA datasets")
    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/pinar/workspace_pinar/moment/Multivariate_ts_length_fixed",
        help="Path to UEA dataset root (each subdirectory = one dataset)",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--target_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        default=None,
        help="Specific dataset names to evaluate (default: all in data_path)",
    )
    parser.add_argument(
        "--init",
        type=str,
        default="random",
        choices=["pretrained", "random"],
        help="Use pretrained weights or random init (Xavier reset_weights)",
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
        default="MOMENT-1-base",
        choices=["MOMENT-1-small", "MOMENT-1-base", "MOMENT-1-large"],
        help="Which MOMENT size to use",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV (default: auto-named based on init/seed/model)",
    )
    args = parser.parse_args()

    hf_id = f"AutonLab/{args.model}"
    print(f"Using {hf_id} ({args.init} init, seed={args.seed})\n")

    # Global seed (per-dataset reset_weights also re-seeds before applying)
    set_seed(args.seed)

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
                ds_name,
                data_path=args.data_path,
                hf_id=hf_id,
                init=args.init,
                seed=args.seed,
                device=args.device,
                target_len=args.target_len,
                batch_size=args.batch_size,
            )
            if results is not None:
                all_results[ds_name] = results
        except Exception as e:
            print(f"  [ERROR] {e}")
        print()

    # ─── Save results ───
    if all_results:
        import pandas as pd

        if args.output is None:
            tag = args.model.replace("-", "_")
            if args.init == "random":
                args.output = f"{tag}_random_seed{args.seed}_results.csv"
            else:
                args.output = f"{tag}_pretrained_results.csv"

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