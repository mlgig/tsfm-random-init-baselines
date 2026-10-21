"""Mantis frozen-representation evaluation on UEA multivariate datasets."""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import warnings
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.downstream import PROBE_ORDER, evaluate_probes
from common.uea import load_uea_collection, pad_train_test_to_common_length, right_pad_collection
from mantis.architecture import Mantis8M, MantisV2
from mantis.trainer import MantisTrainer


MANTIS_REVISIONS = {
    "Mantis8M": "bc7d5ab40c02133386a28e2c127f35c17c86901d",
    "MantisV2": "8f6ca35cb54ab14b120618943c6fca5ddf5a76a6",
}

warnings.filterwarnings("ignore")


def set_seed(seed: int):
    """Set all relevant seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resize_to_multiple_of_32(X, target_len: int = 512) -> np.ndarray:
    """Right-pad ragged UEA cases, then interpolate every case to ``target_len``."""
    X_tensor = torch.tensor(right_pad_collection(X), dtype=torch.float)
    X_resized = F.interpolate(X_tensor, size=target_len, mode='linear', align_corners=False)
    return X_resized.numpy()


def evaluate_dataset(dataset_name, data_path, model, device, target_len=512):
    """Evaluate Mantis on a single dataset. Returns dict of classifier -> accuracy."""
    train_file = os.path.join(data_path, dataset_name, f"{dataset_name}_TRAIN.ts")
    test_file = os.path.join(data_path, dataset_name, f"{dataset_name}_TEST.ts")

    if not os.path.exists(train_file) or not os.path.exists(test_file):
        print(f"  [SKIP] Missing .ts files for {dataset_name}")
        return None

    # Load data
    X_train, y_train = load_uea_collection(train_file)
    X_test, y_test = load_uea_collection(test_file)

    # Resize to target_len (must be multiple of 32)
    X_train, X_test = pad_train_test_to_common_length(X_train, X_test)
    X_train = resize_to_multiple_of_32(X_train, target_len)
    X_test = resize_to_multiple_of_32(X_test, target_len)

    # Extract features with Mantis
    Z_train = model.transform(X_train)
    Z_test = model.transform(X_test)

    probe_results = evaluate_probes(Z_train, y_train, Z_test, y_test)
    results = {}
    for clf_name in PROBE_ORDER:
        probe = probe_results[clf_name]
        results[clf_name] = probe["accuracy"]
        if probe["error"]:
            print(f"  {clf_name}: FAILED — {probe['error']}")
        else:
            print(f"  {clf_name}: {probe['accuracy']:.4f} (best params: {probe['best_params']})")

    return results


def main():
    parser = argparse.ArgumentParser(description="Mantis zero-shot eval on UEA datasets")
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to UEA dataset root (each subdirectory = one dataset)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device (defaults to CUDA when available, otherwise CPU)",
    )
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
        "--revision",
        type=str,
        default=None,
        help="Hugging Face checkpoint revision (defaults to the pinned revision used for the paper)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file for results (default: auto-named based on init/seed/model)",
    )
    args = parser.parse_args()

    if args.target_len <= 0 or args.target_len % 32:
        parser.error("--target_len must be a positive multiple of 32 for Mantis")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available; pass --device cpu")

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
    revision = args.revision or MANTIS_REVISIONS[args.model]

    if args.init == "pretrained":
        # Mantis8M exposes a classmethod loader, so explicitly pass the device
        # rather than relying on the instance created immediately above.
        network = network.from_pretrained(
            hf_id, revision=revision, device=args.device
        )
        print(f"  Loaded pretrained weights from {hf_id}@{revision}")
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
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, float_format="%.17g")
        print(f"\nResults saved to {args.output}")
        print(f"\n{'='*60}")
        print(df.to_string(float_format=lambda x: f"{x:.4f}"))
        print(f"{'='*60}")
    else:
        print("No results to save.")


if __name__ == "__main__":
    main()
