"""MOMENT frozen-representation evaluation on UEA multivariate datasets."""

import os
import sys
import json
import numpy as np
import torch
import warnings
import argparse
from pathlib import Path
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from huggingface_hub import hf_hub_download
from momentfm import MOMENTPipeline

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.downstream import PROBE_ORDER, evaluate_probes
from common.uea import load_uea_collection, right_pad_collection


MOMENT_REVISIONS = {
    "MOMENT-1-small": "411e288267f82cce86296dbe4d6c8bc533cc162f",
    "MOMENT-1-base": "5e44b0ea26376a176360f87831124e018f876d96",
    "MOMENT-1-large": "ca58581bc7bea2ebed4e80dc0a3e4b8b609c6ecc",
}

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


# ═══════════════════════════════════════════════════════════
# Data utilities
# ═══════════════════════════════════════════════════════════

def load_from_tsfile(file_path):
    """Load a .ts split, keeping a ragged UEA collection until context shaping."""
    return load_uea_collection(str(file_path))


def encode_train_test_labels(train_labels, test_labels):
    """Encode labels with one map derived from training labels only."""
    classes = np.unique(train_labels)
    label_map = {label: index for index, label in enumerate(classes)}
    unknown = set(np.asarray(test_labels)) - set(classes)
    if unknown:
        raise ValueError(f"test labels not present in training labels: {sorted(unknown)!r}")
    return (
        np.asarray([label_map[label] for label in train_labels], dtype=np.int64),
        np.asarray([label_map[label] for label in test_labels], dtype=np.int64),
    )


def pad_or_truncate(X, target_len=512):
    """Right-pad/truncate every UEA case to MOMENT's fixed context length."""
    return right_pad_collection(X, target_length=target_len)


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

# ═══════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════

def load_moment_config(hf_id: str, revision: str) -> dict:
    """Fetch the pinned architecture JSON required by ``momentfm==0.1.4``."""
    config_path = hf_hub_download(
        repo_id=hf_id,
        filename="config.json",
        revision=revision,
    )
    with open(config_path, encoding="utf-8") as config_file:
        return json.load(config_file)


def build_moment(hf_id, revision, n_channels, num_class, init, seed, device):
    """Build an architecture-matched MOMENT encoder.

    ``momentfm==0.1.4`` requires the model repository's ``config.json`` to be
    supplied explicitly to ``from_pretrained``. For random initialization we
    download only that pinned architecture JSON, seed, and construct a new
    pipeline. Thus the final encoder is initialized by MOMENT/PyTorch's
    official construction path and contains no checkpoint parameters or
    buffers.
    """
    model_kwargs = {
        "task_name": "classification",
        "n_channels": n_channels,
        "num_class": num_class,
    }
    config = load_moment_config(hf_id, revision)
    if init == "pretrained":
        model = MOMENTPipeline.from_pretrained(
            hf_id,
            revision=revision,
            config=config,
            model_kwargs=model_kwargs,
        )
        model.init()
    else:
        set_seed(seed)
        model = MOMENTPipeline(config, model_kwargs=dict(model_kwargs))
        model.init()

    return model.to(device).float().eval()


def evaluate_dataset(
    dataset_name, data_path, hf_id, revision, init, seed, device,
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
    X_train, train_labels = load_from_tsfile(train_file)
    X_test, test_labels = load_from_tsfile(test_file)
    y_train, y_test = encode_train_test_labels(train_labels, test_labels)

    # Pad/truncate every raw or ragged UEA case to the model context length.
    X_train = pad_or_truncate(X_train, target_len)
    X_test = pad_or_truncate(X_test, target_len)

    n_channels = X_train.shape[1]
    num_class = len(np.unique(y_train))

    # Build a fresh model per dataset (because n_channels / num_class vary)
    model = build_moment(hf_id, revision, n_channels, num_class, init, seed, device)

    train_loader = DataLoader(TSDataset(X_train, y_train), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TSDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    # Extract features
    Z_train, y_train_emb = get_embeddings(model, train_loader, device)
    Z_test, y_test_emb = get_embeddings(model, test_loader, device)

    probe_results = evaluate_probes(Z_train, y_train_emb, Z_test, y_test_emb)
    results = {}
    for clf_name in PROBE_ORDER:
        probe = probe_results[clf_name]
        results[clf_name] = probe["accuracy"]
        if probe["error"]:
            print(f"  {clf_name}: FAILED — {probe['error']}")
        else:
            print(f"  {clf_name}: {probe['accuracy']:.4f} (best params: {probe['best_params']})")

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
        required=True,
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
        help="Use released pretrained weights or a fresh, seeded MOMENT initialization",
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
        "--revision",
        type=str,
        default=None,
        help="Hugging Face checkpoint revision (defaults to the pinned revision used for the paper)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV (default: auto-named based on init/seed/model)",
    )
    args = parser.parse_args()

    hf_id = f"AutonLab/{args.model}"
    revision = args.revision or MOMENT_REVISIONS[args.model]
    print(f"Using {hf_id}@{revision} ({args.init} init, seed={args.seed})\n")

    # The random condition reseeds immediately before each fresh construction.
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
                revision=revision,
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
