"""UniTS frozen-representation evaluation on UEA multivariate datasets."""
import os
import sys
import csv
import yaml
import argparse
from collections.abc import Mapping
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.downstream import PROBE_ORDER, evaluate_probes

# UniTS repo imports
from models.UniTS import Model as UniTSModel
from data_provider.data_factory import data_provider
from types import SimpleNamespace


# ═══════════════════════════════════════════════════════════
# Seed + Random Init helpers
# ═══════════════════════════════════════════════════════════

def set_seed(seed: int):
    """Set all relevant seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_pretrained_weights(model: torch.nn.Module, checkpoint: str | Path) -> None:
    """Load a public UniTS checkpoint and make partial target-task loading explicit.

    Release artifacts use either a bare DataParallel state dictionary or a
    ``student``/``state_dict`` wrapper.  A target-task YAML intentionally has
    different prompt and class-token entries from the source training YAML, so
    only shape-compatible tensors can be transferred.  Reject a checkpoint
    that would otherwise silently load no representation weights.
    """
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(raw, Mapping):
        raise TypeError(f"{checkpoint_path} is not a UniTS state dictionary")

    for wrapper in ("state_dict", "model", "student"):
        candidate = raw.get(wrapper)
        if isinstance(candidate, Mapping):
            raw = candidate
            break

    state = {
        str(key).removeprefix("module."): value
        for key, value in raw.items()
        if isinstance(value, torch.Tensor)
    }
    if not state:
        raise ValueError(f"{checkpoint_path} contains no tensor state")

    target_state = model.state_dict()
    compatible = {
        key: value
        for key, value in state.items()
        if key in target_state and target_state[key].shape == value.shape
    }
    mismatched = [
        key for key, value in state.items()
        if key in target_state and target_state[key].shape != value.shape
    ]
    if not compatible:
        raise RuntimeError(
            f"{checkpoint_path} has no tensors compatible with this UniTS architecture. "
            "Check --d_model, --e_layers, --prompt_num, --patch_len, --stride, and the task YAML."
        )

    loaded = model.load_state_dict(compatible, strict=False)
    print(
        "  Loaded "
        f"{len(compatible)}/{len(state)} checkpoint tensors from {checkpoint_path} "
        f"({len(loaded.missing_keys)} target-only, {len(mismatched)} shape-mismatched skipped)"
    )


# ═══════════════════════════════════════════════════════════
# UniTS arg + loader helpers
# ═══════════════════════════════════════════════════════════

def build_args(task_cfg, global_args):
    """Build the argparse-like namespace UniTS expects for a single task."""
    a = SimpleNamespace(
        task_name=task_cfg["task_name"],
        is_training=0,
        root_path=task_cfg["root_path"],
        data=task_cfg["data"],
        data_path="",
        embed=task_cfg["embed"],
        seq_len=task_cfg["seq_len"],
        label_len=task_cfg["label_len"],
        pred_len=task_cfg["pred_len"],
        enc_in=task_cfg["enc_in"],
        num_class=task_cfg["num_class"],
        features="M",
        target="OT",
        freq="h",
        batch_size=64,
        num_workers=global_args.num_workers,
        prompt_num=global_args.prompt_num,
        patch_len=global_args.patch_len,
        stride=global_args.stride,
        e_layers=global_args.e_layers,
        d_model=global_args.d_model,
        n_heads=global_args.n_heads,
        dropout=0.1,
        task_data_config_path=global_args.yaml,
        subsample_pct=None,
        label_soft_name=None,
        p_hidden_dims=[128, 128],
    )
    return a


def get_loader(task_cfg, flag, g_args):
    args = build_args(task_cfg, g_args)
    data_set, data_loader = data_provider(args, task_cfg, flag, ddp=False)
    return data_set, data_loader


# ═══════════════════════════════════════════════════════════
# Feature extraction
# ═══════════════════════════════════════════════════════════

@torch.no_grad()
def extract_features(model, loader, device, feat_store, task_id=0):
    model.eval()
    feats, labels = [], []
    for batch in loader:
        if len(batch) == 3:
            x, y, pad = batch
        else:
            x, y = batch[0], batch[1]
            pad = None

        x = x.float().to(device)
        if pad is not None:
            pad = pad.float().to(device)

        feat_store.clear()

        try:
            _ = model(x, pad, None, None, task_id=task_id, task_name="classification")
        except Exception:
            _ = model.classification(x, pad, 1)

        h = feat_store.get("out")
        if h is None:
            raise RuntimeError("Hook did not capture features.")

        # h might be (B, L, D) or (B, L, D1, D2)
        if h.dim() >= 3:
            h = h.mean(dim=1)

        # Flatten anything past batch dim
        if h.dim() > 2:
            h = h.reshape(h.size(0), -1)

        feats.append(h.cpu().numpy())
        labels.append(y.numpy().reshape(-1))

    return np.concatenate(feats, axis=0), np.concatenate(labels, axis=0)


# ═══════════════════════════════════════════════════════════
# Classifiers (the shared camera-ready protocol)
# ═══════════════════════════════════════════════════════════

def fit_and_score(tr_x, tr_y, te_x, te_y):
    """Fit all classifiers, return dict {clf_name: {'acc':..., 'mf1':..., 'best_params':...}}."""
    probe_results = evaluate_probes(tr_x, tr_y, te_x, te_y)
    out = {}
    for clf_name in PROBE_ORDER:
        probe = probe_results[clf_name]
        out[clf_name] = {
            "acc": probe["accuracy"],
            "mf1": probe["macro_f1"],
            "best_params": probe["best_params"],
        }
        if probe["error"]:
            print(f"    {clf_name:7s} FAILED — {probe['error']}")
        else:
            print(
                f"    {clf_name:7s} Acc={probe['accuracy']:.4f} "
                f"MF1={probe['macro_f1']:.4f} | best: {probe['best_params']}"
            )

    return out


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None,
                    help="Checkpoint path (required if --init pretrained)")
    ap.add_argument("--yaml", required=True)
    ap.add_argument("--out", default=None,
                    help="Output CSV (default: auto-named based on init/seed)")
    ap.add_argument("--prompt_num", type=int, default=10)
    ap.add_argument("--patch_len", type=int, default=16)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--e_layers", type=int, default=3)
    ap.add_argument("--d_model", type=int, default=64)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=0,
                    help="DataLoader workers; 0 is the most portable default")
    ap.add_argument("--init", type=str, default="pretrained",
                    choices=["pretrained", "random"],
                    help="Use a pretrained checkpoint or a fresh, seeded official initialization")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed (only relevant for --init random)")
    g_args = ap.parse_args()

    if g_args.init == "pretrained" and g_args.ckpt is None:
        ap.error("--ckpt is required when --init pretrained")

    # Auto-name output if not provided
    if g_args.out is None:
        if g_args.init == "random":
            g_args.out = f"units_random_seed{g_args.seed}_results.csv"
        else:
            g_args.out = "units_pretrained_results.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(g_args.yaml) as f:
        cfg = yaml.safe_load(f)
    tasks = cfg["task_dataset"]

    configs_list = [[str(name), t_cfg] for name, t_cfg in tasks.items()]

    model_configs = SimpleNamespace(
        task_data_config=tasks,
        prompt_num=g_args.prompt_num,
        patch_len=g_args.patch_len,
        stride=g_args.stride,
        e_layers=g_args.e_layers,
        d_model=g_args.d_model,
        n_heads=g_args.n_heads,
        dropout=0.1,
        task_name="classification",
    )

    # ─── Set seed BEFORE building the model ───
    set_seed(g_args.seed)

    print(f"Building UniTS ({g_args.init} init, seed={g_args.seed})...")
    model = UniTSModel(model_configs, configs_list).to(device)

    if g_args.init == "pretrained":
        load_pretrained_weights(model, g_args.ckpt)
    else:
        print(f"  Using fresh official initialization (seed={g_args.seed}); no checkpoint loaded")

    model.eval()

    # ─── Register feature hook ───
    feat_store = {}
    def hook(_m, _inp, out):
        if isinstance(out, tuple):
            out = out[0]
        feat_store["out"] = out.detach()

    target = None
    for attr in ["blocks", "encoder", "layers"]:
        mod = getattr(model, attr, None)
        if mod is not None and len(list(mod.children())) > 0:
            target = list(mod.children())[-1]
            print(f"[hook] attached to model.{attr}[-1]")
            break

    if target is None:
        raise RuntimeError("Could not find encoder blocks.")
    target.register_forward_hook(hook)

    # ─── Evaluate per task ───
    output_path = Path(g_args.out).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    g_args.out = str(output_path)

    CLF_NAMES = list(PROBE_ORDER)
    results = []
    for idx, (task_name, task_cfg) in enumerate(tasks.items()):
        print(f"\n=== {task_name} ===")
        try:
            # The upstream provider uses lowercase values to select evaluation
            # batching and avoid dropping a final incomplete test batch.
            _, tr_loader = get_loader(task_cfg, "train", g_args)
            _, te_loader = get_loader(task_cfg, "test", g_args)

            tr_x, tr_y = extract_features(model, tr_loader, device, feat_store, task_id=idx)
            te_x, te_y = extract_features(model, te_loader, device, feat_store, task_id=idx)

            clf_results = fit_and_score(tr_x, tr_y, te_x, te_y)

            row = {
                "dataset": task_cfg["dataset"],
                "n_train": len(tr_y),
                "n_test": len(te_y),
                "feat_dim": tr_x.shape[1],
            }
            for c in CLF_NAMES:
                row[f"{c}_acc"] = clf_results[c]["acc"]
                row[f"{c}_mf1"] = clf_results[c]["mf1"]
            # Keep the RBF-SVM best parameters for reference.
            svm_best = clf_results["RBF_SVM"]["best_params"] or {}
            row["best_C"] = svm_best.get("svm__C")
            row["best_gamma"] = svm_best.get("svm__gamma")

            results.append(row)
        except Exception as e:
            print(f"  FAIL: {e}")
            results.append({
                "dataset": task_cfg["dataset"],
                **{f"{c}_acc": float("nan") for c in CLF_NAMES},
                **{f"{c}_mf1": float("nan") for c in CLF_NAMES},
            })

        # Incremental save
        with open(g_args.out, "w", newline="") as f:
            if results:
                # Union of all keys across rows (failed rows may have fewer)
                fieldnames = list(dict.fromkeys(k for r in results for k in r.keys()))
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(results)

    # ─── Summary ───
    print("\n" + "=" * 70)
    print("Mean accuracy per classifier:")
    for c in CLF_NAMES:
        col = [r.get(f"{c}_acc", float("nan")) for r in results]
        col = [x for x in col if not (isinstance(x, float) and np.isnan(x))]
        if col:
            print(f"  {c:7s} mean Acc over {len(col)} datasets: {np.mean(col):.4f}")
        else:
            print(f"  {c:7s} no valid results")

    print(f"\nResults saved to {g_args.out}")


if __name__ == "__main__":
    main()
