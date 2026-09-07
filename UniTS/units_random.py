"""
UniTS frozen feature extraction + multi-classifier probe on UEA datasets.
- Pretrained: loads weights from --ckpt
- Random: skips checkpoint, applies Xavier reset_weights with --seed

Random init uses the same reset_weights strategy as the MOMENT / Mantis / NuTime
random-init baselines, for fair comparison across foundation models.

Classifiers: SVM (GridSearch), LogReg (GridSearch), RidgeClassifierCV, RandomForest.
"""
import os
import sys
import csv
import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

# UniTS repo imports
from models.UniTS import Model as UniTSModel
from data_provider.data_factory import data_provider
from types import SimpleNamespace


# Example invocations:
#
# Pretrained:
#   python svm_random.py \
#     --ckpt /home/pinar/workspace_pinar/UniTS/checkpoints/units_x64_supervised_checkpoint.pth \
#     --yaml data_provider/fewshot_new_task.yaml \
#     --init random --seed 0
#
# Random init (same seeds as Mantis / MOMENT / NuTime):
#   python units_feature_svm.py \
#     --yaml data_provider/uea_zeroshot.yaml \
#     --init random --seed 42

# Select a GPU outside the script, for example: CUDA_VISIBLE_DEVICES=0 bash scripts/run_units.sh ...


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
        num_workers=4,
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
# Classifiers (same set used across Mantis / MOMENT / NuTime)
# ═══════════════════════════════════════════════════════════

def get_classifiers():
    classifiers = {}

    # SVM (RBF) — GridSearch over C and gamma
    classifiers["SVM"] = {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", random_state=42)),
        ]),
        "param_grid": {
            "svm__C": [0.1, 1, 10, 100],
            "svm__gamma": ["scale", 0.01, 0.001],
        },
        "use_gridsearch": True,
    }

    # Logistic Regression — GridSearch over C
    classifiers["LogReg"] = {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(max_iter=2000, random_state=42)),
        ]),
        "param_grid": {"logreg__C": [0.1, 1, 10]},
        "use_gridsearch": True,
    }

    # RidgeClassifierCV — built-in CV over alphas
    classifiers["Ridge"] = {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))),
        ]),
        "param_grid": None,
        "use_gridsearch": False,
    }

    # Random Forest
    classifiers["RF"] = {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(n_estimators=200, random_state=42)),
        ]),
        "param_grid": None,
        "use_gridsearch": False,
    }

    return classifiers


def fit_and_score(tr_x, tr_y, te_x, te_y, n_splits=3):
    """Fit all classifiers, return dict {clf_name: {'acc':..., 'mf1':..., 'best_params':...}}."""
    if tr_x.ndim > 2:
        tr_x = tr_x.reshape(tr_x.shape[0], -1)
        te_x = te_x.reshape(te_x.shape[0], -1)

    classifiers = get_classifiers()
    out = {}
    for clf_name, cfg in classifiers.items():
        try:
            if cfg["use_gridsearch"] and n_splits >= 2:
                gs = GridSearchCV(
                    cfg["pipeline"],
                    cfg["param_grid"],
                    cv=n_splits,
                    scoring="accuracy",
                    n_jobs=-1,
                    refit=True,
                )
                gs.fit(tr_x, tr_y)
                y_pred = gs.predict(te_x)
                best = gs.best_params_
            else:
                pipe = cfg["pipeline"]
                pipe.fit(tr_x, tr_y)
                y_pred = pipe.predict(te_x)
                best = None

            acc = accuracy_score(te_y, y_pred)
            mf1 = f1_score(te_y, y_pred, average="macro")
            out[clf_name] = {"acc": acc, "mf1": mf1, "best_params": best}
            best_str = f" | best: {best}" if best is not None else ""
            print(f"    {clf_name:7s} Acc={acc:.4f} MF1={mf1:.4f}{best_str}")
        except Exception as e:
            print(f"    {clf_name:7s} FAILED — {type(e).__name__}: {e}")
            out[clf_name] = {"acc": float("nan"), "mf1": float("nan"), "best_params": None}

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
    ap.add_argument("--init", type=str, default="pretrained",
                    choices=["pretrained", "random"],
                    help="Use pretrained checkpoint or random init (Xavier reset_weights)")
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
        sd = torch.load(g_args.ckpt, map_location=device)
        if "state_dict" in sd:
            sd = sd["state_dict"]
        elif "model" in sd:
            sd = sd["model"]
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)
        print(f"  Loaded pretrained weights from {g_args.ckpt}")
    else:
        # Re-seed right before reset so the same seed -> same random weights
        set_seed(g_args.seed)
        model.apply(reset_weights)
        print(f"  Applied reset_weights (Xavier init, seed={g_args.seed})")

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
    CLF_NAMES = ["SVM", "LogReg", "Ridge", "RF"]
    results = []
    for idx, (task_name, task_cfg) in enumerate(tasks.items()):
        print(f"\n=== {task_name} ===")
        try:
            _, tr_loader = get_loader(task_cfg, "TRAIN", g_args)
            _, te_loader = get_loader(task_cfg, "TEST", g_args)

            tr_x, tr_y = extract_features(model, tr_loader, device, feat_store, task_id=idx)
            te_x, te_y = extract_features(model, te_loader, device, feat_store, task_id=idx)

            clf_results = fit_and_score(tr_x, tr_y, te_x, te_y, n_splits=3)

            row = {
                "dataset": task_cfg["dataset"],
                "n_train": len(tr_y),
                "n_test": len(te_y),
                "feat_dim": tr_x.shape[1],
            }
            for c in CLF_NAMES:
                row[f"{c}_acc"] = clf_results[c]["acc"]
                row[f"{c}_mf1"] = clf_results[c]["mf1"]
            # Keep SVM best params for reference
            svm_best = clf_results["SVM"]["best_params"] or {}
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