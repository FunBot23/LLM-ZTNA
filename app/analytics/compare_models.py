"""Side-by-side comparison of Random Forest, TabTransformer, and GNN.

For each model:
  * Retrain N times with different random seeds
  * Evaluate on the held-out split
  * Report mean +/- std for accuracy, precision, recall, F1, ROC-AUC, FPR

Outputs:
  * Console table with side-by-side metrics (mean +/- std)
  * PNG bar chart with error bars and value labels (comparison_error_bars.png)
  * PNG confusion matrices, one panel per model (confusion_matrices.png)
  * CSV of raw per-run metrics and confusion counts (comparison_runs.csv)

Usage:
    python -m app.analytics.compare_models --csv Train_Test_Network.csv --runs 5
    python -m app.analytics.compare_models --csv Train_Test_Network.csv --runs 5 --skip-gnn
    python -m app.analytics.compare_models --csv Train_Test_Network.csv --gnn-csv gnn_data.csv --runs 5
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from app.analytics.features import (
    CATEGORICAL_COLS, CATEGORICAL_VOCABS, CONTINUOUS_COLS,
)
from app.analytics.model import TabTransformer
from app.analytics.train import load_and_encode


# ============================================================================
# Metric helpers
# ============================================================================
def compute_metrics(y_true, y_pred, y_prob) -> dict:
    """Compute the six comparison metrics (plus raw confusion counts)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_true, y_prob) if len(set(y_true.tolist())) > 1 else float("nan"),
        "fpr":       fp / (fp + tn) if (fp + tn) else 0.0,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


# ============================================================================
# Model runners — each returns (metrics_dict, train_seconds)
# ============================================================================
def run_random_forest(cat, cont, y, seed: int) -> tuple[dict, float]:
    """Train and evaluate a Random Forest baseline."""
    # RF works directly on concatenated features
    X = np.concatenate([cat.astype(np.float32), cont.astype(np.float32)], axis=1)

    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y,
    )

    t0 = time.time()
    model = RandomForestClassifier(
        n_estimators=100,
        n_jobs=-1,
        random_state=seed,
        class_weight="balanced",
    )
    model.fit(X_tr, y_tr)
    train_s = time.time() - t0

    y_prob = model.predict_proba(X_va)[:, 1]
    y_pred = (y_prob >= 0.5).astype(np.int32)
    return compute_metrics(y_va, y_pred, y_prob), train_s


def run_tabtransformer(cat, cont, y, seed: int, epochs: int = 5) -> tuple[dict, float]:
    """Train and evaluate the TabTransformer."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    cont_mean = cont.mean(axis=0)
    cont_std  = cont.std(axis=0) + 1e-6
    cont_norm = (cont - cont_mean) / cont_std

    x_cat_tr, x_cat_va, x_c_tr, x_c_va, y_tr, y_va = train_test_split(
        cat, cont_norm, y, test_size=0.2, random_state=seed, stratify=y,
    )

    tr_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x_cat_tr),
            torch.from_numpy(x_c_tr).float(),
            torch.from_numpy(y_tr).float(),
        ),
        batch_size=512, shuffle=True,
    )

    model = TabTransformer(
        cat_cardinalities=[len(CATEGORICAL_VOCABS[c]) for c in CATEGORICAL_COLS],
        num_continuous=len(CONTINUOUS_COLS),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    pos = float(y_tr.sum()); neg = float(len(y_tr) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    t0 = time.time()
    for _ in range(epochs):
        model.train()
        for cb, xb, yb in tr_loader:
            opt.zero_grad()
            logits = model(cb, xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
    train_s = time.time() - t0

    # Evaluate
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(y_va), 4096):
            cb = torch.from_numpy(x_cat_va[i:i+4096])
            xb = torch.from_numpy(x_c_va[i:i+4096]).float()
            probs.append(torch.sigmoid(model(cb, xb)).numpy())
    y_prob = np.concatenate(probs)
    y_pred = (y_prob >= 0.5).astype(np.int32)
    return compute_metrics(y_va, y_pred, y_prob), train_s


def run_gnn(csv_path: Path, seed: int, epochs: int = 8) -> tuple[dict, float]:
    """Train and evaluate the GraphSAGE GNN. Requires a 'ts' column."""
    # Import here so RF/Transformer runs don't require torch-geometric
    from app.analytics.gnn_model import GraphSAGENodeClassifier
    from app.analytics.gnn_train import build_windows
    from app.analytics.graph_window import NODE_FEATURES
    from torch_geometric.loader import DataLoader as GeoLoader

    torch.manual_seed(seed)
    np.random.seed(seed)

    windows = build_windows(csv_path, window_seconds=60.0)
    all_x = torch.cat([w.x[w.train_mask] for w in windows], dim=0)
    mean = all_x.mean(dim=0); std = all_x.std(dim=0) + 1e-6
    for w in windows:
        w.x = (w.x - mean) / std

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(windows))
    split = max(1, int(len(idx) * 0.8))
    train_w = [windows[i] for i in idx[:split]]
    val_w   = [windows[i] for i in idx[split:]] or [windows[idx[-1]]]

    model = GraphSAGENodeClassifier(in_channels=len(NODE_FEATURES))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    all_y = torch.cat([w.y[w.train_mask] for w in train_w])
    pos = float(all_y.sum()); neg = float(len(all_y) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    t0 = time.time()
    for _ in range(epochs):
        model.train()
        for batch in GeoLoader(train_w, batch_size=8, shuffle=True):
            opt.zero_grad()
            logits = model(batch.x, batch.edge_index)
            m = batch.train_mask
            loss = loss_fn(logits[m], batch.y[m])
            loss.backward()
            opt.step()
    train_s = time.time() - t0

    model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for batch in GeoLoader(val_w, batch_size=8):
            logits = model(batch.x, batch.edge_index)
            m = batch.train_mask if hasattr(batch, "train_mask") else torch.ones_like(batch.y, dtype=torch.bool)
            y_prob.append(torch.sigmoid(logits[m]).numpy())
            y_true.append(batch.y[m].numpy())
    y_prob = np.concatenate(y_prob)
    y_true = np.concatenate(y_true).astype(np.int32)
    y_pred = (y_prob >= 0.5).astype(np.int32)
    return compute_metrics(y_true, y_pred, y_prob), train_s


# ============================================================================
# Main comparison loop
# ============================================================================
def compare(csv_path: Path, runs: int, skip_gnn: bool, gnn_csv_path: Path | None = None, cm_mode: str = "count") -> None:
    print(f"Loading dataset {csv_path} ...")
    cat, cont, y = load_and_encode(csv_path)
    print(f"  {len(y):,} rows, {int(y.sum())} attack / {int((1-y).sum())} normal\n")

    gnn_csv = gnn_csv_path or csv_path
    gnn_key = "GraphSAGE*" if gnn_csv != csv_path else "GraphSAGE"

    results: dict[str, list[dict]] = {"RandomForest": [], "TabTransformer": []}
    times:   dict[str, list[float]] = {"RandomForest": [], "TabTransformer": []}
    if not skip_gnn:
        results[gnn_key] = []
        times[gnn_key]   = []

    for run_idx in range(runs):
        seed = 42 + run_idx
        print(f"----- Run {run_idx+1}/{runs} (seed={seed}) -----")

        print("  Random Forest ...", end=" ", flush=True)
        m, t = run_random_forest(cat, cont, y, seed)
        results["RandomForest"].append(m); times["RandomForest"].append(t)
        print(f"F1={m['f1']:.4f} AUC={m['roc_auc']:.4f} ({t:.1f}s)")

        print("  TabTransformer ...", end=" ", flush=True)
        m, t = run_tabtransformer(cat, cont, y, seed)
        results["TabTransformer"].append(m); times["TabTransformer"].append(t)
        print(f"F1={m['f1']:.4f} AUC={m['roc_auc']:.4f} ({t:.1f}s)")

        if not skip_gnn:
            print("  GraphSAGE ...", end=" ", flush=True)
            try:
                m, t = run_gnn(gnn_csv, seed)
                results[gnn_key].append(m); times[gnn_key].append(t)
                print(f"F1={m['f1']:.4f} AUC={m['roc_auc']:.4f} ({t:.1f}s)")
            except Exception as e:
                print(f"SKIPPED ({e})")

    _print_comparison_table(results, times)
    if gnn_key.endswith("*"):
        print(f"\n* GraphSAGE trained/evaluated on {gnn_csv.name}; "
              f"other models used {csv_path.name}.")

    _save_raw_csv(results, "comparison_runs.csv")
    _save_bar_chart_with_error_bars(results, str(Path.cwd() / "comparison_error_bars.png"))
    _save_confusion_matrices(results, str(Path.cwd() / "confusion_matrices.png"))


# ============================================================================
# Reporting
# ============================================================================
def _print_comparison_table(results, times) -> None:
    metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc", "fpr"]

    print("\n" + "=" * 88)
    print("SIDE-BY-SIDE COMPARISON (mean +/- std over runs)")
    print("=" * 88)
    header = f"  {'Metric':<12}"
    for model in results:
        header += f"  {model:>22}"
    print(header)
    print("  " + "-" * (12 + 24 * len(results)))

    for metric in metric_names:
        row = f"  {metric:<12}"
        for model in results:
            vals = [r[metric] for r in results[model]]
            if not vals:
                row += f"  {'n/a':>22}"
            else:
                mean = np.mean(vals); std = np.std(vals)
                row += f"  {mean:>10.4f} +/- {std:.4f}  "
        print(row)

    # Timing
    row = f"  {'train_s':<12}"
    for model in results:
        vals = times[model]
        if not vals:
            row += f"  {'n/a':>22}"
        else:
            mean = np.mean(vals); std = np.std(vals)
            row += f"  {mean:>10.2f} +/- {std:.2f}  "
    print(row)


def _save_raw_csv(results, path: str) -> None:
    rows = []
    for model, runs in results.items():
        for i, r in enumerate(runs):
            rows.append({"model": model, "run": i, **r})
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"\nRaw per-run metrics -> {path}")


def _save_bar_chart_with_error_bars(results, path: str) -> None:
    """Models on the x-axis; bars are F1, accuracy, precision, recall."""
    metrics = ["f1", "accuracy", "precision", "recall"]
    metric_labels = {"f1": "F1", "accuracy": "Accuracy",
                     "precision": "Precision", "recall": "Recall"}

    # Only plot models that actually produced results (e.g. GNN may be skipped)
    models = [m for m in results if results[m]]

    means = {k: [np.mean([r[k] for r in results[m]]) for m in models] for k in metrics}
    stds  = {k: [np.std ([r[k] for r in results[m]]) for m in models] for k in metrics}

    x = np.arange(len(models))
    width = 0.8 / len(metrics)

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, k in enumerate(metrics):
        bars = ax.bar(
            x + (i - (len(metrics) - 1) / 2) * width,
            means[k], width,
            yerr=stds[k], capsize=4, label=metric_labels[k],
        )
        ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Model comparison (mean +/- std across runs)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=len(metrics), frameon=False)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Bar chart with error bars -> {path}")


def _save_confusion_matrices(results, path: str, mode:str = "count") -> None:
    """One confusion-matrix panel per model, averaged over all runs."""
    models = [m for m in results if results[m]]
    class_names = ["Normal", "Attack"]

    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4.8), squeeze=False)

    for ax, model in zip(axes[0], models):
        runs = results[model]

        # Shape (n_runs, 2, 2). Rows = true class, columns = predicted class.
        cms = np.array(
            [[[r["tn"], r["fp"]], [r["fn"], r["tp"]]] for r in runs],
            dtype=float,
        )
        mean_cm = cms.mean(axis=0)
        std_cm  = cms.std(axis=0)

        # Row-normalise so each true class sums to 100%
        row_sums = np.maximum(mean_cm.sum(axis=1, keepdims=True), 1.0)
        pct = mean_cm / row_sums * 100

        ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
        for i in range(2):
            for j in range(2):
                ax.text(
                    j, i,
                    f"{pct[i, j]:.2f}%\n{mean_cm[i, j]:,.0f} ± {std_cm[i, j]:,.0f}",
                    ha="center", va="center", fontsize=10,
                    color="white" if pct[i, j] > 50 else "black",
                )

        ax.set_xticks([0, 1]); ax.set_xticklabels(class_names)
        ax.set_yticks([0, 1]); ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{model}\n(mean of {len(runs)} runs)")

    fig.suptitle("Confusion matrices (cell: row % / mean count ± std)", y=1.02)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Confusion matrices -> {path}")


# ============================================================================
# CLI
# ============================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--runs", default=5, type=int,
                    help="Number of retraining runs per model (default 5)")
    ap.add_argument("--skip-gnn", action="store_true",
                    help="Skip the GraphSAGE run (needs a 'ts' column and torch-geometric)")
    ap.add_argument("--gnn-csv", type=Path, default=None,
                    help="Separate CSV for the GraphSAGE run (defaults to --csv)")
    ap.add_argument("--cm-mode", choices=["row", "col", "count"], default="count",
                    help="Confusion matrix normalisation (default: count)")
    args = ap.parse_args()
    compare(args.csv, args.runs, args.skip_gnn, args.gnn_csv)
