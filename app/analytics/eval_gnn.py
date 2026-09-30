"""Evaluate the trained GraphSAGE GNN on the held-out window split.

Uses the same 80/20 window split (seed 42) as gnn_train.py, so this
evaluates on graph windows the model has never seen.

Reports the same metric suite as eval.py for the TabTransformer, so the
two models can be compared side-by-side in the report.

Usage:
    python -m app.analytics.eval_gnn --csv <TON_IoT>.csv
    python -m app.analytics.eval_gnn --csv <TON_IoT>.csv --threshold 0.4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, precision_recall_curve,
    recall_score, roc_auc_score, auc,
)
from torch_geometric.loader import DataLoader

from app.analytics.gnn_model import GraphSAGENodeClassifier
from app.analytics.gnn_train import build_windows
from app.analytics.graph_window import NODE_FEATURES
from app.analytics.cm_plot import save_confusion_matrix

DEFAULT_MODEL = Path("app/analytics/gnn_model.pt")


def evaluate(csv_path: Path, model_path: Path, threshold: float, window_seconds: float) -> None:
    print(f"[eval_gnn] loading {csv_path}")
    windows = build_windows(csv_path, window_seconds)
    print(f"[eval_gnn] built {len(windows)} graph windows")

    checkpoint = torch.load(model_path, map_location="cpu")
    mean = torch.tensor(checkpoint["feature_mean"])
    std  = torch.tensor(checkpoint["feature_std"])
    for w in windows:
        w.x = (w.x - mean) / (std + 1e-8)

    # Same split logic as gnn_train.py
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(windows))
    split = max(1, int(len(idx) * 0.8))
    val_w = [windows[i] for i in idx[split:]] or [windows[idx[-1]]]
    print(f"[eval_gnn] held-out validation windows: {len(val_w)}")

    model = GraphSAGENodeClassifier(in_channels=len(NODE_FEATURES))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    y_true, y_prob = [], []
    with torch.no_grad():
        for batch in DataLoader(val_w, batch_size=8):
            logits = model(batch.x, batch.edge_index)
            m = batch.train_mask if hasattr(batch, "train_mask") else torch.ones_like(batch.y, dtype=torch.bool)
            p = torch.sigmoid(logits[m]).numpy()
            y_prob.append(p)
            y_true.append(batch.y[m].numpy())
    y_prob = np.concatenate(y_prob)
    y_true = np.concatenate(y_true).astype(np.int32)
    y_pred = (y_prob >= threshold).astype(np.int32)

    print(f"[eval_gnn] scored {len(y_true):,} nodes "
          f"({int(y_true.sum())} suspicious / {int((1 - y_true).sum())} clean)")

    # Metrics
    if len(set(y_true.tolist())) > 1:
        roc = roc_auc_score(y_true, y_prob)
        prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(rec_curve, prec_curve)
    else:
        roc = pr_auc = float("nan")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    accuracy    = accuracy_score(y_true, y_pred)
    precision   = precision_score(y_true, y_pred, zero_division=0)
    recall      = recall_score(y_true, y_pred, zero_division=0)
    f1          = f1_score(y_true, y_pred, zero_division=0)
    #specificity = tn / (tn + fp) if (tn + fp) else 0.0
    fpr         = fp / (fp + tn) if (fp + tn) else 0.0
    
    save_confusion_matrix(
        np.array([[tn, fp], [fn, tp]]),
        Path.cwd() / "gnn_confusion_matrix.png",
        f"GraphSAGE GNN confusion matrix (threshold = {threshold:.2f})",
        ["Clean", "Suspicious"],
    )

    print()
    print("=" * 62)
    print(f"GraphSAGE GNN evaluation  (threshold = {threshold:.2f})")
    print("=" * 62)
    print(f"\nConfusion matrix (rows = actual, columns = predicted):")
    print(f"                  predicted clean    predicted suspicious")
    print(f"  actual clean    {tn:>15,d}    {fp:>20,d}")
    print(f"  actual suspect  {fn:>15,d}    {tp:>20,d}")
    print(f"\nThreshold-independent metrics:")
    print(f"  ROC-AUC             = {roc:.4f}")
    print(f"  PR-AUC              = {pr_auc:.4f}")
    print(f"\nAt threshold {threshold:.2f}:")
    print(f"  Accuracy            = {accuracy:.4f}")
    print(f"  Precision           = {precision:.4f}  (of nodes we flagged, % truly suspicious)")
    print(f"  Recall / Sensitivity= {recall:.4f}  (of suspicious nodes, % we caught)")
    print(f"  F1 score            = {f1:.4f}")
    #print(f"  Specificity         = {specificity:.4f}  (of clean nodes, % correctly cleared)")
    print(f"  False positive rate = {fpr:.4f}")
    print(f"\nPer-class breakdown:")
    print(classification_report(y_true, y_pred, target_names=["clean", "suspicious"], digits=4))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--model", default=DEFAULT_MODEL, type=Path)
    ap.add_argument("--threshold", default=0.5, type=float)
    ap.add_argument("--window-seconds", default=60.0, type=float,
                    help="MUST match the value used at training time (default 60).")
    args = ap.parse_args()
    evaluate(args.csv, args.model, args.threshold, args.window_seconds)
