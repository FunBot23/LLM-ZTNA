""" 
Usage:
    python -m app.analytics.eval --csv <TON_IoT>.csv
    python -m app.analytics.eval --csv <TON_IoT>.csv --threshold 0.4
Evaluate the TabTransformer on the held-out validation split.

Output format matches the project's standard evaluation display:
  - precision / recall / f1-score / support table
  - macro and weighted averages
  - ROC-AUC
  - Confusion matrix (rows=true, cols=pred)
  - False positive rate
  - Per-attack-type recall
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
)
from sklearn.model_selection import train_test_split

from app.analytics.features import (
    CATEGORICAL_COLS, CATEGORICAL_VOCABS, CONTINUOUS_COLS,
)
from app.analytics.model import TabTransformer
from app.analytics.train import load_and_encode
from app.analytics.cm_plot import save_confusion_matrix
DEFAULT_MODEL = Path("app/analytics/model.pt")


def evaluate(csv_path: Path, model_path: Path, threshold: float = 0.5) -> dict:
    cat, cont, y = load_and_encode(csv_path)
    types = pd.read_csv(csv_path, low_memory=False, usecols=["type"])["type"].to_numpy()

    checkpoint = torch.load(model_path, map_location="cpu")
    cont_mean = np.array(checkpoint["cont_mean"])
    cont_std  = np.array(checkpoint["cont_std"])
    cont_norm = (cont - cont_mean) / (cont_std + 1e-8)

    _, x_cat_va, _, x_c_va, _, y_va, _, types_va = train_test_split(
        cat, cont_norm, y, types,
        test_size=0.2, random_state=42, stratify=y,
    )

    model = TabTransformer(
        cat_cardinalities=[len(CATEGORICAL_VOCABS[c]) for c in CATEGORICAL_COLS],
        num_continuous=len(CONTINUOUS_COLS),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    probs = []
    with torch.no_grad():
        for i in range(0, len(y_va), 4096):
            cat_b  = torch.from_numpy(x_cat_va[i:i+4096])
            cont_b = torch.from_numpy(x_c_va[i:i+4096]).float()
            probs.append(torch.sigmoid(model(cat_b, cont_b)).numpy())
    p_mal  = np.concatenate(probs)
    y_pred = (p_mal >= threshold).astype(np.int32)

    return _print_report(y_va, y_pred, p_mal, types_va, title="EVALUATION (balanced)", threshold=threshold)


def _print_report(y_true, y_pred, y_prob, types, title: str, threshold:float = 0.5) -> dict:
    print("=" * 66)
    print(title)
    print("=" * 66)

    # 1. Precision / recall / f1 table
    print(classification_report(
        y_true, y_pred,
        target_names=["normal", "attack"],
        digits=2,
    ))

    # 2. ROC-AUC
    roc = roc_auc_score(y_true, y_prob) if len(set(y_true.tolist())) > 1 else float("nan")
    print(f"ROC-AUC: {roc:.4f}")

    # 3. Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    print("Confusion matrix (rows=true, cols=pred) [normal, attack]:")
    print(f"[[ {tn:>5d}  {fp:>5d}]")
    print(f" [ {fn:>5d}  {tp:>5d}]]")
    
    save_confusion_matrix(
        cm,
        Path.cwd() / "tabtransformer_confusion_matrix.png",
        f"TabTransformer confusion matrix (threshold = {threshold:.2f})",
        ["Normal", "Attack"],
    )

    # 4. False positive rate
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    print(f"False positive rate (normal misclassified as attack): {fpr:.4f}")

    # 5. Per-attack-type recall
    print("Per-type recall (fraction of each attack type correctly flagged as label=1):")
    print("type")
    for t in sorted(set(types)):
        if t == "normal":
            continue
        mask = types == t
        if mask.sum() == 0:
            continue
        recall_t = (y_pred[mask] == 1).mean()
        print(f"{t:<12} {recall_t:.4f}")
    print("dtype: float64")

    return {
        "roc_auc": float(roc),
        "fpr": float(fpr),
        "confusion_matrix": cm.tolist(),
        "precision_attack": float(tp / (tp + fp)) if (tp + fp) else 0.0,
        "recall_attack":    float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "precision_normal": float(tn / (tn + fn)) if (tn + fn) else 0.0,
        "recall_normal":    float(tn / (tn + fp)) if (tn + fp) else 0.0,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--model", default=DEFAULT_MODEL, type=Path)
    ap.add_argument("--threshold", default=0.5, type=float)
    args = ap.parse_args()
    evaluate(args.csv, args.model, args.threshold)