"""Evaluate the saved Random Forest baseline on its held-out validation set.

Loads the trained checkpoint (rf_baseline_balanced.joblib), selects the 58
feature columns listed in feature_columns.json from val.parquet, and scores
the model. No retraining happens here.

Prints the same report as eval.py / eval_gnn.py and saves the confusion
matrix as random_forest_confusion_matrix.png.

Usage (run from the LLM-ZTNA folder):
    python -m app.analytics.eval_rf --model rf_baseline_balanced.joblib ^
        --val val.parquet --features feature_columns.json
    python -m app.analytics.eval_rf --model rf_baseline_balanced.joblib ^
        --val val.parquet --features feature_columns.json --threshold 0.4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
)

from app.analytics.cm_plot import save_confusion_matrix


def evaluate(model_path: Path, val_path: Path, features_path: Path,
             threshold: float = 0.5, out_dir: Path | None = None) -> None:
    out_dir = out_dir or Path.cwd()

    feature_cols = json.loads(features_path.read_text())
    model = joblib.load(model_path)

    df = pd.read_parquet(val_path)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"{val_path.name} is missing feature columns: {missing}")

    # Same columns, same order as at training time
    X_va = df[feature_cols]
    y_va = df["label"].to_numpy().astype(np.int32)
    types = df["type"].to_numpy() if "type" in df.columns else None

    y_prob = model.predict_proba(X_va)[:, 1]
    y_pred = (y_prob >= threshold).astype(np.int32)

    print("=" * 66)
    print(f"RANDOM FOREST EVALUATION (threshold = {threshold:.2f})")
    print("=" * 66)
    print(f"Validation rows: {len(y_va):,} "
          f"({int(y_va.sum()):,} attack / {int((1 - y_va).sum()):,} normal)\n")
    print(classification_report(y_va, y_pred, target_names=["normal", "attack"], digits=4))

    roc = roc_auc_score(y_va, y_prob) if len(set(y_va.tolist())) > 1 else float("nan")
    print(f"ROC-AUC: {roc:.4f}")

    cm = confusion_matrix(y_va, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    print("Confusion matrix (rows=true, cols=pred) [normal, attack]:")
    print(f"[[ {tn:>5d}  {fp:>5d}]")
    print(f" [ {fn:>5d}  {tp:>5d}]]")
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    print(f"False positive rate (normal misclassified as attack): {fpr:.4f}")

    if types is not None:
        print("Per-type recall (fraction of each attack type correctly flagged):")
        for t in sorted(set(types)):
            if t == "normal":
                continue
            mask = types == t
            print(f"{t:<12} {(y_pred[mask] == 1).mean():.4f}")

    save_confusion_matrix(
        cm,
        out_dir / "random_forest_confusion_matrix.png",
        f"Random Forest confusion matrix (threshold = {threshold:.2f})",
        ["Normal", "Attack"],
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=Path,
                    help="Path to rf_baseline_balanced.joblib")
    ap.add_argument("--val", required=True, type=Path,
                    help="Path to val.parquet")
    ap.add_argument("--features", required=True, type=Path,
                    help="Path to feature_columns.json")
    ap.add_argument("--threshold", default=0.5, type=float)
    ap.add_argument("--out", default=None, type=Path,
                    help="Output folder for the PNG (default: current folder)")
    args = ap.parse_args()
    evaluate(args.model, args.val, args.features, args.threshold, args.out)
