"""Plot single-run confusion matrices (raw counts), like the GNN evaluation printout.

Rows = actual, columns = predicted, threshold 0.50.

Usage (run from the LLM-ZTNA folder):
    # 1) Exactly the matrix from the GNN evaluation screenshot
    python -m app.analytics.plot_confusion

    # 2) One panel per model, taken from a single run in comparison_runs.csv
    python -m app.analytics.plot_confusion --from-csv comparison_runs.csv --run 0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Confusion matrix from the GraphSAGE evaluation printout: [[TN, FP], [FN, TP]]
GNN_SCREENSHOT_CM = np.array([[4158, 3], [1, 129]])


def _metrics_line(cm: np.ndarray) -> str:
    """Precision / recall / F1 / FPR computed from a 2x2 count matrix."""
    tn, fp = cm[0]
    fn, tp = cm[1]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr       = fp / (fp + tn) if (fp + tn) else 0.0
    return f"Precision {precision:.4f}  |  Recall {recall:.4f}  |  F1 {f1:.4f}  |  FPR {fpr:.4f}"


def _draw_cm(ax, cm: np.ndarray, title: str, class_names: list[str]) -> None:
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(cm.max(), 1))
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, f"{int(cm[i, j]):,}",
                ha="center", va="center", fontsize=16, fontweight="bold",
                color="white" if cm[i, j] > 0.5 * cm.max() else "black",
            )
    ax.set_xticks([0, 1]); ax.set_xticklabels([f"Predicted\n{c}" for c in class_names])
    ax.set_yticks([0, 1]); ax.set_yticklabels([f"Actual\n{c}" for c in class_names])
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(_metrics_line(cm), fontsize=8.5, labelpad=12)


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    try:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix -> {path}")
    except OSError as e:
        print(f"Could not save {path}: {e}")
    finally:
        plt.close(fig)


def plot_screenshot_matrix(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    _draw_cm(ax, GNN_SCREENSHOT_CM,
             "GraphSAGE GNN confusion matrix (threshold = 0.50)",
             ["Clean", "Suspicious"])
    _save(fig, out / "gnn_confusion_matrix.png")


def plot_from_csv(csv_path: Path, run: int, out: Path) -> None:
    df = pd.read_csv(csv_path)
    needed = {"tn", "fp", "fn", "tp"}
    if not needed.issubset(df.columns):
        raise SystemExit(
            f"{csv_path} has no confusion counts (tn/fp/fn/tp). "
            "Rerun compare_models.py once to regenerate it."
        )

    models = list(dict.fromkeys(df["model"]))  # keep original order
    fig, axes = plt.subplots(1, len(models), figsize=(5.5 * len(models), 5.2), squeeze=False)

    for ax, model in zip(axes[0], models):
        sub = df[(df["model"] == model) & (df["run"] == run)]
        if sub.empty:
            raise SystemExit(f"No run {run} found for {model}.")
        r = sub.iloc[0]
        cm = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]], dtype=int)
        _draw_cm(ax, cm, f"{model} (run {run})", ["Normal", "Attack"])

    fig.suptitle("Confusion matrices (counts, threshold = 0.50)", y=1.02)
    _save(fig, out / "confusion_matrices_single_run.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-csv", type=Path, default=None,
                    help="comparison_runs.csv to plot one run per model from")
    ap.add_argument("--run", type=int, default=0,
                    help="Run index to plot when using --from-csv (default 0)")
    ap.add_argument("--out", type=Path, default=Path.cwd(),
                    help="Output folder (default: current folder)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    if args.from_csv:
        plot_from_csv(args.from_csv, args.run, args.out)
    else:
        plot_screenshot_matrix(args.out)
