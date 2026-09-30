"""Shared confusion-matrix plot (raw counts) for the eval scripts.

Both eval.py and eval_gnn.py call save_confusion_matrix() with the same
counts they print, so the figure can never disagree with the console output.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _metrics_line(cm: np.ndarray) -> str:
    tn, fp = cm[0]
    fn, tp = cm[1]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr       = fp / (fp + tn) if (fp + tn) else 0.0
    return f"Precision {precision:.4f}  |  Recall {recall:.4f}  |  F1 {f1:.4f}  |  FPR {fpr:.4f}"


def save_confusion_matrix(cm, path: Path, title: str, class_names: list[str]) -> None:
    """Save a 2x2 count matrix. Rows = actual, columns = predicted."""
    cm = np.asarray(cm, dtype=int).reshape(2, 2)

    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(cm.max(), 1))
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, f"{cm[i, j]:,}",
                ha="center", va="center", fontsize=16, fontweight="bold",
                color="white" if cm[i, j] > 0.5 * cm.max() else "black",
            )
    ax.set_xticks([0, 1]); ax.set_xticklabels([f"Predicted\n{c}" for c in class_names])
    ax.set_yticks([0, 1]); ax.set_yticklabels([f"Actual\n{c}" for c in class_names])
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(_metrics_line(cm), fontsize=8.5, labelpad=12)
    fig.tight_layout()

    try:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"\nConfusion matrix plot -> {path}")
    except OSError as e:
        print(f"\nCould not save {path}: {e}")
    finally:
        plt.close(fig)
