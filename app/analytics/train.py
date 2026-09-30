"""Train the flow-level Transformer on TON_IoT's Train_Test_Network.csv.

Usage from the project root:
    python -m app.analytics.train --csv path/to/Train_Test_Network.csv

The checkpoint includes the model weights, the continuous-feature
normalisation stats, and the train-set medians used for imputation at
inference. Everything the scorer needs is in one file.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from app.analytics.features import (
    CATEGORICAL_COLS,
    CATEGORICAL_VOCABS,
    CONTINUOUS_COLS,
    encode_categorical,
)
from app.analytics.model import TabTransformer

DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "model.pt"


def load_and_encode(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path, low_memory=False)

    missing = [c for c in CATEGORICAL_COLS + CONTINUOUS_COLS + ["label"] if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {missing}. "
            f"Expected TON_IoT Train_Test_Network.csv schema."
        )

    # Categoricals → vocab indices.
    cat = np.stack(
        [df[col].map(lambda v: encode_categorical(col, v)).to_numpy() for col in CATEGORICAL_COLS],
        axis=1,
    ).astype(np.int64)

    # Continuous → coerce to float, clip negatives, log1p (heavy-tailed).
    cont_raw = df[CONTINUOUS_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    cont_raw = cont_raw.clip(lower=0.0)
    cont = np.log1p(cont_raw.to_numpy(dtype=np.float32))

    y = df["label"].astype(int).to_numpy()
    return cat, cont, y


def train(csv_path: Path, out_path: Path, epochs: int, batch_size: int, lr: float) -> None:
    print(f"[train] loading {csv_path}")
    cat, cont, y = load_and_encode(csv_path)
    print(f"[train] {len(y):,} rows; class balance: {int(y.sum())} pos / {int((1 - y).sum())} neg")

    # Save normalisation stats + imputation medians from the full pre-split data.
    cont_mean = cont.mean(axis=0)
    cont_std = cont.std(axis=0) + 1e-6
    cont_median = np.median(cont, axis=0)

    cont_norm = (cont - cont_mean) / cont_std

    x_cat_tr, x_cat_va, x_c_tr, x_c_va, y_tr, y_va = train_test_split(
        cat, cont_norm, y, test_size=0.2, random_state=42, stratify=y,
    )

    tr = DataLoader(
        TensorDataset(
            torch.from_numpy(x_cat_tr), torch.from_numpy(x_c_tr).float(),
            torch.from_numpy(y_tr).float(),
        ),
        batch_size=batch_size, shuffle=True,
    )
    va = DataLoader(
        TensorDataset(
            torch.from_numpy(x_cat_va), torch.from_numpy(x_c_va).float(),
            torch.from_numpy(y_va).float(),
        ),
        batch_size=batch_size,
    )

    model = TabTransformer(
        cat_cardinalities=[len(CATEGORICAL_VOCABS[c]) for c in CATEGORICAL_COLS],
        num_continuous=len(CONTINUOUS_COLS),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Positive class weight to counter imbalance if any.
    pos = float(y_tr.sum()); neg = float(len(y_tr) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0; n = 0
        for xc, xn, yb in tr:
            opt.zero_grad()
            logits = model(xc, xn)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total += loss.item() * yb.size(0); n += yb.size(0)

        model.eval()
        with torch.no_grad():
            all_p, all_y = [], []
            for xc, xn, yb in va:
                p = torch.sigmoid(model(xc, xn)).cpu().numpy()
                all_p.append(p); all_y.append(yb.numpy())
            p = np.concatenate(all_p); yv = np.concatenate(all_y)
            try:
                auc = roc_auc_score(yv, p)
            except ValueError:
                auc = float("nan")
            acc = float(((p >= 0.5) == yv).mean())
        print(f"[train] epoch {epoch:2d}  train_loss={total/n:.4f}  val_auc={auc:.4f}  val_acc={acc:.4f}")

    torch.save(
        {
            "model_state": model.state_dict(),
            "cat_cardinalities": [len(CATEGORICAL_VOCABS[c]) for c in CATEGORICAL_COLS],
            "num_continuous": len(CONTINUOUS_COLS),
            "cont_mean": cont_mean.tolist(),
            "cont_std": cont_std.tolist(),
            "cont_median_log1p": cont_median.tolist(),
            "feature_schema": {
                "categorical": CATEGORICAL_COLS,
                "continuous": CONTINUOUS_COLS,
            },
        },
        out_path,
    )
    print(f"[train] saved checkpoint to {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--out", default=DEFAULT_CHECKPOINT, type=Path)
    ap.add_argument("--epochs", default=8, type=int)
    ap.add_argument("--batch-size", default=512, type=int)
    ap.add_argument("--lr", default=1e-3, type=float)
    args = ap.parse_args()
    train(args.csv, args.out, args.epochs, args.batch_size, args.lr)


if __name__ == "__main__":
    main()
