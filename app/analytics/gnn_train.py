"""Train the short-window context GNN on TON_IoT flow CSV.
Usage from project root: python -m app.analytics.gnn_train --csv path/to/Train_Test_Network.csv

Bins flows by timestamp into `window_seconds`-wide graphs. Each graph
is trained on independently; no cross-window state.
Requires the CSV to have columns: ts, src_ip, dst_ip, dst_port, proto,
conn_state, duration, src_bytes, src_pkts, label.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from app.analytics.gnn_model import GraphSAGENodeClassifier
from app.analytics.graph_window import (
    NODE_FEATURES,
    build_edge_index,
    collect_ips,
    compute_node_features,
)

DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "gnn_model.pt"

REQUIRED_COLS = {
    "ts", "src_ip", "dst_ip", "dst_port",
    "proto", "conn_state", "duration",
    "src_bytes", "src_pkts", "label",
}


def _safe_float(value, default: float = 0.0) -> float:
    """Coerce to float; non-numeric junk (e.g. corrupted rows with an IP
    string leaked into a numeric column) falls back to default instead
    of crashing the whole run."""
    try:
        f = float(value)
        if f != f:  # NaN check
            return default
        return f
    except (ValueError, TypeError):
        return default
    
    
def _row_to_flow(row) -> dict:
    """Row from CSV → flow dict with keys matching the live aggregator."""
    return {
        "src": str(row["src_ip"]),
        "dst": str(row["dst_ip"]),
        "dst_port": int(row.get("dst_port", 0) or 0),
        "proto": str(row.get("proto", "tcp")).lower(),
        "conn_state": str(row.get("conn_state", "OTH")),
        "duration": _safe_float(row.get("duration", 0) or 0),
        "src_bytes": _safe_float(row.get("src_bytes", 0) or 0),
        "src_pkts": _safe_float(row.get("src_pkts", 0) or 0),
        "label": int(row["label"]),
    }


def build_windows(csv_path: Path, window_seconds: float) -> list[Data]:
    df = pd.read_csv(csv_path, low_memory=False)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

    t0 = float(df["ts"].iloc[0])
    df["bin"] = ((df["ts"] - t0) // window_seconds).astype(int)

    windows: list[Data] = []
    for _bin_id, group in df.groupby("bin"):
        flows = [_row_to_flow(r) for _, r in group.iterrows()]
        if len(flows) < 2:
            continue

        ips = collect_ips(flows)
        if len(ips) < 2:
            continue
        ip_to_idx = {ip: i for i, ip in enumerate(ips)}

        x = compute_node_features(flows, ips)
        edge_index = build_edge_index(flows, ip_to_idx)
        if edge_index.shape[1] == 0:
            continue

        # Node label = max flow-label over this node's src-flows.
        y = np.zeros(len(ips), dtype=np.float32)
        mask = np.zeros(len(ips), dtype=bool)
        for f in flows:
            i = ip_to_idx.get(f["src"])
            if i is None:
                continue
            mask[i] = True
            if f["label"] == 1:
                y[i] = 1.0

        windows.append(
            Data(
                x=torch.from_numpy(x).float(),
                edge_index=torch.from_numpy(edge_index),
                y=torch.from_numpy(y),
                train_mask=torch.from_numpy(mask),
            )
        )

    print(f"[gnn_train] built {len(windows)} graph windows (window={window_seconds}s)")
    if not windows:
        raise RuntimeError("no valid windows produced — check `ts` and label distribution")
    return windows


def train(csv_path: Path, out_path: Path, epochs: int, lr: float, window_seconds: float) -> None:
    windows = build_windows(csv_path, window_seconds)

    # Feature normalisation stats (over all training-node rows).
    all_x = torch.cat([w.x[w.train_mask] for w in windows], dim=0)
    mean = all_x.mean(dim=0)
    std = all_x.std(dim=0) + 1e-6
    for w in windows:
        w.x = (w.x - mean) / std

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(windows))
    split = max(1, int(len(idx) * 0.8))
    train_w = [windows[i] for i in idx[:split]]
    val_w = [windows[i] for i in idx[split:]] or [windows[idx[-1]]]

    tr = DataLoader(train_w, batch_size=8, shuffle=True)
    va = DataLoader(val_w, batch_size=8)

    model = GraphSAGENodeClassifier(in_channels=len(NODE_FEATURES))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    all_y = torch.cat([w.y[w.train_mask] for w in train_w])
    pos = float(all_y.sum()); neg = float(len(all_y) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0; n = 0
        for batch in tr:
            opt.zero_grad()
            logits = model(batch.x, batch.edge_index)
            m = batch.train_mask
            loss = loss_fn(logits[m], batch.y[m])
            loss.backward()
            opt.step()
            total += loss.item() * int(m.sum()); n += int(m.sum())

        model.eval()
        with torch.no_grad():
            all_p, all_yv = [], []
            for batch in va:
                logits = model(batch.x, batch.edge_index)
                m = batch.train_mask
                all_p.append(torch.sigmoid(logits[m]).cpu().numpy())
                all_yv.append(batch.y[m].cpu().numpy())
            p = np.concatenate(all_p); yv = np.concatenate(all_yv)
            try:
                auc = roc_auc_score(yv, p) if len(set(yv.tolist())) > 1 else float("nan")
            except ValueError:
                auc = float("nan")
            acc = float(((p >= 0.5) == yv).mean())

        print(f"[gnn_train] epoch {epoch:2d}  loss={total/max(n,1):.4f}  val_auc={auc:.4f}  val_acc={acc:.4f}")

    torch.save(
        {
            "model_state": model.state_dict(),
            "in_channels": len(NODE_FEATURES),
            "feature_mean": mean.tolist(),
            "feature_std": std.tolist(),
            "node_features": NODE_FEATURES,
            "window_seconds": window_seconds,
        },
        out_path,
    )
    print(f"[gnn_train] saved to {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--out", default=DEFAULT_CHECKPOINT, type=Path)
    ap.add_argument("--epochs", default=15, type=int)
    ap.add_argument("--lr", default=5e-3, type=float)
    ap.add_argument("--window-seconds", default=60.0, type=float)
    args = ap.parse_args()
    train(args.csv, args.out, args.epochs, args.lr, args.window_seconds)


if __name__ == "__main__":
    main()
