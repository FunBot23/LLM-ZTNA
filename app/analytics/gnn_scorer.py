"""Context scorer: rolling-window GNN inference for live flows.

Every observed flow calls `.observe(flow)`; scoring pulls the current window graph, runs GraphSAGE, returns the source-node's p(suspicious).
If the checkpoint is missing the scorer stays None and the PDP simply skips the context term.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from app.analytics.gnn_model import GraphSAGENodeClassifier
from app.analytics.graph_window import (
    RollingWindow,
    build_edge_index,
    collect_ips,
    compute_node_features,
)

DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "gnn_model.pt"


class ContextScorer:
    def __init__(self, checkpoint_path: Path) -> None:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.model = GraphSAGENodeClassifier(in_channels=ckpt["in_channels"])
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.mean = torch.tensor(ckpt["feature_mean"], dtype=torch.float32)
        self.std = torch.tensor(ckpt["feature_std"], dtype=torch.float32)
        self.window = RollingWindow(window_seconds=ckpt.get("window_seconds", 60.0))

    def observe(self, flow: dict) -> None:
        self.window.add(flow)

    @torch.no_grad()
    def score(self, src_ip: str) -> tuple[float, str]:
        """Returns (p_suspicious, short note) for src_ip in current window."""
        flows = self.window.current()
        if len(flows) < 2:
            return 0.5, "warming up"

        ips = collect_ips(flows)
        if src_ip not in ips or len(ips) < 2:
            return 0.5, "src not in current graph"

        ip_to_idx = {ip: i for i, ip in enumerate(ips)}
        edge_index = build_edge_index(flows, ip_to_idx)
        if edge_index.shape[1] == 0:
            return 0.5, "no edges in graph"

        x = torch.from_numpy(compute_node_features(flows, ips))
        x = (x - self.mean) / self.std
        ei = torch.from_numpy(edge_index)

        logits = self.model(x, ei)
        p = float(torch.sigmoid(logits[ip_to_idx[src_ip]]).item())

        peers = len({f["dst"] for f in flows if f["src"] == src_ip and f.get("dst")})
        return p, f"win={len(flows)}f/{len(ips)}n, {peers} peers"


_scorer: Optional[ContextScorer] = None
_load_attempted = False


def get_context_scorer() -> Optional[ContextScorer]:
    global _scorer, _load_attempted
    if _load_attempted:
        return _scorer
    _load_attempted = True
    if DEFAULT_CHECKPOINT.exists():
        try:
            _scorer = ContextScorer(DEFAULT_CHECKPOINT)
            print(f"[context_scorer] loaded {DEFAULT_CHECKPOINT}")
        except Exception as e:
            print(f"[context_scorer] failed to load ({e}); skipping context layer")
            _scorer = None
    else:
        print(f"[context_scorer] no checkpoint at {DEFAULT_CHECKPOINT}; skipping context layer")
    return _scorer
