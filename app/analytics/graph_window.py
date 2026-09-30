"""Graph window construction — shared by GNN training and live inference.

Two responsibilities, kept in one file so the training and inference
paths cannot drift on node-feature definitions:

  1. compute_node_features / build_edge_index: pure functions that turn a
     list of flow dicts into a graph (node features + edge_index).
  2. RollingWindow: time-based sliding window of recent flows for live
     inference. Evicts flows older than `window_seconds`.

This is *short-window neighborhood behavior*, not "device
reputation over time". Windows are independent,nothing carries across
them.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Iterable

import numpy as np

# Per-node features, computed for each unique IP in a window.
# Order matters — model was trained on this ordering.
NODE_FEATURES = [
    "num_flows_as_src",         # activity volume
    "num_distinct_dst",         # fan-out (unique peers)
    "num_distinct_dst_ports",   # port diversity (scan indicator)
    "total_src_bytes_log",      # data volume
    "total_src_pkts_log",       # packet volume
    "avg_duration",             # short flows = scans/probes
    "frac_tcp",                 # protocol mix
    "frac_s0",                  # SYN-only fraction (unreplied → scans)
    "frac_rej",                 # rejected fraction (blocked attempts)
]


def compute_node_features(flows: list[dict], ips: list[str]) -> np.ndarray:
    """Aggregate per-source features from `flows` for each ip in `ips`.

    Nodes that only appear as destinations (never as src in this window)
    get all-zero feature rows — that's fine; the GNN can still route
    messages through them.
    """
    ip_to_idx = {ip: i for i, ip in enumerate(ips)}
    x = np.zeros((len(ips), len(NODE_FEATURES)), dtype=np.float32)

    by_src: dict[str, list[dict]] = defaultdict(list)
    for f in flows:
        by_src[f["src"]].append(f)

    for src, src_flows in by_src.items():
        if src not in ip_to_idx:
            continue
        i = ip_to_idx[src]
        n = len(src_flows)
        distinct_dst = len({f.get("dst") for f in src_flows if f.get("dst")})
        distinct_ports = len({int(f.get("dst_port", 0)) for f in src_flows})
        total_bytes = sum(float(f.get("src_bytes", 0)) for f in src_flows)
        total_pkts = sum(float(f.get("src_pkts", 0)) for f in src_flows)
        avg_dur = float(np.mean([float(f.get("duration", 0)) for f in src_flows]))
        frac_tcp = sum(1 for f in src_flows if str(f.get("proto", "")).lower() == "tcp") / n
        frac_s0 = sum(1 for f in src_flows if f.get("conn_state") == "S0") / n
        frac_rej = sum(1 for f in src_flows if f.get("conn_state") == "REJ") / n

        x[i] = [
            n, distinct_dst, distinct_ports,
            float(np.log1p(total_bytes)),
            float(np.log1p(total_pkts)),
            avg_dur,
            frac_tcp, frac_s0, frac_rej,
        ]
    return x


def build_edge_index(flows: list[dict], ip_to_idx: dict[str, int]) -> np.ndarray:
    """Directed edges from src → dst. Shape (2, num_edges) int64."""
    edges = []
    for f in flows:
        s = f.get("src"); d = f.get("dst")
        if s in ip_to_idx and d in ip_to_idx:
            edges.append([ip_to_idx[s], ip_to_idx[d]])
    if not edges:
        return np.zeros((2, 0), dtype=np.int64)
    return np.asarray(edges, dtype=np.int64).T


def collect_ips(flows: Iterable[dict]) -> list[str]:
    ips: set[str] = set()
    for f in flows:
        s = f.get("src")
        if s: ips.add(s)
        d = f.get("dst")
        if d: ips.add(d)
    return sorted(ips)


class RollingWindow:
    """Time-based sliding window of flow events for live inference.

    Time is monotonic wall-clock (time.monotonic()), not the packet
    timestamp. This means at PCAP replay speeds >1x the window covers
    the same wall-clock seconds regardless of playback speed — which is
    what we want for a live dashboard.
    """

    def __init__(self, window_seconds: float = 60.0) -> None:
        self.window = window_seconds
        self._buf: deque[tuple[float, dict]] = deque()

    def add(self, flow: dict) -> None:
        now = time.monotonic()
        self._buf.append((now, flow))
        self._evict(now)

    def _evict(self, now: float) -> None:
        cutoff = now - self.window
        while self._buf and self._buf[0][0] < cutoff:
            self._buf.popleft()

    def current(self) -> list[dict]:
        self._evict(time.monotonic())
        return [f for _, f in self._buf]

    def __len__(self) -> int:
        self._evict(time.monotonic())
        return len(self._buf)
