"""
api_flow_tracker.py — derives synthetic flow features from live HTTP
requests to the mock device API, so requests that don't come from a
PCAP-derived flow (i.e. anything hitting mock_api.py directly) still get
scored by the real Transformer/GNN pipeline instead of falling back to
the static per-device hash stub.

Produces a dict shaped identically to FlowAggregator._finalize()'s output,
so scorer.py and gnn_scorer.py consume it exactly as they would a real
pcap-derived flow.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

# Rolling window over which we judge "burstiness" for a device.
WINDOW_SECONDS = 60.0
# Requests within this window count as a "high frequency" burst.
BURST_THRESHOLD = 8

# Rough per-request byte estimates by HTTP method — not exact, just
# plausible enough to give the model believable magnitudes.
_METHOD_BYTES = {
    "GET": (180, 260),      # (request bytes, response bytes)
    "POST": (420, 240),
}

# All API traffic terminates at our own server — modeling it as a single
# node gives the GNN a realistic star-topology graph (every device really
# does talk to one central management API).
SERVER_NODE_ID = "ztf-server"

# HTTP traffic is modeled on this synthetic port so service_from_port()
# resolves it to "http", a category the model has real training signal for.
SYNTHETIC_DST_PORT = 8080


@dataclass
class _DeviceWindow:
    timestamps: deque[float] = field(default_factory=deque)

    def record(self, now: float) -> None:
        self.timestamps.append(now)
        cutoff = now - WINDOW_SECONDS
        while self.timestamps and self.timestamps[0] < cutoff:
            self.timestamps.popleft()

    def count(self) -> int:
        return len(self.timestamps)

    def span(self, now: float) -> float:
        if not self.timestamps:
            return 0.0
        return max(0.0, now - self.timestamps[0])


class ApiFlowTracker:
    def __init__(self) -> None:
        self._windows: dict[str, _DeviceWindow] = {}

    def observe(self, device_id: str, method: str, path: str) -> dict:
        """Record this request and return a flow-shaped feature dict."""
        now = time.monotonic()
        window = self._windows.setdefault(device_id, _DeviceWindow())
        window.record(now)

        count = window.count()
        span = window.span(now)
        is_burst = count >= BURST_THRESHOLD

        req_bytes, resp_bytes = _METHOD_BYTES.get(method.upper(), (200, 200))

        return {
            "src": device_id,
            "dst": SERVER_NODE_ID,
            "src_port": 0,
            "dst_port": SYNTHETIC_DST_PORT,
            "proto": "tcp",
            "service": "http",
            # Rapid repeated requests look like an incomplete/scanning
            # pattern (S0); steady, spaced-out requests look like a
            # normal completed exchange (SF) — mirrors what the model
            # already learned to associate with each state.
            "conn_state": "S0" if is_burst else "SF",
            "duration": span,
            "src_bytes": float(req_bytes),
            "dst_bytes": float(resp_bytes),
            "src_pkts": float(count),
            "dst_pkts": float(count),
            "src_ip_bytes": float(req_bytes + 40 * count),
            "dst_ip_bytes": float(resp_bytes + 40 * count),
        }


_tracker: ApiFlowTracker | None = None


def get_tracker() -> ApiFlowTracker:
    global _tracker
    if _tracker is None:
        _tracker = ApiFlowTracker()
    return _tracker
