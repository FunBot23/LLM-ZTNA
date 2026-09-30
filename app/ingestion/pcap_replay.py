"""PCAP replay ingestion.

Reads a packet capture, aggregates packets into flows, and feeds
completed flow events into the decision pipeline at (scaled) real time.

We now emit per-flow events (with features the Transformer
scorer needs) instead of per-packet signals.  
"""
import asyncio
import os
import time
from pathlib import Path
from typing import Optional

from scapy.all import PcapReader

from app.enforcement.pipeline import process_event
from app.ingestion.flow_aggregator import FlowAggregator

MAX_EVENTS_PER_SECOND = float(os.environ.get("ZTF_PCAP_MAX_RPS", "3"))


def _action_for(flow: dict) -> str:
    return f"{flow['proto'].upper()} \u2192 {flow['dst']}:{flow['dst_port']}"


async def replay(
    pcap_path: str,
    speed: float = 10.0,
    loop: bool = True,
    idle_timeout: float = 15.0,
    min_event_interval: float = 0.0,
) -> None:
    """Replay a PCAP into the decision pipeline as completed flow events."""
    path = Path(pcap_path)
    if not path.exists():
        print(f"[ingestion] PCAP not found: {path.resolve()}")
        return

    print(f"[ingestion] replaying {path.name} at {speed}x (loop={loop}, idle_timeout={idle_timeout}s)")

    events_this_second = 0
    second_start = time.monotonic()
    last_emit_at: Optional[float] = None

    async def emit(flow: dict) -> None:
        nonlocal events_this_second, second_start,last_emit_at
        now = time.monotonic()
        
        if min_event_interval > 0 and last_emit_at is not None:
            wait = min_event_interval - (now - last_emit_at)
            if wait > 0:
                await asyncio.sleep(wait)
            now = time.monotonic()
        if now - second_start >= 1.0:
            second_start = now
            events_this_second = 0
        if events_this_second >= MAX_EVENTS_PER_SECOND:
            return  # drop under load rather than backing up
        events_this_second += 1
        await process_event(
            src=flow["src"],
            action=_action_for(flow),
            features=flow,
        )

    while True:
        agg = FlowAggregator(idle_timeout=idle_timeout)
        first_pkt_ts: Optional[float] = None
        wall_start = time.monotonic()

        with PcapReader(str(path)) as reader:
            for pkt in reader:
                pkt_ts = float(pkt.time) if hasattr(pkt, "time") else time.time()

                if speed != float("inf"):
                    if first_pkt_ts is None:
                        first_pkt_ts = pkt_ts
                    target_offset = (pkt_ts - first_pkt_ts) / speed
                    delay = target_offset - (time.monotonic() - wall_start)
                    if delay > 0:
                        await asyncio.sleep(delay)

                await agg.observe(pkt, pkt_ts, emit)

        # Flush anything still open at end-of-file.
        await agg.flush(emit)

        if not loop:
            print(f"[ingestion] {path.name} finished")
            return
        print(f"[ingestion] {path.name} finished, restarting")
