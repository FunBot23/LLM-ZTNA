"""Flow aggregator.

Accumulates packets by 5-tuple, tracks direction (originator vs responder),
and emits a completed flow event when either:
  * a FIN or RST is seen on the connection, or
  * the flow has been idle for `idle_timeout` seconds.

Emitted event carries the feature set the Transformer scorer expects:
proto, service, conn_state, duration, {src,dst}_{bytes,pkts,ip_bytes}.

Direction convention: the first packet of a flow defines the originator
("src" from the model's perspective). Subsequent packets in the reverse
direction accumulate on the "dst" side. Matches Zeek's convention.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

from scapy.all import IP, TCP, UDP

from app.analytics.features import conn_state_from_flags, service_from_port


@dataclass
class _FlowState:
    orig_ip: str
    resp_ip: str
    orig_port: int
    resp_port: int
    proto: str
    first_ts: float
    last_ts: float
    src_pkts: int = 0
    dst_pkts: int = 0
    src_bytes: int = 0        # L4 payload bytes from originator
    dst_bytes: int = 0        # L4 payload bytes from responder
    src_ip_bytes: int = 0     # total IP bytes from originator
    dst_ip_bytes: int = 0     # total IP bytes from responder
    flags: set[str] = field(default_factory=set)  # union of all TCP flags seen
    had_response: bool = False


class FlowAggregator:
    def __init__(self, idle_timeout: float = 15.0) -> None:
        self.idle_timeout = idle_timeout
        # Key is the *unordered* 5-tuple so both directions land in the same slot.
        self._flows: dict[tuple, _FlowState] = {}

    def _key(self, a_ip, a_port, b_ip, b_port, proto) -> tuple:
        # Symmetric key so (A→B) and (B→A) hash to the same flow.
        endpoints = tuple(sorted([(a_ip, a_port), (b_ip, b_port)]))
        return (*endpoints[0], *endpoints[1], proto)

    async def observe(
        self,
        pkt,
        pkt_ts: float,
        on_flow: Callable[[dict], Awaitable[None]],
    ) -> None:
        """Feed one packet in. Emits completed flows via `on_flow` callback."""
        if IP not in pkt:
            return

        ip = pkt[IP]
        if TCP in pkt:
            l4 = pkt[TCP]
            proto = "tcp"
            payload_len = len(bytes(l4.payload))
            flag_bits = str(l4.flags)  # e.g. "SA", "PA", "FA"
        elif UDP in pkt:
            l4 = pkt[UDP]
            proto = "udp"
            payload_len = max(0, int(l4.len) - 8)  # UDP header is 8 bytes
            flag_bits = ""
        else:
            return

        ip_bytes = int(ip.len) if hasattr(ip, "len") and ip.len else len(bytes(ip))

        key = self._key(ip.src, int(l4.sport), ip.dst, int(l4.dport), proto)
        state = self._flows.get(key)

        if state is None:
            # First packet defines the originator direction.
            state = _FlowState(
                orig_ip=ip.src, resp_ip=ip.dst,
                orig_port=int(l4.sport), resp_port=int(l4.dport),
                proto=proto,
                first_ts=pkt_ts, last_ts=pkt_ts,
            )
            self._flows[key] = state

        state.last_ts = pkt_ts
        from_orig = (ip.src == state.orig_ip and int(l4.sport) == state.orig_port)

        if from_orig:
            state.src_pkts += 1
            state.src_bytes += payload_len
            state.src_ip_bytes += ip_bytes
        else:
            state.dst_pkts += 1
            state.dst_bytes += payload_len
            state.dst_ip_bytes += ip_bytes
            state.had_response = True

        # Track TCP flags; letters used mirror Zeek/conn_state_from_flags.
        if proto == "tcp":
            if "S" in flag_bits: state.flags.add("S")
            if "F" in flag_bits: state.flags.add("F")
            if "R" in flag_bits: state.flags.add("R")
            if "A" in flag_bits: state.flags.add("A")

            # Emit immediately on graceful close or reset.
            if "F" in flag_bits or "R" in flag_bits:
                await on_flow(self._finalize(key, state))
                return

        # Otherwise let sweep_timeouts handle emission.
        await self.sweep_timeouts(pkt_ts, on_flow)

    async def sweep_timeouts(
        self,
        now_ts: float,
        on_flow: Callable[[dict], Awaitable[None]],
    ) -> None:
        expired = [k for k, s in self._flows.items() if now_ts - s.last_ts >= self.idle_timeout]
        for k in expired:
            await on_flow(self._finalize(k, self._flows[k]))

    async def flush(self, on_flow: Callable[[dict], Awaitable[None]]) -> None:
        """Emit and clear everything still open. Call at PCAP EOF."""
        for k in list(self._flows.keys()):
            await on_flow(self._finalize(k, self._flows[k]))

    def _finalize(self, key: tuple, state: _FlowState) -> dict:
        del self._flows[key]
        return {
            "src": state.orig_ip,
            "dst": state.resp_ip,
            "src_port": state.orig_port,
            "dst_port": state.resp_port,
            "proto": state.proto,
            "service": service_from_port(state.resp_port),
            "conn_state": conn_state_from_flags(state.flags, state.had_response, state.proto),
            "duration": max(0.0, state.last_ts - state.first_ts),
            "src_bytes": state.src_bytes,
            "dst_bytes": state.dst_bytes,
            "src_pkts": state.src_pkts,
            "dst_pkts": state.dst_pkts,
            "src_ip_bytes": state.src_ip_bytes,
            "dst_ip_bytes": state.dst_ip_bytes,
        }
