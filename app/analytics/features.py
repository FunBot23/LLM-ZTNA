"""Feature schema.

Both the training script (reading Train_Test_Network.csv) and the live
scorer (reading Scapy-derived flow events) map into these same columns
with the same encoders.

Feature choice rationale: we only include columns extractable both from
TON_IoT's CSV and from live Scapy flows without DPI.
"""
from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Vocabularies — fixed at training time. OOV values map to index 0.
# ---------------------------------------------------------------------------

PROTO_VOCAB = ["<unk>", "tcp", "udp", "icmp"]

# TON_IoT service labels commonly present; extend if your CSV has more.
SERVICE_VOCAB = [
    "<unk>", "-", "dns", "http", "ssl", "ssh", "telnet", "ftp",
    "smtp", "rdp", "smb", "mqtt", "dhcp", "irc", "radius",
]

# Zeek/TON_IoT conn_state codes; "<unk>" catches anything unfamiliar.
CONN_STATE_VOCAB = [
    "<unk>", "SF", "S0", "REJ", "RSTO", "RSTR",
    "RSTOS0", "RSTRH", "SH", "SHR", "OTH", "S1", "S2", "S3",
]

CATEGORICAL_COLS = ["proto", "service", "conn_state"]
CATEGORICAL_VOCABS = {
    "proto": PROTO_VOCAB,
    "service": SERVICE_VOCAB,
    "conn_state": CONN_STATE_VOCAB,
}

CONTINUOUS_COLS = [
    "duration",
    "src_bytes", "dst_bytes",
    "src_pkts", "dst_pkts",
    "src_ip_bytes", "dst_ip_bytes",
]

# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def encode_categorical(col: str, value: Any) -> int:
    """Map a raw categorical value to its vocab index. Unknown → 0 (<unk>)."""
    vocab = CATEGORICAL_VOCABS[col]
    try:
        # Normalise to lowercase string for robustness.
        v = str(value).strip().lower() if value is not None else "<unk>"
        return vocab.index(v)
    except ValueError:
        return 0


def encode_continuous(value: Any, median: float = 0.0) -> float:
    """Coerce to float and impute NaN/None with the train-set median."""
    if value is None:
        return median
    try:
        f = float(value)
        if f != f:  # NaN
            return median
        return max(0.0, f)  # clip negatives (these features are all non-negative)
    except (ValueError, TypeError):
        return median


# ---------------------------------------------------------------------------
# Live-side derivations — port → service, TCP flags → conn_state.
# Used by the flow aggregator, kept here so training and inference stay
# textually consistent.
# ---------------------------------------------------------------------------

_PORT_SERVICE = {
    53: "dns", 80: "http", 8080: "http", 443: "ssl", 22: "ssh",
    23: "telnet", 21: "ftp", 25: "smtp", 3389: "rdp",
    445: "smb", 1883: "mqtt", 8883: "mqtt", 67: "dhcp", 68: "dhcp", 1880: "http"
}


def service_from_port(dst_port: int) -> str:
    return _PORT_SERVICE.get(int(dst_port), "-")


def conn_state_from_flags(flags_seen: set[str], had_response: bool, proto: str = "tcp") -> str:
    """Rough Zeek-style conn_state from accumulated TCP flags.

    Not a perfect Zeek replica — Zeek tracks per-direction flag history.
    This heuristic covers the common cases seen in TON_IoT.
    """
    if proto!="tcp":
        return "SF" if had_response else "S0"
    has_syn = "S" in flags_seen
    has_fin = "F" in flags_seen
    has_rst = "R" in flags_seen

    if has_syn and has_fin:
        return "SF"
    if has_syn and has_rst:
        return "REJ" if not had_response else "RSTO"
    if has_syn and not had_response:
        return "S0"
    if has_syn and had_response and not has_fin and not has_rst:
        return "S1"
    return "OTH"
