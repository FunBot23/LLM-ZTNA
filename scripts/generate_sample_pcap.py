"""Generate a small sample PCAP for smoke-testing the ingestion pipeline.

Creates ~30 packets over ~30 seconds of virtual time mixing:
  * benign browsing traffic (a few devices → HTTPS/DNS)
  * one internal SSH connection
  * one blocklisted device doing scans

Run once from the project root:
    python scripts/generate_sample_pcap.py

Then set the env var and start the server:
    $env:ZTF_PCAP_PATH="sample.pcap"
    $env:ZTF_PCAP_SPEED="20.0"
    uvicorn app.main:app --reload

Delete the env vars to go back to the synthetic traffic generator.
"""
import random
from pathlib import Path

from scapy.all import IP, TCP, UDP, Ether, wrpcap

OUT = Path(__file__).resolve().parent.parent / "sample.pcap"

BENIGN_DEVICES = ["192.168.1.47", "192.168.1.12", "192.168.1.88"]
INTERNAL = ["10.0.0.5", "10.0.0.42"]
BLOCKLISTED = "10.0.0.66"
EXTERNAL = ["93.184.216.34", "1.1.1.1", "142.250.80.46"]


def _mk(src, dst, sport, dport, proto, ts):
    l3 = IP(src=src, dst=dst)
    l4 = TCP(sport=sport, dport=dport) if proto == "TCP" else UDP(sport=sport, dport=dport)
    pkt = Ether() / l3 / l4
    pkt.time = ts
    return pkt


def build():
    random.seed(7)
    pkts = []
    t = 0.0

    # Normal browsing: benign devices → HTTPS/DNS
    for _ in range(12):
        src = random.choice(BENIGN_DEVICES)
        dst = random.choice(EXTERNAL)
        pkts.append(_mk(src, dst, random.randint(40000, 60000), 443, "TCP", t))
        t += random.uniform(0.5, 1.5)
        pkts.append(_mk(src, "1.1.1.1", random.randint(40000, 60000), 53, "UDP", t))
        t += random.uniform(0.3, 0.8)

    # One internal SSH session
    pkts.append(_mk(INTERNAL[0], INTERNAL[1], 51022, 22, "TCP", t))
    t += 2.0

    # Blocklisted device port-scans internal subnet
    for port in [22, 23, 80, 445, 3389, 8080]:
        pkts.append(_mk(BLOCKLISTED, random.choice(INTERNAL + BENIGN_DEVICES), 55000, port, "TCP", t))
        t += 0.4

    wrpcap(str(OUT), pkts)
    print(f"wrote {len(pkts)} packets to {OUT}")


if __name__ == "__main__":
    build()
