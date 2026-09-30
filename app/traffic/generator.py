"""Synthetic traffic generator.

Makes real HTTP requests against the mock device API so the whole
PEP→PDP→bus→SSE loop runs continuously with something to watch. This
is a stand-in for real ingestion (Step 3) and gets removed then.

Uses httpx.AsyncClient hitting our own server, so the code path is
identical to what a real device would trigger — no shortcuts around
the PEP.
"""
import asyncio
import os
import random

import httpx

DEFAULT_BASE_URL = os.environ.get("ZTF_BASE_URL", "http://127.0.0.1:8000")
INTERVAL_MIN = float(os.environ.get("ZTF_TRAFFIC_INTERVAL_MIN", "3.0"))
INTERVAL_MAX = float(os.environ.get("ZTF_TRAFFIC_INTERVAL_MAX", "6.0"))

DEVICES = [
    "192.168.1.47",
    "192.168.1.12",
    "192.168.1.88",
    "10.0.0.5",
    "10.0.0.42",
    "10.0.0.66",  # on the blocklist — will always be denied
]

ROUTES = [
    ("GET",  "/api/device/read"),
    ("POST", "/api/device/write"),
    ("POST", "/api/device/admin"),
]


async def run(base_url: str = DEFAULT_BASE_URL) -> None:
    # Wait for uvicorn to be listening.
    await asyncio.sleep(1.0)

    async with httpx.AsyncClient(base_url=base_url, timeout=2.0) as client:
        while True:
            method, path = random.choice(ROUTES)
            device = random.choice(DEVICES)
            try:
                await client.request(
                    method, path, headers={"X-Device-ID": device}
                )
            except httpx.HTTPError:
                # Non-2xx (deny/escalate/etc.) is expected and already logged.
                pass
            await asyncio.sleep(random.uniform(INTERVAL_MIN, INTERVAL_MAX))
