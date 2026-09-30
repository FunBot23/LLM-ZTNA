"""
device_traffic_sim.py — standalone traffic generator for the Zero-Trust
Framework's mock device API.

Run this alongside `uvicorn app.main:app --reload` to simulate real devices
issuing requests over time, exercising the full PEP -> PDP -> scoring ->
(optional) step-up pipeline exactly as a real device would.

Usage:
    python scripts/device_traffic_sim.py
    python scripts/device_traffic_sim.py --base-url http://127.0.0.1:8000 --duration 120
"""

from __future__ import annotations

import argparse
import asyncio
import random
import time
from dataclasses import dataclass, field

import httpx

# ---------------------------------------------------------------------------
# Device profiles
# ---------------------------------------------------------------------------


@dataclass
class DeviceProfile:
    device_id: str
    label: str
    # Actions this device issues, weighted by likelihood.
    actions: list[tuple[str, str]]  # (method, path)
    # Seconds between requests (min, max) — smaller = more aggressive.
    interval_range: tuple[float, float]
    # If True, auto-retry with a step-up token on 401 (mimics a real client).
    honors_step_up: bool = True


NORMAL_DEVICES = [
    DeviceProfile(
        device_id="192.168.1.40",
        label="normal-sensor-1",
        actions=[("GET", "/api/device/read")] * 9 + [("POST", "/api/device/write")],
        interval_range=(3.0, 7.0),
    ),
    DeviceProfile(
        device_id="192.168.1.55",
        label="normal-sensor-2",
        actions=[("GET", "/api/device/read")] * 8 + [("POST", "/api/device/write")] * 2,
        interval_range=(4.0, 8.0),
    ),
    DeviceProfile(
        device_id="192.168.1.61",
        label="normal-gateway",
        actions=[("GET", "/api/device/read")] * 7 + [("POST", "/api/device/write")] * 3,
        interval_range=(2.0, 5.0),
    ),
]

MISBEHAVING_DEVICES = [
    DeviceProfile(
        device_id="192.168.1.201",
        label="aggressive-admin-caller",
        actions=[("POST", "/api/device/admin")] * 6 + [("POST", "/api/device/write")] * 4,
        interval_range=(0.3, 0.8),  # rapid-fire — should push into escalate/deny
    ),
    DeviceProfile(
        device_id="10.0.0.66",
        label="pre-blocked-seed-device",
        actions=[("GET", "/api/device/read")],
        interval_range=(2.0, 4.0),  # should be denied immediately every time
        honors_step_up=False,  # blocked outright, no point retrying
    ),
]


# ---------------------------------------------------------------------------
# Per-device loop
# ---------------------------------------------------------------------------


async def run_device(
    client: httpx.AsyncClient,
    profile: DeviceProfile,
    stop_at: float,
) -> None:
    while time.monotonic() < stop_at:
        method, path = random.choice(profile.actions)
        headers = {"X-Device-ID": profile.device_id}

        resp = await client.request(method, path, headers=headers)
        _log(profile, method, path, resp)

        if resp.status_code == 401 and profile.honors_step_up:
            token = resp.json().get("challenge_token")
            if token:
                step_headers = {
                    "X-Device-ID": profile.device_id,
                    "X-Step-Up-Token": token,
                }
                step_resp = await client.get("/api/device/read", headers=step_headers)
                _log(profile, "GET", "/api/device/read (step-up)", step_resp)

        await asyncio.sleep(random.uniform(*profile.interval_range))


def _log(profile: DeviceProfile, method: str, path: str, resp: httpx.Response) -> None:
    reason = resp.headers.get("X-Decision-Reason", "")
    print(
        f"[{profile.label:<24}] {profile.device_id:<15} "
        f"{method:<4} {path:<28} -> {resp.status_code}"
        + (f"  ({reason})" if reason else "")
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main(base_url: str, duration: float) -> None:
    stop_at = time.monotonic() + duration
    all_devices = NORMAL_DEVICES + MISBEHAVING_DEVICES

    print(f"[sim] starting {len(all_devices)} simulated devices for {duration:.0f}s")
    print(f"[sim] target: {base_url}")
    for d in all_devices:
        print(f"  - {d.label:<24} {d.device_id:<15} interval={d.interval_range}")
    print()

    async with httpx.AsyncClient(base_url=base_url, timeout=10.0) as client:
        await asyncio.gather(*(run_device(client, d, stop_at) for d in all_devices))

    print("\n[sim] finished")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--duration", type=float, default=120.0, help="seconds to run")
    args = ap.parse_args()
    asyncio.run(main(args.base_url, args.duration))
