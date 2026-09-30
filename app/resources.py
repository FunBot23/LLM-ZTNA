"""Real resources served by the mock device API.

Instead of returning canned JSON, the handlers in mock_api.py read from
and write to actual files in the `resources/` folder. Because the PEP
intercepts before the handler runs, a denied request never touches
these files — you can point at the file to prove enforcement worked.

Resource families:
  * sensor_readings.csv   read-tier   telemetry data
  * device_config.csv     write-tier  device settings
  * admin_audit.csv       admin-tier  administrative log
  * portal (HTML)         write-tier  device management webpage
  * actuator_state.json   mixed       simulated physical lock (Wokwi ESP32)

Honest scope note: nothing here prevents someone with laptop access from
opening these files directly. In production, OS-level file permissions
restrict the resource files to the service account running the app. The
framework itself protects the *sanctioned interface* — this module makes
that interface concrete.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

RESOURCES_DIR = Path(__file__).resolve().parent.parent / "resources"

SENSORS_CSV = RESOURCES_DIR / "sensor_readings.csv"
CONFIG_CSV = RESOURCES_DIR / "device_config.csv"
AUDIT_CSV = RESOURCES_DIR / "admin_audit.csv"
ACTUATOR_STATE_FILE = RESOURCES_DIR / "actuator_state.json"


# ---------------------------------------------------------------------------
# Initial seed data — used when the files don't exist yet.
# ---------------------------------------------------------------------------

_SENSORS_SEED = [
    ["timestamp", "location", "temperature_c", "humidity_pct", "co2_ppm"],
    ["2025-11-04T14:00:00", "building-1-lobby",  "22.5", "61", "420"],
    ["2025-11-04T14:05:00", "building-1-lobby",  "22.6", "61", "418"],
    ["2025-11-04T14:10:00", "building-1-lobby",  "22.7", "62", "425"],
    ["2025-11-04T14:00:00", "building-2-server", "18.2", "45", "380"],
    ["2025-11-04T14:05:00", "building-2-server", "18.4", "45", "382"],
    ["2025-11-04T14:10:00", "building-2-server", "18.3", "45", "385"],
    ["2025-11-04T14:00:00", "building-3-office", "21.8", "55", "550"],
    ["2025-11-04T14:05:00", "building-3-office", "21.9", "55", "560"],
    ["2025-11-04T14:10:00", "building-3-office", "22.0", "56", "570"],
]

_CONFIG_SEED = [
    ["device_id", "setpoint_c", "mode", "enabled"],
    ["hvac-controller-1", "20.0", "auto", "true"],
    ["hvac-controller-2", "21.5", "heat", "true"],
    ["hvac-controller-3", "19.0", "cool", "false"],
    ["hvac-controller-4", "22.0", "auto", "true"],
]

_AUDIT_SEED = [
    ["timestamp", "operator", "action", "target"],
]


# ---------------------------------------------------------------------------
# Init / reset
# ---------------------------------------------------------------------------

def init_if_missing() -> None:
    """Create resource files if they don't exist. Called on app startup."""
    RESOURCES_DIR.mkdir(exist_ok=True)
    for path, rows in (
        (SENSORS_CSV, _SENSORS_SEED),
        (CONFIG_CSV, _CONFIG_SEED),
        (AUDIT_CSV, _AUDIT_SEED),
    ):
        if not path.exists():
            with path.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerows(rows)
    if not ACTUATOR_STATE_FILE.exists():
        _write_actuator_state({
            "state": "locked",
            "last_changed_by": None,
            "last_changed_at": None,
        })


def reset() -> None:
    """Delete and recreate all resource files with seed data.

    Useful between demo runs to guarantee identical starting state.
    """
    for path in (SENSORS_CSV, CONFIG_CSV, AUDIT_CSV, ACTUATOR_STATE_FILE):
        if path.exists():
            path.unlink()
    init_if_missing()


# ---------------------------------------------------------------------------
# Read/write operations — called by the handlers in mock_api.py
# ---------------------------------------------------------------------------

def read_sensors(location: Optional[str] = None, limit: int = 10) -> list[dict]:
    """Return the most recent N sensor readings, optionally filtered by location."""
    with SENSORS_CSV.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if location:
        rows = [r for r in rows if r.get("location") == location]
    return rows[-limit:]


def update_device_config(device_id: str, updates: dict) -> Optional[dict]:
    """Update a device's config row in-place. Returns the updated row,
    or None if device_id isn't found.

    Special key `new_device_id` in `updates` renames the device (must not
    collide with another existing device). All other keys must be known
    columns; unknown keys are ignored.
    """
    with CONFIG_CSV.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    target = next((r for r in rows if r.get("device_id") == device_id), None)
    if target is None:
        return None

    # Handle rename request separately.
    new_id = updates.pop("new_device_id", None)
    if new_id is not None:
        new_id = str(new_id).strip()
        if new_id and new_id != device_id:
            if any(r.get("device_id") == new_id for r in rows if r is not target):
                raise ValueError(f"cannot rename to '{new_id}': already exists")
            target["device_id"] = new_id

    for key, value in updates.items():
        if key in fieldnames and key != "device_id":
            target[key] = str(value)

    with CONFIG_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return target


def append_admin_audit(operator: str, action: str, target: str) -> dict:
    """Append a row to the admin audit log. Returns the appended entry."""
    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "operator": operator,
        "action": action,
        "target": target,
    }
    with AUDIT_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "operator", "action", "target"])
        writer.writerow(entry)
    return entry


def list_devices() -> list[str]:
    """Return known device_ids for quick discovery."""
    with CONFIG_CSV.open(encoding="utf-8") as f:
        return [r["device_id"] for r in csv.DictReader(f)]


# ---------------------------------------------------------------------------
# Multi-file read + file operations (used by expanded admin surface)
# ---------------------------------------------------------------------------

import re as _re

_READABLE_FILES = ["sensor_readings.csv", "device_config.csv", "actuator_state.json"]
_PROTECTED_FILES = {
    "sensor_readings.csv", "device_config.csv",
    "admin_audit.csv", "actuator_state.json",
}
_FILENAME_RE = _re.compile(r"^[a-zA-Z0-9_-]+\.(csv|txt|json)$")
_MAX_FILE_SIZE_BYTES = 1_000_000  # 1 MB


def readable_files() -> list[str]:
    """Whitelist exposed to GET /api/device/read."""
    return list(_READABLE_FILES)


def _sanitize_filename(name: str) -> str:
    """Reject path traversal and enforce extension allowlist."""
    if not _FILENAME_RE.match(name or ""):
        raise ValueError(f"invalid filename (allowed: letters/digits/_/-, extensions .csv .txt .json): {name!r}")
    return name


def read_resource_file(name: str, limit: int = 10) -> dict:
    """Read any file in the resources folder. Returns a shape depending on type."""
    safe = _sanitize_filename(name)
    path = RESOURCES_DIR / safe
    if not path.exists():
        raise FileNotFoundError(f"resource not found: {safe}")
    if safe.endswith(".csv"):
        with path.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        rows = rows[-limit:]
        return {"resource": safe, "type": "csv", "count": len(rows), "data": rows}
    if safe.endswith(".json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {"raw": path.read_text(encoding="utf-8")}
        return {"resource": safe, "type": "json", "data": data}
    return {"resource": safe, "type": "text", "content": path.read_text(encoding="utf-8")}


def create_device_config(device_id: str, setpoint_c: float, mode: str, enabled: bool) -> dict:
    """Add a new row to device_config.csv. Fails if device_id already exists."""
    device_id = (device_id or "").strip()
    if not device_id:
        raise ValueError("device_id is required")
    with CONFIG_CSV.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if any(r.get("device_id") == device_id for r in rows):
        raise ValueError(f"device_id already exists: {device_id}")
    new_row = {
        "device_id": device_id,
        "setpoint_c": str(setpoint_c),
        "mode": mode,
        "enabled": str(bool(enabled)).lower(),
    }
    rows.append(new_row)
    with CONFIG_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return new_row


def delete_device_config(device_id: str) -> dict:
    """Remove a row from device_config.csv. Returns the deleted row."""
    device_id = (device_id or "").strip()
    with CONFIG_CSV.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    target = next((r for r in rows if r.get("device_id") == device_id), None)
    if target is None:
        raise ValueError(f"unknown device_id: {device_id}")
    kept = [r for r in rows if r.get("device_id") != device_id]
    with CONFIG_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)
    return target


def list_resource_files() -> list[dict]:
    """Enumerate all files in the resources folder with size and protection flag."""
    out = []
    for p in sorted(RESOURCES_DIR.iterdir()):
        if p.is_file():
            out.append({
                "filename": p.name,
                "size": p.stat().st_size,
                "protected": p.name in _PROTECTED_FILES,
            })
    return out


def create_resource_file(filename: str, content: str) -> dict:
    """Create a new file in the resources folder. Extension must be .csv, .txt, or .json."""
    safe = _sanitize_filename(filename)
    if len(content or "") > _MAX_FILE_SIZE_BYTES:
        raise ValueError(f"content exceeds {_MAX_FILE_SIZE_BYTES} byte limit")
    path = RESOURCES_DIR / safe
    if path.exists():
        raise ValueError(f"file already exists: {safe}")
    path.write_text(content or "", encoding="utf-8")
    return {"filename": safe, "size": path.stat().st_size}


def delete_resource_file(filename: str) -> dict:
    """Delete a file. Refuses to touch the core protected files."""
    safe = _sanitize_filename(filename)
    if safe in _PROTECTED_FILES:
        raise ValueError(f"refusing to delete protected file: {safe}")
    path = RESOURCES_DIR / safe
    if not path.exists():
        raise ValueError(f"file not found: {safe}")
    path.unlink()
    return {"filename": safe, "deleted": True}


# ---------------------------------------------------------------------------
# Portal (HTML) — device management webpage
# ---------------------------------------------------------------------------

def render_portal_page() -> str:
    """Render the device management portal from device_config.csv."""
    with CONFIG_CSV.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    body_rows = ""
    for r in rows:
        enabled = r.get("enabled") == "true"
        badge = (
            '<span class="px-2 py-0.5 rounded bg-green-100 text-green-800 text-xs font-medium">on</span>'
            if enabled else
            '<span class="px-2 py-0.5 rounded bg-gray-100 text-gray-600 text-xs font-medium">off</span>'
        )
        body_rows += (
            f'<tr class="border-t border-gray-200">'
            f'<td class="p-3 font-mono text-sm">{r["device_id"]}</td>'
            f'<td class="p-3 text-sm">{r["setpoint_c"]}\u00b0C</td>'
            f'<td class="p-3 text-sm">{r["mode"]}</td>'
            f'<td class="p-3">{badge}</td>'
            f'</tr>'
        )

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8">
<title>Device Control Portal</title>
<script src="https://cdn.tailwindcss.com"></script>
</head><body class="bg-gray-50 p-6">
<div class="max-w-3xl mx-auto">
<h1 class="text-2xl font-bold text-gray-800 mb-1">Device Control Portal</h1>
<p class="text-sm text-gray-600 mb-6">Sensitive admin surface. Access controlled by the Zero-Trust Framework.</p>
<div class="bg-white border border-gray-200 rounded-lg overflow-hidden shadow-sm">
<table class="w-full">
<thead class="bg-gray-100"><tr>
<th class="p-3 text-left text-xs font-medium text-gray-600 uppercase tracking-wide">Device ID</th>
<th class="p-3 text-left text-xs font-medium text-gray-600 uppercase tracking-wide">Setpoint</th>
<th class="p-3 text-left text-xs font-medium text-gray-600 uppercase tracking-wide">Mode</th>
<th class="p-3 text-left text-xs font-medium text-gray-600 uppercase tracking-wide">Status</th>
</tr></thead>
<tbody>{body_rows}</tbody>
</table>
</div>
<p class="mt-4 text-xs text-gray-400">Rendered at {now}. Data: device_config.csv. If you can see this page, the PEP allowed your request.</p>
</div></body></html>"""


# ---------------------------------------------------------------------------
# Snapshot (SVG) — fake camera feed
# ---------------------------------------------------------------------------

def render_snapshot_svg() -> str:
    """Render a fake security-camera snapshot as SVG.

    Uses SVG so it's inline text (no image file needed) but still renders
    as a proper picture in any browser.
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="640" height="480" viewBox="0 0 640 480">
<rect fill="#1a1a1a" width="640" height="480"/>
<rect fill="#2a2a2a" x="10" y="10" width="620" height="460"/>
<circle cx="600" cy="30" r="6" fill="#ff2020"/>
<text x="588" y="35" font-family="Courier,monospace" font-size="12" fill="#ff2020" text-anchor="end">REC</text>
<text x="320" y="120" font-family="Courier,monospace" font-size="26" fill="#e0e0e0" text-anchor="middle">CAMERA-01</text>
<text x="320" y="150" font-family="Courier,monospace" font-size="14" fill="#909090" text-anchor="middle">Server Room B \u2014 East Wall</text>
<g transform="translate(320,270)">
<rect x="-60" y="-40" width="120" height="80" fill="none" stroke="#4a4a4a" stroke-width="2"/>
<circle cx="0" cy="0" r="30" fill="#333" stroke="#666" stroke-width="2"/>
<circle cx="0" cy="0" r="10" fill="#555"/>
<rect x="-45" y="-55" width="20" height="10" fill="#333"/>
</g>
<text x="320" y="380" font-family="Courier,monospace" font-size="14" fill="#c0c0c0" text-anchor="middle">LIVE FEED</text>
<text x="320" y="405" font-family="Courier,monospace" font-size="12" fill="#909090" text-anchor="middle">{ts}</text>
<text x="20" y="465" font-family="Courier,monospace" font-size="10" fill="#606060">ZTF-Protected Resource</text>
</svg>"""


# ---------------------------------------------------------------------------
# Actuator (JSON state file) — read by the Wokwi ESP32 every 2s
# ---------------------------------------------------------------------------

def _write_actuator_state(state: dict) -> None:
    ACTUATOR_STATE_FILE.write_text(json.dumps(state, indent=2))


def read_actuator_state() -> dict:
    """Return the current actuator state. Auto-init if missing."""
    if not ACTUATOR_STATE_FILE.exists():
        default = {"state": "locked", "last_changed_by": None, "last_changed_at": None}
        _write_actuator_state(default)
        return default
    return json.loads(ACTUATOR_STATE_FILE.read_text())


def set_actuator_state(new_state: str, by: str) -> dict:
    """Set the actuator to 'locked' or 'unlocked' and record who did it."""
    if new_state not in ("locked", "unlocked"):
        raise ValueError(f"invalid actuator state: {new_state}")
    state = {
        "state": new_state,
        "last_changed_by": by,
        "last_changed_at": datetime.now().isoformat(timespec="seconds"),
    }
    _write_actuator_state(state)
    return state
