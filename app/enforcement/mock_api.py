"""Mock resource endpoints — the device-facing surface a PEP protects.

Handlers read and write REAL files in the `resources/` folder. If the
PEP denies a request, the handler never runs and files are not touched
— visible proof that enforcement is real.

Six resource endpoints, three sensitivity tiers:

  Read tier (permissive threshold):
    GET  /api/device/read         sensor telemetry from sensor_readings.csv
    GET  /api/device/snapshot     fake camera image (SVG)
    GET  /api/device/actuator     current actuator state (Wokwi polls this)

  Write tier (medium threshold):
    POST /api/device/write        update device_config.csv
    GET  /api/device/portal       management webpage (elevated read)
    POST /api/device/actuator/lock    set actuator to locked

  Admin tier (strict threshold):
    POST /api/device/admin              append to admin_audit.csv
    POST /api/device/actuator/unlock    set actuator to unlocked  (dangerous)

The router prefix must match PROTECTED_PREFIX in pep_middleware.py.
"""
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, Response

from app import resources

router = APIRouter(prefix="/api/device", tags=["device"])


# ---------------------------------------------------------------------------
# Read tier
# ---------------------------------------------------------------------------

@router.get("/read")
async def read(file: str | None = None, limit: int = 10):
    """Read a resource file. Default: sensor_readings.csv. Query params:

      * file  — one of the readable files (see resources.readable_files())
      * limit — number of rows to return (CSV only, 1..100)
    """
    filename = file or "sensor_readings.csv"
    if filename not in resources.readable_files():
        raise HTTPException(
            status_code=400,
            detail=f"file not exposed for reading: {filename}. "
                   f"allowed: {resources.readable_files()}",
        )
    limit = max(1, min(int(limit), 100))
    try:
        result = resources.read_resource_file(filename, limit=limit)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "ok", **result}


@router.get("/snapshot")
async def snapshot():
    """Fake camera feed as an SVG image. Renders directly in a browser."""
    return Response(content=resources.render_snapshot_svg(),
                    media_type="image/svg+xml")


@router.get("/actuator")
async def actuator_state():
    """Current actuator state. The Wokwi ESP32 polls this every 2s."""
    return resources.read_actuator_state()


# ---------------------------------------------------------------------------
# Write tier
# ---------------------------------------------------------------------------

@router.post("/write")
async def write(request: Request):
    """Update a device's configuration in device_config.csv."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="expected JSON body")

    device_id = body.get("device_id")
    if not device_id:
        raise HTTPException(status_code=400, detail="body must include device_id")

    updates = {k: v for k, v in body.items() if k != "device_id"}
    if not updates:
        raise HTTPException(status_code=400, detail="body must include at least one field to update")

    try:
        updated = resources.update_device_config(device_id, updates)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if updated is None:
        raise HTTPException(status_code=404, detail=f"unknown device_id: {device_id}")

    return {"status": "ok", "resource": "device_config.csv",
            "updated_row": updated, "note": "changes have been written to disk"}


@router.get("/portal", response_class=HTMLResponse)
async def portal():
    """Device management webpage. Sensitive admin surface."""
    return HTMLResponse(resources.render_portal_page())


@router.post("/actuator/lock")
async def actuator_lock(request: Request):
    """Lock the physical actuator. Wokwi LED goes red."""
    device_id = request.headers.get("X-Device-ID", "unknown")
    state = resources.set_actuator_state("locked", by=device_id)
    return {"status": "ok", "resource": "actuator_state.json", "state": state}


# ---------------------------------------------------------------------------
# Admin tier
# ---------------------------------------------------------------------------

@router.post("/admin")
async def admin(request: Request):
    """Append an administrative action to admin_audit.csv."""
    try:
        body = await request.json()
    except Exception:
        body = {}

    entry = resources.append_admin_audit(
        operator=str(body.get("operator", "unknown")),
        action=str(body.get("action", "unspecified")),
        target=str(body.get("target", "")),
    )
    return {"status": "ok", "resource": "admin_audit.csv",
            "audit_entry": entry, "note": "action has been permanently recorded"}


@router.post("/actuator/unlock")
async def actuator_unlock(request: Request):
    """UNLOCK the physical actuator. Highest sensitivity — Wokwi LED goes green."""
    device_id = request.headers.get("X-Device-ID", "unknown")
    state = resources.set_actuator_state("unlocked", by=device_id)
    return {"status": "ok", "resource": "actuator_state.json", "state": state}


# ---------------------------------------------------------------------------
# Admin-tier: device_config.csv row operations
# ---------------------------------------------------------------------------

@router.post("/config/create")
async def config_create(request: Request):
    """Add a new device row to device_config.csv.

    Body: {"device_id": "hvac-controller-5", "setpoint_c": 21.0, "mode": "auto", "enabled": true}
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="expected JSON body")
    try:
        row = resources.create_device_config(
            device_id=str(body.get("device_id", "")),
            setpoint_c=float(body.get("setpoint_c", 20.0)),
            mode=str(body.get("mode", "auto")),
            enabled=bool(body.get("enabled", True)),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    user = getattr(request.state, "user", None)
    return {"status": "ok", "resource": "device_config.csv",
            "created_row": row, "created_by": user}


@router.post("/config/delete")
async def config_delete(request: Request):
    """Remove a device row from device_config.csv. Body: {"device_id": "..."}."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="expected JSON body")
    try:
        row = resources.delete_device_config(str(body.get("device_id", "")))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    user = getattr(request.state, "user", None)
    return {"status": "ok", "resource": "device_config.csv",
            "deleted_row": row, "deleted_by": user}


# ---------------------------------------------------------------------------
# Admin-tier: resource file operations
# ---------------------------------------------------------------------------

@router.get("/files/list")
async def files_list():
    """List every file in the resources folder, with size and protection flag."""
    return {"status": "ok", "files": resources.list_resource_files()}


@router.post("/files/create")
async def files_create(request: Request):
    """Create a new file in the resources folder.

    Body: {"filename": "notes.txt", "content": "…"}
    Extensions restricted to .csv .txt .json. Content capped at 1MB.
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="expected JSON body")
    try:
        result = resources.create_resource_file(
            filename=str(body.get("filename", "")),
            content=str(body.get("content", "")),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    user = getattr(request.state, "user", None)
    return {"status": "ok", "created": result, "created_by": user}


@router.post("/files/delete")
async def files_delete(request: Request):
    """Delete a resource file. Body: {"filename": "notes.txt"}. Refuses core files."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="expected JSON body")
    try:
        result = resources.delete_resource_file(str(body.get("filename", "")))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    user = getattr(request.state, "user", None)
    return {"status": "ok", **result, "deleted_by": user}
