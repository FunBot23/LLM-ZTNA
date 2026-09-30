"""Policy Enforcement Point — real ASGI middleware.

Intercepts every request under /api/device/*. Consults the PDP, then:

  * allow             → forward to handler (200)
  * deny              → 403 with reason
  * escalate (fresh)  → create pending review + fire LLM advisor →
                        return 202 with X-Pending-ID + Retry-After
  * escalate (poll)   → device retried with X-Pending-ID → resolve:
                          - pending → 202 again (keep polling)
                          - allowed → forward to handler (200)
                          - denied  → 403
                          - retry   → 429 with Retry-After
                          - expired → 403 (auto-timeout)

Replaces the earlier step-up/challenge flow with a human-in-the-loop
async review pattern. LLM advises the human; human decides.
"""
from __future__ import annotations

import asyncio

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from app.enforcement.api_flow_tracker import get_tracker

from app import auth
from app.bus import bus
from app.enforcement.pipeline import process_event
from app.enforcement.review_queue import review_queue
from app.orchestrator.llm_advisor import advise

PROTECTED_PREFIX = "/api/device/"

# Admin-tier paths require an authenticated user session on top of the
# usual PDP verdict. A high-trust device without a logged-in user cannot
# do admin — both identity axes must pass.
ADMIN_PATHS = {
    "/api/device/admin",
    "/api/device/actuator/unlock",
    "/api/device/config/create",
    "/api/device/config/delete",
    "/api/device/files/create",
    "/api/device/files/delete",
}


def _ascii_safe(value: str, limit: int = 200) -> str:
    return value.encode("ascii", errors="replace").decode("ascii")[:limit]


class PEPMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if not path.startswith(PROTECTED_PREFIX):
            return await call_next(request)

        device_id = request.headers.get("X-Device-ID")
        if not device_id:
            # Fall back to the client's IP as its identity. This is the
            # natural default: a device on the network is identified by
            # its network address. Programmatic clients (curl, Wokwi,
            # firmware) can still override with X-Device-ID.
            device_id = request.client.host if request.client else None
        if not device_id:
            return JSONResponse(
                {"detail": "unable to identify requesting device"},
                status_code=400,
            )

        # Optional user session. Admin operations REQUIRE it; other tiers
        # accept its absence (machine clients without cookies).
        session_token = request.cookies.get("ztf_session")
        username = auth.get_session(session_token)
        request.state.user = username  # handlers may read this

        if path in ADMIN_PATHS and username is None:
            return JSONResponse(
                {
                    "detail": "admin operations require an authenticated user session; "
                              "please log in at /login",
                    "requires_auth": True,
                },
                status_code=401,
            )

        # Polling on an existing pending review?
        pending_id = request.headers.get("X-Pending-ID")
        if pending_id:
            return await self._handle_poll(request, device_id, pending_id, call_next)

        # Fresh request → normal PDP flow.
        action = f"{request.method} {path}"
        features = get_tracker().observe(device_id, request.method, path)
        decision = await process_event(src=device_id, action=action, features=features, user=username)

        if decision.verdict == "deny":
            return JSONResponse(
                {"detail": decision.reason},
                status_code=403,
                headers={"X-Decision-Reason": _ascii_safe(decision.reason)},
            )

        if decision.verdict == "escalate":
            pid = decision.pending_review_id
            #item = review_queue.create(
                #src=device_id, action=action, features=None,
                #pdp_reason=decision.reason, pdp_score=decision.score,
            #)
            # Fire LLM advisor + notify dashboard, don't block on either.
            #asyncio.create_task(
                #advise(item.id, device_id, action, None, decision.reason, decision.score)
            #)
            #asyncio.create_task(
                #bus.publish({"event_type": "review_pending", "id": item.id})
            #)
            return JSONResponse(
                {
                    "detail": "escalated to human review",
                    "pending_id": pid,
                    "pdp_reason": decision.reason,
                    "expires_in_seconds": review_queue.timeout,
                    "instructions": "poll this URL with header  X-Pending-ID: <pending_id>",
                    "retry_after_seconds": 5,
                },
                status_code=202,
                headers={
                    "X-Pending-ID": pid,
                    "Retry-After": "5",
                    "X-Decision-Reason": _ascii_safe(decision.reason),
                },
            )

        # allow
        return await call_next(request)

    async def _handle_poll(self, request, device_id, pending_id, call_next):
        item = review_queue.get(pending_id)
        if item is None:
            return JSONResponse(
                {"detail": "unknown or expired pending review; start a fresh request"},
                status_code=404,
            )
        if item.src != device_id:
            return JSONResponse(
                {"detail": "pending review belongs to a different device"},
                status_code=403,
            )
        expected_action = f"{request.method} {request.url.path}"
        if item.action != expected_action:
            return JSONResponse(
                {
                    "detail": (
                        f"pending review is for {item.action}; "
                        f"polling request is {expected_action}"
                    ),
                },
                status_code=400,
            )

        if item.status == "pending":
            return JSONResponse(
                {
                    "detail": "still awaiting review",
                    "pending_id": item.id,
                    "llm_recommendation": item.llm_recommendation,  # may be None if advice hasn't landed yet
                    "retry_after_seconds": 5,
                },
                status_code=202,
                headers={"X-Pending-ID": item.id, "Retry-After": "5"},
            )

        if item.status == "allowed":
            return await call_next(request)

        if item.status == "denied":
            reason = f"pending {item.id} denied by {item.resolved_by}"
            return JSONResponse(
                {"detail": reason, "pending_id": item.id, "resolved_by": item.resolved_by},
                status_code=403,
                headers={"X-Decision-Reason": _ascii_safe(reason)},
            )

        if item.status == "retry":
            wait = int(item.retry_after_seconds or 60)
            return JSONResponse(
                {
                    "detail": f"human asked device to retry after {wait} seconds",
                    "pending_id": item.id,
                    "retry_after_seconds": wait,
                },
                status_code=429,
                headers={"Retry-After": str(wait)},
            )

        return JSONResponse(
            {"detail": f"unexpected pending status: {item.status}"},
            status_code=500,
        )
