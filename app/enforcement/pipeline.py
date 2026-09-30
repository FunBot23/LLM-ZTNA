"""Shared decision pipeline.

One canonical path from an event to a verdict:
    caller -> PDP -> audit log -> live stream -> return verdict

Both the mock device API (in-app HTTP requests) and the ingestion layer
(real captured flows) funnel through here. Adding a new event source in
the future means calling process_event() — nothing else changes.

Also exposes log_and_publish_resolution(): when a pending review is
resolved (by human or timeout) and the device picks up the answer, a
follow-up audit row is written so time-window queries like "denies
today" include human-resolved escalates.
"""
from datetime import datetime
from typing import Optional

from app.bus import bus
from app.db import log_decision
from app.enforcement.pdp import Decision, DecisionRequest, decide
import asyncio
from app.enforcement.review_queue import review_queue
from app.orchestrator.llm_advisor import advise


async def process_event(
    src: str,
    action: str,
    features: Optional[dict] = None,
    extra: Optional[dict] = None,
    user: Optional[str] = None,
) -> Decision:
    #decision = decide(DecisionRequest(src=src, action=action, features=features))
    """Score, log, publish. Returns the Decision so callers can act on it
    (e.g. the mock API returns 403 for deny).

    `user` is the authenticated username making the request, or None for
    unauthenticated machine clients. Included in the audit log for
    accountability on admin operations.
    """
    import time
    decision = decide(DecisionRequest(src=src, action=action, features=features))
    
    if decision.verdict == "escalate":
        item = review_queue.create(
            src=src, action=action, features=features,
            pdp_reason=decision.reason, pdp_score=decision.score,
        )
        decision.pending_review_id = item.id
        asyncio.create_task(
            advise(item.id, src, action, features, decision.reason, decision.score)
        )
        asyncio.create_task(
            bus.publish({"event_type": "review_pending", "id": item.id})
        )

    now = time.time()
    event = {
        "ts": datetime.now().strftime("%H:%M:%S"),
        "created_at": now,
        "src": src,
        "user": user,
        "action": action,
        "verdict": decision.verdict,
        "score": round(decision.score, 3),
        "reason": decision.reason,
    }
    if extra:
        event["extra"] = extra

    # Durable audit first, then best-effort live stream.
    await log_decision(event)
    await bus.publish(event)
    return decision


# Track which pending reviews have already been logged as resolved, so
# repeated polls don't create duplicate audit rows.
_resolutions_logged: set[str] = set()


async def log_and_publish_resolution(item, resolved_by_user: Optional[str] = None) -> None:
    """Log the resolution of a pending review as a follow-up audit row.

    Idempotent: repeated polls on the same resolved pending only log once.
    """
    if item.id in _resolutions_logged:
        return
    _resolutions_logged.add(item.id)

    import time
    verdict = {"allowed": "allow", "denied": "deny", "retry": "escalate"}.get(item.status, "escalate")
    llm_note = (
        f" (LLM advised {item.llm_recommendation})" if item.llm_recommendation else ""
    )
    reason = f"pending {item.id} resolved by {item.resolved_by}{llm_note}"

    event = {
        "ts": datetime.now().strftime("%H:%M:%S"),
        "created_at": time.time(),
        "src": item.src,
        "user": resolved_by_user,
        "action": item.action,
        "verdict": verdict,
        "score": round(item.pdp_score, 3),
        "reason": reason,
    }
    await log_decision(event)
    await bus.publish(event)
