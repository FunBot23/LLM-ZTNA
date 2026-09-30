"""Shared decision pipeline.

One canonical path from an event to a verdict:
    caller -> PDP -> audit log -> live stream -> return verdict

"""
from datetime import datetime
from typing import Optional

from app.bus import bus
from app.db import log_decision
from app.enforcement.pdp import Decision, DecisionRequest, decide


async def process_event(
    src: str,
    action: str,
    features: Optional[dict] = None,
    extra: Optional[dict] = None,
) -> Decision:
    """Score, log, publish. Returns the Decision so callers can act on it
    (e.g. the mock API returns 403 for deny).
    """
    import time
    decision = decide(DecisionRequest(src=src, action=action, features=features))

    now = time.time()
    event = {
        "ts": datetime.now().strftime("%H:%M:%S"),
        "created_at": now,
        "src": src,
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
