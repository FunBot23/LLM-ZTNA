"""Human-review queue for escalate events.

Every escalate becomes a pending record awaiting operator action. The
device polls with X-Pending-ID until the review resolves. If nobody
reviews within `timeout_seconds`, the item auto-resolves to deny.

The LLM advisor writes its recommendation into the same record, but
does NOT resolve it, that's the human's job. LLM output is advisory.
"""
from __future__ import annotations

import secrets
import time
from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class PendingReview:
    id: str
    src: str
    action: str
    features: Optional[dict]
    pdp_reason: str
    pdp_score: float
    created_at: float
    expires_at: float
    status: str = "pending"                    # 'pending' | 'allowed' | 'denied' | 'retry' | 'expired'
    llm_recommendation: Optional[str] = None   # 'allow' | 'deny' | 'retry' | None
    llm_reasoning: Optional[str] = None
    llm_confidence: Optional[float] = None
    llm_retry_seconds: Optional[int] = None
    retry_after_seconds: Optional[int] = None
    resolved_at: Optional[float] = None
    resolved_by: Optional[str] = None          # 'human' | 'timeout'

    def to_public_dict(self) -> dict:
        d = asdict(self)
        now = time.time()
        d["age_seconds"] = max(0.0, now - self.created_at)
        d["expires_in_seconds"] = max(0.0, self.expires_at - now) if self.status == "pending" else None
        return d


class ReviewQueue:
    def __init__(self, timeout_seconds: float = 60.0, prune_after_seconds: float = 3600.0) -> None:
        self.timeout = timeout_seconds
        self.prune_after = prune_after_seconds
        self._items: dict[str, PendingReview] = {}

    def create(
        self, src: str, action: str, features: Optional[dict],
        pdp_reason: str, pdp_score: float,
    ) -> PendingReview:
        pid = f"pnd_{secrets.token_urlsafe(8)}"
        now = time.time()
        item = PendingReview(
            id=pid, src=src, action=action, features=features,
            pdp_reason=pdp_reason, pdp_score=float(pdp_score),
            created_at=now, expires_at=now + self.timeout,
        )
        self._items[pid] = item
        return item

    def get(self, pid: str) -> Optional[PendingReview]:
        return self._items.get(pid)

    def resolve(
        self, pid: str, resolution: str,
        retry_after_seconds: Optional[int] = None,
        by: str = "human",
    ) -> Optional[PendingReview]:
        """resolution ∈ {'allowed', 'denied', 'retry'}. Returns the item on success."""
        item = self._items.get(pid)
        if item is None or item.status != "pending":
            return None
        if resolution not in ("allowed", "denied", "retry"):
            return None
        item.status = resolution
        item.retry_after_seconds = retry_after_seconds
        item.resolved_at = time.time()
        item.resolved_by = by
        return item

    def attach_llm_advice(
        self, pid: str, recommendation: Optional[str],
        reasoning: str, confidence: float,
        retry_seconds: Optional[int] = None,
    ) -> Optional[PendingReview]:
        item = self._items.get(pid)
        if item is None:
            return None
        item.llm_recommendation = recommendation
        item.llm_reasoning = reasoning
        item.llm_confidence = confidence
        item.llm_retry_seconds = retry_seconds
        return item

    def sweep_expired(self) -> list[PendingReview]:
        """Auto-deny anything past its expiry. Returns the items just expired."""
        now = time.time()
        expired: list[PendingReview] = []
        for item in list(self._items.values()):
            if item.status == "pending" and item.expires_at < now:
                item.status = "denied"
                item.resolved_at = now
                item.resolved_by = "timeout"
                expired.append(item)
        # Also prune old resolved items to bound memory.
        cutoff = now - self.prune_after
        for pid in [p for p, it in self._items.items()
                    if it.status != "pending" and (it.resolved_at or 0) < cutoff]:
            del self._items[pid]
        return expired

    def pending(self) -> list[PendingReview]:
        """Currently-pending items, newest first."""
        items = [i for i in self._items.values() if i.status == "pending"]
        return sorted(items, key=lambda i: i.created_at, reverse=True)

    def recent(self, limit: int = 25) -> list[PendingReview]:
        """All items (pending + resolved), newest first."""
        items = sorted(self._items.values(), key=lambda i: i.created_at, reverse=True)
        return items[:limit]


# Module-level singleton.
review_queue = ReviewQueue()
