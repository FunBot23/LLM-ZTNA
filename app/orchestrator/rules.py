"""Versioned, hot-reloadable rule set.

Read by the PDP synchronously on every decision. Written by the LLM
worker asynchronously — but Python's asyncio is cooperative single-
threaded, so no locks needed: any `apply_*` method is atomic between
event-loop yields.

Every change bumps `version` and appends a RuleChange to `_changes` so
the dashboard (and audit) can show policy evolution.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class BlockEntry:
    src: str
    expires_at: Optional[float]   # unix ts; None for permanent
    reason: str


@dataclass
class RuleChange:
    version: int
    ts: str
    kind: str                    # 'add_blocklist' | 'remove_blocklist' | 'override_thresholds'
    target: str
    detail: dict
    source: str                  # 'seed' | 'llm' | 'admin'
    confidence: float


@dataclass
class RuleSet:
    version: int = 0
    action_thresholds: dict = field(default_factory=lambda: {
        # Read tier — permissive
        "GET /api/device/read":             [0.50, 0.30],
        "GET /api/device/snapshot":         [0.50, 0.30],
        "GET /api/device/actuator":         [0.30, 0.10],
        "GET /api/device/files/list":       [0.65, 0.45],
        # Write tier — medium
        "POST /api/device/write":           [0.70, 0.50],
        "GET /api/device/portal":           [0.65, 0.45],
        "POST /api/device/actuator/lock":   [0.65, 0.45],
        # Admin tier — strict (all require an authenticated session too)
        "POST /api/device/admin":           [0.85, 0.65],
        "POST /api/device/actuator/unlock": [0.85, 0.65],
        "POST /api/device/config/create":   [0.85, 0.65],
        "POST /api/device/config/delete":   [0.85, 0.65],
        "POST /api/device/files/create":    [0.85, 0.65],
        "POST /api/device/files/delete":    [0.85, 0.65],
    })
    default_threshold: list = field(default_factory=lambda: [0.70, 0.50])
    _blocklist: dict = field(default_factory=dict)         # src -> BlockEntry
    _overrides: dict = field(default_factory=dict)         # src -> [allow_min, escalate_min]
    _changes: list = field(default_factory=list)

    # ---- read paths (called by PDP) --------------------------------------

    def is_blocked(self, src: str) -> tuple[bool, str]:
        entry = self._blocklist.get(src)
        if entry is None:
            return False, ""
        # None expiry = permanent; otherwise check expiry
        if entry.expires_at is not None and entry.expires_at < time.time():
            del self._blocklist[src]
            return False, ""
        return True, entry.reason

    def thresholds_for(self, action: str, src: str) -> tuple[float, float]:
        if src in self._overrides:
            t = self._overrides[src]
            return float(t[0]), float(t[1])
        t = self.action_thresholds.get(action, self.default_threshold)
        return float(t[0]), float(t[1])

    # ---- write paths (called by LLM worker / seed) ------------------------

    def add_blocklist(self, src: str, ttl: Optional[float], reason: str, source: str, confidence: float) -> RuleChange:
        """ttl in seconds; pass None for a permanent block."""
        expires = None if ttl is None else time.time() + ttl
        self._blocklist[src] = BlockEntry(src=src, expires_at=expires, reason=reason)
        return self._record("add_blocklist", src, {"ttl_seconds": ttl, "reason": reason}, source, confidence)

    def remove_blocklist(self, src: str, reason: str, source: str, confidence: float) -> RuleChange:
        self._blocklist.pop(src, None)
        return self._record("remove_blocklist", src, {"reason": reason}, source, confidence)

    def override_thresholds(
        self, src: str, allow_min: float, escalate_min: float, source: str, confidence: float,
    ) -> RuleChange:
        self._overrides[src] = [float(allow_min), float(escalate_min)]
        return self._record(
            "override_thresholds", src,
            {"allow_min": float(allow_min), "escalate_min": float(escalate_min)},
            source, confidence,
        )

    def tune_action_thresholds(
        self, action: str, allow_min: float, escalate_min: float, source: str, confidence: float,
    ) -> RuleChange:
        """Adjust the GLOBAL thresholds for a specific action (not per-source)."""
        self.action_thresholds[action] = [float(allow_min), float(escalate_min)]
        return self._record(
            "tune_action_thresholds", action,
            {"allow_min": float(allow_min), "escalate_min": float(escalate_min)},
            source, confidence,
        )

    def _record(self, kind, target, detail, source, confidence) -> RuleChange:
        self.version += 1
        change = RuleChange(
            version=self.version,
            ts=datetime.now().strftime("%H:%M:%S"),
            kind=kind, target=target, detail=detail,
            source=source, confidence=confidence,
        )
        self._changes.append(change)
        # cap history to keep the endpoint response bounded
        if len(self._changes) > 200:
            self._changes = self._changes[-200:]
        return change

    # ---- introspection ---------------------------------------------------

    def snapshot(self) -> dict:
        now = time.time()
        return {
            "version": self.version,
            "blocklist": [
                {
                    "src": e.src,
                    "reason": e.reason,
                    "expires_in_seconds": (None if e.expires_at is None
                                           else max(0.0, e.expires_at - now)),
                }
                for e in self._blocklist.values()
                if e.expires_at is None or e.expires_at > now
            ],
            "device_overrides": dict(self._overrides),
            "action_thresholds": dict(self.action_thresholds),
            "recent_changes": [asdict(c) for c in self._changes[-30:]],
        }


# --------------------------------------------------------------------------
# Singleton — seeded with the previously-hardcoded blocklist entry.
# --------------------------------------------------------------------------

_ruleset: Optional[RuleSet] = None


def get_ruleset() -> RuleSet:
    global _ruleset
    if _ruleset is None:
        _ruleset = RuleSet()
        _ruleset.add_blocklist(
            "10.0.0.66", ttl=None,
            reason="initial seed rule", source="seed", confidence=1.0,
        )
    return _ruleset
