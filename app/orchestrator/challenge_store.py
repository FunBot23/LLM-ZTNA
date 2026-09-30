"""Challenge tokens + step-up session tracking.

When the PDP escalates, the PEP issues a one-time token. The client
proves it can complete step-up by returning that token on the retry.
On successful redemption, a short-lived degraded session opens: reads
are allowed, writes are not.

In production the step-up would be OTP/MFA/hardware-key verification.
Here we simulate: possessing the token *is* the proof, since it was
delivered over the (assumed-secure) response channel to that device.
Tokens are single-use, source-bound, and time-limited.
"""
from __future__ import annotations

import secrets
import time
from typing import Optional


class ChallengeStore:
    def __init__(
        self,
        token_ttl_seconds: float = 300.0,
        session_ttl_seconds: float = 120.0,
    ) -> None:
        self.token_ttl = token_ttl_seconds
        self.session_ttl = session_ttl_seconds
        self._challenges: dict[str, tuple[str, float]] = {}   # token -> (src, expires_at)
        self._sessions: dict[str, float] = {}                 # src   -> session_expires_at

    def issue(self, src: str, session_ttl_seconds: Optional[float] = None) -> str:
        """Mint a fresh challenge token bound to `src`.

        `session_ttl_seconds` overrides the default session TTL for the
        session that this token will open when redeemed. Lets the PEP
        honor per-source policy from the RuleSet.
        """
        token = secrets.token_urlsafe(16)
        # Store token → (src, token_expiry, session_ttl_when_redeemed)
        self._challenges[token] = (
            src,
            time.time() + self.token_ttl,
            self.session_ttl if session_ttl_seconds is None else float(session_ttl_seconds),
        )
        return token

    def redeem(self, token: str, src: str) -> bool:
        """Consume a token. Returns True → step-up session begins for `src`."""
        entry = self._challenges.pop(token, None)
        if entry is None:
            return False
        bound_src, expires, session_ttl = entry
        if bound_src != src or expires < time.time():
            return False
        self._sessions[src] = time.time() + session_ttl
        return True

    def has_active_session(self, src: str) -> bool:
        exp = self._sessions.get(src)
        if exp is None:
            return False
        if exp < time.time():
            del self._sessions[src]
            return False
        return True

    def session_remaining(self, src: str) -> float:
        exp = self._sessions.get(src, 0.0)
        return max(0.0, exp - time.time())


# Module-level singleton — shared across middleware invocations.
challenge_store = ChallengeStore()
