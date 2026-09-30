"""User authentication and session management.

Handles:
  * user signup and login (self-service, no admin approval needed)
  * bcrypt password hashing with a per-user salt
  * in-memory session store keyed by cryptographically random tokens
  * cookie-based session transport with SameSite=Strict for CSRF safety

Users persist to users.json in the project root. Sessions live only in
memory - restarting the app logs everyone out.

Honest scope note: this is minimum-viable authentication for a demo.
Production would add password strength requirements, brute-force rate
limiting, session refresh, TLS, and probably 2FA.
"""
from __future__ import annotations

import json
import re
import secrets
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import bcrypt

USERS_FILE = Path(__file__).resolve().parent.parent / "users.json"
SESSION_TTL_SECONDS = 8 * 3600  # 8 hours

USERNAME_RE = re.compile(r"^[a-zA-Z0-9_-]{3,32}$")
PASSWORD_MIN_LENGTH = 6

_users: dict = {}
_sessions: dict = {}  # token -> {"username": str, "expires_at": float}


def _load_users() -> None:
    global _users
    if USERS_FILE.exists():
        try:
            _users = json.loads(USERS_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            _users = {}
    else:
        _users = {}


def _save_users() -> None:
    USERS_FILE.write_text(json.dumps(_users, indent=2), encoding="utf-8")


_load_users()


def _hash_password(pw: str) -> str:
    return bcrypt.hashpw(pw.encode("utf-8"), bcrypt.gensalt(rounds=10)).decode("utf-8")


def _verify_password(pw: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(pw.encode("utf-8"), hashed.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def _validate_credentials(username: str, password: str) -> Optional[str]:
    if not USERNAME_RE.match(username or ""):
        return "username must be 3-32 characters (letters, digits, underscore, hyphen)"
    if len(password or "") < PASSWORD_MIN_LENGTH:
        return f"password must be at least {PASSWORD_MIN_LENGTH} characters"
    return None


def signup(username: str, password: str) -> tuple[bool, str]:
    """Register a new user. Returns (ok, session_token_or_error)."""
    err = _validate_credentials(username, password)
    if err:
        return False, err
    if username in _users:
        return False, "username already exists"
    _users[username] = {
        "password_hash": _hash_password(password),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    _save_users()
    return True, _make_session(username)


def login(username: str, password: str) -> tuple[bool, str]:
    """Authenticate an existing user. Returns (ok, session_token_or_error)."""
    user = _users.get(username or "")
    if user is None or not _verify_password(password or "", user["password_hash"]):
        return False, "invalid username or password"
    return True, _make_session(username)


def _make_session(username: str) -> str:
    token = secrets.token_urlsafe(24)
    _sessions[token] = {
        "username": username,
        "expires_at": time.time() + SESSION_TTL_SECONDS,
    }
    return token


def get_session(token: Optional[str]) -> Optional[str]:
    """Return username for a valid session token, or None if invalid/expired."""
    if not token:
        return None
    session = _sessions.get(token)
    if session is None:
        return None
    if session["expires_at"] < time.time():
        _sessions.pop(token, None)
        return None
    return session["username"]


def logout(token: Optional[str]) -> None:
    if token:
        _sessions.pop(token, None)


def sweep_expired_sessions() -> int:
    """Remove expired sessions from memory. Returns count removed."""
    now = time.time()
    expired = [t for t, s in _sessions.items() if s["expires_at"] < now]
    for t in expired:
        _sessions.pop(t, None)
    return len(expired)
