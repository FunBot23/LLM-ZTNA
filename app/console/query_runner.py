"""Constrained query runner for the natural-language console.

The console never lets the LLM (or user) write SQL directly. Instead
the parser translates a query into one of a fixed vocabulary of
operations declared here, and this runner executes it safely against
SQLite (via aiosqlite) and the live RuleSet.

Operations:
  * list_decisions(verdict?, src?, since?, limit=25)
  * list_blocklist()
  * list_rule_changes(source?, since?, limit=25)
  * count_decisions(verdict?, since?)
  * top_sources(verdict?, since?, limit=10)
  * describe_policy()

Time filters use created_at (unix ts) so "last hour" is trivially
correct across day boundaries. "today" = since local midnight.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Any, Optional

import aiosqlite

from app.db import DB_PATH
from app.orchestrator.rules import get_ruleset

VALID_VERDICTS = {"allow", "escalate", "deny"}
VALID_RULE_SOURCES = {"seed", "llm", "admin"}
MAX_LIMIT = 100


def _resolve_since(since: Optional[str]) -> Optional[float]:
    """Map an English time reference to a unix cutoff (or None for 'all')."""
    if not since:
        return None
    s = since.strip().lower()
    if s in ("all", "any", "ever"):
        return None
    now = time.time()
    if s in ("today",):
        midnight = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        return midnight.timestamp()
    if s in ("hour", "last hour", "1h", "60m"):
        return now - 3600
    if s in ("10m", "10 minutes", "last 10 minutes"):
        return now - 600
    if s in ("5m", "5 minutes"):
        return now - 300
    if s in ("day", "24h", "last 24 hours"):
        return now - 86400
    # Fallback: try "Nm" / "Nh" / "Ns"
    try:
        if s.endswith("s"):
            return now - float(s[:-1])
        if s.endswith("m"):
            return now - float(s[:-1]) * 60
        if s.endswith("h"):
            return now - float(s[:-1]) * 3600
    except ValueError:
        pass
    return None


async def run(operation: str, args: Optional[dict] = None) -> dict:
    """Dispatch to the named operation. Returns {"columns": [...], "rows": [...], "meta": {...}}."""
    args = args or {}

    if operation == "list_decisions":
        return await _list_decisions(args)
    if operation == "list_blocklist":
        return _list_blocklist()
    if operation == "list_rule_changes":
        return _list_rule_changes(args)
    if operation == "count_decisions":
        return await _count_decisions(args)
    if operation == "top_sources":
        return await _top_sources(args)
    if operation == "describe_policy":
        return _describe_policy()
    return {"columns": [], "rows": [], "meta": {"error": f"unknown operation: {operation}"}}


# ---------------------------------------------------------------------------
# SQLite-backed operations
# ---------------------------------------------------------------------------

async def _list_decisions(args: dict) -> dict:
    where: list[str] = []
    params: list[Any] = []
    if (v := args.get("verdict")) and v in VALID_VERDICTS:
        where.append("verdict = ?")
        params.append(v)
    if (s := args.get("src")):
        where.append("src = ?")
        params.append(str(s))
    cutoff = _resolve_since(args.get("since"))
    if cutoff is not None:
        where.append("created_at >= ?")
        params.append(cutoff)
    limit = min(int(args.get("limit", 25) or 25), MAX_LIMIT)

    sql = "SELECT ts, src, action, verdict, score, reason FROM decisions"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(sql, params)
        rows = [dict(r) for r in await cur.fetchall()]

    return {
        "columns": ["ts", "src", "action", "verdict", "score", "reason"],
        "rows": rows,
        "meta": {"count": len(rows), "filters": {k: v for k, v in args.items() if v}},
    }


async def _count_decisions(args: dict) -> dict:
    where: list[str] = []
    params: list[Any] = []
    if (v := args.get("verdict")) and v in VALID_VERDICTS:
        where.append("verdict = ?")
        params.append(v)
    cutoff = _resolve_since(args.get("since"))
    if cutoff is not None:
        where.append("created_at >= ?")
        params.append(cutoff)

    sql = "SELECT COUNT(*) AS n FROM decisions"
    if where:
        sql += " WHERE " + " AND ".join(where)

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(sql, params)
        row = await cur.fetchone()

    return {
        "columns": ["count"],
        "rows": [{"count": int(row["n"] if row else 0)}],
        "meta": {"filters": {k: v for k, v in args.items() if v}},
    }


async def _top_sources(args: dict) -> dict:
    where: list[str] = []
    params: list[Any] = []
    if (v := args.get("verdict")) and v in VALID_VERDICTS:
        where.append("verdict = ?")
        params.append(v)
    cutoff = _resolve_since(args.get("since"))
    if cutoff is not None:
        where.append("created_at >= ?")
        params.append(cutoff)
    limit = min(int(args.get("limit", 10) or 10), MAX_LIMIT)

    sql = "SELECT src, COUNT(*) AS n FROM decisions"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " GROUP BY src ORDER BY n DESC LIMIT ?"
    params.append(limit)

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(sql, params)
        rows = [{"src": r["src"], "count": int(r["n"])} for r in await cur.fetchall()]

    return {
        "columns": ["src", "count"],
        "rows": rows,
        "meta": {"filters": {k: v for k, v in args.items() if v}},
    }


# ---------------------------------------------------------------------------
# RuleSet-backed operations
# ---------------------------------------------------------------------------

def _list_blocklist() -> dict:
    snap = get_ruleset().snapshot()
    return {
        "columns": ["src", "reason", "expires_in_seconds"],
        "rows": snap["blocklist"],
        "meta": {"count": len(snap["blocklist"]), "rule_version": snap["version"]},
    }


def _list_rule_changes(args: dict) -> dict:
    snap = get_ruleset().snapshot()
    changes = snap["recent_changes"]

    if (src := args.get("source")) and src in VALID_RULE_SOURCES:
        changes = [c for c in changes if c["source"] == src]

    cutoff = _resolve_since(args.get("since"))
    if cutoff is not None:
        # RuleChange only stores wall-clock string "ts" — best-effort filter
        # by parsing HH:MM:SS relative to today.
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        kept = []
        for c in changes:
            try:
                hh, mm, ss = c["ts"].split(":")
                dt = today.replace(hour=int(hh), minute=int(mm), second=int(ss))
                if dt.timestamp() >= cutoff:
                    kept.append(c)
            except (ValueError, KeyError):
                kept.append(c)
        changes = kept

    limit = min(int(args.get("limit", 25) or 25), MAX_LIMIT)
    changes = changes[-limit:]

    return {
        "columns": ["version", "ts", "kind", "target", "source", "confidence"],
        "rows": [
            {k: c.get(k) for k in ("version", "ts", "kind", "target", "source", "confidence")}
            for c in changes
        ],
        "meta": {"count": len(changes), "current_rule_version": snap["version"]},
    }


def _describe_policy() -> dict:
    snap = get_ruleset().snapshot()
    return {
        "columns": ["key", "value"],
        "rows": [
            {"key": "rule_version", "value": snap["version"]},
            {"key": "blocklist_entries", "value": len(snap["blocklist"])},
            {"key": "device_overrides", "value": len(snap["device_overrides"])},
            {"key": "session_policies", "value": len(snap.get("session_policies", {}))},
            {"key": "action_thresholds", "value": snap["action_thresholds"]},
            {"key": "default_session_ttl_seconds",
             "value": snap.get("defaults", {}).get("session_ttl_seconds")},
            {"key": "default_session_methods",
             "value": snap.get("defaults", {}).get("session_methods")},
        ],
        "meta": {"rule_version": snap["version"]},
    }
