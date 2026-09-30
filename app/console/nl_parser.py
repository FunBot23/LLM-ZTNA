"""Natural-language query parser for the security console.

Translates admin-typed English into a structured operation from a
constrained vocabulary. The heavy lifting on safety is done by the
runner (query_runner.py) this file just turns text into intent.

Two backends:
  * Gemini (when GEMINI_API_KEY is set), asked for strict JSON output
  * Deterministic heuristic (always available) — pattern-matches
    keywords so the console still works without an API key.

Both return the same shape:
    {"operation": str, "args": dict, "explanation": str}
"""
from __future__ import annotations

import json
import os
import re
from typing import Optional

import httpx

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
)


PROMPT = """You are a security console assistant. Translate the admin's plain-English query into a structured operation from the fixed vocabulary below.

Admin query:
{query}

Vocabulary (return one of these as "operation"):
  * list_decisions        args: {{"verdict": "allow"|"escalate"|"deny"|null, "src": "<ip>"|null,
                                  "since": "today"|"hour"|"10m"|"day"|null, "limit": <int, default 25>}}
  * list_blocklist        args: {{}}
  * list_rule_changes     args: {{"source": "seed"|"llm"|"admin"|null, "since": <same as above>|null, "limit": <int>}}
  * count_decisions       args: {{"verdict": ..., "since": ...}}
  * top_sources           args: {{"verdict": ..., "since": ..., "limit": <int, default 10>}}
  * describe_policy       args: {{}}

Return strict JSON only:
{{
  "operation": "<one of the above>",
  "args": {{...}},
  "explanation": "<one plain-English sentence recapping what will be run>"
}}

If the query is unclear or outside this vocabulary, pick the closest match and explain the compromise in "explanation".
"""


async def parse(query: str) -> dict:
    """Turn `query` into {"operation": ..., "args": ..., "explanation": ...}."""
    if GEMINI_API_KEY:
        parsed = await _gemini_parse(query)
        if parsed is not None:
            return parsed
    return _heuristic_parse(query)


# ---------------------------------------------------------------------------
# LLM backend
# ---------------------------------------------------------------------------

async def _gemini_parse(query: str) -> Optional[dict]:
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{GEMINI_URL}?key={GEMINI_API_KEY}",
                json={
                    "contents": [{"parts": [{"text": PROMPT.format(query=query)}]}],
                    "generationConfig": {
                        "responseMimeType": "application/json",
                        "temperature": 0.1,
                    },
                },
            )
            resp.raise_for_status()
            body = resp.json()
            text = body["candidates"][0]["content"]["parts"][0]["text"]
            parsed = json.loads(text)
            if not isinstance(parsed, dict) or "operation" not in parsed:
                return None
            parsed.setdefault("args", {})
            parsed.setdefault("explanation", "")
            return parsed
    except Exception as e:
        print(f"[console] gemini parse failed: {e}; falling back to heuristic")
        return None


# ---------------------------------------------------------------------------
# Heuristic backend
# ---------------------------------------------------------------------------

_IP_PATTERN = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
_LIMIT_PATTERN = re.compile(r"\b(?:top|first|last|show|limit)\s+(\d{1,3})\b")


def _extract_since(q: str) -> Optional[str]:
    ql = q.lower()
    if "today" in ql:
        return "today"
    if "last 24" in ql or "24h" in ql or "past day" in ql or "day" in ql.split():
        return "day"
    if "hour" in ql or "1h" in ql or "60m" in ql:
        return "hour"
    if "10 min" in ql or "10m" in ql or "10-min" in ql:
        return "10m"
    if "5 min" in ql or "5m" in ql:
        return "5m"
    return None


def _extract_verdict(q: str) -> Optional[str]:
    ql = q.lower()
    if "blocked" in ql or "denied" in ql or "denies" in ql or "deny" in ql or "reject" in ql:
        return "deny"
    if "escalat" in ql or "challeng" in ql or "step-up" in ql or "step up" in ql:
        return "escalate"
    if "allowed" in ql or "allows" in ql or "granted" in ql or "allow" in ql:
        return "allow"
    return None


def _heuristic_parse(query: str) -> dict:
    ql = query.lower().strip()
    since = _extract_since(ql)
    verdict = _extract_verdict(ql)
    ip_matches = _IP_PATTERN.findall(query)
    limit_match = _LIMIT_PATTERN.search(ql)
    limit = int(limit_match.group(1)) if limit_match else None

    # -- policy / rules queries ---------------------------------------
    if any(k in ql for k in ("policy", "policies", "current rules", "threshold", "current setting")):
        return {
            "operation": "describe_policy",
            "args": {},
            "explanation": "Snapshot of the current policy: thresholds, blocklist size, and defaults.",
        }

    if any(k in ql for k in ("rule change", "policy change", "rule update", "policy update", "rule history")):
        args: dict = {}
        if since:
            args["since"] = since
        if limit:
            args["limit"] = limit
        for word in ("llm", "seed", "admin"):
            if word in ql:
                args["source"] = word
                break
        return {
            "operation": "list_rule_changes",
            "args": args,
            "explanation": f"Recent rule changes{' from ' + args['source'] if args.get('source') else ''}"
                           f"{' (' + since + ')' if since else ''}.",
        }

    if "blocklist" in ql or "blocked device" in ql or "who is blocked" in ql \
            or ("blocked" in ql and "today" not in ql and "hour" not in ql):
        return {
            "operation": "list_blocklist",
            "args": {},
            "explanation": "Currently-active blocklist entries.",
        }

    # -- count queries ------------------------------------------------
    if re.search(r"\bhow many\b|\bcount\b", ql):
        args = {}
        if verdict:
            args["verdict"] = verdict
        if since:
            args["since"] = since
        return {
            "operation": "count_decisions",
            "args": args,
            "explanation": f"Count of {verdict or 'all'} decisions{' (' + since + ')' if since else ''}.",
        }

    # -- top / most --------------------------------------------------
    if re.search(r"\btop\b|\bmost\b|\bworst\b|\bnoisiest\b", ql):
        args = {}
        if verdict:
            args["verdict"] = verdict
        if since:
            args["since"] = since
        if limit:
            args["limit"] = limit
        return {
            "operation": "top_sources",
            "args": args,
            "explanation": f"Top sources by {verdict or 'total'} decisions"
                           f"{' (' + since + ')' if since else ''}.",
        }

    # -- default: list_decisions ------------------------------------
    args = {}
    if verdict:
        args["verdict"] = verdict
    if since:
        args["since"] = since
    if ip_matches:
        args["src"] = ip_matches[0]
    if limit:
        args["limit"] = limit
    return {
        "operation": "list_decisions",
        "args": args,
        "explanation": (
            f"Recent {verdict or ''} decisions"
            f"{' for ' + args['src'] if args.get('src') else ''}"
            f"{' (' + since + ')' if since else ''}."
        ).replace("  ", " ").strip(),
    }
