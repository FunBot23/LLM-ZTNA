"""Per-escalate LLM advisor.

Different from llm_worker.py (which does batch, autonomous rule triage).
This one is fired PER escalate event and its output is *advisory only*
Attached to a pending review record so the human operator sees it as
guidance. The human still makes the actual decision.

Backends: Gemini (when GEMINI_API_KEY is set) or a heuristic. Both
return the same shape.
"""
from __future__ import annotations

import json
import os
from typing import Optional

import httpx

from app.bus import bus
from app.enforcement.review_queue import review_queue

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
)


async def advise(
    pending_id: str, src: str, action: str,
    features: Optional[dict], pdp_reason: str, pdp_score: float,
) -> None:
    """Fire LLM (or heuristic), attach advice to the pending record,
    publish a review_updated bus event so the dashboard refreshes.
    """
    result: Optional[dict] = None
    if GEMINI_API_KEY:
        result = await _gemini_advise(src, action, features, pdp_reason, pdp_score)
    if result is None:
        result = _heuristic_advise(src, action, features, pdp_reason, pdp_score)

    review_queue.attach_llm_advice(
        pending_id,
        recommendation=result.get("recommendation"),
        reasoning=str(result.get("reasoning", ""))[:400],
        confidence=float(result.get("confidence", 0.5)),
        retry_seconds=result.get("retry_after_seconds"),
    )
    await bus.publish({"event_type": "review_updated", "id": pending_id})


async def _gemini_advise(
    src: str, action: str, features: Optional[dict],
    pdp_reason: str, pdp_score: float,
) -> Optional[dict]:
    features_str = json.dumps(features)[:400] if features else "(none — request via mock API)"
    prompt = f"""You are advising a security operator on a single zero-trust escalate event.

Event:
  Source device:    {src}
  Action:           {action}
  PDP trust score:  {pdp_score:.2f}
  PDP reason:       {pdp_reason}
  Flow features:    {features_str}

Recommend one of: allow | deny | retry.
Give a short (1-2 sentence) reasoning the operator can read.

Strict JSON only:
{{
  "recommendation": "allow" | "deny" | "retry",
  "retry_after_seconds": <int, only when recommendation is "retry">,
  "reasoning": "<1-2 sentences>",
  "confidence": <0.0-1.0>
}}
Output only the JSON."""

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                f"{GEMINI_URL}?key={GEMINI_API_KEY}",
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "responseMimeType": "application/json",
                        "temperature": 0.2,
                    },
                },
            )
            resp.raise_for_status()
            body = resp.json()
            text = body["candidates"][0]["content"]["parts"][0]["text"]
            parsed = json.loads(text)
            if not isinstance(parsed, dict) or "recommendation" not in parsed:
                return None
            return parsed
    except Exception as e:
        print(f"[advisor] gemini failed: {e}; falling back to heuristic")
        return None


def _heuristic_advise(
    src: str, action: str, features: Optional[dict],
    pdp_reason: str, pdp_score: float,
) -> dict:
    """Deterministic stand-in.

    Rules of thumb, tuned to be conservative (in doubt → retry):
      * very low score (<0.3) → deny
      * upper-band escalate (>0.65) → allow with modest confidence
      * middle band → retry in 60s (let the pattern develop)
    """
    if pdp_score < 0.30:
        return {
            "recommendation": "deny",
            "reasoning": f"Very low trust score ({pdp_score:.2f}); safest to deny.",
            "confidence": 0.75,
        }
    if pdp_score >= 0.65:
        return {
            "recommendation": "allow",
            "reasoning": f"Score {pdp_score:.2f} is at the upper edge of the escalate band; likely benign.",
            "confidence": 0.60,
        }
    return {
        "recommendation": "retry",
        "retry_after_seconds": 60,
        "reasoning": (
            f"Score {pdp_score:.2f} is inconclusive; suggest retry in 60s "
            "to let short-term behavior develop."
        ),
        "confidence": 0.55,
    }
