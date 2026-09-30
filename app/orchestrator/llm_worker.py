"""Async LLM policy orchestrator.

Subscribes to the decision bus. Buffers escalate-band events per source
IP. Every CHECK_INTERVAL_SECONDS, hands qualifying batches to the LLM
(Gemini via direct HTTP), applies proposed rule changes to the active
RuleSet, and publishes rule_update events back to the bus so the
dashboard sees policy evolve in real time.
If GEMINI_API_KEY is unset, a deterministic
heuristic stands in so the demo is always runnable.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from collections import defaultdict, deque
from typing import Optional

import httpx

from app.bus import bus
from app.orchestrator.rules import RuleChange, get_ruleset

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
)

CHECK_INTERVAL_SECONDS = 30.0
BUFFER_SECONDS = 120.0
MIN_ESCALATES_PER_SRC = 3

# Safety asymmetry — see module docstring.
TIGHTEN_MIN_CONFIDENCE = 0.5
LOOSEN_MIN_CONFIDENCE = 0.85


class LLMTriage:
    def __init__(self) -> None:
        self._buffer: deque = deque()  # (monotonic_ts, event)

    async def run(self) -> None:
        mode = "gemini" if GEMINI_API_KEY else "heuristic (no GEMINI_API_KEY)"
        print(f"[llm_worker] started in {mode} mode; check interval {CHECK_INTERVAL_SECONDS}s")

        queue = bus.subscribe()
        asyncio.create_task(self._periodic())
        try:
            while True:
                event = await queue.get()
                # Ignore our own rule_update events; only track escalate decisions.
                if event.get("verdict") == "escalate":
                    self._buffer.append((time.monotonic(), event))
                    self._prune(time.monotonic())
        finally:
            bus.unsubscribe(queue)

    def _prune(self, now: float) -> None:
        cutoff = now - BUFFER_SECONDS
        while self._buffer and self._buffer[0][0] < cutoff:
            self._buffer.popleft()

    async def _periodic(self) -> None:
        while True:
            await asyncio.sleep(CHECK_INTERVAL_SECONDS)
            try:
                await self._triage()
            except Exception as e:
                print(f"[llm_worker] triage error: {e}")

    async def _triage(self) -> None:
        self._prune(time.monotonic())
        if not self._buffer:
            return

        by_src: dict[str, list] = defaultdict(list)
        for _t, ev in self._buffer:
            by_src[ev["src"]].append(ev)

        candidates = {s: evs for s, evs in by_src.items() if len(evs) >= MIN_ESCALATES_PER_SRC}
        if not candidates:
            return

        print(f"[llm_worker] triaging {len(candidates)} src(s): {list(candidates.keys())}")

        response = await self._call_llm(candidates) if GEMINI_API_KEY else self._heuristic(candidates)
        if not response:
            return

        actions = response.get("actions", []) or []
        analysis = str(response.get("analysis", ""))[:400]
        applied = 0
        for action in actions:
            change = self._apply(action)
            if change is None:
                continue
            applied += 1
            await bus.publish({
                "event_type": "rule_update",
                "ts": change.ts,
                "version": change.version,
                "kind": change.kind,
                "target": change.target,
                "source": change.source,
                "confidence": change.confidence,
                "detail": change.detail,
                "llm_reasoning": analysis,
            })

        # Drop applied srcs from the buffer so we don't keep re-triaging.
        applied_srcs = {a.get("target") for a in actions if a.get("target")}
        if applied_srcs:
            self._buffer = deque(
                (t, e) for t, e in self._buffer if e["src"] not in applied_srcs
            )
        print(f"[llm_worker] applied {applied} rule change(s)")

    # ---- action safety gates --------------------------------------------

    def _apply(self, action: dict) -> Optional[RuleChange]:
        kind = str(action.get("type", ""))
        target = str(action.get("target", ""))
        reason = str(action.get("reason", "llm-proposed"))
        confidence = float(action.get("confidence", 0.5))

        if not kind or not target:
            return None

        ruleset = get_ruleset()

        if kind == "add_blocklist":
            if confidence < TIGHTEN_MIN_CONFIDENCE:
                print(f"[llm_worker] skip add_blocklist({target}) conf={confidence:.2f}<{TIGHTEN_MIN_CONFIDENCE}")
                return None
            ttl = float(action.get("ttl_seconds", 300))
            return ruleset.add_blocklist(target, ttl=ttl, reason=reason, source="llm", confidence=confidence)

        if kind == "override_thresholds":
            try:
                new_allow = float(action["allow_min"])
                new_escalate = float(action["escalate_min"])
            except (KeyError, ValueError):
                return None
            cur_allow, _ = ruleset.thresholds_for("POST /api/device/write", target)
            is_tightening = new_allow >= cur_allow
            required = TIGHTEN_MIN_CONFIDENCE if is_tightening else LOOSEN_MIN_CONFIDENCE
            if confidence < required:
                direction = "tightening" if is_tightening else "loosening"
                print(f"[llm_worker] skip override({target}) {direction} conf={confidence:.2f}<{required}")
                return None
            return ruleset.override_thresholds(target, new_allow, new_escalate, source="llm", confidence=confidence)

        if kind == "remove_blocklist":
            if confidence < LOOSEN_MIN_CONFIDENCE:
                print(f"[llm_worker] skip remove_blocklist({target}) conf={confidence:.2f}<{LOOSEN_MIN_CONFIDENCE}")
                return None
            return ruleset.remove_blocklist(target, reason=reason, source="llm", confidence=confidence)

        if kind == "tune_action_thresholds":
            try:
                new_allow = float(action["allow_min"])
                new_escalate = float(action["escalate_min"])
            except (KeyError, ValueError):
                return None
            # Direction: any component going up = tightening, going down = loosening.
            cur = ruleset.action_thresholds.get(target, ruleset.default_threshold)
            cur_allow, cur_escalate = float(cur[0]), float(cur[1])
            is_loosening = new_allow < cur_allow or new_escalate < cur_escalate
            required = LOOSEN_MIN_CONFIDENCE if is_loosening else TIGHTEN_MIN_CONFIDENCE
            if confidence < required:
                direction = "loosening" if is_loosening else "tightening"
                print(f"[llm_worker] skip tune_action_thresholds({target}) {direction} "
                      f"conf={confidence:.2f}<{required}")
                return None
            return ruleset.tune_action_thresholds(target, new_allow, new_escalate,
                                                   source="llm", confidence=confidence)

        print(f"[llm_worker] unknown action kind: {kind}")
        return None

    # ---- LLM & heuristic backends ---------------------------------------

    async def _call_llm(self, candidates: dict) -> Optional[dict]:
        prompt = self._build_prompt(candidates)
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                for attempt in range(3):
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
                    if resp.status_code == 429:
                        delay = self._parse_retry_delay(resp) or 20.0
                        print(f"[llm_worker] 429; retrying in {delay}s")
                        await asyncio.sleep(delay)
                        continue
                    resp.raise_for_status()
                    body = resp.json()
                    text = body["candidates"][0]["content"]["parts"][0]["text"]
                    return json.loads(text)
        except Exception as e:
            print(f"[llm_worker] gemini call failed: {e}")
        return None

    @staticmethod
    def _parse_retry_delay(resp: httpx.Response) -> Optional[float]:
        try:
            body = resp.json()
            for detail in body.get("error", {}).get("details", []):
                if "retryDelay" in detail:
                    m = re.match(r"(\d+)s", str(detail["retryDelay"]))
                    if m:
                        return float(m.group(1))
        except Exception:
            return None
        return None

    def _build_prompt(self, candidates: dict) -> str:
        lines = []
        for src, events in candidates.items():
            action_counts = defaultdict(int)
            for e in events:
                action_counts[e.get("action", "")] += 1
            last = events[-1]
            lines.append(
                f"- {src}: {len(events)} escalate events; actions={dict(action_counts)}; "
                f"latest score={last.get('score')}; reason={(last.get('reason') or '')[:80]}"
            )
        summary = "\n".join(lines)
        return f"""You are a zero-trust security policy assistant. The enforcement layer has produced these escalate-band events over the last {int(BUFFER_SECONDS)} seconds. Each of these escalates was ALSO reviewed individually by a human via the review console (LLM-advised). Your job is different: propose STANDING rule changes for the near future so similar events don't need per-request human attention.

{summary}

Propose rule updates as strict JSON:
{{
  "analysis": "<one or two sentences>",
  "actions": [
    {{
      "type": "add_blocklist" | "remove_blocklist" | "override_thresholds" | "tune_action_thresholds",
      "target": "<src ip, or action string for tune_action_thresholds>",
      "confidence": <0.0-1.0>,
      "reason": "<short justification>",

      // add_blocklist:
      "ttl_seconds": <int, 300-3600 preferred>,

      // override_thresholds (per-source, decide-time):
      "allow_min": <0.0-1.0>,
      "escalate_min": <0.0-1.0>,

      // tune_action_thresholds (global, per action string like "POST /api/device/write"):
      // uses the same allow_min / escalate_min fields; target is the action
    }}
  ]
}}

Guidance:
- Prefer temporary blocks over permanent
- Only remove_blocklist / relaxations with strong evidence
- Tightening rules need confidence >= 0.5; loosening rules need >= 0.85
- For repeated escalates on the SAME action across many sources, tune_action_thresholds is more efficient than per-source overrides
- Empty "actions" is a valid response — humans are handling individual escalates via the review console

Output ONLY the JSON object, nothing else."""

    def _heuristic(self, candidates: dict) -> dict:
        """Deterministic stand-in for when GEMINI_API_KEY isn't set.

        Mirrors what a conservative LLM would recommend: many escalates
        across multiple actions from one source → temporary block; fewer
        but consistent escalates → tighten thresholds for that source.
        """
        actions = []
        for src, events in candidates.items():
            n = len(events)
            distinct_actions = len({e.get("action", "") for e in events})
            avg_score = sum(float(e.get("score", 0.5)) for e in events) / n

            if n >= 5 and distinct_actions >= 2 and avg_score < 0.4:
                actions.append({
                    "type": "add_blocklist",
                    "target": src,
                    "confidence": 0.7,
                    "reason": f"{n} escalates across {distinct_actions} actions, avg score {avg_score:.2f}",
                    "ttl_seconds": 600,
                })
            elif n >= 3 and avg_score < 0.55:
                actions.append({
                    "type": "override_thresholds",
                    "target": src,
                    "confidence": 0.6,
                    "reason": f"{n} escalates, avg score {avg_score:.2f}",
                    "allow_min": 0.85,
                    "escalate_min": 0.70,
                })
        return {
            "analysis": (
                f"heuristic mode (no LLM key set): {len(candidates)} src analyzed, "
                f"{len(actions)} action(s) proposed"
            ),
            "actions": actions,
        }
