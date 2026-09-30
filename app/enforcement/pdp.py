"""Policy Decision Point.

Combines up to three signals:
  * Blocklist (attribute match, from live rule set) — hard rule, trumps everything.
  * Transformer per-flow score — primary trust signal.
  * GraphSAGE context score — short-window neighborhood behavior.

Score combination is a weighted average with the Transformer as the
primary voice (CONTEXT_WEIGHT=0.3). Both scorers degrade gracefully:
if neither is loaded, we fall back to the identity-hash stub.

The blocklist and per-action thresholds now come from the live RuleSet
singleton — the LLM policy orchestrator can add/remove blocklist entries
and override thresholds at runtime without a restart.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Literal, Optional

from app.analytics.gnn_scorer import get_context_scorer
from app.analytics.scorer import get_scorer
from app.orchestrator.rules import get_ruleset

Verdict = Literal["allow", "escalate", "deny"]

# Weight for the context (GNN) term when combining with per-flow trust.
CONTEXT_WEIGHT = 0.3


@dataclass
class DecisionRequest:
    src: str
    action: str
    features: Optional[dict] = None


@dataclass
class Decision:
    verdict: Verdict
    score: float
    reason: str
    pending_review_id: Optional[str] = None

@dataclass
class Decision:
    verdict: Verdict
    score: float
    reason:str
    pending_review_id: Optional[str] = None

def _stub_trust_score(device_id: str) -> float:
    h = hashlib.sha256(device_id.encode()).digest()
    base = int.from_bytes(h[:2], "big") / 0xFFFF
    noise = (random.random() - 0.5) * 0.15
    return max(0.0, min(1.0, base + noise))


def _combined_trust_score(src: str, features: Optional[dict]) -> tuple[float, str]:
    flow_scorer = get_scorer()
    if features is not None and flow_scorer is not None:
        p_mal, _ = flow_scorer.score(features)
        flow_trust = 1.0 - p_mal
        flow_note = f"flow={flow_trust:.2f}"
    else:
        flow_trust = _stub_trust_score(src)
        flow_note = f"stub={flow_trust:.2f}"

    ctx_scorer = get_context_scorer()
    if features is not None and ctx_scorer is not None:
        ctx_scorer.observe(features)
        p_ctx, ctx_meta = ctx_scorer.score(src)
        ctx_trust = 1.0 - p_ctx
        combined = (1 - CONTEXT_WEIGHT) * flow_trust + CONTEXT_WEIGHT * ctx_trust
        return combined, f"{flow_note} ctx={ctx_trust:.2f} ({ctx_meta})"

    return flow_trust, flow_note


def decide(req: DecisionRequest) -> Decision:
    ruleset = get_ruleset()

    # Rule 1 — dynamic blocklist (may be modified by the LLM orchestrator).
    blocked, block_reason = ruleset.is_blocked(req.src)
    if blocked:
        return Decision("deny", 0.0, f"blocklist v{ruleset.version}: {block_reason}")

    # Rule 2 — score-based, thresholds from the live rule set.
    score, source_note = _combined_trust_score(req.src, req.features)
    allow_min, escalate_min = ruleset.thresholds_for(req.action, req.src)

    if score >= allow_min:
        verdict, band = "allow", f"score {score:.2f} \u2265 {allow_min:.2f}"
    elif score >= escalate_min:
        verdict = "escalate"
        band = f"score {score:.2f} in review band [{escalate_min:.2f}, {allow_min:.2f}]"
    else:
        verdict, band = "deny", f"score {score:.2f} < {escalate_min:.2f}"

    return Decision(verdict, score, f"{band} \u2014 {source_note}")
