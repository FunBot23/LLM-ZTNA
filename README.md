An LLM assisted cybersecurity zero-trust framework for IoT traffic that combines a TabTransformer + GraphSAGE GNN trust pipeline with an LLM policy layer. 
The LLM advises a human reviewer on unclear traffic and proposes policy changes.

How it works:

Traffic (live API requests) is aggregated into flows.

A TabTransformer scores each flow and a GraphSAGE GNN scores its short-window neighborhood context. 

Both scores are then combined into a final trust score.

The PDP turns that score into a verdict: allow / escalate / deny.

Escalates go to a human review queue, with an LLM-generated recommendation and reasoning attached (Gemini, with a heuristic fallback if Gemini is unavailable).

A separate LLM policy loop watches for repeated escalate patterns and proposes time-limited rule/blocklist changes.

Everything streams live to a dashboard: decisions, policy updates, and pending reviews.
