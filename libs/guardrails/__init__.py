"""
Regex-based guardrails: input filtering (block before any LLM call),
output filtering (block after generation, before caching/returning), and
structured-output validation (a safety net on the planner's own JSON,
never stricter than the fallback that already existed).

Deliberately regex-only, no model calls — zero added latency/cost, and
every decision is inspectable as a plain pattern list (see patterns.py).
"""
