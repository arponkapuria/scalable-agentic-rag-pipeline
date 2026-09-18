"""
Heuristic query-complexity -> model-tier router — used by planner.py,
query_rewriter.py, and responder.py's final-answer synthesis. (Originally
scoped to responder only; extended to planner/query-rewriter after a
live-observed classification-quality gap — see planner.py's model= comment.)

Routing LOGIC is fixed here per the locked design ("routing logic fixed
in code"); WHICH model backs each tier is env-driven (MODEL_TIER_SIMPLE/
MODEL_TIER_COMPLEX in .env) so swapping models doesn't touch code.

Complexity signal is intentionally simple — combined query+context length
plus a small fixed keyword list for multi-step/comparative phrasing. This
is a demo router: the point is to make the ROUTING MECHANISM real and
demoable ("here's a cheap model for simple lookups, a bigger one for
harder synthesis"), not to build a state-of-the-art complexity classifier.
"""
from services.api.app.config import settings

_COMPLEX_KEYWORDS = (
    "compare", "comparison", "why", "analyze", "analysis",
    "difference between", "trade-off", "tradeoff", "pros and cons",
    "step by step", "summarize all", "across",
)


def route_model(query: str | None, context_chars: int = 0) -> str:
    """Returns a Groq model id — either MODEL_TIER_SIMPLE or
    MODEL_TIER_COMPLEX, both of which must also appear in GROQ_MODELS so
    the rate limiter tracks them.

    Defensive None/empty guard (found live): a caller's own upstream bug
    — planner.py's plan.get("refined_query", user_query) silently
    returning None for a key present with an explicit JSON null, rather
    than falling back — crashed here with 'NoneType' has no attribute
    'lower'. Fixed at that call site too, but guarding here as well means
    a future caller with the same class of bug degrades to the simple
    tier instead of crashing the whole request."""
    if not query:
        return settings.MODEL_TIER_SIMPLE
    text = query.lower()
    is_long = (len(query) + context_chars) > settings.MODEL_ROUTING_CHAR_THRESHOLD
    has_complex_keyword = any(kw in text for kw in _COMPLEX_KEYWORDS)

    if is_long or has_complex_keyword:
        return settings.MODEL_TIER_COMPLEX
    return settings.MODEL_TIER_SIMPLE
