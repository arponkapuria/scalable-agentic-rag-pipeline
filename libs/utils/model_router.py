"""
Routes a query to a "simple" or "complex" model tier based on query length and a fixed keyword list for comparative or multi-step phrasing. The routing logic is fixed in code; which model backs each tier is set via environment variables, so swapping models doesn't require a code change.
"""
from services.api.app.config import settings

_COMPLEX_KEYWORDS = (
    "compare", "comparison", "why", "analyze", "analysis",
    "difference between", "trade-off", "tradeoff", "pros and cons",
    "step by step", "summarize all", "across",
)


def route_model(query: str | None, context_chars: int = 0) -> str:
    """Picks a model tier based on query length and the presence of complexity keywords.

    Args:
        query: The query text to classify. None or empty falls back to the simple tier.
        context_chars: Length of any retrieved context to include in the length check.

    Returns:
        Either settings.MODEL_TIER_SIMPLE or settings.MODEL_TIER_COMPLEX.
    """
    if not query:
        return settings.MODEL_TIER_SIMPLE
    text = query.lower()
    is_long = (len(query) + context_chars) > settings.MODEL_ROUTING_CHAR_THRESHOLD
    has_complex_keyword = any(kw in text for kw in _COMPLEX_KEYWORDS)

    if is_long or has_complex_keyword:
        return settings.MODEL_TIER_COMPLEX
    return settings.MODEL_TIER_SIMPLE