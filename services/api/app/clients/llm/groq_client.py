"""
Groq backend client, with per-model rate limits and round-robin dispatch since each model has a separate quota.
"""
from services.api.app.clients.llm.openai_compatible import OpenAICompatibleClient
from services.api.app.config import settings
from libs.utils.rate_limiter import BackendRateLimiter

_GROQ_MODEL_LIMITS = {
    "openai/gpt-oss-120b": dict(rpm=30, rpd=1000, tpm=8000, tpd=200_000),
    "qwen/qwen3.8-27b": dict(rpm=30, rpd=1000, tpm=8000, tpd=200_000),
    "openai/gpt-oss-20b": dict(rpm=30, rpd=1000, tpm=8000, tpd=200_000),
}


def _build_groq_rate_limiter(models: list[str]) -> BackendRateLimiter:
    """Builds a per-model rate limiter for the given Groq models.

    Args:
        models: List of Groq model ids to register.

    Returns:
        A BackendRateLimiter with one tracker per model.
    """
    limiter = BackendRateLimiter(shared=False)
    for model in models:
        limits = _GROQ_MODEL_LIMITS.get(model)
        if limits:
            limiter.register(model, **limits)
        else:
            limiter.register(model)  # unknown model — no limits, falls back to reactive handling
    return limiter


class GroqClient(OpenAICompatibleClient):
    """Groq's OpenAI-compatible endpoint, with per-model round-robin dispatch."""

    def __init__(self):
        super().__init__(
            base_url="https://api.groq.com/openai/v1",
            models=settings.GROQ_MODELS,
            api_key=settings.GROQ_API_KEY,
            rate_limiter=_build_groq_rate_limiter(settings.GROQ_MODELS),
            header_style="groq",
            dispatch_strategy="round_robin",
        )