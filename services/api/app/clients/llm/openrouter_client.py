"""
OpenRouter backend client, used as the automatic backup to Groq (or primary, if configured). Account-level quota shared across every model, unlike Groq's per-model limits.
"""
from services.api.app.clients.llm.openai_compatible import OpenAICompatibleClient
from services.api.app.config import settings
from libs.utils.rate_limiter import BackendRateLimiter

_OPENROUTER_RPM = 20
_OPENROUTER_RPD = 50


def _build_openrouter_rate_limiter() -> BackendRateLimiter:
    """Builds the shared, account-level rate limiter for OpenRouter.

    Returns:
        A BackendRateLimiter with one shared tracker.
    """
    limiter = BackendRateLimiter(shared=True)
    limiter.register("_shared", rpm=_OPENROUTER_RPM, rpd=_OPENROUTER_RPD)
    return limiter


class OpenRouterClient(OpenAICompatibleClient):
    """OpenRouter's OpenAI-compatible endpoint."""

    def __init__(self):
        super().__init__(
            base_url="https://openrouter.ai/api/v1",
            models=settings.OPENROUTER_MODELS,
            api_key=settings.OPENROUTER_API_KEY,
            rate_limiter=_build_openrouter_rate_limiter(),
            header_style="openrouter",
        )