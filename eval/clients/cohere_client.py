"""Cohere judge client, an alternative to Gemma, using Cohere's OpenAI-compatible API."""

from services.api.app.clients.llm.openai_compatible import OpenAICompatibleClient
from services.api.app.config import settings
from libs.utils.rate_limiter import BackendRateLimiter

_COHERE_RPM = 20
_COHERE_RPD = 1000  # The real trial cap is 1,000 per month, so this only approximates it per run.


def _build_cohere_rate_limiter() -> BackendRateLimiter:
    """Builds a rate limiter for the Cohere judge model."""
    limiter = BackendRateLimiter(shared=False)
    limiter.register(settings.COHERE_JUDGE_MODEL, rpm=_COHERE_RPM, rpd=_COHERE_RPD)
    return limiter


class CohereClient(OpenAICompatibleClient):
    """Single-model Cohere client with no failover."""

    def __init__(self):
        super().__init__(
            base_url=settings.COHERE_BASE_URL,
            models=[settings.COHERE_JUDGE_MODEL],
            api_key=settings.COHERE_API_KEY,
            rate_limiter=_build_cohere_rate_limiter(),
            header_style="none",
        )


# Shared instance that eval scripts start and close themselves.
cohere_client = CohereClient()