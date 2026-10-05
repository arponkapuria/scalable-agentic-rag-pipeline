"""Gemma judge client for the eval scripts, using Google's OpenAI-compatible API.

It lives under eval/ because only eval scripts use it, and it stays a separate vendor from the generator and captioner.
"""
from services.api.app.clients.llm.openai_compatible import OpenAICompatibleClient
from services.api.app.config import settings
from libs.utils.rate_limiter import BackendRateLimiter

# Limits for gemma-4-31b-it from Google AI Studio.
_GEMMA_RPM = 30
_GEMMA_TPM = 16_000
_GEMMA_RPD = 14_400


def _build_gemma_rate_limiter() -> BackendRateLimiter:
    """Builds a rate limiter for the Gemma judge model."""
    limiter = BackendRateLimiter(shared=False)
    limiter.register(settings.GEMMA_JUDGE_MODEL, rpm=_GEMMA_RPM, tpm=_GEMMA_TPM, rpd=_GEMMA_RPD)
    return limiter


class GemmaClient(OpenAICompatibleClient):
    """Single-model Gemma client with no failover, so a failed call leaves the metric null instead of switching judges."""

    def __init__(self):
        super().__init__(
            base_url=settings.GEMMA_BASE_URL,
            models=[settings.GEMMA_JUDGE_MODEL],
            api_key=settings.GOOGLE_API_KEY,
            rate_limiter=_build_gemma_rate_limiter(),
            header_style="none",
        )


# Shared instance that eval scripts start and close themselves.
gemma_client = GemmaClient()