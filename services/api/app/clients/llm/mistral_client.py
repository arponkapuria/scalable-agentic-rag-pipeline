"""
Standalone Mistral vision client, used only for PDF figure captioning during ingestion — never part of the chat failover pool. No backup configured: a caption failure just drops that figure.
"""
from services.api.app.clients.llm.openai_compatible import OpenAICompatibleClient
from services.api.app.config import settings
from libs.utils.rate_limiter import BackendRateLimiter

_MISTRAL_RPM = 30
_MISTRAL_TPM = 937_500


def _build_mistral_rate_limiter() -> BackendRateLimiter:
    """Builds a single-model rate limiter for the Mistral vision model.

    Returns:
        A BackendRateLimiter registered for settings.MISTRAL_VISION_MODEL.
    """
    limiter = BackendRateLimiter(shared=False)
    limiter.register(settings.MISTRAL_VISION_MODEL, rpm=_MISTRAL_RPM, tpm=_MISTRAL_TPM)
    return limiter


class MistralClient(OpenAICompatibleClient):
    """Single-model vision client for figure captioning."""

    def __init__(self):
        super().__init__(
            base_url="https://api.mistral.ai/v1",
            models=[settings.MISTRAL_VISION_MODEL],
            api_key=settings.MISTRAL_API_KEY,
            rate_limiter=_build_mistral_rate_limiter(),
            header_style="none",
        )


mistral_client = MistralClient()