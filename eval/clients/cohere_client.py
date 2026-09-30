from services.api.app.clients.llm.openai_compatible import OpenAICompatibleClient
from services.api.app.config import settings
from libs.utils.rate_limiter import BackendRateLimiter

_COHERE_RPM = 20
_COHERE_RPD = 1000  # trial cap is 1,000/MONTH — this only approximates it per-run


def _build_cohere_rate_limiter() -> BackendRateLimiter:
    limiter = BackendRateLimiter(shared=False)
    limiter.register(settings.COHERE_JUDGE_MODEL, rpm=_COHERE_RPM, rpd=_COHERE_RPD)
    return limiter


class CohereClient(OpenAICompatibleClient):
    def __init__(self):
        super().__init__(
            base_url=settings.COHERE_BASE_URL,
            models=[settings.COHERE_JUDGE_MODEL],
            api_key=settings.COHERE_API_KEY,
            rate_limiter=_build_cohere_rate_limiter(),
            header_style="none",
        )


cohere_client = CohereClient()