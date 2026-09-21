from services.api.app.clients.llm.mistral_client import MistralClient, _build_mistral_rate_limiter
from services.api.app.config import settings


def test_limiter_matches_mistral_free_plan_limits():
    tracker = _build_mistral_rate_limiter().get(settings.MISTRAL_VISION_MODEL)
    assert tracker.requests_minute.limit == 30       # 0.5 requests/second
    assert tracker.tokens_minute.limit == 937_500
    assert tracker.requests_day is None              # Mistral's free plan is dollar-metered, not per-day


def test_client_targets_mistral_with_the_configured_vision_model():
    client = MistralClient()
    assert client.base_url == "https://api.mistral.ai/v1"
    assert client.models == [settings.MISTRAL_VISION_MODEL]
    assert settings.MISTRAL_VISION_MODEL == "ministral-14b-2512"
