"""
Local Ollama backend — offline, local development only. Manual-select only (LLM_BACKEND=ollama); never an auto-failover target.
"""
from services.api.app.clients.llm.openai_compatible import OpenAICompatibleClient
from services.api.app.config import settings


class OllamaClient(OpenAICompatibleClient):
    """Local Ollama server, reached via its OpenAI-compatible endpoint."""

    def __init__(self):
        super().__init__(
            base_url=f"{settings.OLLAMA_BASE_URL.rstrip('/')}/v1",
            models=settings.OLLAMA_MODELS,
            api_key=None,
        )