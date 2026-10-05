"""
Selects and builds the active LLMClient from config. For the API backend, wraps Groq and OpenRouter as an auto-failover pair; Ollama and vLLM are manual-select only.
"""

import logging

from services.api.app.config import settings
from services.api.app.clients.llm.base import LLMClient
from services.api.app.clients.llm.groq_client import GroqClient
from services.api.app.clients.llm.openrouter_client import OpenRouterClient
from services.api.app.clients.llm.ollama_client import OllamaClient
from services.api.app.clients.llm.vllm_client import VLLMClient
from services.api.app.clients.llm.openai_compatible import ModelExhaustedError

logger = logging.getLogger(__name__)


class FailoverLLMClient(LLMClient):
    """Wraps a primary and backup LLMClient, failing over once the primary's model list is fully exhausted."""

    def __init__(self, primary: LLMClient, backup: LLMClient):
        self.primary = primary
        self.backup = backup
        self.last_backend_used: str = ""
        self.last_model_used: str = ""
        self.last_finish_reason: str = ""

    async def start(self):
        """Starts both the primary and backup clients."""
        await self.primary.start()
        await self.backup.start()

    async def close(self):
        """Closes both the primary and backup clients."""
        await self.primary.close()
        await self.backup.close()

    async def chat_completion(self, messages, temperature=0.3, json_mode=False, model=None, max_tokens=1024) -> str:
        """Tries the primary client, failing over to the backup if the primary is fully exhausted.

        Args:
            messages: Chat messages in OpenAI format.
            temperature: Sampling temperature.
            json_mode: Whether to request a structured JSON response.
            model: Optional model pin, in the primary backend's own namespace.
            max_tokens: Caps response length.

        Returns:
            The assistant's text response, from whichever client succeeded.
        """
        try:
            result = await self.primary.chat_completion(messages, temperature, json_mode, model, max_tokens)
            self.last_backend_used = self.primary.__class__.__name__
            self.last_model_used = getattr(self.primary, "last_model_used", "")
            self.last_finish_reason = getattr(self.primary, "last_finish_reason", "")
            return result
        except ModelExhaustedError as e:
            logger.warning(f"Primary backend exhausted, failing over to backup: {e}")
            result = await self.backup.chat_completion(messages, temperature, json_mode, max_tokens=max_tokens)
            self.last_backend_used = self.backup.__class__.__name__
            self.last_model_used = getattr(self.backup, "last_model_used", "")
            self.last_finish_reason = getattr(self.backup, "last_finish_reason", "")
            return result


def build_llm_client() -> LLMClient:
    """Builds the active LLMClient based on settings.LLM_BACKEND.

    Returns:
        A GroqClient, OpenRouterClient, FailoverLLMClient, OllamaClient, or VLLMClient.

    Raises:
        ValueError: If LLM_BACKEND is not a recognized value.
    """
    backend = settings.LLM_BACKEND

    if backend == "api":
        if not settings.ENABLE_OPENROUTER_FALLBACK:
            return GroqClient() if settings.API_PRIMARY == "groq" else OpenRouterClient()
        groq, openrouter = GroqClient(), OpenRouterClient()
        primary, backup = (groq, openrouter) if settings.API_PRIMARY == "groq" else (openrouter, groq)
        return FailoverLLMClient(primary, backup)
    if backend == "ollama":
        return OllamaClient()
    if backend == "vllm_local":
        return VLLMClient(variant="metal")
    if backend == "vllm_modal":
        return VLLMClient(variant="modal")

    raise ValueError(f"Unknown LLM_BACKEND: {backend!r} (expected api|ollama|vllm_local|vllm_modal)")


llm_client: LLMClient = build_llm_client()