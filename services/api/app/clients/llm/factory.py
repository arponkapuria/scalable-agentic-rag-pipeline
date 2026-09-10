"""
Selects the active LLMClient from LLM_BACKEND.

LLM_BACKEND=api wraps Groq + OpenRouter as an auto-failover pair: primary
is set by API_PRIMARY, the other is automatic backup once the primary has
exhausted its own model-priority list (see openai_compatible.py). Ollama
and vLLM are manual-select only, per the locked design — they're never
wired into failover.
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
    """Wraps a primary + backup client. Falls over only after the primary
    has exhausted its own model-priority list, not on the first error."""

    def __init__(self, primary: LLMClient, backup: LLMClient):
        self.primary = primary
        self.backup = backup
        # Inspectable after a chat_completion() call, not part of the
        # LLMClient interface — adding a return-value/metadata contract
        # would ripple through every caller (planner, responder, graph
        # extractor, tool nodes, enhancers). Only responder.py currently
        # needs this (Phase 5's backend_used cache field), so an
        # attribute is the lower-blast-radius choice here.
        self.last_backend_used: str = ""

    async def start(self):
        await self.primary.start()
        await self.backup.start()

    async def close(self):
        await self.primary.close()
        await self.backup.close()

    async def chat_completion(self, messages, temperature=0.3, json_mode=False, model=None, max_tokens=1024) -> str:
        try:
            result = await self.primary.chat_completion(messages, temperature, json_mode, model, max_tokens)
            self.last_backend_used = self.primary.__class__.__name__
            return result
        except ModelExhaustedError as e:
            logger.warning(f"Primary backend exhausted, failing over to backup: {e}")
            # A `model` pin names a model in the PRIMARY's namespace (e.g.
            # a Groq model id from heuristic routing) — meaningless to a
            # different backend, so the backup falls back to its own
            # priority list rather than being handed an id it doesn't have.
            result = await self.backup.chat_completion(messages, temperature, json_mode, max_tokens=max_tokens)
            self.last_backend_used = self.backup.__class__.__name__
            return result


def build_llm_client() -> LLMClient:
    backend = settings.LLM_BACKEND

    if backend == "api":
        if not settings.ENABLE_OPENROUTER_FALLBACK:
            # Config-gated off, not deleted (see PROGRESS.md carry-over
            # notes) — Groq runs alone, no failover wrapper at all.
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

    raise ValueError(
        f"Unknown LLM_BACKEND: {backend!r} (expected api|ollama|vllm_local|vllm_modal)"
    )


# Global instance (managed by lifespan in main.py) — mirrors clients/ray_llm.py's pattern.
llm_client: LLMClient = build_llm_client()
