"""
Common interface every LLM backend (Groq, OpenRouter, Ollama, vLLM) implements.
Callers (agent nodes, factory) depend on this, never on a concrete backend —
that's what makes LLM_BACKEND a config swap instead of a code change.
"""

from abc import ABC, abstractmethod
from typing import Dict, List


class LLMClient(ABC):
    @abstractmethod
    async def start(self) -> None:
        """Called once during app startup (see main.py lifespan)."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Called once during app shutdown."""
        ...

    @abstractmethod
    async def chat_completion(
        self,
        messages: List[Dict],
        temperature: float = 0.3,
        json_mode: bool = False,
        model: str = None,
        max_tokens: int = 1024,
    ) -> str:
        """
        Returns the assistant's text response.
        Callers order `messages` with stable content (system prompt, tool
        schema) first and variable content (retrieved context, question)
        last — this earns Groq's automatic prompt caching for free and
        costs nothing on backends that don't support it.

        `model`: optional pin to one specific model, bypassing the
        backend's own priority-list walk entirely (used by heuristic
        model-tier routing — see model_router.py). Ignored by backends
        with only one configured model.

        `max_tokens`: caps response length. Callers with a known-short
        expected output (query rewriting, classification) should pass a
        small value — it's a hard backstop against a model ignoring its
        prompt instructions and free-writing a long answer where a short
        one was expected (observed with query rewriting: a model returned
        a full essay instead of a rewritten search query).
        """
        ...
