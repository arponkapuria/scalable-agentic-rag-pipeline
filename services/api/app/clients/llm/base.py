"""
Common interface every LLM backend implements. Callers depend on this, never on a concrete backend.
"""

from abc import ABC, abstractmethod
from typing import Dict, List


class LLMClient(ABC):
    @abstractmethod
    async def start(self) -> None:
        """Opens the client's connection. Called once at app startup."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Closes the client's connection. Called once at app shutdown."""
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
        """Sends a chat completion request and returns the assistant's text response.

        Args:
            messages: Chat messages in OpenAI format. Stable content (system prompt) first,
                variable content (context, question) last — enables prompt caching.
            temperature: Sampling temperature.
            json_mode: Whether to request a structured JSON response.
            model: Optional pin to one specific model, bypassing priority-list selection.
            max_tokens: Caps response length.

        Returns:
            The assistant's text response.
        """
        ...