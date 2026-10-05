"""Adapter that lets ragas use the Gemma and Cohere chat clients as judge LLMs."""
import re
import typing as t
import asyncio 

from eval import _ragas_compat  # noqa: F401 — must load before ragas
from ragas.llms.base import InstructorBaseRagasLLM

from eval.clients.gemma_client import gemma_client
from eval.clients.cohere_client import cohere_client

T = t.TypeVar("T")

_THOUGHT_BLOCK = re.compile(r"<thought>.*?</thought>", re.DOTALL)


def _extract_json(raw: str) -> str:
    """Pulls the JSON object out of a model reply.

    Gemma sometimes prepends a <thought> block even in JSON mode, so this strips it and keeps everything from the first "{" to the last "}".
    """
    cleaned = _THOUGHT_BLOCK.sub("", raw).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        return cleaned  # No JSON found, so let validation raise a clear error.
    return cleaned[start : end + 1]


class _ChatClientInstructorLLM(InstructorBaseRagasLLM):
    """Wraps a chat client as a ragas judge that returns validated pydantic objects."""

    def __init__(self, client, min_interval: float = 0.0):
        """Stores the chat client and the minimum delay in seconds before each call, used to stay under rate limits."""
        self._client = client
        self._min_interval = min_interval

    async def agenerate(self, prompt: str, response_model: t.Type[T]) -> T:
        """Sends the prompt in JSON mode and parses the reply into response_model.

        Raises:
            pydantic.ValidationError: If the reply doesn't match the model's schema.
        """
        await asyncio.sleep(self._min_interval)
        raw = await self._client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            json_mode=True,
            max_tokens=2048,
        )
        return response_model.model_validate_json(_extract_json(raw))

    def generate(self, prompt: str, response_model: t.Type[T]) -> T:
        """Not supported, use agenerate instead."""
        raise NotImplementedError("async-only")


gemma_instructor_llm = _ChatClientInstructorLLM(gemma_client, min_interval=2.1)     # 30 RPM
cohere_instructor_llm = _ChatClientInstructorLLM(cohere_client, min_interval=3.1)   # 20 RPM