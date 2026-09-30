"""
Wraps GemmaClient as ragas's InstructorBaseRagasLLM interface so ragas's
real metric implementations (ragas.metrics.collections.*) can call it.

ragas's InstructorBaseRagasLLM contract: implement
agenerate(prompt: str, response_model: Type[T]) -> T, returning a
validated instance of whatever pydantic model the calling metric asks
for. Each metric's own prompt (built internally via
`self.prompt.to_string(...)`) already embeds the full JSON-schema
instructions for that response_model directly in the prompt text (with
worked examples) — so all this wrapper does is send that prompt to
Gemma with json_mode=True and validate the JSON response against
response_model. No Instructor-library function-calling machinery is
needed at runtime; ragas doesn't care how a valid response_model
instance was produced, only that one comes back.
"""
import re
import typing as t
import asyncio 

from eval import _ragas_compat  # noqa: F401 — import-order fix, see that module
from ragas.llms.base import InstructorBaseRagasLLM

from eval.clients.gemma_client import gemma_client
from eval.clients.cohere_client import cohere_client

T = t.TypeVar("T")

_THOUGHT_BLOCK = re.compile(r"<thought>.*?</thought>", re.DOTALL)


def _extract_json(raw: str) -> str:
    """Gemma (gemma-4-31b-it via Google's OpenAI-compat endpoint) leaks a
    <thought>...</thought> reasoning preamble into the content string even
    with json_mode/response_format=json_object set — verified live (raw
    responses observed starting with a literal '<thought>' block before
    the actual JSON object, breaking model_validate_json at column 1).
    Strip that block, then fall back to slicing from the first '{' to the
    last '}' in whatever remains, so any stray preamble — tagged or not —
    never reaches model_validate_json as leading/trailing garbage."""
    cleaned = _THOUGHT_BLOCK.sub("", raw).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        return cleaned  # let model_validate_json raise its own clear error
    return cleaned[start : end + 1]


class _ChatClientInstructorLLM(InstructorBaseRagasLLM):
    def __init__(self, client, min_interval: float = 0.0):
        self._client = client
        self._min_interval = min_interval

    async def agenerate(self, prompt: str, response_model: t.Type[T]) -> T:
        await asyncio.sleep(self._min_interval)
        raw = await self._client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            json_mode=True,
            max_tokens=2048,
        )
        return response_model.model_validate_json(_extract_json(raw))

    def generate(self, prompt: str, response_model: t.Type[T]) -> T:
        raise NotImplementedError("async-only")


gemma_instructor_llm = _ChatClientInstructorLLM(gemma_client, min_interval=2.1)     # 30 RPM
cohere_instructor_llm = _ChatClientInstructorLLM(cohere_client, min_interval=3.1)   # 20 RPM