"""
Judge-only Gemma client — Google's OpenAI-compatible
endpoint, same OpenAICompatibleClient base every other backend
(Groq/OpenRouter/Ollama/vLLM/Mistral) already uses. Lives under eval/, not
services/api/app/clients/llm/ alongside the production clients: this is
never part of the chat app's dependency graph or lifespan — only eval
scripts start/stop it. Same isolation reasoning as MistralClient
(captioning): the judge must not share a vendor with either the generator
(Groq) or the captioner (Mistral), and must not compete with either for
its own rate-limit budget.

Limits from the project owner's Google AI Studio console (gemma-4-31b-it):
30 RPM / 16K TPM / 14,400 RPD — comfortably covers the ~275-call judge
budget EVALUATION_DESIGN.md sizes this run at (finishes in minutes, not
hours). Re-verify at aistudio.google.com if behavior looks off, same
caveat as every other free-tier client in this project.
"""
from services.api.app.clients.llm.openai_compatible import OpenAICompatibleClient
from services.api.app.config import settings
from libs.utils.rate_limiter import BackendRateLimiter

_GEMMA_RPM = 30
_GEMMA_TPM = 16_000
_GEMMA_RPD = 14_400


def _build_gemma_rate_limiter() -> BackendRateLimiter:
    limiter = BackendRateLimiter(shared=False)
    limiter.register(settings.GEMMA_JUDGE_MODEL, rpm=_GEMMA_RPM, tpm=_GEMMA_TPM, rpd=_GEMMA_RPD)
    return limiter


class GemmaClient(OpenAICompatibleClient):
    """Single-model judge client — no failover, no fallback backend. A
    judge call failing just leaves that metric null for the question
    (see run_judge.py) rather than silently falling over to a different
    judge model, which would make scores incomparable across questions."""

    def __init__(self):
        super().__init__(
            base_url=settings.GEMMA_BASE_URL,
            models=[settings.GEMMA_JUDGE_MODEL],
            api_key=settings.GOOGLE_API_KEY,
            rate_limiter=_build_gemma_rate_limiter(),
            header_style="none",
        )


# Global instance — started/closed by whichever eval script needs it
# (run_judge.py, judge_check.py), same lifecycle pattern as the app's own
# llm_client/mistral_client globals, just not wired into main.py's
# lifespan since this never runs inside the live app process.
gemma_client = GemmaClient()