"""
Standalone Gemini client — used ONLY for PDF figure captioning
(opendataloader_pdf.py), never part of the Groq/OpenRouter chat pool and
never wired into FailoverLLMClient. Captioning is a separate workload
with its own free-tier budget; sharing it with chat's failover pair would
mean a captioning burst could burn quota the chat path needs, and vice
versa. No backup configured here — a caption failure just drops that one
figure (soft-fail, same as before), not a fallback to another provider.

Uses Google's OpenAI-compatible endpoint (generativelanguage.googleapis.com
/v1beta/openai) so it can reuse OpenAICompatibleClient as-is — Gemini
speaks the same request/response shape as everything else here, including
multimodal `image_url` content blocks.

Free-tier limits (Gemini 3.5 Flash Lite, as told directly by the person
running this project — re-verify at ai.google.dev/gemini-api/docs/rate-limits
if behavior seems off): ~15 RPM, ~1,500 RPD, ~1M TPM. header_style="none":
unlike Groq, Gemini's OpenAI-compat layer doesn't return live rate-limit
headers, so budget tracking here is proactive/local-count-only — same
limitation OpenRouter has.
"""
from services.api.app.clients.llm.openai_compatible import OpenAICompatibleClient
from services.api.app.config import settings
from libs.utils.rate_limiter import BackendRateLimiter


def _build_gemini_rate_limiter() -> BackendRateLimiter:
    limiter = BackendRateLimiter(shared=False)
    limiter.register(settings.GEMINI_VISION_MODEL, rpm=15, rpd=1500, tpm=1_000_000)
    return limiter


class GeminiClient(OpenAICompatibleClient):
    """Single-model vision client for figure captioning."""

    def __init__(self):
        super().__init__(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai",
            models=[settings.GEMINI_VISION_MODEL],
            api_key=settings.GEMINI_API_KEY,
            rate_limiter=_build_gemini_rate_limiter(),
            header_style="none",
        )


# Global instance — started/closed by main.py's lifespan alongside
# llm_client, since in-process ingestion (see pipelines/ingestion/
# pipeline.py) now runs inside the same FastAPI process/event loop.
gemini_client = GeminiClient()
