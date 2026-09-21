"""
Standalone Mistral vision client — used ONLY for PDF figure captioning
(pipelines/ingestion/loaders/docling_loader.py), never part of the
Groq/OpenRouter chat pool and never wired into FailoverLLMClient.
Captioning is a separate workload with its own free-tier budget; sharing
it with chat's failover pair would mean a captioning burst could burn quota
the chat path needs, and vice versa. No backup configured here — a caption
failure just drops that one figure (soft-fail), not a fallback to another
provider.

Why Mistral specifically: captions are inlined into the chunks the answer
is generated from AND that the evaluation judge later scores, so the
captioner should not share a vendor with either. Generator = OpenAI
(gpt-oss), judge = Google (Gemma), captioner = Mistral. Image input on the
free plan was confirmed with a real call (ministral-14b-2512).

Uses Mistral's OpenAI-compatible endpoint, so OpenAICompatibleClient is
reused as-is, including multimodal `image_url` content blocks.

Free-plan limits for ministral-14b-2512, read from the project owner's
Mistral console (Admin > Limits — re-verify there if behavior seems off):
0.5 requests/second, 937,500 tokens/minute, plus a $10/month included
API allowance. header_style="none": Mistral's rate-limit headers use
their own names (x-ratelimit-*-req-minute), so budget tracking here is
proactive/local-count only — same as OpenRouter.
"""
from services.api.app.clients.llm.openai_compatible import OpenAICompatibleClient
from services.api.app.config import settings
from libs.utils.rate_limiter import BackendRateLimiter

# 0.5 requests/second = 30/minute. The real limit is per-SECOND, which a
# per-minute window cannot express — the caller paces requests
# (CAPTION_MIN_INTERVAL_SECONDS in docling_loader.py) so bursts never hit it.
_MISTRAL_RPM = 30
_MISTRAL_TPM = 937_500


def _build_mistral_rate_limiter() -> BackendRateLimiter:
    limiter = BackendRateLimiter(shared=False)
    limiter.register(settings.MISTRAL_VISION_MODEL, rpm=_MISTRAL_RPM, tpm=_MISTRAL_TPM)
    return limiter


class MistralClient(OpenAICompatibleClient):
    """Single-model vision client for figure captioning."""

    def __init__(self):
        super().__init__(
            base_url="https://api.mistral.ai/v1",
            models=[settings.MISTRAL_VISION_MODEL],
            api_key=settings.MISTRAL_API_KEY,
            rate_limiter=_build_mistral_rate_limiter(),
            header_style="none",
        )


# Global instance — started/closed by main.py's lifespan alongside
# llm_client, since in-process ingestion (see pipelines/ingestion/
# pipeline.py) runs inside the same FastAPI process/event loop.
mistral_client = MistralClient()
