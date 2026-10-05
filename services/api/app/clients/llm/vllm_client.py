"""
vLLM backend client — a locally-hosted or cloud-GPU-hosted model server. Manual-select only; cold starts make it unsuitable for auto-failover.
"""
from services.api.app.clients.llm.openai_compatible import OpenAICompatibleClient
from services.api.app.config import settings


class VLLMClient(OpenAICompatibleClient):
    """vLLM's OpenAI-compatible server. "metal" runs locally on Apple Silicon; "modal" runs on Modal-hosted GPU infrastructure."""

    def __init__(self, variant: str = "metal"):
        """Args:
            variant: "metal" for a local vLLM server, "modal" for a Modal-hosted one.

        Raises:
            RuntimeError: If variant is "modal" but VLLM_MODAL_URL isn't configured.
            ValueError: If variant is neither "metal" nor "modal".
        """
        if variant == "metal":
            base_url = settings.VLLM_METAL_URL
            models = settings.VLLM_METAL_MODELS
            api_key = None
        elif variant == "modal":
            if not settings.VLLM_MODAL_URL:
                raise RuntimeError("VLLM_MODAL_URL is not set — required for LLM_BACKEND=vllm_modal")
            base_url = settings.VLLM_MODAL_URL
            models = settings.VLLM_MODAL_MODELS
            api_key = settings.VLLM_MODAL_API_KEY
        else:
            raise ValueError(f"Unknown vLLM variant: {variant!r} (expected 'metal' or 'modal')")

        super().__init__(base_url=base_url, models=models, api_key=api_key)
        self.variant = variant