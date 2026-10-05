"""
Shared base class for every LLM backend speaking the OpenAI /chat/completions schema. Thin subclasses just supply base_url/models/api_key.
"""

import httpx
import logging
from typing import Dict, List, Optional

from services.api.app.clients.llm.base import LLMClient
from libs.utils.backoff import exponential_backoff
from libs.utils.circuit_breaker import CircuitBreaker, CircuitOpenError
from libs.utils.rate_limiter import BackendRateLimiter, estimate_tokens

logger = logging.getLogger(__name__)


class ModelExhaustedError(Exception):
    """Every model in this backend's priority list failed or was already rate-limited."""


class OpenAICompatibleClient(LLMClient):
    """Tries each configured model in order, skipping rate-limited ones, and falls over to the next model on failure.

    A circuit breaker sits in front of the whole backend: after repeated exhaustion, calls fail
    fast for a cooldown window. A pin via chat_completion(model=X) is tried first but still
    falls through to the rest of the list on failure.
    """

    def __init__(
        self,
        base_url: str,
        models: List[str],
        api_key: Optional[str] = None,
        timeout: float = 120.0,
        rate_limiter: Optional[BackendRateLimiter] = None,
        header_style: str = "none",  # "groq" | "openrouter" | "none"
        dispatch_strategy: str = "priority",  # "priority" | "round_robin"
    ):
        self.base_url = base_url.rstrip("/")
        self.models = models
        self.api_key = api_key
        self.timeout = timeout
        self.client: Optional[httpx.AsyncClient] = None
        self._circuit = CircuitBreaker(failure_threshold=3, cooldown_seconds=30.0)
        self._rate_limiter = rate_limiter
        self._header_style = header_style
        self._dispatch_strategy = dispatch_strategy
        self._rr_index = 0
        self.last_model_used: str = ""
        self.last_finish_reason: str = ""

    async def start(self):
        """Opens the underlying HTTP client.

        Raises:
            RuntimeError: If no models are configured.
        """
        if not self.models:
            raise RuntimeError(f"{self.__class__.__name__}: no models configured")
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        limits = httpx.Limits(max_keepalive_connections=20, max_connections=50)
        self.client = httpx.AsyncClient(
            base_url=self.base_url, headers=headers, timeout=self.timeout, limits=limits
        )
        logger.info(f"{self.__class__.__name__} initialized ({len(self.models)} model(s) configured).")

    async def close(self):
        """Closes the underlying HTTP client."""
        if self.client:
            await self.client.aclose()
            logger.info(f"{self.__class__.__name__} closed.")

    async def chat_completion(
        self,
        messages: List[Dict],
        temperature: float = 0.3,
        json_mode: bool = False,
        model: Optional[str] = None,
        max_tokens: int = 1024,
    ) -> str:
        """Sends a chat completion request, trying models in order until one succeeds.

        Args:
            messages: Chat messages in OpenAI format.
            temperature: Sampling temperature.
            json_mode: Whether to request a structured JSON response.
            model: Optional pin to one specific model, tried first.
            max_tokens: Caps response length.

        Returns:
            The assistant's text response.

        Raises:
            ModelExhaustedError: If every model in the list is rate-limited or fails.
        """
        if not self.client:
            raise RuntimeError(f"{self.__class__.__name__} not started. Call start() first.")

        try:
            self._circuit.before_call()
        except CircuitOpenError as e:
            raise ModelExhaustedError(str(e)) from e

        if model is not None:
            models_to_try = [model] + [m for m in self.models if m != model]
        elif self._dispatch_strategy == "round_robin" and self.models:
            i = self._rr_index % len(self.models)
            models_to_try = self.models[i:] + self.models[:i]
            self._rr_index += 1
        else:
            models_to_try = self.models

        estimated_tokens = estimate_tokens(messages)
        last_error: Optional[Exception] = None
        any_attempted = False

        for model in models_to_try:
            tracker = self._rate_limiter.get(model) if self._rate_limiter else None

            if tracker is not None and not tracker.can_proceed(estimated_tokens):
                logger.info(f"{self.__class__.__name__}: skipping '{model}' — rate-limited.")
                continue

            any_attempted = True
            if tracker is not None:
                tracker.record_attempt(estimated_tokens)

            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if json_mode:
                payload["response_format"] = {"type": "json_object"}

            try:
                response = await self._post_with_retry(payload)
                self._record_headers(tracker, response.headers)
                choice = response.json()["choices"][0]
                content = choice["message"]["content"]

                if not content or not content.strip():
                    raise ValueError(f"model '{model}' returned empty content")

                self._circuit.record_success()
                self.last_model_used = model
                self.last_finish_reason = choice.get("finish_reason") or ""
                if self.last_finish_reason == "length":
                    logger.warning(f"{self.__class__.__name__}: '{model}' hit max_tokens={max_tokens} — answer truncated.")
                return content
            except (httpx.HTTPStatusError, httpx.TransportError) as e:
                last_error = e
                if isinstance(e, httpx.HTTPStatusError):
                    self._record_headers(tracker, e.response.headers)
                logger.warning(f"{self.__class__.__name__}: model '{model}' failed ({e}); trying next.")
                continue
            except (KeyError, IndexError, ValueError) as e:
                last_error = e
                logger.error(f"{self.__class__.__name__}: bad response from '{model}': {e}")
                continue

        self._circuit.record_failure()
        if not any_attempted:
            logger.warning(f"{self.__class__.__name__}: all {len(models_to_try)} model(s) skipped — rate-limited.")
        raise ModelExhaustedError(f"{self.__class__.__name__}: all {len(models_to_try)} model(s) exhausted") from last_error

    def _record_headers(self, tracker, headers) -> None:
        """Updates the rate-limit tracker from the provider's response headers, if any.

        Args:
            tracker: The ModelRateLimiter for the model just called, or None.
            headers: The response headers dict.
        """
        if tracker is None:
            return
        if self._header_style == "groq":
            tracker.record_groq_headers(headers)
        elif self._header_style == "openrouter":
            tracker.record_openrouter_headers(headers)

    @exponential_backoff(max_retries=2)
    async def _post_with_retry(self, payload: dict) -> httpx.Response:
        """Posts to /chat/completions, retrying on failure.

        Args:
            payload: The request body.

        Returns:
            The HTTP response.

        Raises:
            httpx.HTTPStatusError: On a 4xx/5xx response, with the response body in the message.
        """
        response = await self.client.post("/chat/completions", json=payload)
        if response.status_code >= 400:
            try:
                detail = response.json()
            except Exception:
                detail = response.text[:500]
            raise httpx.HTTPStatusError(
                f"{response.status_code} {response.reason_phrase} for url '{response.url}' — body: {detail}",
                request=response.request,
                response=response,
            )
        return response