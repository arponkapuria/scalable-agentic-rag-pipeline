import asyncio
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from services.api.app.config import settings
from services.api.app.clients.llm.openai_compatible import (
    OpenAICompatibleClient,
    ModelExhaustedError,
)
from services.api.app.clients.llm.factory import (
    FailoverLLMClient,
    build_llm_client,
)
from services.api.app.clients.llm.groq_client import GroqClient
from services.api.app.clients.llm.openrouter_client import OpenRouterClient
from services.api.app.clients.llm.ollama_client import OllamaClient
from services.api.app.clients.llm.vllm_client import VLLMClient


def run(coro):
    return asyncio.run(coro)


def _mock_response(content: str) -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.raise_for_status = MagicMock()
    resp.headers = {}
    resp.json.return_value = {"choices": [{"message": {"content": content}}]}
    return resp


def _http_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://example.com/chat/completions")
    response = httpx.Response(status, request=request)
    return httpx.HTTPStatusError("boom", request=request, response=response)


# --- Config: comma-separated string -> priority-ordered list ---

def test_settings_splits_model_lists_in_priority_order():
    assert settings.GROQ_MODELS == ["model-a", "model-b"]
    assert settings.OPENROUTER_MODELS == ["or-model-a", "or-model-b"]


# --- Model-priority fallback within a single backend ---

def test_tries_next_model_on_retryable_failure():
    client = OpenAICompatibleClient(
        base_url="https://example.com", models=["m1", "m2"], api_key="k"
    )
    run(client.start())
    client._post_with_retry = AsyncMock(
        side_effect=[_http_error(429), _mock_response("answer from m2")]
    )

    result = run(client.chat_completion(messages=[{"role": "user", "content": "hi"}]))

    assert result == "answer from m2"
    assert client._post_with_retry.call_count == 2
    # first attempt used m1, second used m2 — priority order respected
    assert client._post_with_retry.call_args_list[0].args[0]["model"] == "m1"
    assert client._post_with_retry.call_args_list[1].args[0]["model"] == "m2"


def test_first_model_succeeds_second_never_tried():
    client = OpenAICompatibleClient(
        base_url="https://example.com", models=["m1", "m2"], api_key="k"
    )
    run(client.start())
    client._post_with_retry = AsyncMock(return_value=_mock_response("answer from m1"))

    result = run(client.chat_completion(messages=[{"role": "user", "content": "hi"}]))

    assert result == "answer from m1"
    assert client._post_with_retry.call_count == 1


def test_all_models_exhausted_raises():
    client = OpenAICompatibleClient(
        base_url="https://example.com", models=["m1", "m2"], api_key="k"
    )
    run(client.start())
    client._post_with_retry = AsyncMock(side_effect=_http_error(500))

    with pytest.raises(ModelExhaustedError):
        run(client.chat_completion(messages=[{"role": "user", "content": "hi"}]))

    assert client._post_with_retry.call_count == 2


def test_start_raises_if_no_models_configured():
    client = OpenAICompatibleClient(base_url="https://example.com", models=[], api_key="k")
    with pytest.raises(RuntimeError):
        run(client.start())


# --- Groq <-> OpenRouter auto-failover (LLM_BACKEND=api) ---

def test_failover_falls_back_to_backup_after_primary_exhausted():
    primary = AsyncMock()
    backup = AsyncMock()
    primary.chat_completion.side_effect = ModelExhaustedError("primary dead")
    backup.chat_completion.return_value = "answer from backup"

    client = FailoverLLMClient(primary=primary, backup=backup)
    messages = [{"role": "user", "content": "hi"}]

    result = run(client.chat_completion(messages))

    assert result == "answer from backup"
    primary.chat_completion.assert_awaited_once()
    backup.chat_completion.assert_awaited_once()


def test_failover_does_not_call_backup_when_primary_succeeds():
    primary = AsyncMock()
    backup = AsyncMock()
    primary.chat_completion.return_value = "answer from primary"

    client = FailoverLLMClient(primary=primary, backup=backup)

    result = run(client.chat_completion([{"role": "user", "content": "hi"}]))

    assert result == "answer from primary"
    backup.chat_completion.assert_not_awaited()


# --- Backend selection from LLM_BACKEND / API_PRIMARY ---

def test_build_llm_client_api_backend_groq_primary(monkeypatch):
    monkeypatch.setattr(settings, "LLM_BACKEND", "api")
    monkeypatch.setattr(settings, "API_PRIMARY", "groq")
    monkeypatch.setattr(settings, "ENABLE_OPENROUTER_FALLBACK", True)

    client = build_llm_client()

    assert isinstance(client, FailoverLLMClient)
    assert isinstance(client.primary, GroqClient)
    assert isinstance(client.backup, OpenRouterClient)


def test_build_llm_client_api_backend_openrouter_primary(monkeypatch):
    monkeypatch.setattr(settings, "LLM_BACKEND", "api")
    monkeypatch.setattr(settings, "API_PRIMARY", "openrouter")
    monkeypatch.setattr(settings, "ENABLE_OPENROUTER_FALLBACK", True)

    client = build_llm_client()

    assert isinstance(client.primary, OpenRouterClient)
    assert isinstance(client.backup, GroqClient)


def test_build_llm_client_ollama_backend(monkeypatch):
    monkeypatch.setattr(settings, "LLM_BACKEND", "ollama")

    client = build_llm_client()

    assert isinstance(client, OllamaClient)


def test_build_llm_client_vllm_local_backend(monkeypatch):
    monkeypatch.setattr(settings, "LLM_BACKEND", "vllm_local")

    client = build_llm_client()

    assert isinstance(client, VLLMClient)
    assert client.variant == "metal"


def test_build_llm_client_vllm_modal_backend(monkeypatch):
    monkeypatch.setattr(settings, "LLM_BACKEND", "vllm_modal")
    monkeypatch.setattr(settings, "VLLM_MODAL_URL", "https://example.modal.run")
    monkeypatch.setattr(settings, "VLLM_MODAL_MODELS", ["m1"])

    client = build_llm_client()

    assert isinstance(client, VLLMClient)
    assert client.variant == "modal"


def test_build_llm_client_vllm_modal_requires_url(monkeypatch):
    monkeypatch.setattr(settings, "LLM_BACKEND", "vllm_modal")
    monkeypatch.setattr(settings, "VLLM_MODAL_URL", None)

    with pytest.raises(RuntimeError):
        build_llm_client()


def test_build_llm_client_unknown_backend_raises(monkeypatch):
    monkeypatch.setattr(settings, "LLM_BACKEND", "nonsense")

    with pytest.raises(ValueError):
        build_llm_client()
