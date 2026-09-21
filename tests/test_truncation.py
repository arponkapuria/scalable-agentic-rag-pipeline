import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx

from libs.utils.refusal import TRUNCATION_NOTE, is_truncated
from services.api.app.agents.nodes import responder
from services.api.app.clients.llm.factory import FailoverLLMClient
from services.api.app.clients.llm.openai_compatible import OpenAICompatibleClient


def _response(finish_reason, content="partial answer") -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.raise_for_status = MagicMock()
    resp.headers = {}
    resp.json.return_value = {"choices": [{"message": {"content": content}, "finish_reason": finish_reason}]}
    return resp


def _complete(finish_reason):
    client = OpenAICompatibleClient(base_url="https://example.com", models=["m1"], api_key="k")
    asyncio.run(client.start())
    client._post_with_retry = AsyncMock(return_value=_response(finish_reason))
    asyncio.run(client.chat_completion(messages=[{"role": "user", "content": "q"}], max_tokens=64))
    return client


def test_client_records_length_and_warns(caplog):
    with caplog.at_level(logging.WARNING):
        client = _complete("length")
    assert client.last_finish_reason == "length"
    assert "hit max_tokens=64" in caplog.text


def test_client_records_normal_stop_without_warning(caplog):
    with caplog.at_level(logging.WARNING):
        client = _complete("stop")
    assert client.last_finish_reason == "stop"
    assert "hit max_tokens" not in caplog.text


def test_missing_finish_reason_is_tolerated():
    assert _complete(None).last_finish_reason == ""


def test_failover_wrapper_mirrors_finish_reason():
    primary = MagicMock(last_finish_reason="length")
    primary.chat_completion = AsyncMock(return_value="x")
    wrapper = FailoverLLMClient(primary, MagicMock())
    asyncio.run(wrapper.chat_completion([{"role": "user", "content": "q"}]))
    assert wrapper.last_finish_reason == "length"


def test_responder_flags_only_truncated_answers(monkeypatch):
    monkeypatch.setattr(responder, "llm_client", SimpleNamespace(last_finish_reason="length"))
    flagged = responder._flag_truncation("cut of")
    assert flagged == "cut of" + TRUNCATION_NOTE and is_truncated(flagged)

    monkeypatch.setattr(responder, "llm_client", SimpleNamespace(last_finish_reason="stop"))
    assert responder._flag_truncation("complete.") == "complete."
    assert not is_truncated("complete.")


def test_responder_passes_the_configured_answer_cap(monkeypatch):
    llm = SimpleNamespace(last_finish_reason="stop", last_model_used="m", last_backend_used="b",
                          chat_completion=AsyncMock(return_value="a complete answer"))
    monkeypatch.setattr(responder, "llm_client", llm)
    monkeypatch.setattr(responder.settings, "ANSWER_MAX_TOKENS", 1234)
    asyncio.run(responder._answer_from_documents("q?", ["some context"], False))
    assert llm.chat_completion.call_args.kwargs["max_tokens"] == 1234
