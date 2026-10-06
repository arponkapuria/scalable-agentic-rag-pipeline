"""
Tests for inbound rate limiting: fixed-window counting, per-subject isolation, window rollover, Retry-After, fail-open on Redis errors, trusted-proxy IP resolution, and the upload size cap (declared and stored).
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from redis.exceptions import RedisError

from services.api.app.config import settings
from services.api.app.session import rate_limit
from services.api.app.session.rate_limit import _enforce, client_ip


class FakeRedis:
    """Minimal in-memory stand-in for the INCR/EXPIRE pipeline the limiter uses."""

    def __init__(self):
        self.counts = {}

    def pipeline(self, transaction=True):
        return FakePipeline(self)


class FakePipeline:
    def __init__(self, redis):
        self.redis, self.ops = redis, []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def incr(self, key):
        self.ops.append(("incr", key))

    def expire(self, key, seconds):
        self.ops.append(("expire", key))

    async def execute(self):
        results = []
        for op, key in self.ops:
            if op == "incr":
                self.redis.counts[key] = self.redis.counts.get(key, 0) + 1
                results.append(self.redis.counts[key])
            else:
                results.append(True)
        return results


@pytest.fixture
def fake_redis(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(rate_limit.redis_client, "get_client", lambda: fake)
    return fake


def _run(coro):
    return asyncio.run(coro)


def test_allows_up_to_limit_then_429_with_retry_after(fake_redis):
    for _ in range(3):
        _run(_enforce("chat:ip", "1.1.1.1", 3, 60))
    with pytest.raises(HTTPException) as exc:
        _run(_enforce("chat:ip", "1.1.1.1", 3, 60))
    assert exc.value.status_code == 429
    assert 1 <= int(exc.value.headers["Retry-After"]) <= 60


def test_subjects_are_isolated(fake_redis):
    for _ in range(2):
        _run(_enforce("chat:ip", "1.1.1.1", 2, 60))
    _run(_enforce("chat:ip", "2.2.2.2", 2, 60))  # different IP, own counter
    _run(_enforce("chat:session", "1.1.1.1", 2, 60))  # same subject string, different scope


def test_window_rollover_resets_counter(fake_redis, monkeypatch):
    monkeypatch.setattr(rate_limit.time, "time", lambda: 1000.0)
    for _ in range(2):
        _run(_enforce("chat:ip", "1.1.1.1", 2, 60))
    monkeypatch.setattr(rate_limit.time, "time", lambda: 1000.0 + 60)
    _run(_enforce("chat:ip", "1.1.1.1", 2, 60))


def test_fails_open_when_redis_errors(monkeypatch):
    def boom():
        raise RedisError("down")

    monkeypatch.setattr(rate_limit.redis_client, "get_client", boom)
    _run(_enforce("chat:ip", "1.1.1.1", 1, 60))


def _request(xff=None, host="9.9.9.9"):
    headers = {"x-forwarded-for": xff} if xff else {}
    return SimpleNamespace(headers=headers, client=SimpleNamespace(host=host))


def test_client_ip_ignores_xff_unless_trusted(monkeypatch):
    monkeypatch.setattr(settings, "TRUSTED_PROXY", False)
    assert client_ip(_request(xff="6.6.6.6")) == "9.9.9.9"
    monkeypatch.setattr(settings, "TRUSTED_PROXY", True)
    assert client_ip(_request(xff="6.6.6.6, 7.7.7.7")) == "7.7.7.7"  # rightmost = proxy-appended
    assert client_ip(_request()) == "9.9.9.9"


def test_upload_rejects_declared_oversize(monkeypatch):
    from services.api.app.routes.upload import PresignedURLRequest, generate_upload_url

    monkeypatch.setattr(settings, "FILE_UPLOAD_MAX_SIZE_MB", 1)
    req = PresignedURLRequest(filename="a.pdf", content_type="application/pdf", file_size=2 * 1024 * 1024)
    with pytest.raises(HTTPException) as exc:
        _run(generate_upload_url(req, corpus_id="c1"))
    assert exc.value.status_code == 413


def test_ingestion_deletes_and_rejects_stored_oversize(monkeypatch):
    from pipelines.ingestion import pipeline

    monkeypatch.setattr(settings, "FILE_UPLOAD_MAX_SIZE_MB", 1)
    s3 = MagicMock()
    s3.head_object.return_value = {"ContentLength": 2 * 1024 * 1024}
    monkeypatch.setattr(pipeline, "get_s3_client", lambda: s3)
    with pytest.raises(pipeline.FileTooLargeError):
        _run(pipeline._fetch_object("bucket", "uploads/c1/f1.pdf"))
    s3.delete_object.assert_called_once()
    s3.get_object.assert_not_called()