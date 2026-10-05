"""
Tests for session management: SessionStore (Redis sorted-set create, touch, TTL validity, expired-session purge) and the get_corpus_id FastAPI dependency (401 on a missing or invalid cookie, touch-and-return on a valid one). Redis is mocked.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from services.api.app.session.store import SessionStore, ACTIVE_SESSIONS_KEY
from services.api.app.session.dependency import get_corpus_id


def run(coro):
    return asyncio.run(coro)


def _mock_redis_client():
    """Patches session.store's redis_client.get_client() to return an
    AsyncMock standing in for redis.asyncio's client."""
    fake_redis = AsyncMock()
    patcher = patch(
        "services.api.app.session.store.redis_client.get_client",
        return_value=fake_redis,
    )
    patcher.start()
    return fake_redis, patcher


# --- SessionStore ---

def test_create_session_zadds_with_current_score():
    fake_redis, patcher = _mock_redis_client()
    try:
        store = SessionStore()
        corpus_id = run(store.create_session())

        assert corpus_id  # non-empty uuid string
        fake_redis.zadd.assert_awaited_once()
        args, kwargs = fake_redis.zadd.call_args
        assert args[0] == ACTIVE_SESSIONS_KEY
        assert corpus_id in args[1]
    finally:
        patcher.stop()


def test_touch_uses_xx_so_unknown_ids_are_not_resurrected():
    fake_redis, patcher = _mock_redis_client()
    try:
        store = SessionStore()
        run(store.touch("some-corpus-id"))

        _, kwargs = fake_redis.zadd.call_args
        assert kwargs.get("xx") is True
    finally:
        patcher.stop()


def test_is_valid_true_within_ttl():
    fake_redis, patcher = _mock_redis_client()
    try:
        fake_redis.zscore.return_value = time.time() - 30  # 30s ago
        store = SessionStore()

        assert run(store.is_valid("corpus-1", ttl_minutes=1)) is True
    finally:
        patcher.stop()


def test_is_valid_false_when_expired():
    fake_redis, patcher = _mock_redis_client()
    try:
        fake_redis.zscore.return_value = time.time() - 120  # 2 min ago
        store = SessionStore()

        assert run(store.is_valid("corpus-1", ttl_minutes=1)) is False
    finally:
        patcher.stop()


def test_is_valid_false_when_unknown():
    fake_redis, patcher = _mock_redis_client()
    try:
        fake_redis.zscore.return_value = None
        store = SessionStore()

        assert run(store.is_valid("nonexistent", ttl_minutes=60)) is False
    finally:
        patcher.stop()


def test_purge_expired_removes_and_returns_stale_ids():
    fake_redis, patcher = _mock_redis_client()
    try:
        fake_redis.zrangebyscore.return_value = ["stale-1", "stale-2"]
        store = SessionStore()

        purged = run(store.purge_expired(ttl_minutes=60))

        assert purged == ["stale-1", "stale-2"]
        fake_redis.zrem.assert_awaited_once_with(
            ACTIVE_SESSIONS_KEY, "stale-1", "stale-2"
        )
    finally:
        patcher.stop()


def test_purge_expired_skips_zrem_when_nothing_stale():
    fake_redis, patcher = _mock_redis_client()
    try:
        fake_redis.zrangebyscore.return_value = []
        store = SessionStore()

        purged = run(store.purge_expired(ttl_minutes=60))

        assert purged == []
        fake_redis.zrem.assert_not_awaited()
    finally:
        patcher.stop()


# --- get_corpus_id dependency ---

def _mock_request(cookie_value):
    request = MagicMock()
    request.cookies = {"corpus_id": cookie_value} if cookie_value else {}
    return request


def test_get_corpus_id_missing_cookie_raises_401():
    request = _mock_request(None)

    with pytest.raises(HTTPException) as exc_info:
        run(get_corpus_id(request))
    assert exc_info.value.status_code == 401


def test_get_corpus_id_invalid_cookie_raises_401():
    request = _mock_request("expired-corpus")
    with patch(
        "services.api.app.session.dependency.session_store.is_valid",
        new=AsyncMock(return_value=False),
    ):
        with pytest.raises(HTTPException) as exc_info:
            run(get_corpus_id(request))
        assert exc_info.value.status_code == 401


def test_get_corpus_id_valid_cookie_touches_and_returns_it():
    request = _mock_request("live-corpus")
    with patch(
        "services.api.app.session.dependency.session_store.is_valid",
        new=AsyncMock(return_value=True),
    ), patch(
        "services.api.app.session.dependency.session_store.touch",
        new=AsyncMock(),
    ) as mock_touch:
        result = run(get_corpus_id(request))

        assert result == "live-corpus"
        mock_touch.assert_awaited_once_with("live-corpus")