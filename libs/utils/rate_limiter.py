"""
Client-side rate limiter for LLM API calls, checked before a call is attempted rather than relying on an HTTP 429 response. Tracks usage locally per time window (requests and tokens, per minute and per day), and switches to a provider's own live rate-limit headers when they're available, since those are more accurate than a local estimate.
"""
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class _Window:
    """Tracks usage within one rolling time window, e.g. "requests per minute"."""
    limit: Optional[int]
    period_seconds: float
    count: int = 0
    window_start: float = field(default_factory=time.monotonic)
    # Live, provider-reported values — None until a real header arrives, then preferred over the local estimate until it expires.
    live_remaining: Optional[int] = None
    live_reset_at: Optional[float] = None

    def _maybe_roll(self):
        now = time.monotonic()
        if now - self.window_start >= self.period_seconds:
            self.window_start = now
            self.count = 0

    def remaining(self) -> Optional[int]:
        """Returns remaining quota for this window — a live provider value if fresh, else the local estimate.

        Returns:
            None if this window has no configured limit, else the remaining count.
        """
        if self.limit is None:
            return None  # no cap tracked for this window
        now = time.monotonic()
        if self.live_remaining is not None and self.live_reset_at and now < self.live_reset_at:
            return self.live_remaining
        self._maybe_roll()
        return max(0, self.limit - self.count)

    def record_attempt(self, amount: int = 1):
        """Records local usage against this window and the live estimate, if one exists.

        Args:
            amount: Units consumed — 1 for a request window, token count for a token window.
        """
        self._maybe_roll()
        self.count += amount
        if self.live_remaining is not None:
            self.live_remaining = max(0, self.live_remaining - amount)

    def record_live(self, remaining: int, reset_in_seconds: float):
        """Overwrites the local estimate with a provider-reported live value.

        Args:
            remaining: Remaining quota per the provider's own response headers.
            reset_in_seconds: Seconds until this window resets, per the provider.
        """
        self.live_remaining = remaining
        self.live_reset_at = time.monotonic() + max(reset_in_seconds, 0.0)


class ModelRateLimiter:
    """Tracks request and token limits for one model, or one account-level pool shared across models."""

    def __init__(
        self,
        rpm: Optional[int] = None,
        rpd: Optional[int] = None,
        tpm: Optional[int] = None,
        tpd: Optional[int] = None,
    ):
        """Args:
            rpm: Requests per minute limit, if any.
            rpd: Requests per day limit, if any.
            tpm: Tokens per minute limit, if any.
            tpd: Tokens per day limit, if any.
        """
        self.requests_minute = _Window(rpm, 60) if rpm else None
        self.requests_day = _Window(rpd, 86400) if rpd else None
        self.tokens_minute = _Window(tpm, 60) if tpm else None
        self.tokens_day = _Window(tpd, 86400) if tpd else None

    def can_proceed(self, estimated_tokens: int = 0) -> bool:
        """Checks every configured window before a call is attempted.

        Args:
            estimated_tokens: Estimated token cost of the upcoming call (see estimate_tokens()).

        Returns:
            False if any request or token window is exhausted; True otherwise.
        """
        for window in (self.requests_minute, self.requests_day):
            if window is not None:
                remaining = window.remaining()
                if remaining is not None and remaining <= 0:
                    return False
        for window in (self.tokens_minute, self.tokens_day):
            if window is not None:
                remaining = window.remaining()
                if remaining is not None and remaining < estimated_tokens:
                    return False
        return True

    def record_attempt(self, estimated_tokens: int = 0):
        """Records one call attempt across every configured request and token window.

        Args:
            estimated_tokens: Token cost to charge against the token windows.
        """
        for window in (self.requests_minute, self.requests_day):
            if window is not None:
                window.record_attempt(1)
        for window in (self.tokens_minute, self.tokens_day):
            if window is not None:
                window.record_attempt(estimated_tokens)

    def record_groq_headers(self, headers) -> None:
        """Updates live usage from Groq's rate-limit response headers.

        Args:
            headers: The response headers dict from a Groq API call.
        """
        try:
            if "x-ratelimit-remaining-requests" in headers:
                remaining = int(headers["x-ratelimit-remaining-requests"])
                reset_s = _parse_groq_reset(headers.get("x-ratelimit-reset-requests", "60s"))
                target = self.requests_day or self.requests_minute
                if target:
                    target.record_live(remaining, reset_s)
            if "x-ratelimit-remaining-tokens" in headers:
                remaining = int(headers["x-ratelimit-remaining-tokens"])
                reset_s = _parse_groq_reset(headers.get("x-ratelimit-reset-tokens", "60s"))
                target = self.tokens_minute or self.tokens_day
                if target:
                    target.record_live(remaining, reset_s)
        except (ValueError, TypeError):
            pass  # malformed header — keep the local estimate, don't crash the call

    def record_openrouter_headers(self, headers) -> None:
        """Updates live usage from OpenRouter's rate-limit response headers (present on error responses only).

        Args:
            headers: The response headers dict from an OpenRouter API call.
        """
        try:
            if "x-ratelimit-remaining" in headers:
                remaining = int(headers["x-ratelimit-remaining"])
                reset_s = _parse_openrouter_reset(headers.get("x-ratelimit-reset"))
                target = self.requests_day or self.requests_minute
                if target:
                    target.record_live(remaining, reset_s)
        except (ValueError, TypeError):
            pass


def _parse_groq_reset(value: str) -> float:
    """Parses Groq's reset-duration format, e.g. "1.2s", "120ms", "2m59.56s".

    Args:
        value: The raw header value.

    Returns:
        Seconds until reset, or 60.0 if the value can't be parsed.
    """
    value = value.strip()
    try:
        if value.endswith("ms"):
            return float(value[:-2]) / 1000
        if "m" in value and value.endswith("s"):
            minutes, seconds = value.split("m")
            return float(minutes) * 60 + float(seconds.rstrip("s"))
        if value.endswith("s"):
            return float(value[:-1])
        return float(value)
    except ValueError:
        return 60.0


def _parse_openrouter_reset(value) -> float:
    """Parses OpenRouter's reset timestamp (Unix ms) into seconds remaining.

    Args:
        value: The raw header value.

    Returns:
        Seconds until reset, or 60.0 if the value is missing or can't be parsed.
    """
    if not value:
        return 60.0
    try:
        reset_at_ms = float(value)
        return max(0.0, reset_at_ms / 1000 - time.time())
    except (ValueError, TypeError):
        return 60.0


class BackendRateLimiter:
    """Owns per-model trackers, or one shared tracker for an account-level pool."""

    def __init__(self, shared: bool = False):
        """Args:
            shared: True for one account-level pool shared across every model; False for one tracker per model.
        """
        self._shared = shared
        self._trackers: dict[str, ModelRateLimiter] = {}
        self._shared_tracker: Optional[ModelRateLimiter] = None

    def register(self, model: str, **limits):
        """Creates and stores a ModelRateLimiter for the given model, or as the shared tracker.

        Args:
            model: Model id. Ignored, but still required, when shared=True.
            **limits: rpm/rpd/tpm/tpd kwargs forwarded to ModelRateLimiter.
        """
        tracker = ModelRateLimiter(**limits)
        if self._shared:
            self._shared_tracker = tracker
        else:
            self._trackers[model] = tracker

    def get(self, model: str) -> Optional[ModelRateLimiter]:
        """Returns the tracker for this model, or the shared tracker if shared=True.

        Args:
            model: Model id to look up.

        Returns:
            The matching ModelRateLimiter, or None if never registered.
        """
        if self._shared:
            return self._shared_tracker
        return self._trackers.get(model)


def estimate_tokens(messages: list[dict]) -> int:
    """Estimates token count from character count, using a rough ~4 chars/token heuristic.

    Args:
        messages: Chat messages (OpenAI-compatible dicts with a "content" key).

    Returns:
        Estimated total token count, minimum 1.
    """
    total_chars = sum(len(m.get("content", "")) for m in messages)
    return max(1, total_chars // 4)