"""
Circuit breaker for backend clients. After repeated consecutive failures, calls fail immediately instead of retrying into a downed or rate-limited service. After a cooldown period, one trial call is allowed through to test whether the service has recovered.
"""
import time
from enum import Enum


class CircuitState(Enum):
    CLOSED = "closed"        # normal operation
    OPEN = "open"             # failing fast, cooling down
    HALF_OPEN = "half_open"   # cooldown elapsed, next call is a trial


class CircuitOpenError(Exception):
    """Raised instead of making a call while the circuit is open."""


class CircuitBreaker:
    """Tracks failures for one backend and decides whether calls should be allowed through."""

    def __init__(self, failure_threshold: int = 5, cooldown_seconds: float = 30.0):
        """Args:
            failure_threshold: Consecutive failures before the circuit opens.
            cooldown_seconds: How long the circuit stays open before allowing a trial call.
        """
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self._consecutive_failures = 0
        self._opened_at: float | None = None
        self._state = CircuitState.CLOSED

    def _current_state(self) -> CircuitState:
        """Returns the current state, moving OPEN to HALF_OPEN once the cooldown has elapsed."""
        if self._state == CircuitState.OPEN and self._opened_at is not None:
            if time.monotonic() - self._opened_at >= self.cooldown_seconds:
                self._state = CircuitState.HALF_OPEN
        return self._state

    def before_call(self):
        """Call before attempting a request.

        Raises:
            CircuitOpenError: If the circuit is open and cooldown hasn't elapsed.
        """
        if self._current_state() == CircuitState.OPEN:
            raise CircuitOpenError(
                f"Circuit open ({self._consecutive_failures} consecutive failures) — "
                f"cooling down for {self.cooldown_seconds}s before retrying"
            )

    def record_success(self):
        """Resets the failure count and closes the circuit."""
        self._consecutive_failures = 0
        self._state = CircuitState.CLOSED
        self._opened_at = None

    def record_failure(self):
        """Increments the failure count and opens the circuit once the threshold is reached."""
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.failure_threshold:
            self._state = CircuitState.OPEN
            self._opened_at = time.monotonic()