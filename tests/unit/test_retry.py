"""Unit tests for resilience.retry helpers."""

import asyncpg
import pytest

from pg_mcp.resilience.retry import RETRYABLE_DB_EXCEPTIONS, compute_backoff_delay


class TestComputeBackoffDelay:
    """Test exponential backoff delay computation."""

    def test_first_retry_uses_base_delay(self) -> None:
        """Attempt 0 scales the base delay by factor^0 = 1."""
        assert compute_backoff_delay(0, base_delay=1.0, backoff_factor=2.0) == 1.0

    def test_delay_grows_exponentially(self) -> None:
        """Attempts 0, 1, 2 with factor 2 double each time."""
        assert compute_backoff_delay(1, base_delay=1.0, backoff_factor=2.0) == 2.0
        assert compute_backoff_delay(2, base_delay=1.0, backoff_factor=2.0) == 4.0
        assert compute_backoff_delay(3, base_delay=1.0, backoff_factor=2.0) == 8.0

    def test_respects_custom_base_delay(self) -> None:
        """Base delay scales results linearly."""
        assert compute_backoff_delay(0, base_delay=0.5, backoff_factor=2.0) == 0.5
        assert compute_backoff_delay(2, base_delay=0.5, backoff_factor=2.0) == 2.0

    def test_capped_at_max_delay(self) -> None:
        """Delay never exceeds max_delay."""
        result = compute_backoff_delay(10, base_delay=5.0, backoff_factor=3.0, max_delay=30.0)
        assert result == 30.0

    def test_default_max_delay_is_30_seconds(self) -> None:
        """Without an explicit cap the delay tops out at 30s."""
        result = compute_backoff_delay(20, base_delay=10.0, backoff_factor=10.0)
        assert result == 30.0

    def test_negative_attempt_treated_as_zero(self) -> None:
        """Degenerate negative attempt values do not shrink the delay."""
        assert compute_backoff_delay(-1, base_delay=1.0, backoff_factor=2.0) == 1.0

    def test_fractional_backoff_factor(self) -> None:
        """Factor < 1 decays the delay but never below 0 attempts' scale."""
        assert compute_backoff_delay(1, base_delay=2.0, backoff_factor=0.5) == 1.0
        assert compute_backoff_delay(2, base_delay=2.0, backoff_factor=0.5) == 0.5


class TestRetryableDbExceptions:
    """Test the retryable exception tuple composition."""

    def test_contains_connection_errors(self) -> None:
        """Transient connection failures are retryable."""
        assert asyncpg.ConnectionDoesNotExistError in RETRYABLE_DB_EXCEPTIONS
        assert asyncpg.ConnectionFailureError in RETRYABLE_DB_EXCEPTIONS
        assert asyncpg.InterfaceError in RETRYABLE_DB_EXCEPTIONS

    def test_excludes_data_errors(self) -> None:
        """Data integrity errors must not trigger a retry."""
        assert asyncpg.UniqueViolationError not in RETRYABLE_DB_EXCEPTIONS
        assert asyncpg.NotNullViolationError not in RETRYABLE_DB_EXCEPTIONS

    def test_is_a_tuple_of_exception_types(self) -> None:
        """All entries are actual exception classes."""
        for exc in RETRYABLE_DB_EXCEPTIONS:
            assert isinstance(exc, type)
            assert issubclass(exc, BaseException)

    @pytest.mark.parametrize(
        ("exc_class",),
        [
            (asyncpg.ConnectionDoesNotExistError,),
            (asyncpg.ConnectionFailureError,),
            (asyncpg.InterfaceError,),
        ],
    )
    def test_catchable_via_tuple(self, exc_class: type[BaseException]) -> None:
        """A raised retryable exception is caught by the tuple in except."""
        try:
            raise exc_class("transient failure")
        except RETRYABLE_DB_EXCEPTIONS:
            pass  # expected: caught as retryable
