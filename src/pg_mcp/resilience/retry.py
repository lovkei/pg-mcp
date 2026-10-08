"""Retry helpers: exponential backoff computation and retryable DB exceptions."""

import asyncpg

# Transient connection-level failures worth retrying once with a fresh
# connection. Excludes data errors (e.g. unique violation) and cancellations.
RETRYABLE_DB_EXCEPTIONS: tuple[type[Exception], ...] = (
    asyncpg.ConnectionDoesNotExistError,
    asyncpg.ConnectionFailureError,
    asyncpg.InterfaceError,
)


def compute_backoff_delay(
    attempt: int,
    base_delay: float,
    backoff_factor: float,
    max_delay: float = 30.0,
) -> float:
    """Compute an exponential backoff delay for the given retry attempt.

    Args:
        attempt: Zero-based attempt index (0 = first retry, 1 = second, ...).
        base_delay: Delay in seconds before any backoff scaling.
        backoff_factor: Multiplier applied per attempt (e.g. 2.0 = doubling).
        max_delay: Upper bound in seconds to avoid unbounded waits.

    Returns:
        float: Delay in seconds to sleep before the next attempt, capped at
            ``max_delay``. Always positive, even for degenerate inputs.
    """
    delay = base_delay * (backoff_factor**max(0, attempt))
    return min(delay, max_delay)
