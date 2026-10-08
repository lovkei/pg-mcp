"""Unit tests for database pool management.

Uses fake pool objects (no real PostgreSQL) to exercise create/close logic.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import asyncpg
import pytest

from pg_mcp.config.settings import DatabaseConfig
from pg_mcp.db.pool import close_pools, create_pool, create_pools


def _config(name: str = "db1") -> DatabaseConfig:
    return DatabaseConfig(
        host="localhost", port=5432, name=name, user="postgres", password="pw"
    )


class TestCreatePool:
    """Tests for single-pool creation."""

    async def test_create_pool_passes_config(self) -> None:
        fake_pool = MagicMock(spec=asyncpg.Pool)
        with patch(
            "pg_mcp.db.pool.asyncpg.create_pool", new=AsyncMock(return_value=fake_pool)
        ) as mk:
            pool = await create_pool(_config("mydb"))

        assert pool is fake_pool
        kwargs = mk.call_args.kwargs
        assert kwargs["database"] == "mydb"
        assert kwargs["host"] == "localhost"
        assert kwargs["user"] == "postgres"

    async def test_create_pool_none_raises_runtime_error(self) -> None:
        with (
            patch("pg_mcp.db.pool.asyncpg.create_pool", new=AsyncMock(return_value=None)),
            pytest.raises(RuntimeError, match="Failed to create connection pool"),
        ):
            await create_pool(_config())


class TestCreatePools:
    """Tests for multi-database pool creation."""

    async def test_creates_one_pool_per_config(self) -> None:
        fake_pool = MagicMock(spec=asyncpg.Pool)
        with patch("pg_mcp.db.pool.asyncpg.create_pool", new=AsyncMock(return_value=fake_pool)):
            pools = await create_pools([_config("db1"), _config("db2")])

        assert set(pools) == {"db1", "db2"}

    async def test_empty_config_list_gives_empty_dict(self) -> None:
        assert await create_pools([]) == {}


class TestClosePools:
    """Tests for graceful pool shutdown."""

    def _pool(self) -> MagicMock:
        pool = MagicMock(spec=asyncpg.Pool)
        pool.close = AsyncMock()
        pool.terminate = MagicMock()
        return pool

    async def test_graceful_close(self) -> None:
        pool = self._pool()
        await close_pools({"db1": pool})

        pool.close.assert_awaited_once()
        pool.terminate.assert_not_called()

    async def test_close_timeout_terminates_pool(self) -> None:
        pool = self._pool()

        async def hang() -> None:
            import asyncio

            await asyncio.sleep(100)

        pool.close = hang
        with patch("pg_mcp.db.pool.asyncio.wait_for", side_effect=TimeoutError):
            await close_pools({"db1": pool}, grace_timeout=0.1)

        pool.terminate.assert_called_once()

    async def test_close_error_terminates_and_continues(self) -> None:
        pool_a, pool_b = self._pool(), self._pool()
        pool_a.close = AsyncMock(side_effect=RuntimeError("boom"))

        await close_pools({"a": pool_a, "b": pool_b})

        pool_a.terminate.assert_called_once()
        pool_b.close.assert_awaited_once()
