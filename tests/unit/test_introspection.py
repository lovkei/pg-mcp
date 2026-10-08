"""Unit tests for the schema introspection service.

All database access goes through a fake asyncpg connection; no PostgreSQL
server is needed.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from pg_mcp.db.introspection import SchemaIntrospector


class FakeConn:
    """Fake asyncpg connection returning canned rows per query keyword."""

    def __init__(self) -> None:
        self.queries: list[str] = []

    async def fetchval(self, query: str, *args: object) -> object:
        self.queries.append(query)
        if "version()" in query:
            return "PostgreSQL 15.4 (Debian), 64-bit"
        if "reltuples" in query:
            return 123
        if "contype = 'u'" in query:
            return False
        return None

    async def fetch(self, query: str, *args: object) -> list[dict]:
        self.queries.append(query)
        if "relkind = 'r'" in query:
            return [
                {"schema_name": "public", "table_name": "users", "comment": "User accounts"}
            ]
        if "relkind = 'v'" in query:
            return [
                {"schema_name": "public", "table_name": "active_users", "comment": None}
            ]
        if "typtype = 'e'" in query:
            return [
                {
                    "schema_name": "public",
                    "type_name": "user_status",
                    "values": ["active", "banned"],
                }
            ]
        if "attnum > 0" in query:
            return [
                {
                    "column_name": "id",
                    "data_type": "integer",
                    "is_nullable": False,
                    "default_value": "nextval('users_id_seq')",
                    "comment": None,
                },
                {
                    "column_name": "email",
                    "data_type": "character varying(255)",
                    "is_nullable": False,
                    "default_value": None,
                    "comment": "Login email",
                },
            ]
        if "indisprimary" in query and "pg_am" not in query:
            return [{"column_name": "id"}]
        if "contype = 'f'" in query:
            return [
                {
                    "constraint_name": "fk_posts_user",
                    "column_name": "user_id",
                    "referenced_table": "users",
                    "referenced_column": "id",
                }
            ]
        if "pg_am am ON" in query:
            return [
                {
                    "index_name": "idx_users_email",
                    "is_unique": True,
                    "index_type": "btree",
                    "columns": ["email"],
                }
            ]
        return []


@pytest.fixture
def introspector() -> SchemaIntrospector:
    pool = MagicMock()
    # asyncpg's pool.acquire() is called without await and returns an async
    # context manager, so a plain MagicMock returning a fake CM is correct.
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=None)
    ctx.__aexit__ = AsyncMock(return_value=False)
    pool.acquire = MagicMock(return_value=ctx)
    return SchemaIntrospector(pool, database_name="testdb")


def _wire_conn(introspector: SchemaIntrospector, conn: FakeConn) -> None:
    """Point the introspector's pool at the given fake connection."""
    introspector.pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)


async def test_introspect_builds_full_schema(introspector: SchemaIntrospector) -> None:
    conn = FakeConn()
    _wire_conn(introspector, conn)

    schema = await introspector.introspect()

    assert schema.database_name == "testdb"
    assert schema.version == "PostgreSQL 15.4 (Debian)"
    # One table + one view
    assert len(schema.tables) == 2
    users = schema.tables[0]
    assert users.table_name == "users"
    assert users.comment == "User accounts"
    # Columns enriched with PK flag and unique check
    assert users.columns[0].name == "id"
    assert users.columns[0].is_primary_key is True
    assert users.columns[1].name == "email"
    assert users.columns[1].comment == "Login email"
    assert users.row_count_estimate == 123
    # Foreign key and index populated
    assert users.foreign_keys[0].referenced_table == "users"
    assert users.indexes[0].is_unique is True
    # Enum types captured
    assert schema.enum_types[0].type_name == "user_status"
    assert schema.enum_types[0].values == ["active", "banned"]


async def test_introspect_handles_null_version(introspector: SchemaIntrospector) -> None:
    conn = FakeConn()

    async def null_version(query: str, *args: object) -> object:
        return None if "version()" in query else await FakeConn().fetchval(query, *args)

    conn.fetchval = null_version  # type: ignore[method-assign]
    _wire_conn(introspector, conn)

    schema = await introspector.introspect()
    assert schema.version is None


async def test_introspect_row_count_none_defaults_to_zero(
    introspector: SchemaIntrospector,
) -> None:
    conn = FakeConn()
    conn.queries = []

    async def null_count(query: str, *args: object) -> object:
        if "reltuples" in query:
            return None
        return await FakeConn().fetchval(query, *args)

    conn.fetchval = null_count  # type: ignore[method-assign]
    _wire_conn(introspector, conn)

    schema = await introspector.introspect()
    assert schema.tables[0].row_count_estimate == 0
