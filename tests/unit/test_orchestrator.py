"""Unit tests for QueryOrchestrator.

This module tests the orchestrator's coordination of the query pipeline:
multi-database routing, retry/backoff logic (validation and LLM transient
failures), circuit-breaker accounting, low-confidence regeneration, rate
limiting, metrics instrumentation, and request tracing.
"""

import asyncio
import re
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pg_mcp.config.settings import ResilienceConfig, ValidationConfig
from pg_mcp.models.errors import (
    DatabaseError,
    ErrorCode,
    LLMError,
    LLMTimeoutError,
    LLMUnavailableError,
    SecurityViolationError,
    SQLParseError,
)
from pg_mcp.models.generation import GenerationResult
from pg_mcp.models.query import QueryRequest, ResultValidationResult, ReturnType
from pg_mcp.models.schema import ColumnInfo, DatabaseSchema, TableInfo
from pg_mcp.resilience.circuit_breaker import CircuitState
from pg_mcp.resilience.rate_limiter import MultiRateLimiter
from pg_mcp.resilience.retry import compute_backoff_delay
from pg_mcp.services.orchestrator import QueryOrchestrator

# Small retry delay so tests exercise real backoff sleeps without slowing down.
RETRY_DELAY = 0.1
BACKOFF_FACTOR = 2.0


def make_generation(sql: str, tokens: int | None = None) -> GenerationResult:
    """Build a GenerationResult the way the real generator would."""
    return GenerationResult(sql=sql, tokens_used=tokens)


def make_sql_validator() -> MagicMock:
    """Build a validator mock whose validate_or_raise accepts everything."""
    validator = MagicMock()
    validator.validate_or_raise.return_value = None
    return validator


def make_schema_cache(schema: DatabaseSchema) -> MagicMock:
    """Build a schema cache mock that always has the schema cached."""
    cache = MagicMock()
    cache.get.return_value = schema
    return cache


def mock_schema() -> DatabaseSchema:
    """Create mock database schema with a users table."""
    return DatabaseSchema(
        database_name="test_db",
        tables=[
            TableInfo(
                schema_name="public",
                table_name="users",
                columns=[
                    ColumnInfo(
                        name="id",
                        data_type="integer",
                        is_nullable=False,
                        is_primary_key=True,
                    ),
                    ColumnInfo(
                        name="name",
                        data_type="varchar(255)",
                        is_nullable=False,
                    ),
                ],
            )
        ],
        version="15.0",
    )


def make_orchestrator(
    *,
    generator: Any | None = None,
    sql_validator: Any | None = None,
    executors: dict[str, Any] | None = None,
    result_validator: Any | None = None,
    schema_cache: Any | None = None,
    pools: dict[str, Any] | None = None,
    resilience: ResilienceConfig | None = None,
    validation: ValidationConfig | None = None,
    rate_limiter: Any | None = None,
    metrics: Any | None = None,
) -> QueryOrchestrator:
    """Build a QueryOrchestrator with mocked components and fast retries."""
    if executors is None:
        executors = {"test_db": AsyncMock()}
    if pools is None:
        pools = {name: MagicMock() for name in executors}
    if generator is None:
        generator = AsyncMock()
        generator.generate.return_value = make_generation("SELECT 1;")
    if sql_validator is None:
        sql_validator = make_sql_validator()
    if result_validator is None:
        result_validator = AsyncMock()
    if schema_cache is None:
        schema_cache = MagicMock()
        schema_cache.get.return_value = None
        schema_cache.load = AsyncMock(return_value=mock_schema())
    if resilience is None:
        resilience = ResilienceConfig(retry_delay=RETRY_DELAY, backoff_factor=BACKOFF_FACTOR)

    return QueryOrchestrator(
        sql_generator=generator,
        sql_validator=sql_validator,
        executors=executors,
        result_validator=result_validator,
        schema_cache=schema_cache,
        pools=pools,
        resilience_config=resilience,
        validation_config=validation if validation is not None else ValidationConfig(),
        rate_limiter=rate_limiter,
        metrics_collector=metrics,
    )


def result_validation(confidence: int) -> ResultValidationResult:
    """Build a ResultValidationResult with the given confidence."""
    return ResultValidationResult(
        confidence=confidence,
        explanation="explanation",
        suggestion=None,
        is_acceptable=confidence >= 70,
    )


class TestDatabaseResolution:
    """Test database name resolution and routing table logic."""

    def test_resolve_database_specified_valid(self) -> None:
        """Test resolving a specified valid database."""
        orchestrator = make_orchestrator(
            executors={"db1": AsyncMock(), "db2": AsyncMock()},
        )
        assert orchestrator._resolve_database("db1") == "db1"

    def test_resolve_database_specified_invalid(self) -> None:
        """Test resolving a specified but unknown database lists the available ones."""
        orchestrator = make_orchestrator(
            executors={"db1": AsyncMock(), "db2": AsyncMock()},
        )
        with pytest.raises(DatabaseError) as exc_info:
            orchestrator._resolve_database("nonexistent")

        assert "not found" in str(exc_info.value).lower()
        assert exc_info.value.details["available_databases"] == ["db1", "db2"]

    def test_resolve_database_auto_select_single(self) -> None:
        """Test auto-selecting when only one database is routable."""
        orchestrator = make_orchestrator(executors={"only_db": AsyncMock()})
        assert orchestrator._resolve_database(None) == "only_db"

    def test_resolve_database_auto_select_multiple_fails(self) -> None:
        """Test that auto-select fails when multiple databases are available."""
        orchestrator = make_orchestrator(
            executors={"db1": AsyncMock(), "db2": AsyncMock()},
        )
        with pytest.raises(DatabaseError) as exc_info:
            orchestrator._resolve_database(None)

        assert "multiple databases" in str(exc_info.value).lower()
        assert "db1" in exc_info.value.details["available_databases"]

    def test_resolve_database_no_databases(self) -> None:
        """Test error when no databases are configured."""
        orchestrator = make_orchestrator(executors={})
        with pytest.raises(DatabaseError) as exc_info:
            orchestrator._resolve_database(None)

        assert "no databases configured" in str(exc_info.value).lower()

    def test_pool_without_executor_is_not_routable(self) -> None:
        """A pool without a matching executor must not be selectable (routing safety)."""
        orchestrator = make_orchestrator(
            executors={"db1": AsyncMock()},
            pools={"db1": MagicMock(), "ghost": MagicMock()},
        )

        with pytest.raises(DatabaseError, match="not found"):
            orchestrator._resolve_database("ghost")
        # Auto-select only considers pool+executor intersections
        assert orchestrator._resolve_database(None) == "db1"
        assert orchestrator._available_databases() == ["db1"]


class TestSQLGenerationWithRetry:
    """Test SQL generation with validation retry logic."""

    async def test_generate_sql_success_first_attempt(self) -> None:
        """Test successful SQL generation on the first attempt."""
        generator = AsyncMock()
        generator.generate.return_value = make_generation("SELECT * FROM users;", tokens=55)

        orchestrator = make_orchestrator(
            generator=generator,
            sql_validator=make_sql_validator(),
        )

        sql, validation_result, tokens = await orchestrator._generate_sql_with_retry(
            question="Get all users",
            schema=mock_schema(),
            request_id="test-123",
        )

        assert sql == "SELECT * FROM users;"
        assert validation_result.is_valid is True
        assert validation_result.is_select is True
        assert tokens == 55
        generator.generate.assert_called_once()

    async def test_generate_sql_retry_on_validation_failure(self) -> None:
        """Test retry with error feedback when validation fails."""
        generator = AsyncMock()
        generator.generate.side_effect = [
            make_generation("SELECT * FROM user;"),
            make_generation("SELECT * FROM users;"),
        ]
        validator = MagicMock()
        validator.validate_or_raise.side_effect = [
            SQLParseError('relation "user" does not exist'),
            None,
        ]

        orchestrator = make_orchestrator(generator=generator, sql_validator=validator)

        sql, validation_result, _tokens = await orchestrator._generate_sql_with_retry(
            question="Get all users",
            schema=mock_schema(),
            request_id="test-123",
        )

        assert sql == "SELECT * FROM users;"
        assert validation_result.is_valid is True
        assert generator.generate.call_count == 2
        assert validator.validate_or_raise.call_count == 2

        second_call = generator.generate.call_args_list[1]
        assert second_call.kwargs["previous_attempt"] == "SELECT * FROM user;"
        assert 'relation "user" does not exist' in second_call.kwargs["error_feedback"]

    async def test_validation_retry_backs_off_between_attempts(self) -> None:
        """Validation retries sleep for the exponential backoff delay."""
        generator = AsyncMock()
        generator.generate.side_effect = [
            make_generation("SELECT * FROM user;"),
            make_generation("SELECT * FROM users;"),
        ]
        validator = MagicMock()
        validator.validate_or_raise.side_effect = [
            SQLParseError("bad sql"),
            None,
        ]

        orchestrator = make_orchestrator(generator=generator, sql_validator=validator)

        with patch("asyncio.sleep", new=AsyncMock()) as mock_sleep:
            await orchestrator._generate_sql_with_retry(
                question="Get all users",
                schema=mock_schema(),
                request_id="test-123",
            )

        mock_sleep.assert_awaited_once_with(
            compute_backoff_delay(0, RETRY_DELAY, BACKOFF_FACTOR)
        )

    async def test_generate_sql_fails_after_max_retries(self) -> None:
        """Test failure after exhausting all retries (circuit-breaker probe)."""
        generator = AsyncMock()
        generator.generate.return_value = make_generation("DELETE FROM users;")
        validator = MagicMock()
        validator.validate_or_raise.side_effect = SecurityViolationError(
            "DELETE statements are not allowed"
        )

        orchestrator = make_orchestrator(
            generator=generator,
            sql_validator=validator,
            resilience=ResilienceConfig(
                max_retries=2, retry_delay=RETRY_DELAY, backoff_factor=BACKOFF_FACTOR
            ),
        )

        with pytest.raises(SecurityViolationError) as exc_info:
            await orchestrator._generate_sql_with_retry(
                question="Delete all users",
                schema=mock_schema(),
                request_id="test-123",
            )

        assert "DELETE statements are not allowed" in str(exc_info.value)
        # Initial + 2 retries = 3 generation attempts
        assert generator.generate.call_count == 3
        # Exactly one terminal failure recorded on the circuit breaker
        assert orchestrator.circuit_breaker.failure_count == 1

    async def test_generate_sql_circuit_breaker_open(self) -> None:
        """Test that an open circuit breaker prevents SQL generation."""
        orchestrator = make_orchestrator()
        orchestrator.circuit_breaker._state = CircuitState.OPEN
        orchestrator.circuit_breaker._failure_count = 5

        with pytest.raises(LLMError) as exc_info:
            await orchestrator._generate_sql_with_retry(
                question="Get all users",
                schema=mock_schema(),
                request_id="test-123",
            )

        assert "temporarily unavailable" in str(exc_info.value).lower()
        assert "circuit breaker" in str(exc_info.value).lower()

    async def test_generate_sql_unexpected_error(self) -> None:
        """Unexpected generation errors map to LLMError (circuit-breaker probe)."""
        generator = AsyncMock()
        generator.generate.side_effect = RuntimeError("Unexpected error")

        orchestrator = make_orchestrator(generator=generator)

        with pytest.raises(LLMError) as exc_info:
            await orchestrator._generate_sql_with_retry(
                question="Get all users",
                schema=mock_schema(),
                request_id="test-123",
            )

        assert "unexpectedly" in str(exc_info.value).lower()
        assert orchestrator.circuit_breaker.failure_count == 1


class TestLLMTransientRetry:
    """Test LLM transient-failure retry with exponential backoff."""

    async def test_transient_llm_failure_retried_with_backoff(self) -> None:
        """Retryable unavailability is retried after a backoff sleep."""
        generator = AsyncMock()
        generator.generate.side_effect = [
            LLMUnavailableError("LLM service temporarily down"),
            make_generation("SELECT 1;"),
        ]
        orchestrator = make_orchestrator(generator=generator)

        with patch("asyncio.sleep", new=AsyncMock()) as mock_sleep:
            result = await orchestrator._generate_sql_with_retry(
                question="Get all users",
                schema=mock_schema(),
                request_id="test-123",
            )

        assert result[0] == "SELECT 1;"
        mock_sleep.assert_awaited_once_with(
            compute_backoff_delay(0, RETRY_DELAY, BACKOFF_FACTOR)
        )
        assert generator.generate.call_count == 2
        # A successful retry records success, which resets the failure count.
        assert orchestrator.circuit_breaker.failure_count == 0

    async def test_llm_timeout_retried(self) -> None:
        """LLM timeouts are transient and retried with backoff."""
        generator = AsyncMock()
        generator.generate.side_effect = [
            LLMTimeoutError("request timed out"),
            make_generation("SELECT 1;"),
        ]
        orchestrator = make_orchestrator(generator=generator)

        with patch("asyncio.sleep", new=AsyncMock()) as mock_sleep:
            result = await orchestrator._generate_sql_with_retry(
                question="Get all users",
                schema=mock_schema(),
                request_id="test-123",
            )

        assert result[0] == "SELECT 1;"
        assert mock_sleep.await_count == 1

    async def test_fatal_llm_failure_not_retried(self) -> None:
        """Non-retryable failures (e.g. bad credentials) raise immediately."""
        generator = AsyncMock()
        generator.generate.side_effect = LLMUnavailableError(
            "authentication failed", retryable=False
        )
        orchestrator = make_orchestrator(generator=generator)

        with pytest.raises(LLMUnavailableError):
            await orchestrator._generate_sql_with_retry(
                question="Get all users",
                schema=mock_schema(),
                request_id="test-123",
            )

        assert generator.generate.call_count == 1
        # Fatal failure recorded exactly once on the circuit breaker
        assert orchestrator.circuit_breaker.failure_count == 1

    async def test_transient_llm_failure_exhausts_retries(self) -> None:
        """Exhausted retries re-raise the last transient error; each failure counts."""
        generator = AsyncMock()
        generator.generate.side_effect = LLMUnavailableError("still down")
        orchestrator = make_orchestrator(
            generator=generator,
            resilience=ResilienceConfig(
                max_retries=2, retry_delay=RETRY_DELAY, backoff_factor=BACKOFF_FACTOR
            ),
        )

        with (
            patch("asyncio.sleep", new=AsyncMock()) as mock_sleep,
            pytest.raises(LLMUnavailableError),
        ):
            await orchestrator._generate_sql_with_retry(
                question="Get all users",
                schema=mock_schema(),
                request_id="test-123",
            )

        # 3 attempts (initial + 2 retries), backoff sleeps between them
        assert generator.generate.call_count == 3
        assert mock_sleep.await_count == 2
        # One failure recorded per failed attempt
        assert orchestrator.circuit_breaker.failure_count == 3


class TestResultValidation:
    """Test result validation logic."""

    async def test_validate_results_success(self) -> None:
        """Test successful result validation."""
        result_validator = AsyncMock()
        result_validator.validate.return_value = result_validation(85)

        orchestrator = make_orchestrator(
            result_validator=result_validator,
            validation=ValidationConfig(enabled=True),
        )

        confidence = await orchestrator._validate_results_safely(
            question="Count users",
            sql="SELECT COUNT(*) FROM users",
            results=[{"count": 42}],
            row_count=1,
            request_id="test-123",
        )

        assert confidence == 85
        result_validator.validate.assert_called_once()

    async def test_validate_results_disabled(self) -> None:
        """Test that validation is skipped when disabled."""
        result_validator = AsyncMock()
        orchestrator = make_orchestrator(
            result_validator=result_validator,
            validation=ValidationConfig(enabled=False),
        )

        confidence = await orchestrator._validate_results_safely(
            question="Count users",
            sql="SELECT COUNT(*) FROM users",
            results=[{"count": 42}],
            row_count=1,
            request_id="test-123",
        )

        assert confidence == 100
        result_validator.validate.assert_not_called()

    async def test_validate_results_failure_does_not_raise(self) -> None:
        """Test that validation failures don't raise exceptions."""
        result_validator = AsyncMock()
        result_validator.validate.side_effect = Exception("Validation failed")

        orchestrator = make_orchestrator(
            result_validator=result_validator,
            validation=ValidationConfig(enabled=True),
        )

        confidence = await orchestrator._validate_results_safely(
            question="Count users",
            sql="SELECT COUNT(*) FROM users",
            results=[{"count": 42}],
            row_count=1,
            request_id="test-123",
        )

        assert confidence == 100


class TestExecuteQueryFlow:
    """Test complete query execution flow."""

    async def test_execute_query_sql_only(self) -> None:
        """Test executing query with return_type=SQL."""
        generator = AsyncMock()
        generator.generate.return_value = make_generation("SELECT * FROM users;", tokens=77)

        orchestrator = make_orchestrator(
            generator=generator,
            schema_cache=make_schema_cache(mock_schema()),
        )

        request = QueryRequest(
            question="Get all users",
            database="test_db",
            return_type=ReturnType.SQL,
        )
        response = await orchestrator.execute_query(request)

        assert response.success is True
        assert response.generated_sql == "SELECT * FROM users;"
        assert response.validation is not None
        assert response.validation.is_valid is True
        assert response.data is None  # No execution for SQL-only
        assert response.error is None
        assert response.tokens_used == 77
        # Request tracing: response carries the generated request ID
        assert response.request_id is not None
        assert re.fullmatch(r"[0-9a-f-]{36}", response.request_id)

    async def test_execute_query_with_results(self) -> None:
        """Test executing query with return_type=RESULT."""
        generator = AsyncMock()
        generator.generate.return_value = make_generation(
            "SELECT id, name FROM users;", tokens=123
        )
        executor = AsyncMock()
        executor.execute.return_value = (
            [
                {"id": 1, "name": "Alice"},
                {"id": 2, "name": "Bob"},
            ],
            2,
        )
        result_validator = AsyncMock()
        result_validator.validate.return_value = result_validation(90)

        orchestrator = make_orchestrator(
            generator=generator,
            executors={"test_db": executor},
            result_validator=result_validator,
            schema_cache=make_schema_cache(mock_schema()),
            validation=ValidationConfig(enabled=True),
        )

        request = QueryRequest(
            question="Get all users",
            database="test_db",
            return_type=ReturnType.RESULT,
        )
        response = await orchestrator.execute_query(request)

        assert response.success is True
        assert response.generated_sql == "SELECT id, name FROM users;"
        assert response.data is not None
        assert response.data.row_count == 2
        assert len(response.data.rows) == 2
        assert response.data.columns == ["id", "name"]
        assert response.confidence == 90
        assert response.tokens_used == 123
        assert response.warning is None
        assert response.error is None
        assert response.request_id is not None
        # Confidence above threshold: no regeneration attempted
        generator.generate.assert_called_once()

    async def test_execute_query_schema_not_cached(self) -> None:
        """Test loading schema when not in cache."""
        cache = MagicMock()
        cache.get.return_value = None
        cache.load = AsyncMock(return_value=mock_schema())
        pool = MagicMock()

        orchestrator = make_orchestrator(
            schema_cache=cache,
            executors={"test_db": AsyncMock()},
            pools={"test_db": pool},
        )

        request = QueryRequest(
            question="Test query",
            database="test_db",
            return_type=ReturnType.SQL,
        )
        response = await orchestrator.execute_query(request)

        cache.load.assert_called_once_with("test_db", pool)
        assert response.success is True

    async def test_execute_query_schema_load_fails(self) -> None:
        """Test handling of schema load failure (with request tracing)."""
        cache = MagicMock()
        cache.get.return_value = None
        cache.load = AsyncMock(side_effect=Exception("DB connection failed"))

        orchestrator = make_orchestrator(schema_cache=cache)

        request = QueryRequest(
            question="Test query",
            database="test_db",
            return_type=ReturnType.SQL,
        )
        response = await orchestrator.execute_query(request)

        assert response.success is False
        assert response.error is not None
        assert "schema" in response.error.message.lower()
        assert response.generated_sql is None
        assert response.request_id is not None

    async def test_execute_query_validation_error(self) -> None:
        """Test handling of SQL validation errors."""
        generator = AsyncMock()
        generator.generate.return_value = make_generation("DELETE FROM users;")
        validator = MagicMock()
        validator.validate_or_raise.side_effect = SecurityViolationError("DELETE not allowed")

        orchestrator = make_orchestrator(
            generator=generator,
            sql_validator=validator,
            schema_cache=make_schema_cache(mock_schema()),
        )

        request = QueryRequest(
            question="Delete all users",
            database="test_db",
            return_type=ReturnType.SQL,
        )
        response = await orchestrator.execute_query(request)

        assert response.success is False
        assert response.error is not None
        assert "DELETE not allowed" in response.error.message
        assert response.error.code == ErrorCode.SECURITY_VIOLATION
        assert response.request_id is not None

    async def test_execute_query_execution_error(self) -> None:
        """Test handling of SQL execution errors."""
        executor = AsyncMock()
        executor.execute.side_effect = DatabaseError("Query execution failed")

        orchestrator = make_orchestrator(
            executors={"test_db": executor},
            schema_cache=make_schema_cache(mock_schema()),
        )

        request = QueryRequest(
            question="Get all users",
            database="test_db",
            return_type=ReturnType.RESULT,
        )
        response = await orchestrator.execute_query(request)

        assert response.success is False
        assert response.error is not None
        assert "execution failed" in response.error.message.lower()
        assert response.error.code == ErrorCode.DATABASE_ERROR

    async def test_execute_query_unexpected_error(self) -> None:
        """Test handling of unexpected errors."""
        cache = MagicMock()
        cache.get.side_effect = RuntimeError("Unexpected error")

        orchestrator = make_orchestrator(schema_cache=cache)

        request = QueryRequest(
            question="Get all users",
            database="test_db",
            return_type=ReturnType.SQL,
        )
        response = await orchestrator.execute_query(request)

        assert response.success is False
        assert response.error is not None
        assert response.error.code == ErrorCode.INTERNAL_ERROR
        assert "internal server error" in response.error.message.lower()

    async def test_execute_query_auto_select_database(self) -> None:
        """Test auto-selecting database when only one is available."""
        cache = make_schema_cache(mock_schema())

        orchestrator = make_orchestrator(
            schema_cache=cache,
            executors={"only_db": AsyncMock()},
        )

        request = QueryRequest(
            question="Test query",
            database=None,
            return_type=ReturnType.SQL,
        )
        response = await orchestrator.execute_query(request)

        assert response.success is True
        cache.get.assert_called_once_with("only_db")


class TestMultiDatabaseRouting:
    """Test true multi-database routing (request.database picks the executor)."""

    async def test_execute_query_routes_to_requested_database(self) -> None:
        """The request's database selects the executor that runs the SQL."""
        executor1 = AsyncMock()
        executor2 = AsyncMock()
        executor2.execute.return_value = ([{"n": 1}], 1)

        result_validator = AsyncMock()
        result_validator.validate.return_value = result_validation(90)

        schema = mock_schema()
        orchestrator = make_orchestrator(
            executors={"db1": executor1, "db2": executor2},
            pools={"db1": MagicMock(), "db2": MagicMock()},
            result_validator=result_validator,
            schema_cache=make_schema_cache(schema),
            validation=ValidationConfig(enabled=True),
        )

        request = QueryRequest(
            question="Count rows",
            database="db2",
            return_type=ReturnType.RESULT,
        )
        response = await orchestrator.execute_query(request)

        assert response.success is True
        executor2.execute.assert_called_once()
        executor1.execute.assert_not_called()
        orchestrator.schema_cache.get.assert_called_with("db2")

    async def test_execute_query_unknown_database_lists_available(self) -> None:
        """An unknown database returns an explicit error naming available ones."""
        orchestrator = make_orchestrator(
            executors={"db1": AsyncMock(), "db2": AsyncMock()},
            pools={"db1": MagicMock(), "db2": MagicMock()},
        )

        request = QueryRequest(
            question="Count rows",
            database="missing_db",
            return_type=ReturnType.RESULT,
        )
        response = await orchestrator.execute_query(request)

        assert response.success is False
        assert response.error is not None
        assert response.error.code == ErrorCode.DATABASE_ERROR
        assert "not found" in response.error.message.lower()
        assert response.error.details is not None
        assert response.error.details["available_databases"] == ["db1", "db2"]


class TestLowConfidenceRegeneration:
    """Test the low-confidence regeneration loop (PRD §2.1)."""

    def regeneration_setup(
        self,
        confidences: list[int],
        sqls: list[str],
        execute_results: list[Any] | None = None,
        budget: int = 2,
    ) -> tuple[QueryOrchestrator, AsyncMock, AsyncMock]:
        """Wire an orchestrator for a regeneration scenario."""
        generator = AsyncMock()
        generator.generate.side_effect = [make_generation(sql) for sql in sqls]

        result_validator = AsyncMock()
        result_validator.validate.side_effect = [result_validation(c) for c in confidences]

        executor = AsyncMock()
        if execute_results is None:
            executor.execute.return_value = ([{"n": 1}], 1)
        else:
            executor.execute.side_effect = execute_results

        orchestrator = make_orchestrator(
            generator=generator,
            executors={"test_db": executor},
            result_validator=result_validator,
            schema_cache=make_schema_cache(mock_schema()),
            validation=ValidationConfig(
                enabled=True, confidence_threshold=70, max_regeneration_attempts=budget
            ),
        )
        return orchestrator, generator, executor

    async def test_regeneration_improves_confidence(self) -> None:
        """A low-confidence first attempt triggers regeneration (50 -> 90)."""
        orchestrator, generator, executor = self.regeneration_setup(
            confidences=[50, 90],
            sqls=["SELECT first_col FROM users;", "SELECT second_col FROM users;"],
        )

        request = QueryRequest(
            question="Count users",
            database="test_db",
            return_type=ReturnType.RESULT,
        )
        response = await orchestrator.execute_query(request)

        assert response.success is True
        assert response.confidence == 90
        assert response.warning is None
        # Response reports the SQL that actually produced the returned data
        assert response.generated_sql == "SELECT second_col FROM users;"
        assert generator.generate.call_count == 2
        assert executor.execute.call_count == 2

        feedback_call = generator.generate.call_args_list[1]
        assert feedback_call.kwargs["previous_attempt"] == "SELECT first_col FROM users;"
        assert "low confidence" in feedback_call.kwargs["error_feedback"]

    async def test_regeneration_budget_exhausted_sets_warning(self) -> None:
        """When the budget is exhausted the best attempt is returned with a warning."""
        orchestrator, generator, _executor = self.regeneration_setup(
            confidences=[55, 55, 55],
            sqls=["SELECT a;", "SELECT b;", "SELECT c;"],
            budget=2,
        )

        request = QueryRequest(
            question="Count users",
            database="test_db",
            return_type=ReturnType.RESULT,
        )
        response = await orchestrator.execute_query(request)

        assert response.success is True
        assert response.confidence == 55
        assert response.warning is not None
        assert "below the acceptable threshold" in response.warning
        # 55 >= 50: below threshold, but not low enough for rephrasing advice
        assert "rephrasing" not in response.warning
        # Initial + 2 regeneration attempts
        assert generator.generate.call_count == 3

    async def test_regeneration_very_low_confidence_suggests_rephrasing(self) -> None:
        """Confidence below 50 additionally suggests rephrasing the question."""
        orchestrator, _generator, _executor = self.regeneration_setup(
            confidences=[30, 30],
            sqls=["SELECT a;", "SELECT b;"],
            budget=1,
        )

        request = QueryRequest(
            question="Count users",
            database="test_db",
            return_type=ReturnType.RESULT,
        )
        response = await orchestrator.execute_query(request)

        assert response.success is True
        assert response.confidence == 30
        assert response.warning is not None
        assert "rephrasing" in response.warning

    async def test_regeneration_failure_keeps_previous_result(self) -> None:
        """A failure during regeneration keeps the previous best result."""
        orchestrator, generator, _executor = self.regeneration_setup(
            confidences=[50],
            sqls=["SELECT first_col FROM users;"],
        )
        # First generation succeeds; regeneration attempt fails fatally
        generator.generate.side_effect = [
            make_generation("SELECT first_col FROM users;"),
            LLMUnavailableError("fatal auth failure", retryable=False),
        ]

        request = QueryRequest(
            question="Count users",
            database="test_db",
            return_type=ReturnType.RESULT,
        )
        response = await orchestrator.execute_query(request)

        assert response.success is True
        assert response.confidence == 50
        assert response.generated_sql == "SELECT first_col FROM users;"
        assert response.warning is not None

    async def test_regeneration_execution_failure_keeps_previous_result(self) -> None:
        """A runtime failure of regenerated SQL keeps the previous best result."""
        orchestrator, _generator, _executor = self.regeneration_setup(
            confidences=[50, 50],
            sqls=["SELECT a;", "SELECT b;"],
            execute_results=[([{"n": 1}], 1), DatabaseError("regenerated SQL blew up")],
        )

        request = QueryRequest(
            question="Count users",
            database="test_db",
            return_type=ReturnType.RESULT,
        )
        response = await orchestrator.execute_query(request)

        assert response.success is True
        assert response.confidence == 50
        assert response.generated_sql == "SELECT a;"
        assert response.data is not None
        assert response.data.rows == [{"n": 1}]
        assert response.warning is not None


class TestRateLimiting:
    """Test concurrency-based rate limiting on the query path."""

    async def test_rate_limit_returns_rate_limited_response(self) -> None:
        """When no query slot is free the request is rejected without generation."""
        limiter = MultiRateLimiter(query_limit=1, llm_limit=1)
        orchestrator = make_orchestrator(
            rate_limiter=limiter,
            resilience=ResilienceConfig(rate_limit_timeout=0.1),
        )
        generator = orchestrator.sql_generator

        # Hold the only query slot
        assert await limiter.query_limiter.acquire()

        request = QueryRequest(
            question="Get all users",
            database="test_db",
            return_type=ReturnType.SQL,
        )
        response = await orchestrator.execute_query(request)

        assert response.success is False
        assert response.error is not None
        assert response.error.code == ErrorCode.RATE_LIMIT_EXCEEDED
        assert response.request_id is not None
        generator.generate.assert_not_called()

        # Cleanup: release the manually held slot
        limiter.query_limiter.release()

    async def test_query_slot_released_after_success(self) -> None:
        """The query slot is released after a successful query."""
        limiter = MultiRateLimiter(query_limit=1, llm_limit=1)
        orchestrator = make_orchestrator(
            rate_limiter=limiter,
            schema_cache=make_schema_cache(mock_schema()),
        )

        request = QueryRequest(
            question="Get all users",
            database="test_db",
            return_type=ReturnType.SQL,
        )
        response = await orchestrator.execute_query(request)

        assert response.success is True
        # release() decrements the counter asynchronously; give it a tick
        await asyncio.sleep(0.05)
        assert limiter.query_limiter.active_count == 0


class TestMetricsInstrumentation:
    """Test metrics instrumentation wiring on the query path."""

    async def test_metrics_recorded_on_success(self) -> None:
        """Query, LLM, token, and duration metrics are recorded for a success."""
        metrics = MagicMock()
        generator = AsyncMock()
        generator.generate.return_value = make_generation(
            "SELECT id FROM users;", tokens=321
        )
        result_validator = AsyncMock()
        result_validator.validate.return_value = result_validation(90)
        executor = AsyncMock()
        executor.execute.return_value = ([{"id": 1}], 1)

        orchestrator = make_orchestrator(
            generator=generator,
            result_validator=result_validator,
            schema_cache=make_schema_cache(mock_schema()),
            validation=ValidationConfig(enabled=True),
            metrics=metrics,
            executors={"test_db": executor},
        )

        request = QueryRequest(
            question="Get all users",
            database="test_db",
            return_type=ReturnType.RESULT,
        )
        response = await orchestrator.execute_query(request)

        assert response.success is True
        metrics.increment_query_request.assert_called_once_with("success", "test_db")
        metrics.query_duration.observe.assert_called_once()
        # Both generation and result validation are LLM calls on the RESULT path.
        assert metrics.increment_llm_call.call_count == 2
        metrics.increment_llm_call.assert_any_call("generate_sql")
        metrics.increment_llm_call.assert_any_call("validate_result")
        assert metrics.increment_llm_tokens.call_count == 1
        metrics.increment_llm_tokens.assert_called_once_with("generate_sql", 321)
        metrics.observe_llm_latency.assert_called()

    async def test_metrics_recorded_on_error(self) -> None:
        """Failed queries are recorded with the error code as status."""
        metrics = MagicMock()
        cache = MagicMock()
        cache.get.side_effect = RuntimeError("boom")

        orchestrator = make_orchestrator(schema_cache=cache, metrics=metrics)

        request = QueryRequest(
            question="Get all users",
            database="test_db",
            return_type=ReturnType.SQL,
        )
        await orchestrator.execute_query(request)

        metrics.increment_query_request.assert_called_once_with("internal_error", "test_db")
