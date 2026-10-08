"""Query orchestrator for coordinating the complete query flow.

This module provides the QueryOrchestrator class that coordinates all components
of the query processing pipeline: SQL generation, validation, execution, and result
validation. It implements retry logic with exponential backoff, rate limiting,
circuit-breaker accounting, low-confidence regeneration, metrics collection, and
request tracing.
"""

import asyncio
import logging
import time
from collections.abc import Coroutine
from typing import TYPE_CHECKING, Any, TypeVar

from asyncpg import Pool

from pg_mcp.cache.schema_cache import SchemaCache
from pg_mcp.config.settings import ResilienceConfig, ValidationConfig
from pg_mcp.models.errors import (
    DatabaseError,
    ErrorCode,
    LLMError,
    LLMTimeoutError,
    LLMUnavailableError,
    PgMcpError,
    SchemaLoadError,
    SecurityViolationError,
    SQLParseError,
)
from pg_mcp.models.generation import GenerationResult
from pg_mcp.models.query import (
    ErrorDetail,
    QueryRequest,
    QueryResponse,
    QueryResult,
    ReturnType,
    ValidationResult,
)
from pg_mcp.observability.tracing import request_context
from pg_mcp.resilience.circuit_breaker import CircuitBreaker
from pg_mcp.resilience.rate_limiter import MultiRateLimiter
from pg_mcp.resilience.retry import compute_backoff_delay
from pg_mcp.services.result_validator import ResultValidator
from pg_mcp.services.sql_executor import SQLExecutor
from pg_mcp.services.sql_generator import SQLGenerator
from pg_mcp.services.sql_validator import SQLValidator

if TYPE_CHECKING:
    from pg_mcp.models.schema import DatabaseSchema
    from pg_mcp.observability.metrics import MetricsCollector

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


class QueryOrchestrator:
    """Orchestrates the complete query processing pipeline.

    This class coordinates SQL generation, validation, execution, and result
    validation. It implements retry logic with error feedback and exponential
    backoff, the circuit breaker pattern for fault tolerance, concurrency-based
    rate limiting, low-confidence SQL regeneration, per-request tracing, and
    optional metrics collection.

    Example:
        >>> orchestrator = QueryOrchestrator(
        ...     sql_generator=generator,
        ...     sql_validator=validator,
        ...     executors={"mydb": executor},
        ...     result_validator=result_validator,
        ...     schema_cache=cache,
        ...     pools={"mydb": pool},
        ...     resilience_config=resilience_config,
        ...     validation_config=validation_config,
        ... )
        >>> response = await orchestrator.execute_query(QueryRequest(
        ...     question="How many users?",
        ...     database="mydb"
        ... ))
    """

    def __init__(
        self,
        sql_generator: SQLGenerator,
        sql_validator: SQLValidator,
        executors: dict[str, SQLExecutor],
        result_validator: ResultValidator,
        schema_cache: SchemaCache,
        pools: dict[str, Pool],
        resilience_config: ResilienceConfig,
        validation_config: ValidationConfig,
        rate_limiter: MultiRateLimiter | None = None,
        metrics_collector: "MetricsCollector | None" = None,
    ) -> None:
        """Initialize query orchestrator.

        Args:
            sql_generator: SQL generation service.
            sql_validator: SQL validation service.
            executors: Mapping of database names to their SQL executors (true
                multi-database routing: the request's database picks the entry).
            result_validator: Result validation service.
            schema_cache: Schema cache instance.
            pools: Dictionary mapping database names to connection pools.
            resilience_config: Resilience configuration for retries and circuit breaker.
            validation_config: Validation configuration including thresholds.
            rate_limiter: Optional concurrency limiter for queries and LLM calls.
            metrics_collector: Optional metrics collector; instrumentation is
                skipped entirely when None (e.g. metrics disabled locally).
        """
        self.sql_generator = sql_generator
        self.sql_validator = sql_validator
        self.executors = executors
        self.result_validator = result_validator
        self.schema_cache = schema_cache
        self.pools = pools
        self.resilience_config = resilience_config
        self.validation_config = validation_config
        self.rate_limiter = rate_limiter
        self.metrics_collector = metrics_collector

        # Create circuit breaker for LLM calls
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=resilience_config.circuit_breaker_threshold,
            recovery_timeout=resilience_config.circuit_breaker_timeout,
        )

    async def execute_query(self, request: QueryRequest) -> QueryResponse:
        """Execute complete query flow from question to results.

        This method orchestrates the entire pipeline:
        1. Generate request_id for tracking and enter a request context
        2. Resolve and validate database name, acquire a query slot (rate limit)
        3. Load schema from cache
        4. Generate and validate SQL with retry logic and backoff
        5. Execute SQL against the routed database (if return_type == RESULT)
        6. Validate results and optionally regenerate on low confidence
        7. Return structured response with request_id attached

        Args:
            request: Query request containing question and parameters.

        Returns:
            QueryResponse: Complete response with SQL, results, or error information.
        """
        request_id = _new_request_id()
        async with request_context(request_id):
            response = await self._execute_query_inner(request, request_id)
        return response.model_copy(update={"request_id": request_id})

    async def _execute_query_inner(self, request: QueryRequest, request_id: str) -> QueryResponse:
        """Run the pipeline body with tracing already set up.

        Args:
            request: Query request containing question and parameters.
            request_id: Pre-generated request ID for this invocation.

        Returns:
            QueryResponse: Pipeline outcome (request_id attached by the caller).
        """
        started = time.perf_counter()
        database_name = ""
        status = "error"

        # Concurrency gate: acquire a query slot up-front so the whole pipeline
        # (including execution) is bounded by max_concurrent_queries.
        acquired = await self._acquire_query_slot()
        if not acquired:
            logger.warning(
                "Query rejected: rate limit exceeded",
                extra={"request_id": request_id},
            )
            if self.metrics_collector is not None:
                self.metrics_collector.increment_query_request("rate_limited", database_name)
            return QueryResponse(
                success=False,
                generated_sql=None,
                validation=None,
                data=None,
                error=ErrorDetail(
                    code=ErrorCode.RATE_LIMIT_EXCEEDED,
                    message="Server is busy: too many concurrent queries, try again shortly",
                ),
                confidence=0,
                tokens_used=None,
            )

        try:
            # Step 1: Resolve database name
            database_name = self._resolve_database(request.database)
            logger.debug(
                "Resolved database",
                extra={"request_id": request_id, "database": database_name},
            )

            # Step 2: Get schema from cache
            schema = self.schema_cache.get(database_name)
            if schema is None:
                # Schema not in cache, load it
                pool = self.pools.get(database_name)
                if pool is None:
                    raise DatabaseError(
                        message=f"No connection pool available for database '{database_name}'",
                        details={"database": database_name},
                    )
                try:
                    schema = await self.schema_cache.load(database_name, pool)
                except Exception as e:
                    raise SchemaLoadError(
                        message=f"Failed to load schema for database '{database_name}': {e!s}",
                        details={"database": database_name, "error": str(e)},
                    ) from e

            logger.debug(
                "Schema loaded",
                extra={
                    "request_id": request_id,
                    "database": database_name,
                    "tables": len(schema.tables),
                },
            )

            # Step 3: Generate and validate SQL with retry logic
            generated_sql, validation_result, tokens_used = await self._generate_sql_with_retry(
                question=request.question,
                schema=schema,
                request_id=request_id,
            )

            # Step 4: If return_type is SQL, return early
            if request.return_type == ReturnType.SQL:
                logger.info(
                    "Returning SQL only",
                    extra={"request_id": request_id, "sql_length": len(generated_sql)},
                )
                status = "success"
                return QueryResponse(
                    success=True,
                    generated_sql=generated_sql,
                    validation=validation_result,
                    data=None,
                    error=None,
                    confidence=100,
                    tokens_used=tokens_used,
                )

            # Step 5: Execute SQL against the routed database's executor
            logger.debug("Executing SQL", extra={"request_id": request_id})
            executor = self._executor_for(database_name)
            start_time = self._get_current_time_ms()

            results, total_count = await executor.execute(generated_sql)

            execution_time_ms = self._get_current_time_ms() - start_time
            logger.info(
                "SQL executed successfully",
                extra={
                    "request_id": request_id,
                    "row_count": total_count,
                    "execution_time_ms": execution_time_ms,
                },
            )

            # Step 6: Validate results; regenerate on low confidence within budget
            (
                result_confidence,
                best_sql,
                best_results,
                _best_total_count,
                best_execution_time_ms,
                warning,
            ) = await self._validate_with_regeneration(
                question=request.question,
                generated_sql=generated_sql,
                validation_result=validation_result,
                schema=schema,
                executor=executor,
                results=results,
                total_count=total_count,
                execution_time_ms=execution_time_ms,
                request_id=request_id,
            )

            # Step 7: Build successful response
            query_result = QueryResult(
                columns=list(best_results[0].keys()) if best_results else [],
                rows=best_results,
                row_count=len(best_results),  # Limited row count (after max_rows applied)
                execution_time_ms=best_execution_time_ms,
            )

            status = "success"
            return QueryResponse(
                success=True,
                generated_sql=best_sql,
                validation=validation_result,
                data=query_result,
                error=None,
                confidence=result_confidence,
                tokens_used=tokens_used,
                warning=warning,
            )

        except PgMcpError as e:
            # Handle known application errors
            logger.warning(
                "Query execution failed with known error",
                extra={
                    "request_id": request_id,
                    "error_code": e.code,
                    "error_message": str(e),
                },
            )
            status = e.code.value
            return QueryResponse(
                success=False,
                generated_sql=None,
                validation=None,
                data=None,
                error=ErrorDetail(
                    code=e.code,
                    message=e.message,
                    details=e.details,
                ),
                confidence=0,
                tokens_used=None,
            )
        except Exception as e:
            # Handle unexpected errors
            logger.exception(
                "Query execution failed with unexpected error",
                extra={"request_id": request_id},
            )
            status = "internal_error"
            return QueryResponse(
                success=False,
                generated_sql=None,
                validation=None,
                data=None,
                error=ErrorDetail(
                    code=ErrorCode.INTERNAL_ERROR,
                    message=f"Internal server error: {e!s}",
                    details={"error_type": type(e).__name__},
                ),
                confidence=0,
                tokens_used=None,
            )
        finally:
            self._release_query_slot()
            if self.metrics_collector is not None:
                self.metrics_collector.increment_query_request(status, database_name)
                self.metrics_collector.query_duration.observe(time.perf_counter() - started)

    def _resolve_database(self, database: str | None) -> str:
        """Resolve database name from request or auto-select.

        If database is specified, validate that both a pool and an executor
        exist for it. If not specified and only one database is available,
        auto-select it.

        Args:
            database: Database name from request (optional).

        Returns:
            str: Resolved database name.

        Raises:
            DatabaseError: If database is invalid or cannot be auto-selected.
        """
        if database is not None:
            # Validate specified database exists (routing target + pool)
            if database not in self.pools or database not in self.executors:
                raise DatabaseError(
                    message=f"Database '{database}' not found",
                    details={
                        "requested_database": database,
                        "available_databases": self._available_databases(),
                    },
                )
            return database

        # Auto-select if only one database available
        available_dbs = self._available_databases()
        if len(available_dbs) == 0:
            raise DatabaseError(
                message="No databases configured",
                details={},
            )
        if len(available_dbs) == 1:
            return available_dbs[0]

        # Multiple databases, must specify
        raise DatabaseError(
            message="Multiple databases available, please specify which to query",
            details={"available_databases": available_dbs},
        )

    def _available_databases(self) -> list[str]:
        """List databases usable for routing (pool + executor present).

        Returns:
            list[str]: Sorted names usable as routing targets.
        """
        return sorted(set(self.pools) & set(self.executors))

    def _executor_for(self, database_name: str) -> SQLExecutor:
        """Return the executor configured for the given database.

        Args:
            database_name: Resolved database name.

        Returns:
            SQLExecutor: The executor bound to that database's pool/config.

        Raises:
            DatabaseError: If no executor is registered (should not happen after
                _resolve_database, but kept as a defensive check).
        """
        executor = self.executors.get(database_name)
        if executor is None:
            raise DatabaseError(
                message=f"No SQL executor available for database '{database_name}'",
                details={"database": database_name},
            )
        return executor

    async def _acquire_query_slot(self) -> bool:
        """Acquire a query concurrency slot if a rate limiter is configured.

        Returns:
            bool: True when no limiter is configured or a slot was acquired;
                False when the limiter timed out (caller must return a
                rate-limit response).
        """
        if self.rate_limiter is None:
            return True
        return await self.rate_limiter.query_limiter.acquire(
            timeout=self.resilience_config.rate_limit_timeout,
        )

    def _release_query_slot(self) -> None:
        """Release the query concurrency slot acquired by _acquire_query_slot."""
        if self.rate_limiter is not None:
            self.rate_limiter.query_limiter.release()

    async def _acquire_llm_slot(self) -> bool:
        """Acquire an LLM concurrency slot if a rate limiter is configured.

        Returns:
            bool: True when no limiter is configured or a slot was acquired;
                False when the limiter timed out.
        """
        if self.rate_limiter is None:
            return True
        return await self.rate_limiter.llm_limiter.acquire(
            timeout=self.resilience_config.rate_limit_timeout,
        )

    def _release_llm_slot(self) -> None:
        """Release the LLM concurrency slot acquired by _acquire_llm_slot."""
        if self.rate_limiter is not None:
            self.rate_limiter.llm_limiter.release()

    async def _call_llm_guarded(
        self,
        operation: str,
        request_id: str,
        call: Coroutine[Any, Any, _T],
    ) -> _T:
        """Run one LLM call under the concurrency limiter with metrics.

        Wraps the call in the LLM rate-limit slot, records latency and call
        counts, and propagates token usage from the result.

        Args:
            operation: Metric label for the LLM operation (e.g. "generate_sql").
            request_id: Request ID for logging.
            call: Awaitable invoking the LLM (e.g. returning GenerationResult
                or ResultValidationResult).

        Returns:
            _T: Whatever the awaitable ``call`` resolves to.

        Raises:
            LLMError: If the rate-limit slot cannot be acquired (as a
                retryable LLMUnavailableError) or the call itself fails.
        """
        acquired = await self._acquire_llm_slot()
        if not acquired:
            raise LLMUnavailableError(
                message="LLM concurrency limit reached, try again shortly",
                retryable=True,
            )

        started = time.perf_counter()
        try:
            if self.metrics_collector is not None:
                self.metrics_collector.increment_llm_call(operation)
            result = await call
            if (
                self.metrics_collector is not None
                and isinstance(result, GenerationResult)
                and result.tokens_used is not None
            ):
                self.metrics_collector.increment_llm_tokens(operation, result.tokens_used)
            return result
        finally:
            if self.metrics_collector is not None:
                self.metrics_collector.observe_llm_latency(operation, time.perf_counter() - started)
            self._release_llm_slot()
            logger.debug(
                "LLM call completed",
                extra={"request_id": request_id, "operation": operation},
            )

    async def _generate_sql_with_retry(
        self,
        question: str,
        schema: "DatabaseSchema",
        request_id: str,
    ) -> tuple[str, ValidationResult, int | None]:
        """Generate and validate SQL with retry logic on validation failures.

        This method implements a retry loop that:
        1. Checks circuit breaker state
        2. Generates SQL using LLM (transient LLM failures are retried with
           exponential backoff; fatal ones are raised immediately)
        3. Validates the generated SQL
        4. On validation failure, sleeps for a backoff interval and retries
           with error feedback
        5. Records success/failure to circuit breaker exactly once per attempt

        Args:
            question: User's natural language question.
            schema: Database schema for context.
            request_id: Request ID for tracking.

        Returns:
            tuple: (generated_sql, validation_result, tokens_used)

        Raises:
            LLMError: If circuit breaker is open or generation fails.
            SecurityViolationError: If SQL fails validation after all retries.
            SQLParseError: If SQL cannot be parsed.
        """
        # Check circuit breaker
        if not self.circuit_breaker.allow_request():
            raise LLMError(
                message="SQL generation service is temporarily unavailable (circuit breaker open)",
                details={
                    "circuit_state": self.circuit_breaker.state,
                    "failure_count": self.circuit_breaker.failure_count,
                },
            )

        previous_sql: str | None = None
        error_feedback: str | None = None
        max_retries = self.resilience_config.max_retries
        tokens_used: int | None = None

        for attempt in range(max_retries + 1):
            try:
                logger.debug(
                    "Generating SQL",
                    extra={
                        "request_id": request_id,
                        "attempt": attempt + 1,
                        "max_retries": max_retries + 1,
                    },
                )

                # Generate SQL (with transient-failure backoff retry)
                generation = await self._generate_with_llm_retry(
                    question=question,
                    schema=schema,
                    previous_attempt=previous_sql,
                    error_feedback=error_feedback,
                    request_id=request_id,
                )
                generated_sql = generation.sql
                tokens_used = generation.tokens_used

                logger.debug(
                    "SQL generated",
                    extra={
                        "request_id": request_id,
                        "sql_length": len(generated_sql),
                    },
                )

                # Validate SQL
                try:
                    self.sql_validator.validate_or_raise(generated_sql)
                except (SecurityViolationError, SQLParseError) as validation_error:
                    if attempt < max_retries:
                        # Record as failure and retry with feedback
                        logger.warning(
                            "SQL validation failed, retrying with feedback",
                            extra={
                                "request_id": request_id,
                                "attempt": attempt + 1,
                                "error": str(validation_error),
                            },
                        )
                        previous_sql = generated_sql
                        error_feedback = str(validation_error)
                        # Exponential backoff between validation retries
                        await asyncio.sleep(
                            compute_backoff_delay(
                                attempt=attempt,
                                base_delay=self.resilience_config.retry_delay,
                                backoff_factor=self.resilience_config.backoff_factor,
                            )
                        )
                        continue
                    else:
                        # Out of retries, record failure and raise
                        self.circuit_breaker.record_failure()
                        logger.error(
                            "SQL validation failed after all retries",
                            extra={
                                "request_id": request_id,
                                "attempts": attempt + 1,
                                "error": str(validation_error),
                            },
                        )
                        raise

                # Validation successful
                self.circuit_breaker.record_success()
                logger.info(
                    "SQL generated and validated successfully",
                    extra={
                        "request_id": request_id,
                        "attempts": attempt + 1,
                    },
                )

                # Build validation result
                validation_result = ValidationResult(
                    is_valid=True,
                    is_select=True,
                    allows_data_modification=False,
                    uses_blocked_functions=[],
                    error_message=None,
                )

                return generated_sql, validation_result, tokens_used

            except LLMError:
                # Already accounted for by _generate_with_llm_retry
                raise
            except (SecurityViolationError, SQLParseError):
                # Validation-terminal errors from the block above
                raise
            except Exception as e:
                # Unexpected error during generation
                self.circuit_breaker.record_failure()
                logger.exception(
                    "Unexpected error during SQL generation",
                    extra={"request_id": request_id},
                )
                raise LLMError(
                    message=f"SQL generation failed unexpectedly: {e!s}",
                    details={"error_type": type(e).__name__},
                ) from e

        # Should not reach here, but just in case
        self.circuit_breaker.record_failure()
        raise LLMError(
            message="SQL generation failed after all retry attempts",
            details={"max_retries": max_retries},
        )

    async def _generate_with_llm_retry(
        self,
        question: str,
        schema: "DatabaseSchema",
        previous_attempt: str | None,
        error_feedback: str | None,
        request_id: str,
    ) -> GenerationResult:
        """Call the LLM, retrying transient failures with backoff.

        Transient failures (timeout, connection loss, rate limit, 5xx) are
        retried while retry budget remains; fatal failures (bad credentials)
        are raised immediately. Every failure is recorded on the circuit
        breaker, and one success heals it.

        Args:
            question: User's natural language question.
            schema: Database schema for context.
            previous_attempt: Previously failed SQL (for feedback), if any.
            error_feedback: Error message from the previous attempt, if any.
            request_id: Request ID for tracking.

        Returns:
            GenerationResult: Generated SQL plus token usage.

        Raises:
            LLMError: If the LLM call fails fatally or exhausts retries.
        """
        max_retries = self.resilience_config.max_retries
        last_error: LLMError | None = None

        for llm_attempt in range(max_retries + 1):
            try:
                return await self._call_llm_guarded(
                    operation="generate_sql",
                    request_id=request_id,
                    call=self.sql_generator.generate(
                        question=question,
                        schema=schema,
                        previous_attempt=previous_attempt,
                        error_feedback=error_feedback,
                    ),
                )
            except LLMUnavailableError as e:
                last_error = e
                if not e.retryable or llm_attempt >= max_retries:
                    self.circuit_breaker.record_failure()
                    raise
                logger.warning(
                    "Transient LLM failure, backing off before retry",
                    extra={
                        "request_id": request_id,
                        "attempt": llm_attempt + 1,
                        "error": str(e),
                    },
                )
                self.circuit_breaker.record_failure()
                await asyncio.sleep(
                    compute_backoff_delay(
                        attempt=llm_attempt,
                        base_delay=self.resilience_config.retry_delay,
                        backoff_factor=self.resilience_config.backoff_factor,
                    )
                )
            except LLMError as e:
                # Timeouts are transient like unavailability
                last_error = e
                transient = isinstance(e, LLMTimeoutError) or getattr(e, "retryable", False)
                if not transient or llm_attempt >= max_retries:
                    self.circuit_breaker.record_failure()
                    raise
                logger.warning(
                    "LLM timeout, backing off before retry",
                    extra={
                        "request_id": request_id,
                        "attempt": llm_attempt + 1,
                        "error": str(e),
                    },
                )
                self.circuit_breaker.record_failure()
                await asyncio.sleep(
                    compute_backoff_delay(
                        attempt=llm_attempt,
                        base_delay=self.resilience_config.retry_delay,
                        backoff_factor=self.resilience_config.backoff_factor,
                    )
                )

        self.circuit_breaker.record_failure()
        raise (
            last_error
            if last_error is not None
            else LLMError(
                message="SQL generation failed after all retry attempts",
                details={"max_retries": max_retries},
            )
        )

    async def _validate_with_regeneration(
        self,
        question: str,
        generated_sql: str,
        validation_result: ValidationResult,
        schema: "DatabaseSchema",
        executor: SQLExecutor,
        results: list[dict[str, Any]],
        total_count: int,
        execution_time_ms: float,
        request_id: str,
    ) -> tuple[int, str, list[dict[str, Any]], int, float, str | None]:
        """Validate results, regenerating SQL when confidence is below threshold.

        Runs result validation; while confidence stays below the configured
        threshold and regeneration budget remains, regenerates SQL with
        feedback and re-executes, keeping the best attempt seen. When the
        budget is exhausted, returns the best attempt and a warning.

        Args:
            question: User's original question.
            generated_sql: SQL that produced ``results``.
            validation_result: SQL validation result (echoed unchanged).
            schema: Database schema (context for regeneration).
            executor: Executor to re-run regenerated SQL.
            results: Rows from the first execution.
            total_count: Total row count from the first execution.
            execution_time_ms: First execution duration in ms.
            request_id: Request ID for tracking.

        Returns:
            tuple: (confidence, best_sql, best_results, best_total_count,
                best_execution_time_ms, warning).
        """
        best_sql = generated_sql
        best_results = results
        best_total_count = total_count
        best_execution_time_ms = execution_time_ms

        confidence = await self._validate_results_safely(
            question=question,
            sql=best_sql,
            results=best_results,
            row_count=best_total_count,
            request_id=request_id,
        )
        best_confidence = confidence

        regen_budget = self.validation_config.max_regeneration_attempts
        threshold = self.validation_config.confidence_threshold
        regens_used = 0
        current_sql = generated_sql
        warning: str | None = None

        while best_confidence < threshold and regens_used < regen_budget:
            regens_used += 1
            logger.info(
                "Low confidence, regenerating SQL with feedback",
                extra={
                    "request_id": request_id,
                    "confidence": best_confidence,
                    "threshold": threshold,
                    "regeneration": regens_used,
                },
            )

            try:
                generation = await self._call_llm_guarded(
                    operation="regenerate_sql",
                    request_id=request_id,
                    call=self.sql_generator.generate(
                        question=question,
                        schema=schema,
                        previous_attempt=current_sql,
                        error_feedback=(
                            f"Previous SQL returned results with low confidence "
                            f"({best_confidence}% < {threshold}%). "
                            f"Generate a more accurate query."
                        ),
                    ),
                )
                self.sql_validator.validate_or_raise(generation.sql)

                regen_started = self._get_current_time_ms()
                new_results, new_total = await executor.execute(generation.sql)
                new_execution_time_ms = self._get_current_time_ms() - regen_started
            except PgMcpError as e:
                # Regeneration failure keeps the previous best result
                logger.warning(
                    "SQL regeneration failed, keeping previous result",
                    extra={"request_id": request_id, "error": str(e)},
                )
                break

            new_confidence = await self._validate_results_safely(
                question=question,
                sql=generation.sql,
                results=new_results,
                row_count=new_total,
                request_id=request_id,
            )

            if new_confidence >= best_confidence:
                best_sql = generation.sql
                best_results = new_results
                best_total_count = new_total
                best_execution_time_ms = new_execution_time_ms
                best_confidence = new_confidence

            current_sql = generation.sql

        if best_confidence < threshold:
            warning = (
                f"Result confidence {best_confidence}% is below the acceptable "
                f"threshold {threshold}%"
            )
            if best_confidence < 50:
                warning += (
                    "; the generated query may not match the question - "
                    "consider rephrasing the question"
                )

        return (
            best_confidence,
            best_sql,
            best_results,
            best_total_count,
            best_execution_time_ms,
            warning,
        )

    async def _validate_results_safely(
        self,
        question: str,
        sql: str,
        results: list[dict[str, Any]],
        row_count: int,
        request_id: str,
    ) -> int:
        """Validate query results with error handling (non-blocking).

        This method attempts to validate results using LLM, but failures
        don't cause the overall query to fail. Returns a confidence score.

        Args:
            question: User's original question.
            sql: Generated SQL query.
            results: Query results.
            row_count: Total row count.
            request_id: Request ID for tracking.

        Returns:
            int: Confidence score (0-100). Returns 100 if validation disabled/fails.
        """
        if not self.validation_config.enabled:
            return 100

        try:
            logger.debug(
                "Validating results",
                extra={"request_id": request_id},
            )

            validation_result = await self._call_llm_guarded(
                operation="validate_result",
                request_id=request_id,
                call=self.result_validator.validate(
                    question=question,
                    sql=sql,
                    results=results,
                    row_count=row_count,
                ),
            )

            logger.info(
                "Result validation completed",
                extra={
                    "request_id": request_id,
                    "confidence": validation_result.confidence,
                    "is_acceptable": validation_result.is_acceptable,
                },
            )

            return validation_result.confidence

        except Exception as e:
            # Log but don't fail the query
            logger.warning(
                "Result validation failed, continuing with default confidence",
                extra={
                    "request_id": request_id,
                    "error": str(e),
                },
            )
            return 100  # Default to high confidence if validation fails

    @staticmethod
    def _get_current_time_ms() -> float:
        """Get current time in milliseconds.

        Returns:
            float: Current time in milliseconds since epoch.
        """
        return time.time() * 1000


def _new_request_id() -> str:
    """Generate a fresh request ID.

    Returns:
        str: UUID4-based request ID.
    """
    import uuid

    return str(uuid.uuid4())
