"""Unit tests for tracing utilities.

Covers request ID generation, context variable propagation, request_context,
trace decorators, and TracingLogger.
"""

import logging

from pg_mcp.observability.tracing import (
    TracingLogger,
    clear_request_id,
    generate_request_id,
    get_request_id,
    get_tracing_logger,
    request_context,
    set_request_id,
    trace_async,
    trace_sync,
)


class TestRequestIdHelpers:
    """Tests for request ID generation and context get/set/clear."""

    def test_generate_request_id_is_uuid(self) -> None:
        request_id = generate_request_id()
        assert len(request_id) == 36
        assert request_id.count("-") == 4

    def test_generate_request_id_is_unique(self) -> None:
        assert generate_request_id() != generate_request_id()

    def test_default_request_id_is_none(self) -> None:
        clear_request_id()
        assert get_request_id() is None

    def test_set_and_get_request_id(self) -> None:
        set_request_id("custom-id")
        try:
            assert get_request_id() == "custom-id"
        finally:
            clear_request_id()

    def test_clear_request_id(self) -> None:
        set_request_id("id")
        clear_request_id()
        assert get_request_id() is None


class TestRequestContext:
    """Tests for the request_context context manager."""

    async def test_generates_id_when_absent(self) -> None:
        clear_request_id()
        async with request_context() as request_id:
            assert request_id is not None
            assert get_request_id() == request_id
        assert get_request_id() is None

    async def test_restores_previous_id_after_exit(self) -> None:
        set_request_id("outer")
        try:
            async with request_context("inner"):
                assert get_request_id() == "inner"
            assert get_request_id() == "outer"
        finally:
            clear_request_id()

    async def test_context_propagates_into_tasks(self) -> None:
        import asyncio

        seen: dict[str, str | None] = {}

        async def child() -> None:
            seen["request_id"] = get_request_id()

        async with request_context("parent-id"):
            await asyncio.create_task(child())

        assert seen["request_id"] == "parent-id"


class TestTraceDecorators:
    """Tests for the trace_async / trace_sync decorators."""

    async def test_trace_async_passes_through_return_value(self) -> None:
        @trace_async(operation="op")
        async def work(value: int) -> int:
            return value * 2

        assert await work(21) == 42

    async def test_trace_async_without_context_still_works(self) -> None:
        clear_request_id()

        @trace_async()
        async def work() -> str:
            return "ok"

        assert await work() == "ok"

    def test_trace_sync_return_value(self) -> None:
        @trace_sync(operation="sync-op")
        def work(value: int) -> int:
            return value + 1

        assert work(1) == 2

    def test_trace_sync_restores_log_record_factory(self) -> None:
        @trace_sync()
        def work() -> None:
            return None

        before = logging.getLogRecordFactory()
        work()
        assert logging.getLogRecordFactory() is before

    async def test_trace_async_restores_log_record_factory(self) -> None:
        @trace_async()
        async def work() -> None:
            return None

        before = logging.getLogRecordFactory()
        await work()
        assert logging.getLogRecordFactory() is before


class TestTracingLogger:
    """Tests for the context-aware logger wrapper."""

    def test_includes_request_id_in_extra(self) -> None:
        tlog = get_tracing_logger("tracing-test")
        records: list[logging.LogRecord] = []

        handler = logging.Handler()
        handler.emit = records.append  # type: ignore[method-assign]
        tlog._logger.addHandler(handler)
        tlog._logger.setLevel(logging.INFO)

        set_request_id("req-9")
        try:
            tlog.info("message here")
        finally:
            clear_request_id()

        assert len(records) == 1
        assert getattr(records[0], "request_id", None) == "req-9"

    def test_existing_extra_request_id_not_overwritten(self) -> None:
        tlog = TracingLogger("tracing-test")
        records: list[logging.LogRecord] = []

        handler = logging.Handler()
        handler.emit = records.append  # type: ignore[method-assign]
        tlog._logger.addHandler(handler)
        tlog._logger.setLevel(logging.INFO)

        set_request_id("auto")
        try:
            tlog.info("msg", extra={"request_id": "explicit"})
        finally:
            clear_request_id()

        assert getattr(records[0], "request_id", None) == "explicit"

    def test_exception_logs_traceback(self) -> None:
        tlog = TracingLogger("tracing-test")
        records: list[logging.LogRecord] = []

        handler = logging.Handler()
        handler.emit = records.append  # type: ignore[method-assign]
        tlog._logger.addHandler(handler)
        tlog._logger.setLevel(logging.ERROR)

        try:
            raise ValueError("kaboom")
        except ValueError:
            tlog.exception("failed")

        assert records[0].exc_info is not None
