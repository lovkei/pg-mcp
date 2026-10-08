"""Unit tests for structured logging utilities.

Covers the sensitive-data filter, JSON/text formatters, and configure_logging.
"""

import json
import logging

import pytest

from pg_mcp.observability.logging import (
    JSONFormatter,
    SensitiveDataFilter,
    TextFormatter,
    configure_logging,
    get_logger,
)


def _record(msg: str = "hello", **extra: object) -> logging.LogRecord:
    """Build a LogRecord with extra attributes set, as logging.Logger does."""
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname=__file__, lineno=1,
        msg=msg, args=(), exc_info=None, func=None,
    )
    for key, value in extra.items():
        setattr(record, key, value)
    return record


class TestSensitiveDataFilter:
    """Tests for log record sanitization."""

    def setup_method(self) -> None:
        self.filt = SensitiveDataFilter()

    def test_redacts_sensitive_extra_fields(self) -> None:
        record = _record(password="hunter2", token="abc123")
        assert self.filt.filter(record) is True
        assert record.password == "***REDACTED***"
        assert record.token == "***REDACTED***"

    def test_redacts_nested_dict_values(self) -> None:
        record = _record(payload={"user": "bob", "api_key": "sk-secret"})
        self.filt.filter(record)
        assert record.payload == {"user": "bob", "api_key": "***REDACTED***"}

    def test_sanitizes_args_tuples(self) -> None:
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname=__file__, lineno=1,
            msg="connect with %s", args=("secret",), exc_info=None, func=None,
        )
        self.filt.filter(record)
        assert record.args == ("secret",)  # non-dict args pass through unchanged

    def test_sanitizes_args_dict(self) -> None:
        record = _record(msg="connect with %(pwd)s")
        record.args = {"pwd": "hunter2"}
        self.filt.filter(record)
        assert record.args == {"pwd": "***REDACTED***"}

    def test_dict_containing_list_is_sanitized(self) -> None:
        # Top-level extras are sanitized when dict-valued; lists inside them
        # are walked recursively.
        record = _record(items={"rows": [{"token": "t1"}, {"name": "n1"}]})
        self.filt.filter(record)
        assert record.items == {"rows": [{"token": "***REDACTED***"}, {"name": "n1"}]}


class TestJSONFormatter:
    """Tests for JSON log formatting."""

    def setup_method(self) -> None:
        self.formatter = JSONFormatter()

    def test_basic_fields_present(self) -> None:
        data = json.loads(self.formatter.format(_record()))
        assert data["level"] == "INFO"
        assert data["message"] == "hello"
        assert data["logger"] == "test"

    def test_request_id_included_when_present(self) -> None:
        data = json.loads(self.formatter.format(_record(request_id="req-42")))
        assert data["request_id"] == "req-42"

    def test_extra_fields_grouped(self) -> None:
        data = json.loads(self.formatter.format(_record(database="db1")))
        # Python 3.12+ adds a taskName attribute to log records.
        extra = {k: v for k, v in data["extra"].items() if k != "taskName"}
        assert extra == {"database": "db1"}

    def test_exception_included(self) -> None:
        try:
            raise ValueError("boom")
        except ValueError:
            import sys

            record = _record()
            record.exc_info = sys.exc_info()
        formatted = self.formatter.format(record)
        assert "boom" in formatted


class TestTextFormatter:
    """Tests for human-readable log formatting."""

    def setup_method(self) -> None:
        self.formatter = TextFormatter()

    def test_basic_format(self) -> None:
        formatted = self.formatter.format(_record())
        assert "[INFO]" in formatted
        assert "test - hello" in formatted

    def test_request_id_appended(self) -> None:
        formatted = self.formatter.format(_record(request_id="req-7"))
        assert "[request_id=req-7]" in formatted


class TestConfigureLogging:
    """Tests for the logging entry point."""

    def teardown_method(self) -> None:
        # Restore a clean root logger so other tests are unaffected.
        root = logging.getLogger()
        for handler in root.handlers[:]:
            root.removeHandler(handler)

    def test_configure_json_adds_handler(self) -> None:
        configure_logging(level="INFO", log_format="json")
        root = logging.getLogger()
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0].formatter, JSONFormatter)
        assert root.level == logging.INFO

    def test_configure_text_adds_text_formatter(self) -> None:
        configure_logging(level="debug", log_format="text")
        root = logging.getLogger()
        assert isinstance(root.handlers[0].formatter, TextFormatter)
        assert root.level == logging.DEBUG

    def test_sensitive_filter_added_by_default(self) -> None:
        configure_logging(enable_sensitive_filter=True)
        assert any(
            isinstance(f, SensitiveDataFilter) for f in logging.getLogger().handlers[0].filters
        )

    def test_sensitive_filter_omitted_when_disabled(self) -> None:
        configure_logging(enable_sensitive_filter=False)
        assert logging.getLogger().handlers[0].filters == []

    def test_third_party_loggers_quieted(self) -> None:
        configure_logging()
        assert logging.getLogger("asyncpg").level == logging.WARNING
        assert logging.getLogger("openai").level == logging.WARNING


def test_get_logger_returns_named_logger() -> None:
    logger = get_logger("some.module")
    assert logger.name == "some.module"


@pytest.mark.parametrize("fmt", ["json", "text"])
def test_formatters_roundtrip_through_handler(fmt: str) -> None:
    """A record flowing through a configured handler produces valid output."""
    import io

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JSONFormatter() if fmt == "json" else TextFormatter())
    logger = logging.getLogger("roundtrip")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    try:
        logger.info("round trip %s", "message")
    finally:
        logger.removeHandler(handler)

    output = stream.getvalue()
    assert "round trip message" in output
