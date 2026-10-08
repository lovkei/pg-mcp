"""Unit tests for the MetricsCollector helper methods.

The singleton is reset per-test via conftest; here we verify each wrapper
method touches the underlying Prometheus metric.
"""

from pg_mcp.observability.metrics import MetricsCollector


def _collector() -> MetricsCollector:
    return MetricsCollector()


class TestQueryMetrics:
    """Query request counter and duration histogram."""

    def test_increment_query_request(self) -> None:
        m = _collector()
        m.increment_query_request(status="success", database="db1")

        assert m.query_requests.labels(status="success", database="db1")._value.get() == 1

    def test_observe_query_duration(self) -> None:
        m = _collector()
        m.query_duration.observe(0.25)
        assert m.query_duration._sum.get() == 0.25


class TestLLMMetrics:
    """LLM call/latency/token metrics."""

    def test_increment_llm_call(self) -> None:
        m = _collector()
        m.increment_llm_call("generate_sql")
        assert m.llm_calls.labels(operation="generate_sql")._value.get() == 1

    def test_observe_llm_latency(self) -> None:
        m = _collector()
        m.observe_llm_latency("generate_sql", 1.5)
        assert m.llm_latency.labels(operation="generate_sql")._sum.get() == 1.5

    def test_increment_llm_tokens(self) -> None:
        m = _collector()
        m.increment_llm_tokens("generate_sql", 120)
        assert m.llm_tokens_used.labels(operation="generate_sql")._value.get() == 120


class TestSecurityAndDbMetrics:
    """Rejection counter, connection gauge, and DB latency."""

    def test_increment_sql_rejected(self) -> None:
        m = _collector()
        m.increment_sql_rejected("blocked_table")
        assert m.sql_rejected.labels(reason="blocked_table")._value.get() == 1

    def test_set_db_connections_active(self) -> None:
        m = _collector()
        m.set_db_connections_active("db1", 7)
        assert m.db_connections_active.labels(database="db1")._value.get() == 7

    def test_observe_db_query_duration(self) -> None:
        m = _collector()
        m.observe_db_query_duration(0.05)
        assert m.db_query_duration._sum.get() == 0.05

    def test_set_schema_cache_age(self) -> None:
        m = _collector()
        m.set_schema_cache_age("db1", 42.0)
        assert m.schema_cache_age.labels(database="db1")._value.get() == 42.0


class TestSingleton:
    """Singleton identity behavior."""

    def test_same_instance_returned(self) -> None:
        assert MetricsCollector() is MetricsCollector()
