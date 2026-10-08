"""Unit tests for the result validator service.

This module tests ResultValidator with a mocked OpenAI client: confidence
parsing, threshold comparison, disabled-validation fast path, JSON error
handling, and OpenAI exception mapping (including the APITimeoutError
regression where the old ``except TimeoutError`` branch was dead code).
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest
from pydantic import SecretStr

from pg_mcp.config.settings import OpenAIConfig, ValidationConfig
from pg_mcp.models.errors import LLMError, LLMTimeoutError, LLMUnavailableError
from pg_mcp.prompts.result_validation import build_validation_prompt
from pg_mcp.services.result_validator import ResultValidator


def _validator(
    *,
    enabled: bool = True,
    threshold: int = 70,
    sample_rows: int = 5,
    timeout_seconds: float = 10.0,
) -> ResultValidator:
    """Create a ResultValidator with test configuration."""
    openai_config = OpenAIConfig(api_key=SecretStr("sk-test-key-12345"))
    validation_config = ValidationConfig(
        enabled=enabled,
        confidence_threshold=threshold,
        sample_rows=sample_rows,
        timeout_seconds=timeout_seconds,
    )
    return ResultValidator(openai_config=openai_config, validation_config=validation_config)


def _response(content: str | None, *, choices: int = 1) -> MagicMock:
    """Build a mock ChatCompletion response."""
    response = MagicMock()
    if choices == 0:
        response.choices = []
    else:
        response.choices = [MagicMock(message=MagicMock(content=content))]
    return response


async def _validate_with_response(
    validator: ResultValidator, response: MagicMock
) -> object:
    """Run validate() with the client patched to return the given response."""
    with patch.object(
        validator.client.chat.completions, "create", new=AsyncMock(return_value=response)
    ):
        return await validator.validate(
            question="How many users are there?",
            sql="SELECT COUNT(*) FROM users",
            results=[{"count": 42}],
            row_count=1,
        )


async def _validate_with_error(
    validator: ResultValidator, error: Exception
) -> object:
    """Run validate() with the client patched to raise the given error."""
    with patch.object(
        validator.client.chat.completions, "create", new=AsyncMock(side_effect=error)
    ):
        return await validator.validate(
            question="How many users are there?",
            sql="SELECT COUNT(*) FROM users",
            results=[{"count": 42}],
            row_count=1,
        )


class TestDisabledValidation:
    """Tests for the validation-disabled fast path."""

    async def test_disabled_returns_full_confidence(self) -> None:
        """When disabled, skip the LLM entirely and report full confidence."""
        validator = _validator(enabled=False)

        with patch.object(
            validator.client.chat.completions,
            "create",
            new=AsyncMock(side_effect=AssertionError("LLM must not be called")),
        ):
            result = await validator.validate(
                question="q", sql="SELECT 1", results=[{"a": 1}], row_count=1
            )

        assert result.confidence == 100
        assert result.is_acceptable is True
        assert "disabled" in result.explanation.lower()
        assert result.suggestion is None


class TestSuccessfulValidation:
    """Tests for the happy-path validation flow."""

    async def test_high_confidence_result(self) -> None:
        """A well-formed high-confidence response is acceptable."""
        validator = _validator()
        content = json.dumps(
            {"confidence": 95, "explanation": "Count matches", "suggestion": None}
        )
        result = await _validate_with_response(validator, _response(content))

        assert result.confidence == 95
        assert result.explanation == "Count matches"
        assert result.suggestion is None
        assert result.is_acceptable is True

    async def test_confidence_at_threshold_is_acceptable(self) -> None:
        """Confidence exactly equal to the threshold is acceptable (>=)."""
        validator = _validator(threshold=70)
        content = json.dumps({"confidence": 70, "explanation": "ok"})
        result = await _validate_with_response(validator, _response(content))

        assert result.confidence == 70
        assert result.is_acceptable is True

    async def test_confidence_below_threshold_not_acceptable(self) -> None:
        """Confidence below the threshold is not acceptable."""
        validator = _validator(threshold=70)
        content = json.dumps({"confidence": 69, "explanation": "uncertain"})
        result = await _validate_with_response(validator, _response(content))

        assert result.is_acceptable is False

    async def test_suggestion_is_passed_through(self) -> None:
        """A non-null suggestion from the LLM is preserved."""
        validator = _validator()
        content = json.dumps(
            {
                "confidence": 80,
                "explanation": "looks right",
                "suggestion": "Add a date filter",
            }
        )
        result = await _validate_with_response(validator, _response(content))

        assert result.suggestion == "Add a date filter"

    async def test_sample_rows_limits_results_sent(self) -> None:
        """Only sample_rows rows are sent to the LLM."""
        validator = _validator(sample_rows=2)
        results = [{"id": i} for i in range(10)]
        content = json.dumps({"confidence": 90, "explanation": "ok"})

        with patch.object(
            validator.client.chat.completions,
            "create",
            new=AsyncMock(return_value=_response(content)),
        ) as mock_create:
            await validator.validate(
                question="q", sql="SELECT * FROM t", results=results, row_count=10
            )

        prompt = mock_create.call_args.kwargs["messages"][1]["content"]
        assert "showing 2 of 10 rows" in prompt

    async def test_request_uses_json_response_format(self) -> None:
        """The API call asks for deterministic JSON output."""
        validator = _validator()
        content = json.dumps({"confidence": 90, "explanation": "ok"})

        with patch.object(
            validator.client.chat.completions,
            "create",
            new=AsyncMock(return_value=_response(content)),
        ) as mock_create:
            await validator.validate(
                question="q", sql="SELECT 1", results=[], row_count=0
            )

        kwargs = mock_create.call_args.kwargs
        assert kwargs["response_format"] == {"type": "json_object"}
        assert kwargs["temperature"] == 0.0


class TestResponseParsing:
    """Tests for malformed / edge-case LLM responses."""

    async def test_empty_choices_raises_llm_error(self) -> None:
        """An empty choices list raises LLMError."""
        validator = _validator()
        with pytest.raises(LLMError, match="empty response"):
            await _validate_with_response(validator, _response(None, choices=0))

    async def test_empty_content_raises_llm_error(self) -> None:
        """None message content raises LLMError."""
        validator = _validator()
        with pytest.raises(LLMError, match="empty message content"):
            await _validate_with_response(validator, _response(None))

    async def test_invalid_json_returns_moderate_confidence(self) -> None:
        """Unparseable JSON degrades to confidence 60, not acceptable."""
        validator = _validator()
        result = await _validate_with_response(validator, _response("not json at all"))

        assert result.confidence == 60
        assert result.is_acceptable is False
        assert "parsing failed" in result.explanation.lower()
        assert result.suggestion is not None
        assert "manual verification" in result.suggestion.lower()

    async def test_non_numeric_confidence_defaults_to_50(self) -> None:
        """A non-numeric confidence value falls back to 50."""
        validator = _validator(threshold=50)
        content = json.dumps({"confidence": "high", "explanation": "ok"})
        result = await _validate_with_response(validator, _response(content))

        assert result.confidence == 50
        assert result.is_acceptable is True

    async def test_out_of_range_confidence_is_clamped(self) -> None:
        """Out-of-range numeric confidence is clamped into [0, 100]."""
        validator = _validator()
        content = json.dumps({"confidence": 180, "explanation": "ok"})
        result = await _validate_with_response(validator, _response(content))

        assert result.confidence == 100

    async def test_float_confidence_is_clamped_to_int(self) -> None:
        """Float confidence values are clamped into the valid range as ints."""
        validator = _validator()
        content = json.dumps({"confidence": 55.5, "explanation": "ok"})
        result = await _validate_with_response(validator, _response(content))

        assert result.confidence == 55

    async def test_missing_fields_use_defaults(self) -> None:
        """Missing confidence/explanation/suggestion fall back to defaults."""
        validator = _validator(threshold=50)
        result = await _validate_with_response(validator, _response("{}"))

        assert result.confidence == 50
        assert result.explanation == "No explanation provided"
        assert result.suggestion is None


class TestExceptionMapping:
    """Tests mapping OpenAI exceptions to project LLM errors."""

    async def test_api_timeout_maps_to_llm_timeout(self) -> None:
        """Regression: openai.APITimeoutError maps to LLMTimeoutError.

        openai.APITimeoutError does NOT inherit the built-in TimeoutError, so
        an ``except TimeoutError`` branch alone is dead code.
        """
        validator = _validator(timeout_seconds=15.0)
        error = openai.APITimeoutError(request=MagicMock())

        with pytest.raises(LLMTimeoutError) as exc_info:
            await _validate_with_error(validator, error)

        assert "timed out" in str(exc_info.value).lower()
        assert exc_info.value.details["timeout"] == 15.0

    async def test_authentication_error_not_retryable(self) -> None:
        """AuthenticationError maps to a non-retryable LLMUnavailableError."""
        validator = _validator()
        error = openai.AuthenticationError(
            message="Invalid API key",
            response=MagicMock(status_code=401, headers={}),
            body=None,
        )

        with pytest.raises(LLMUnavailableError) as exc_info:
            await _validate_with_error(validator, error)

        assert "authentication" in str(exc_info.value).lower()
        assert exc_info.value.retryable is False

    async def test_permission_denied_error_not_retryable(self) -> None:
        """PermissionDeniedError maps to a non-retryable LLMUnavailableError."""
        validator = _validator()
        error = openai.PermissionDeniedError(
            message="Not allowed",
            response=MagicMock(status_code=403, headers={}),
            body=None,
        )

        with pytest.raises(LLMUnavailableError) as exc_info:
            await _validate_with_error(validator, error)

        assert "permission denied" in str(exc_info.value).lower()
        assert exc_info.value.retryable is False

    async def test_rate_limit_error_retryable(self) -> None:
        """RateLimitError maps to a retryable LLMUnavailableError."""
        validator = _validator()
        error = openai.RateLimitError(
            message="Rate limit exceeded",
            response=MagicMock(status_code=429, headers={}),
            body=None,
        )

        with pytest.raises(LLMUnavailableError) as exc_info:
            await _validate_with_error(validator, error)

        assert "rate limit" in str(exc_info.value).lower()
        assert exc_info.value.retryable is True

    async def test_connection_error_retryable(self) -> None:
        """APIConnectionError maps to a retryable LLMUnavailableError."""
        validator = _validator()
        error = openai.APIConnectionError(request=MagicMock())

        with pytest.raises(LLMUnavailableError) as exc_info:
            await _validate_with_error(validator, error)

        assert "cannot connect" in str(exc_info.value).lower()
        assert exc_info.value.retryable is True

    async def test_server_error_retryable(self) -> None:
        """A 5xx APIStatusError maps to a retryable LLMUnavailableError."""
        validator = _validator()
        error = openai.InternalServerError(
            message="Server error",
            response=MagicMock(status_code=503, headers={}),
            body=None,
        )

        with pytest.raises(LLMUnavailableError) as exc_info:
            await _validate_with_error(validator, error)

        assert "503" in str(exc_info.value)
        assert exc_info.value.retryable is True

    async def test_client_status_error_raises_llm_error(self) -> None:
        """A 4xx APIStatusError (non-5xx) maps to plain LLMError."""
        validator = _validator()
        error = openai.BadRequestError(
            message="Bad request",
            response=MagicMock(status_code=400, headers={}),
            body=None,
        )

        with pytest.raises(LLMError) as exc_info:
            await _validate_with_error(validator, error)

        assert "400" in str(exc_info.value)

    async def test_generic_exception_wrapped_in_llm_error(self) -> None:
        """Any unexpected exception is wrapped in LLMError."""
        validator = _validator()

        with pytest.raises(LLMError, match="Result validation failed"):
            await _validate_with_error(validator, ValueError("boom"))

    async def test_llm_error_passes_through_unchanged(self) -> None:
        """LLMError raised inside the try block is re-raised as-is."""
        validator = _validator()
        # Empty choices raise LLMError from inside the try block.
        with pytest.raises(LLMError, match="empty response"):
            await _validate_with_response(validator, _response(None, choices=0))


class TestValidationPromptBuilder:
    """Tests for the shared validation prompt builder."""

    def test_prompt_contains_question_sql_and_results(self) -> None:
        """The prompt embeds question, SQL, and a result preview."""
        prompt = build_validation_prompt(
            question="How many users?",
            sql="SELECT COUNT(*) FROM users",
            results=[{"count": 42}],
            row_count=1,
        )

        assert "How many users?" in prompt
        assert "SELECT COUNT(*) FROM users" in prompt
        assert '"count": 42' in prompt
        assert "showing 1 of 1 rows" in prompt

    def test_prompt_reports_total_row_count(self) -> None:
        """The prompt shows the total row count vs sampled rows."""
        prompt = build_validation_prompt(
            question="q",
            sql="SELECT * FROM t",
            results=[{"id": 1}, {"id": 2}],
            row_count=100,
        )

        assert "showing 2 of 100 rows" in prompt
