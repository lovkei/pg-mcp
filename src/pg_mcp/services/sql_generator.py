"""SQL generation service using OpenAI for natural language to SQL conversion.

This module provides the SQLGenerator class that uses OpenAI's LLM to convert
natural language questions into valid PostgreSQL SQL queries.
"""

import re
from typing import TYPE_CHECKING

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    PermissionDeniedError,
    RateLimitError,
)

from pg_mcp.config.settings import OpenAIConfig
from pg_mcp.models.errors import LLMError, LLMTimeoutError, LLMUnavailableError
from pg_mcp.models.generation import GenerationResult
from pg_mcp.prompts.sql_generation import SQL_GENERATION_SYSTEM_PROMPT, build_user_prompt

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion

    from pg_mcp.models.schema import DatabaseSchema

# Pre-compiled extraction patterns (module-level: compiled once, reused per call)
_CODE_BLOCK_RE = re.compile(r"```(?:sql)?\s*\n?(.*?)\n?```", re.DOTALL | re.IGNORECASE)
_SQL_STATEMENT_RE = re.compile(r"((?:WITH|SELECT)\s+.*?)(?:;|$)", re.DOTALL | re.IGNORECASE)


class SQLGenerator:
    """SQL generator using OpenAI for natural language to SQL conversion.

    This class handles the interaction with OpenAI's API to generate SQL queries
    from natural language questions. It includes robust error handling, SQL extraction
    from various response formats, and support for retry scenarios with error feedback.

    Example:
        >>> config = OpenAIConfig(api_key="sk-...", model="gpt-4")
        >>> generator = SQLGenerator(config)
        >>> result = await generator.generate(
        ...     question="How many users registered today?",
        ...     schema=db_schema
        ... )
        >>> result.sql
        'SELECT COUNT(*) FROM users WHERE ...'
    """

    def __init__(self, config: OpenAIConfig) -> None:
        """Initialize SQL generator with OpenAI configuration.

        Args:
            config: OpenAI configuration including API key and model settings.
        """
        self.config = config
        self.client = AsyncOpenAI(
            api_key=config.api_key.get_secret_value(),
            base_url=config.base_url,
            timeout=config.timeout,
        )

    async def generate(
        self,
        question: str,
        schema: "DatabaseSchema",
        context: str | None = None,
        previous_attempt: str | None = None,
        error_feedback: str | None = None,
    ) -> GenerationResult:
        """Generate SQL statement from natural language question.

        This method sends the question and database schema to OpenAI's API
        and extracts the generated SQL query from the response. It supports
        retry scenarios by accepting previous failed attempts and error feedback.

        Args:
            question: User's natural language question.
            schema: Database schema information for context.
            context: Optional additional context to guide generation.
            previous_attempt: Previously generated SQL that failed (for retry).
            error_feedback: Error message from previous attempt (for retry).

        Returns:
            GenerationResult: Generated SQL plus token usage (None when the
                API response carries no usage information).

        Raises:
            LLMError: If generation fails or response is invalid.
            LLMTimeoutError: If the API request times out (retryable).
            LLMUnavailableError: If the API is unreachable, rate limited, or 5xx (retryable),
                or authentication fails (not retryable).

        Example:
            >>> # Initial generation
            >>> result = await generator.generate(
            ...     question="Count all active users",
            ...     schema=db_schema
            ... )
            >>> # Retry with error feedback
            >>> result = await generator.generate(
            ...     question="Count all active users",
            ...     schema=db_schema,
            ...     previous_attempt="SELECT COUNT(*) FROM user",
            ...     error_feedback='relation "user" does not exist'
            ... )
        """
        user_prompt = build_user_prompt(
            question=question,
            schema=schema,
            context=context,
            previous_attempt=previous_attempt,
            error_feedback=error_feedback,
        )

        try:
            response: ChatCompletion = await self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": SQL_GENERATION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
        except APITimeoutError as e:
            # openai.APITimeoutError does NOT inherit the built-in TimeoutError,
            # so this must be caught explicitly (previously a dead branch).
            raise LLMTimeoutError(
                message=f"OpenAI API request timed out after {self.config.timeout}s",
                details={"timeout": self.config.timeout},
            ) from e
        except AuthenticationError as e:
            raise LLMUnavailableError(
                message="OpenAI API authentication failed - check API key",
                retryable=False,
            ) from e
        except PermissionDeniedError as e:
            raise LLMUnavailableError(
                message="OpenAI API permission denied for this key/model",
                retryable=False,
            ) from e
        except RateLimitError as e:
            raise LLMUnavailableError(
                message="OpenAI API rate limit exceeded",
                retryable=True,
            ) from e
        except APIConnectionError as e:
            raise LLMUnavailableError(
                message="Cannot connect to OpenAI API",
                retryable=True,
            ) from e
        except APIStatusError as e:
            # 5xx responses are transient server-side failures -> retryable
            if e.status_code >= 500:
                raise LLMUnavailableError(
                    message=f"OpenAI API server error (HTTP {e.status_code})",
                    retryable=True,
                ) from e
            raise LLMError(
                message=f"OpenAI API request failed with status {e.status_code}",
            ) from e
        except LLMError:
            raise
        except Exception as e:
            raise LLMError(
                message=f"OpenAI API request failed: {e}",
            ) from e

        # Extract token usage if present
        tokens_used: int | None = None
        if response.usage is not None and response.usage.total_tokens is not None:
            tokens_used = response.usage.total_tokens

        # Extract SQL from response
        if not response.choices:
            raise LLMError(
                message="OpenAI returned empty response",
                details={"tokens_used": tokens_used},
            )

        content = response.choices[0].message.content
        if not content:
            raise LLMError(
                message="OpenAI returned empty message content",
                details={"tokens_used": tokens_used},
            )

        sql = self._extract_sql(content)
        if not sql:
            raise LLMError(
                message="Failed to extract SQL from OpenAI response",
                details={"content": content},
            )

        return GenerationResult(sql=sql, tokens_used=tokens_used)

    def _extract_sql(self, content: str) -> str | None:
        """Extract SQL query from LLM response content.

        This method implements a multi-strategy approach to extract SQL from
        various response formats that LLMs might generate:

        1. Try to match ```sql ... ``` code blocks (preferred format)
        2. Try to match generic ``` ... ``` code blocks
        3. Try to find SELECT/WITH statements in plain text
        4. Check if entire content looks like SQL

        Args:
            content: Raw content from LLM response.

        Returns:
            str | None: Extracted SQL query, or None if extraction fails.

        Example:
            >>> generator._extract_sql("```sql\\nSELECT 1;\\n```")
            'SELECT 1;'
            >>> generator._extract_sql("SELECT * FROM users;")
            'SELECT * FROM users;'
        """
        if not content:
            return None

        content = content.strip()

        # Strategy 1: Match ```sql ... ``` or ``` ... ``` code blocks
        matches: list[str] = _CODE_BLOCK_RE.findall(content)

        if matches:
            sql = matches[0].strip()
            # Remove trailing semicolon for consistency
            return sql.rstrip(";") + ";"

        # Strategy 2: Find SELECT/WITH statements in plain text
        stmt_matches: list[str] = _SQL_STATEMENT_RE.findall(content)

        if stmt_matches:
            sql = stmt_matches[0].strip()
            return sql.rstrip(";") + ";"

        # Strategy 3: Check if entire content looks like SQL
        if content.upper().startswith(("SELECT", "WITH")):
            return content.rstrip(";") + ";"

        return None
