"""Generation result models.

Models describing the outcome of an LLM generation step: the produced SQL
and the token usage reported by the API.
"""

from pydantic import BaseModel, Field


class GenerationResult(BaseModel):
    """Result of an LLM SQL generation call."""

    sql: str = Field(..., min_length=1, description="Generated SQL query")
    tokens_used: int | None = Field(
        default=None, ge=0, description="Total tokens reported by the API (None if absent)"
    )
