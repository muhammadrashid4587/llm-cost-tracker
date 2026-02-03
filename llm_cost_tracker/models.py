"""
Pydantic models for usage records, cost summaries, and budget configuration.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field


class UsageRecord(BaseModel):
    """A single LLM API call record with token counts and computed cost."""

    model: str
    provider: str = "unknown"
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    total_tokens: int = 0
    input_cost: float = 0.0
    output_cost: float = 0.0
    cached_input_cost: float = 0.0
    total_cost: float = 0.0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict = Field(default_factory=dict)

    model_config = {"frozen": False}


class CostSummary(BaseModel):
    """Aggregated cost summary for one or more models."""

    model: str
    provider: str = "unknown"
    call_count: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cached_input_tokens: int = 0
    total_cost: float = 0.0

    def __str__(self) -> str:
        return (
            f"Model: {self.model} | "
            f"Calls: {self.call_count} | "
            f"Input: {self.total_input_tokens} tok | "
            f"Output: {self.total_output_tokens} tok | "
            f"Cost: ${self.total_cost:.5f}"
        )


class BudgetConfig(BaseModel):
    """Configuration for budget guards."""

    soft_limit: Optional[float] = None
    hard_limit: Optional[float] = None
    alert_callback: Optional[Callable[..., Any]] = Field(default=None, exclude=True)

    model_config = {"arbitrary_types_allowed": True}
