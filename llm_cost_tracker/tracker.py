"""
Core CostTracker class -- thread-safe storage and aggregation of usage records.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional, Sequence

from .models import CostSummary, UsageRecord
from .pricing import get_price


class CostTracker:
    """
    Thread-safe tracker that accumulates :class:`UsageRecord` objects and
    provides aggregated cost summaries.

    Parameters
    ----------
    auto_price : bool
        When ``True`` (default), automatically compute costs from the
        built-in pricing table whenever a record is added without
        explicit cost values.
    """

    def __init__(self, *, auto_price: bool = True) -> None:
        self._records: list[UsageRecord] = []
        self._lock = threading.Lock()
        self.auto_price = auto_price

    # ── Recording ─────────────────────────────────────────────────────

    def record(
        self,
        model: str,
        provider: str = "unknown",
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached_input_tokens: int = 0,
        metadata: Optional[dict] = None,
    ) -> UsageRecord:
        """
        Create a :class:`UsageRecord`, compute costs, store it, and return it.
        """
        input_cost = 0.0
        output_cost = 0.0
        cached_input_cost = 0.0

        if self.auto_price:
            try:
                prices = get_price(model, provider if provider != "unknown" else None)
                input_cost = max(0, input_tokens - cached_input_tokens) * prices["input_cost_per_1m"] / 1_000_000
                output_cost = output_tokens * prices["output_cost_per_1m"] / 1_000_000
                cached_rate = prices.get("cached_input_cost_per_1m", prices["input_cost_per_1m"])
                cached_input_cost = cached_input_tokens * cached_rate / 1_000_000
            except KeyError:
                pass  # Unknown model -- costs stay at 0

        total_tokens = input_tokens + output_tokens
        total_cost = input_cost + output_cost + cached_input_cost

        rec = UsageRecord(
            model=model,
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input_tokens,
            total_tokens=total_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            cached_input_cost=cached_input_cost,
            total_cost=total_cost,
            metadata=metadata or {},
        )

        with self._lock:
            self._records.append(rec)

        return rec

    def add_record(self, record: UsageRecord) -> None:
        """Add a pre-built :class:`UsageRecord` directly."""
        with self._lock:
            self._records.append(record)

    # ── Querying ──────────────────────────────────────────────────────

    @property
    def records(self) -> list[UsageRecord]:
        """Return a shallow copy of all recorded entries."""
        with self._lock:
            return list(self._records)

    @property
    def total_cost(self) -> float:
        """Return the sum of all recorded costs."""
        with self._lock:
            return sum(r.total_cost for r in self._records)

    @property
    def total_tokens(self) -> int:
        """Return the sum of all recorded tokens."""
        with self._lock:
            return sum(r.total_tokens for r in self._records)

    def summary(
        self,
        *,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> str:
        """
        Return a human-readable cost summary.

        Optionally filter by *model*, *provider*, or records *since* a
        given UTC datetime.
        """
        summaries = self.get_summaries(model=model, provider=provider, since=since)
        if not summaries:
            return "No usage recorded."
        lines = [str(s) for s in summaries]
        total = sum(s.total_cost for s in summaries)
        lines.append(f"Total cost: ${total:.5f}")
        return "\n".join(lines)

    def get_summaries(
        self,
        *,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> list[CostSummary]:
        """Return per-model :class:`CostSummary` objects."""
        with self._lock:
            filtered = list(self._records)

        if model:
            filtered = [r for r in filtered if r.model == model]
        if provider:
            filtered = [r for r in filtered if r.provider == provider]
        if since:
            if since.tzinfo is None:
                since = since.replace(tzinfo=timezone.utc)
            filtered = [r for r in filtered if r.timestamp >= since]

        groups: dict[tuple[str, str], list[UsageRecord]] = defaultdict(list)
        for r in filtered:
            groups[(r.provider, r.model)].append(r)

        summaries: list[CostSummary] = []
        for (prov, mod), recs in groups.items():
            summaries.append(
                CostSummary(
                    model=mod,
                    provider=prov,
                    call_count=len(recs),
                    total_input_tokens=sum(r.input_tokens for r in recs),
                    total_output_tokens=sum(r.output_tokens for r in recs),
                    total_cached_input_tokens=sum(r.cached_input_tokens for r in recs),
                    total_cost=sum(r.total_cost for r in recs),
                )
            )
        return summaries

    # ── Convenience exports ───────────────────────────────────────────

    def export_csv(self, path: str) -> None:
        """Export all records to a CSV file."""
        from .export import export_csv

        export_csv(self.records, path)

    def export_json(self, path: str) -> None:
        """Export all records to a JSON file."""
        from .export import export_json

        export_json(self.records, path)

    def export_sqlite(self, path: str) -> None:
        """Export all records to a SQLite database."""
        from .export import export_sqlite

        export_sqlite(self.records, path)

    # ── Housekeeping ──────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear all recorded data."""
        with self._lock:
            self._records.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)

    def __repr__(self) -> str:
        return f"CostTracker(records={len(self)}, total_cost=${self.total_cost:.5f})"
