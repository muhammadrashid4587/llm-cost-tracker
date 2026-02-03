"""Tests for CostTracker."""

from __future__ import annotations

import threading
from datetime import datetime, timezone

import pytest

from llm_cost_tracker.tracker import CostTracker
from llm_cost_tracker.models import UsageRecord


class TestCostTracker:
    """Unit tests for the CostTracker class."""

    def test_record_basic(self):
        tracker = CostTracker()
        rec = tracker.record(
            model="gpt-4o",
            provider="openai",
            input_tokens=100,
            output_tokens=50,
        )
        assert rec.model == "gpt-4o"
        assert rec.provider == "openai"
        assert rec.input_tokens == 100
        assert rec.output_tokens == 50
        assert rec.total_tokens == 150
        assert rec.total_cost > 0

    def test_auto_pricing(self):
        tracker = CostTracker(auto_price=True)
        rec = tracker.record(
            model="gpt-4o",
            provider="openai",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
        )
        # gpt-4o: $2.50 per 1M input + $10.00 per 1M output = $12.50
        assert abs(rec.input_cost - 2.50) < 0.001
        assert abs(rec.output_cost - 10.00) < 0.001
        assert abs(rec.total_cost - 12.50) < 0.001

    def test_auto_pricing_disabled(self):
        tracker = CostTracker(auto_price=False)
        rec = tracker.record(
            model="gpt-4o",
            provider="openai",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
        )
        assert rec.total_cost == 0.0

    def test_unknown_model_zero_cost(self):
        tracker = CostTracker()
        rec = tracker.record(
            model="some-unknown-model",
            provider="unknown",
            input_tokens=100,
            output_tokens=50,
        )
        assert rec.total_cost == 0.0

    def test_total_cost_property(self):
        tracker = CostTracker()
        tracker.record(model="gpt-4o-mini", provider="openai", input_tokens=1000, output_tokens=500)
        tracker.record(model="gpt-4o-mini", provider="openai", input_tokens=2000, output_tokens=1000)
        assert tracker.total_cost > 0
        assert len(tracker) == 2

    def test_total_tokens_property(self):
        tracker = CostTracker()
        tracker.record(model="gpt-4o", provider="openai", input_tokens=100, output_tokens=50)
        tracker.record(model="gpt-4o", provider="openai", input_tokens=200, output_tokens=100)
        assert tracker.total_tokens == 450

    def test_records_returns_copy(self):
        tracker = CostTracker()
        tracker.record(model="gpt-4o", provider="openai", input_tokens=10, output_tokens=5)
        records = tracker.records
        records.clear()
        assert len(tracker) == 1  # original not affected

    def test_summary_string(self):
        tracker = CostTracker()
        tracker.record(model="gpt-4o", provider="openai", input_tokens=100, output_tokens=50)
        summary = tracker.summary()
        assert "gpt-4o" in summary
        assert "Calls: 1" in summary
        assert "Input: 100" in summary
        assert "Output: 50" in summary
        assert "$" in summary

    def test_summary_empty(self):
        tracker = CostTracker()
        assert tracker.summary() == "No usage recorded."

    def test_summary_filter_by_model(self):
        tracker = CostTracker()
        tracker.record(model="gpt-4o", provider="openai", input_tokens=100, output_tokens=50)
        tracker.record(model="gpt-4o-mini", provider="openai", input_tokens=100, output_tokens=50)
        summary = tracker.summary(model="gpt-4o-mini")
        assert "gpt-4o-mini" in summary
        assert summary.count("Model:") == 1

    def test_summary_filter_by_provider(self):
        tracker = CostTracker()
        tracker.record(model="gpt-4o", provider="openai", input_tokens=100, output_tokens=50)
        tracker.record(model="claude-sonnet-4-6", provider="anthropic", input_tokens=100, output_tokens=50)
        summaries = tracker.get_summaries(provider="anthropic")
        assert len(summaries) == 1
        assert summaries[0].provider == "anthropic"

    def test_summary_filter_since(self):
        tracker = CostTracker()
        tracker.record(model="gpt-4o", provider="openai", input_tokens=100, output_tokens=50)
        # Filter for future date should return nothing
        future = datetime(2099, 1, 1, tzinfo=timezone.utc)
        summaries = tracker.get_summaries(since=future)
        assert len(summaries) == 0

    def test_add_record(self):
        tracker = CostTracker()
        rec = UsageRecord(
            model="gpt-4o",
            provider="openai",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            total_cost=0.001,
        )
        tracker.add_record(rec)
        assert len(tracker) == 1
        assert tracker.records[0].total_cost == 0.001

    def test_reset(self):
        tracker = CostTracker()
        tracker.record(model="gpt-4o", provider="openai", input_tokens=100, output_tokens=50)
        assert len(tracker) == 1
        tracker.reset()
        assert len(tracker) == 0
        assert tracker.total_cost == 0.0

    def test_repr(self):
        tracker = CostTracker()
        assert "CostTracker" in repr(tracker)
        assert "$" in repr(tracker)

    def test_thread_safety(self):
        tracker = CostTracker()
        errors = []

        def add_records():
            try:
                for _ in range(100):
                    tracker.record(
                        model="gpt-4o-mini",
                        provider="openai",
                        input_tokens=10,
                        output_tokens=5,
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_records) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(tracker) == 1000

    def test_cached_input_tokens(self):
        tracker = CostTracker()
        rec = tracker.record(
            model="gpt-4o",
            provider="openai",
            input_tokens=1_000_000,
            output_tokens=0,
            cached_input_tokens=500_000,
        )
        # 500k regular input at $2.50/1M = $1.25
        # 500k cached input at $1.25/1M = $0.625
        # total = $1.875
        assert abs(rec.input_cost - 1.25) < 0.001
        assert abs(rec.cached_input_cost - 0.625) < 0.001
        assert abs(rec.total_cost - 1.875) < 0.001
