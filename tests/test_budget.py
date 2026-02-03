"""Tests for BudgetGuard."""

from __future__ import annotations

import warnings

import pytest

from llm_cost_tracker.budget import BudgetExceededError, BudgetGuard, BudgetWarning
from llm_cost_tracker.tracker import CostTracker


class TestBudgetGuard:
    """Tests for budget enforcement."""

    def _make_tracker_with_cost(self, cost: float) -> CostTracker:
        """Helper: create a tracker with a pre-set cost via manual record."""
        tracker = CostTracker(auto_price=False)
        from llm_cost_tracker.models import UsageRecord

        rec = UsageRecord(
            model="test-model",
            provider="test",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            total_cost=cost,
        )
        tracker.add_record(rec)
        return tracker

    def test_no_limits_no_error(self):
        tracker = self._make_tracker_with_cost(100.0)
        guard = BudgetGuard(tracker)
        guard.check()  # Should not raise

    def test_hard_limit_exceeded(self):
        tracker = self._make_tracker_with_cost(10.0)
        guard = BudgetGuard(tracker, hard_limit=5.0)
        with pytest.raises(BudgetExceededError) as exc_info:
            guard.check()
        assert exc_info.value.current_cost == 10.0
        assert exc_info.value.hard_limit == 5.0

    def test_hard_limit_not_exceeded(self):
        tracker = self._make_tracker_with_cost(3.0)
        guard = BudgetGuard(tracker, hard_limit=5.0)
        guard.check()  # Should not raise

    def test_soft_limit_warns(self):
        tracker = self._make_tracker_with_cost(2.0)
        guard = BudgetGuard(tracker, soft_limit=1.0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            guard.check()
            assert len(w) == 1
            assert issubclass(w[0].category, BudgetWarning)
            assert "$2.00000" in str(w[0].message)

    def test_soft_limit_warns_once(self):
        tracker = self._make_tracker_with_cost(2.0)
        guard = BudgetGuard(tracker, soft_limit=1.0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            guard.check()
            guard.check()
            guard.check()
            # Only one warning
            budget_warnings = [x for x in w if issubclass(x.category, BudgetWarning)]
            assert len(budget_warnings) == 1

    def test_soft_limit_not_triggered(self):
        tracker = self._make_tracker_with_cost(0.5)
        guard = BudgetGuard(tracker, soft_limit=1.0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            guard.check()
            budget_warnings = [x for x in w if issubclass(x.category, BudgetWarning)]
            assert len(budget_warnings) == 0

    def test_both_limits_hard_takes_precedence(self):
        tracker = self._make_tracker_with_cost(10.0)
        guard = BudgetGuard(tracker, soft_limit=1.0, hard_limit=5.0)
        with pytest.raises(BudgetExceededError):
            guard.check()

    def test_alert_callback_soft(self):
        calls = []
        tracker = self._make_tracker_with_cost(2.0)
        guard = BudgetGuard(
            tracker,
            soft_limit=1.0,
            alert_callback=lambda cost, kind, limit: calls.append((cost, kind, limit)),
        )
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            guard.check()
        assert len(calls) == 1
        assert calls[0] == (2.0, "soft", 1.0)

    def test_alert_callback_hard(self):
        calls = []
        tracker = self._make_tracker_with_cost(10.0)
        guard = BudgetGuard(
            tracker,
            hard_limit=5.0,
            alert_callback=lambda cost, kind, limit: calls.append((cost, kind, limit)),
        )
        with pytest.raises(BudgetExceededError):
            guard.check()
        assert len(calls) == 1
        assert calls[0] == (10.0, "hard", 5.0)

    def test_remaining_with_hard_limit(self):
        tracker = self._make_tracker_with_cost(3.0)
        guard = BudgetGuard(tracker, hard_limit=5.0)
        assert abs(guard.remaining() - 2.0) < 0.001

    def test_remaining_without_hard_limit(self):
        tracker = self._make_tracker_with_cost(3.0)
        guard = BudgetGuard(tracker)
        assert guard.remaining() is None

    def test_remaining_when_exceeded(self):
        tracker = self._make_tracker_with_cost(10.0)
        guard = BudgetGuard(tracker, hard_limit=5.0)
        assert guard.remaining() == 0.0

    def test_reset_warnings(self):
        tracker = self._make_tracker_with_cost(2.0)
        guard = BudgetGuard(tracker, soft_limit=1.0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            guard.check()
            guard.reset_warnings()
            guard.check()
            budget_warnings = [x for x in w if issubclass(x.category, BudgetWarning)]
            assert len(budget_warnings) == 2

    def test_update_soft_limit_resets_warning(self):
        tracker = self._make_tracker_with_cost(2.0)
        guard = BudgetGuard(tracker, soft_limit=1.0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            guard.check()
            guard.soft_limit = 1.5  # Setting new limit resets warning
            guard.check()
            budget_warnings = [x for x in w if issubclass(x.category, BudgetWarning)]
            assert len(budget_warnings) == 2

    def test_repr(self):
        tracker = self._make_tracker_with_cost(1.0)
        guard = BudgetGuard(tracker, soft_limit=2.0, hard_limit=5.0)
        r = repr(guard)
        assert "soft=2.0" in r
        assert "hard=5.0" in r
        assert "$1.00000" in r
