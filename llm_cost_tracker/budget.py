"""
BudgetGuard -- monitor cumulative cost and raise/warn when thresholds are crossed.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Callable, Optional

from .models import BudgetConfig
from .tracker import CostTracker

logger = logging.getLogger("llm_cost_tracker.budget")


class BudgetExceededError(Exception):
    """Raised when cumulative cost exceeds the hard budget limit."""

    def __init__(self, current_cost: float, hard_limit: float) -> None:
        self.current_cost = current_cost
        self.hard_limit = hard_limit
        super().__init__(
            f"Budget hard limit exceeded: ${current_cost:.5f} >= ${hard_limit:.5f}"
        )


class BudgetWarning(UserWarning):
    """Warning issued when cumulative cost exceeds the soft budget limit."""


class BudgetGuard:
    """
    Monitors a :class:`CostTracker` and enforces budget limits.

    Parameters
    ----------
    tracker : CostTracker
        The tracker to monitor.
    soft_limit : float, optional
        When the cumulative cost reaches this value a warning is logged
        (and the optional *alert_callback* is invoked).
    hard_limit : float, optional
        When the cumulative cost reaches this value a
        :class:`BudgetExceededError` is raised.
    alert_callback : callable, optional
        ``callback(current_cost, limit_type, limit_value)`` called on every
        threshold breach.

    Example
    -------
    ::

        tracker = CostTracker()
        guard = BudgetGuard(tracker, soft_limit=1.00, hard_limit=5.00)

        # After each API call, check the budget:
        guard.check()
    """

    def __init__(
        self,
        tracker: CostTracker,
        *,
        soft_limit: Optional[float] = None,
        hard_limit: Optional[float] = None,
        alert_callback: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.tracker = tracker
        self.config = BudgetConfig(
            soft_limit=soft_limit,
            hard_limit=hard_limit,
            alert_callback=alert_callback,
        )
        self._soft_warned = False

    @property
    def soft_limit(self) -> Optional[float]:
        return self.config.soft_limit

    @soft_limit.setter
    def soft_limit(self, value: Optional[float]) -> None:
        self.config.soft_limit = value
        self._soft_warned = False

    @property
    def hard_limit(self) -> Optional[float]:
        return self.config.hard_limit

    @hard_limit.setter
    def hard_limit(self, value: Optional[float]) -> None:
        self.config.hard_limit = value

    def check(self) -> None:
        """
        Check cumulative cost against the configured limits.

        * Soft limit: emits a :class:`BudgetWarning` (only once until the
          limit is changed) and invokes the *alert_callback*.
        * Hard limit: raises :class:`BudgetExceededError`.
        """
        current = self.tracker.total_cost

        # Soft limit
        if (
            self.config.soft_limit is not None
            and current >= self.config.soft_limit
            and not self._soft_warned
        ):
            self._soft_warned = True
            msg = (
                f"Soft budget limit reached: "
                f"${current:.5f} >= ${self.config.soft_limit:.5f}"
            )
            logger.warning(msg)
            warnings.warn(msg, BudgetWarning, stacklevel=2)
            if self.config.alert_callback is not None:
                self.config.alert_callback(current, "soft", self.config.soft_limit)

        # Hard limit
        if (
            self.config.hard_limit is not None
            and current >= self.config.hard_limit
        ):
            if self.config.alert_callback is not None:
                self.config.alert_callback(current, "hard", self.config.hard_limit)
            raise BudgetExceededError(current, self.config.hard_limit)

    def reset_warnings(self) -> None:
        """Allow the soft-limit warning to fire again."""
        self._soft_warned = False

    def remaining(self) -> Optional[float]:
        """
        Return the remaining budget (relative to the hard limit), or
        ``None`` if no hard limit is set.
        """
        if self.config.hard_limit is None:
            return None
        return max(0.0, self.config.hard_limit - self.tracker.total_cost)

    def __repr__(self) -> str:
        return (
            f"BudgetGuard(soft={self.config.soft_limit}, "
            f"hard={self.config.hard_limit}, "
            f"spent=${self.tracker.total_cost:.5f})"
        )
