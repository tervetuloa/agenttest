from __future__ import annotations

import statistics
from collections import Counter

from synkt.trace.storage import get_current_trace


def assert_no_loop(max_iterations: int = 10) -> None:
    """
    Assert that no agent was called excessively (loop detection).

    Args:
        max_iterations: Max times any single agent can appear in trace
    """
    trace = get_current_trace()
    agent_counts = Counter()

    for msg in trace.messages:
        agent_counts[msg.from_agent] += 1
        agent_counts[msg.to_agent] += 1

    for agent, count in agent_counts.items():
        if count > max_iterations:
            raise AssertionError(
                f"Agent '{agent}' appears {count} times (max: {max_iterations}). "
                "Possible infinite loop detected."
            )


class CostPredictor:
    """
    Predict total cost based on early execution patterns.

    Records per-step costs and extrapolates to estimate the total.
    """

    def __init__(self, budget: float) -> None:
        self.budget = budget
        self.step_costs: list[float] = []

    def record_step(self, cost: float) -> None:
        """Record cost of a single step."""
        self.step_costs.append(cost)

    def predict_total_cost(self, max_steps: int = 50) -> float:
        """
        Predict total cost based on current trajectory.

        Uses trend-aware extrapolation with 3+ data points,
        otherwise assumes constant cost per step.
        """
        if not self.step_costs:
            return 0.0

        n = len(self.step_costs)
        spent = sum(self.step_costs)

        if n < 3:
            avg_cost = statistics.mean(self.step_costs)
            return avg_cost * max_steps

        # Linear trend extrapolation
        avg_cost = statistics.mean(self.step_costs)
        trend = (self.step_costs[-1] - self.step_costs[0]) / max(n - 1, 1)

        remaining = max_steps - n
        if remaining <= 0:
            return spent

        projected_remaining = sum(
            max(0, avg_cost + trend * i) for i in range(remaining)
        )
        return spent + projected_remaining

    def will_exceed_budget(self, max_steps: int = 50) -> bool:
        """Check if trajectory suggests budget overrun."""
        return self.predict_total_cost(max_steps) > self.budget


def assert_cost_under(threshold: float, *, predict: bool = False, max_steps: int = 50) -> None:
    """
    Assert that total cost is under threshold.

    Args:
        threshold: Maximum allowed cost in dollars.
        predict: If ``True``, also check *predicted* total cost based on
                 per-message cost trajectory. Catches overruns early.
        max_steps: Assumed maximum step count for prediction.
    """
    trace = get_current_trace()

    if trace.total_cost > threshold:
        raise AssertionError(
            f"Test cost ${trace.total_cost:.2f} exceeds threshold ${threshold:.2f}"
        )

    if predict and len(trace.messages) >= 3:
        # Estimate per-step cost from total cost spread evenly
        n = len(trace.messages)
        avg_step = trace.total_cost / n
        predictor = CostPredictor(threshold)
        for _ in range(n):
            predictor.record_step(avg_step)

        predicted = predictor.predict_total_cost(max_steps)
        if predicted > threshold:
            raise AssertionError(
                f"Cost trajectory will exceed budget!\n"
                f"Current: ${trace.total_cost:.2f} ({n} steps)\n"
                f"Predicted total ({max_steps} steps): ${predicted:.2f}\n"
                f"Budget: ${threshold:.2f}"
            )
