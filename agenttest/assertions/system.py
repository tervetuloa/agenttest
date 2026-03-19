from __future__ import annotations

from collections import Counter

from agenttest.trace.storage import get_current_trace


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


def assert_cost_under(threshold: float) -> None:
    """
    Assert that total cost is under threshold.

    Args:
        threshold: Maximum allowed cost in dollars
    """
    trace = get_current_trace()

    if trace.total_cost > threshold:
        raise AssertionError(
            f"Test cost ${trace.total_cost:.2f} exceeds threshold ${threshold:.2f}"
        )
