from __future__ import annotations

from datetime import datetime
from typing import Optional, Type

from pydantic import BaseModel

from agenttest.trace.storage import get_current_trace


def assert_handoff(
    from_agent: str,
    to_agent: str,
    message_schema: Optional[Type[BaseModel]] = None,
    **kwargs: str,
) -> None:
    """
    Assert that one agent handed off to another.

    Args:
        from_agent: Name of the agent that initiated handoff
        to_agent: Name of the agent that received handoff
        message_schema: Optional Pydantic model to validate message content

    Raises:
        AssertionError: If handoff didn't occur or schema doesn't match

    Examples:
        >>> assert_handoff("triage", "refunds")
    """
    # Backward compatibility with docs/examples using from_node/to_node.
    from_agent = kwargs.get("from_node", from_agent)
    to_agent = kwargs.get("to_node", to_agent)

    trace = get_current_trace()
    handoffs = [
        msg
        for msg in trace.messages
        if msg.from_agent == from_agent and msg.to_agent == to_agent
    ]

    if not handoffs:
        raise AssertionError(
            f"No handoff from '{from_agent}' to '{to_agent}'. "
            f"Found {len(trace.messages)} total messages: "
            f"{[(m.from_agent, m.to_agent) for m in trace.messages]}"
        )

    if message_schema:
        handoff = handoffs[0]
        try:
            message_schema(**handoff.content)
        except Exception as exc:  # pragma: no cover - exact pydantic error varies
            raise AssertionError(
                f"Handoff message doesn't match schema {message_schema.__name__}: {exc}"
            ) from exc


def assert_parallel_execution(agents: list[str], max_time_delta_ms: float = 50.0) -> None:
    """
    Assert that multiple agents executed in near-parallel based on timestamps.

    Args:
        agents: Agent names expected to run in parallel
        max_time_delta_ms: Maximum allowed span between earliest and latest event

    Raises:
        AssertionError: If insufficient events are present or timing is too far apart
    """
    trace = get_current_trace()
    timestamps: list[datetime] = []

    for message in trace.messages:
        if message.from_agent in agents or message.to_agent in agents:
            timestamps.append(message.timestamp)

    if len(timestamps) < len(agents):
        raise AssertionError(
            f"Expected events for {len(agents)} agents, found {len(timestamps)} relevant messages"
        )

    span_ms = (max(timestamps) - min(timestamps)).total_seconds() * 1000
    if span_ms > max_time_delta_ms:
        raise AssertionError(
            f"Expected parallel execution within {max_time_delta_ms}ms, observed {span_ms:.2f}ms"
        )
