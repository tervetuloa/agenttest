from __future__ import annotations

from typing import Any, Optional

from agenttest.trace.storage import get_current_trace


def assert_tool_called(
    tool_name: str,
    args: Optional[dict[str, Any]] = None,
    times: int = 1,
    by_agent: Optional[str] = None,
) -> None:
    """
    Assert that a tool was called.

    Args:
        tool_name: Name of the tool
        args: Optional dict of expected arguments
        times: Expected number of calls (default 1)
        by_agent: Optional agent name filter
    """
    trace = get_current_trace()
    calls = [tc for tc in trace.tool_calls if tc.tool_name == tool_name]

    if by_agent:
        calls = [tc for tc in calls if tc.agent == by_agent]

    if len(calls) != times:
        raise AssertionError(f"Expected {times} calls to '{tool_name}', got {len(calls)}")

    if args:
        for call in calls:
            for key, expected_value in args.items():
                actual_value = call.args.get(key)
                if actual_value != expected_value:
                    raise AssertionError(
                        f"Tool '{tool_name}' called with {key}={actual_value}, "
                        f"expected {expected_value}"
                    )


def assert_no_tool_called(tool_name: str, by_agent: Optional[str] = None) -> None:
    """Assert that a tool was never called."""
    trace = get_current_trace()
    calls = [tc for tc in trace.tool_calls if tc.tool_name == tool_name]

    if by_agent:
        calls = [tc for tc in calls if tc.agent == by_agent]

    if calls:
        raise AssertionError(
            f"Expected no calls to '{tool_name}', but found {len(calls)} calls"
        )
