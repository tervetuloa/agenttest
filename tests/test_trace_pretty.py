from __future__ import annotations

from agenttest.trace.models import AgentTrace
from agenttest.trace.pretty import format_trace, print_trace
from agenttest.trace.storage import set_current_trace


def test_format_trace_includes_messages_and_summary() -> None:
    trace = AgentTrace(total_cost=0.1234, duration_ms=42.0)
    trace.add_message("planner", "research", {"topic": "pricing"})
    trace.add_tool_call(
        agent="research",
        tool_name="web_search",
        args={"query": "pricing strategy"},
        result={"hits": 3},
        duration_ms=12.5,
    )

    output = format_trace(trace, include_content=True)

    assert "Agent Trace Timeline" in output
    assert "planner -> research" in output
    assert "tool:web_search" in output
    assert "messages: 1" in output
    assert "tool_calls: 1" in output
    assert "total_cost: $0.1234" in output


def test_format_trace_uses_current_trace_when_not_passed() -> None:
    trace = AgentTrace()
    trace.add_message("start", "triage", {"input": "refund"})
    set_current_trace(trace)

    output = format_trace()

    assert "start -> triage" in output


def test_print_trace_writes_timeline(capsys) -> None:
    trace = AgentTrace()
    trace.add_message("a", "b", {})

    print_trace(trace)

    captured = capsys.readouterr()
    assert "a -> b" in captured.out
