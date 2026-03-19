from __future__ import annotations

from agenttest.trace.models import AgentTrace


def test_add_message_records_message() -> None:
    trace = AgentTrace()

    trace.add_message("triage", "refunds", {"order_id": "123"})

    assert len(trace.messages) == 1
    assert trace.messages[0].from_agent == "triage"
    assert trace.messages[0].to_agent == "refunds"
    assert trace.messages[0].content["order_id"] == "123"


def test_add_tool_call_records_call() -> None:
    trace = AgentTrace()

    trace.add_tool_call(
        agent="refunds",
        tool_name="process_refund",
        args={"order_id": "123"},
        result={"ok": True},
        duration_ms=12.5,
    )

    assert len(trace.tool_calls) == 1
    call = trace.tool_calls[0]
    assert call.agent == "refunds"
    assert call.tool_name == "process_refund"
    assert call.args["order_id"] == "123"
    assert call.result == {"ok": True}
    assert call.duration_ms == 12.5
