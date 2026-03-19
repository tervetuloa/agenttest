from __future__ import annotations

import pytest
from pydantic import BaseModel

from agenttest.assertions.coordination import assert_handoff, assert_parallel_execution
from agenttest.assertions.system import assert_cost_under, assert_no_loop
from agenttest.assertions.tools import assert_no_tool_called, assert_tool_called
from agenttest.trace.models import AgentTrace
from agenttest.trace.storage import set_current_trace


class RefundRequest(BaseModel):
    order_id: str


def test_assert_handoff_success() -> None:
    trace = AgentTrace()
    trace.add_message("triage", "refunds", {"order_id": "12345"})
    set_current_trace(trace)

    assert_handoff("triage", "refunds")


def test_assert_handoff_missing() -> None:
    trace = AgentTrace()
    set_current_trace(trace)

    with pytest.raises(AssertionError, match="No handoff"):
        assert_handoff("triage", "refunds")


def test_assert_handoff_schema_validation() -> None:
    trace = AgentTrace()
    trace.add_message("triage", "refunds", {"order_id": "12345"})
    set_current_trace(trace)

    assert_handoff("triage", "refunds", message_schema=RefundRequest)


def test_assert_handoff_schema_validation_failure() -> None:
    trace = AgentTrace()
    trace.add_message("triage", "refunds", {"wrong": "field"})
    set_current_trace(trace)

    with pytest.raises(AssertionError, match="doesn't match schema"):
        assert_handoff("triage", "refunds", message_schema=RefundRequest)


def test_assert_tool_called_success() -> None:
    trace = AgentTrace()
    trace.add_tool_call("refunds", "process_refund", {"order_id": "12345"})
    set_current_trace(trace)

    assert_tool_called("process_refund", args={"order_id": "12345"})


def test_assert_tool_called_wrong_count() -> None:
    trace = AgentTrace()
    set_current_trace(trace)

    with pytest.raises(AssertionError, match="Expected 1 calls"):
        assert_tool_called("process_refund")


def test_assert_no_tool_called_success() -> None:
    trace = AgentTrace()
    set_current_trace(trace)

    assert_no_tool_called("send_email")


def test_assert_no_tool_called_failure() -> None:
    trace = AgentTrace()
    trace.add_tool_call("notifications", "send_email", {"to": "a@example.com"})
    set_current_trace(trace)

    with pytest.raises(AssertionError, match="Expected no calls"):
        assert_no_tool_called("send_email")


def test_assert_no_loop_success() -> None:
    trace = AgentTrace()
    trace.add_message("start", "triage", {})
    trace.add_message("triage", "refunds", {})
    set_current_trace(trace)

    assert_no_loop(max_iterations=3)


def test_assert_no_loop_failure() -> None:
    trace = AgentTrace()
    for _ in range(6):
        trace.add_message("triage", "triage", {})
    set_current_trace(trace)

    with pytest.raises(AssertionError, match="Possible infinite loop"):
        assert_no_loop(max_iterations=10)


def test_assert_cost_under_success() -> None:
    trace = AgentTrace(total_cost=0.25)
    set_current_trace(trace)

    assert_cost_under(1.00)


def test_assert_cost_under_failure() -> None:
    trace = AgentTrace(total_cost=1.50)
    set_current_trace(trace)

    with pytest.raises(AssertionError, match="exceeds threshold"):
        assert_cost_under(1.00)


def test_assert_parallel_execution_success() -> None:
    trace = AgentTrace()
    trace.add_message("router", "agent_a", {})
    trace.add_message("router", "agent_b", {})
    set_current_trace(trace)

    assert_parallel_execution(["agent_a", "agent_b"], max_time_delta_ms=1000)
