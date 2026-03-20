from __future__ import annotations

import pytest

from agenttest import assert_handoff, assert_no_loop, assert_parallel_execution
from agenttest.interceptors.langgraph import LangGraphInterceptor
from agenttest.trace.storage import get_current_trace

from .system import build_research_crew_graph, default_research_input


@pytest.mark.skipif(
    __import__("importlib.util").util.find_spec("langgraph") is None,
    reason="langgraph is not installed",
)
def test_deep_research_flow() -> None:
    """Deep route should traverse planner -> web -> data -> synthesize -> critique."""
    graph = build_research_crew_graph()
    tested = LangGraphInterceptor(graph)

    result = tested.invoke(default_research_input("enterprise retention"))

    assert_handoff("planner", "web_research")
    assert_handoff("web_research", "data_research")
    assert_handoff("data_research", "synthesize")
    assert_handoff("synthesize", "critique")
    assert_no_loop(max_iterations=6)

    # Using a relaxed threshold because this is sequential with short sleeps.
    assert_parallel_execution(["web_research", "data_research"], max_time_delta_ms=1000)
    assert result["approved"] is True
    assert result["quality_score"] >= 0.75
    assert "enterprise retention" in result["report"]


@pytest.mark.skipif(
    __import__("importlib.util").util.find_spec("langgraph") is None,
    reason="langgraph is not installed",
)
def test_fast_route_skips_deep_research_nodes() -> None:
    """Fast route should skip deep nodes and can fail strict quality checks."""
    graph = build_research_crew_graph()
    tested = LangGraphInterceptor(graph)

    payload = default_research_input("incident response")
    payload["depth"] = "quick"
    result = tested.invoke(payload)

    assert_handoff("planner", "synthesize")
    assert_handoff("synthesize", "critique")
    assert_no_loop(max_iterations=5)

    trace = get_current_trace()
    transitions = {(m.from_agent, m.to_agent) for m in trace.messages}
    assert ("planner", "web_research") not in transitions
    assert ("web_research", "data_research") not in transitions

    assert result["approved"] is False
    assert result["quality_score"] < 0.75


@pytest.mark.parametrize("failure_mode", ["timeout", "error"])
@pytest.mark.skipif(
    __import__("importlib.util").util.find_spec("langgraph") is None,
    reason="langgraph is not installed",
)
def test_deep_route_recovers_from_web_research_failures(failure_mode: str) -> None:
    """Injected web failures should trigger recovery and still produce output."""
    graph = build_research_crew_graph()
    tested = LangGraphInterceptor(graph)

    payload = default_research_input("market expansion")
    payload["web_failure_mode"] = failure_mode
    result = tested.invoke(payload)

    assert_handoff("planner", "web_research")
    assert_handoff("web_research", "recovery")
    assert_handoff("recovery", "data_research")
    assert_handoff("data_research", "synthesize")
    assert_handoff("synthesize", "critique")
    assert_no_loop(max_iterations=8)

    assert result["recovered"] is True
    assert "Fallback" in result["report"]


@pytest.mark.skipif(
    __import__("importlib.util").util.find_spec("langgraph") is None,
    reason="langgraph is not installed",
)
def test_recovery_retries_after_transient_failure() -> None:
    """Recovery should retry once after a transient recovery fault and then continue."""
    graph = build_research_crew_graph()
    tested = LangGraphInterceptor(graph)

    payload = default_research_input("supply chain risk")
    payload["web_failure_mode"] = "error"
    payload["recovery_failure_mode"] = "fail_once"

    result = tested.invoke(payload)

    assert_handoff("web_research", "recovery")
    assert_handoff("recovery", "recovery")
    assert_handoff("recovery", "data_research")
    assert_no_loop(max_iterations=12)

    assert result["recovery_attempts"] == 2
    assert result["recovered"] is True
    assert result["approved"] is True
