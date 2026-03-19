from __future__ import annotations

import pytest

from agenttest import assert_handoff, assert_no_loop
from agenttest.interceptors.langgraph import LangGraphInterceptor

from .system import build_customer_service_graph


@pytest.mark.skipif(
    __import__("importlib.util").util.find_spec("langgraph") is None,
    reason="langgraph is not installed",
)
def test_refund_flow() -> None:
    """Test that refund requests flow correctly through agents."""
    graph = build_customer_service_graph()
    test_graph = LangGraphInterceptor(graph)

    result = test_graph.invoke({"input": "I want a refund for order #12345"})

    assert_handoff(from_agent="triage", to_agent="refunds")
    assert_no_loop(max_iterations=5)
    assert result["resolution"] is not None
    assert "12345" in result["resolution"]


@pytest.mark.skipif(
    __import__("importlib.util").util.find_spec("langgraph") is None,
    reason="langgraph is not installed",
)
def test_no_infinite_loops() -> None:
    """Verify system doesn't loop infinitely."""
    graph = build_customer_service_graph()
    test_graph = LangGraphInterceptor(graph)

    test_graph.invoke({"input": "help me"})
    assert_no_loop(max_iterations=3)
