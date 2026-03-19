from __future__ import annotations

from typing import Any

from agenttest.interceptors.langgraph import LangGraphInterceptor
from agenttest.trace.storage import get_current_trace


class FakeCompiledGraph:
    def __init__(self, graph: "FakeGraph"):
        self.graph = graph

    def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
        current_state = state
        for node_name in ["triage", "refunds"]:
            current_state = self.graph.nodes[node_name](current_state)
        return current_state


class FakeGraph:
    def __init__(self) -> None:
        self.nodes = {
            "triage": self._triage,
            "refunds": self._refunds,
        }

    def _triage(self, state: dict[str, Any]) -> dict[str, Any]:
        state["order_id"] = "12345"
        return state

    def _refunds(self, state: dict[str, Any]) -> dict[str, Any]:
        state["resolution"] = f"Refund processed for order {state['order_id']}"
        return state

    def compile(self) -> FakeCompiledGraph:
        return FakeCompiledGraph(self)


def test_langgraph_interceptor_records_transitions() -> None:
    graph = FakeGraph()
    intercepted = LangGraphInterceptor(graph)

    result = intercepted.invoke({"input": "refund please"})

    assert "resolution" in result
    trace = get_current_trace()
    assert len(trace.messages) == 2
    assert trace.messages[0].from_agent == "start"
    assert trace.messages[0].to_agent == "triage"
    assert trace.messages[1].from_agent == "triage"
    assert trace.messages[1].to_agent == "refunds"
