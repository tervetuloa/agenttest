from __future__ import annotations

from typing import TypedDict

try:
    from langgraph.graph import StateGraph
except ImportError:  # pragma: no cover - optional dependency in local dev
    StateGraph = None  # type: ignore[assignment]


class CustomerServiceState(TypedDict):
    input: str
    order_id: str | None
    issue_type: str | None
    resolution: str | None


def triage_node(state: CustomerServiceState) -> CustomerServiceState:
    """Classify customer issue."""
    if "refund" in state["input"].lower():
        state["issue_type"] = "refund"
        state["order_id"] = "12345"
    return state


def refund_node(state: CustomerServiceState) -> CustomerServiceState:
    """Process refund."""
    order_id = state.get("order_id")
    if order_id:
        state["resolution"] = f"Refund processed for order {order_id}"
    else:
        state["resolution"] = "Please provide your order ID so we can process a refund."
    return state


def build_customer_service_graph():
    """Build the customer service agent graph."""
    if StateGraph is None:
        raise RuntimeError("langgraph is required for the customer service example")

    graph = StateGraph(CustomerServiceState)
    graph.add_node("triage", triage_node)
    graph.add_node("refunds", refund_node)
    graph.add_edge("triage", "refunds")
    graph.set_entry_point("triage")
    graph.set_finish_point("refunds")
    return graph
