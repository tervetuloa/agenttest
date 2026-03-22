"""Buggy multi-agent workflow with an infinite loop.

Agent A asks Agent B for help.
Agent B asks Agent A for clarification.
They loop forever — in production this cost one company $47K.
"""

from __future__ import annotations

from typing import Any


def _agent_a(state: dict[str, Any]) -> dict[str, Any]:
    return {"messages": state.get("messages", []) + ["A: Need help with task"]}


def _agent_b(state: dict[str, Any]) -> dict[str, Any]:
    return {"messages": state.get("messages", []) + ["B: What task?"]}


def _router(state: dict[str, Any]) -> str:
    """Routes back to agent_a until 10 messages, creating the loop."""
    if len(state.get("messages", [])) < 10:
        return "agent_a"
    return "done"


def build_looping_graph() -> Any:
    """Build a StateGraph with an infinite loop between two agents.

    Returns a compiled-ready StateGraph (caller should use
    ``LangGraphInterceptor`` to wrap and invoke it).
    """
    from langgraph.graph import StateGraph

    class State(dict):
        pass

    workflow = StateGraph(State)
    workflow.add_node("agent_a", _agent_a)
    workflow.add_node("agent_b", _agent_b)
    workflow.add_edge("agent_a", "agent_b")
    workflow.add_conditional_edges("agent_b", _router)
    workflow.set_entry_point("agent_a")

    return workflow
