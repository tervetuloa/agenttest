from __future__ import annotations

import time
from typing import Any, TypedDict

try:
    from langgraph.constants import END
    from langgraph.graph import StateGraph
except ImportError:  # pragma: no cover - optional dependency in local dev
    END = None  # type: ignore[assignment]
    StateGraph = None  # type: ignore[assignment]


class ResearchState(TypedDict):
    topic: str
    depth: str
    constraints: list[str]
    route: str
    web_findings: list[str]
    data_findings: list[str]
    web_failure_mode: str
    web_error: str
    recovery_failure_mode: str
    recovery_attempts: int
    recovered: bool
    report: str
    quality_score: float
    approved: bool


def planner_node(state: ResearchState) -> ResearchState:
    """Plan the task and decide route complexity."""
    depth = state.get("depth", "standard")
    state["route"] = "deep" if depth in {"deep", "thorough"} else "fast"
    state.setdefault("constraints", [])
    state.setdefault("web_findings", [])
    state.setdefault("data_findings", [])
    return state


def web_research_node(state: ResearchState) -> ResearchState:
    """Simulate web-heavy discovery work."""
    topic = state["topic"]
    failure_mode = state.get("web_failure_mode", "")

    if failure_mode == "timeout":
        time.sleep(0.06)
        state["web_error"] = "web_research timeout"
        state["web_findings"] = []
        return state

    if failure_mode == "error":
        state["web_error"] = "web_research simulated failure"
        state["web_findings"] = []
        return state

    # Sleep a bit so timestamps are close but still realistic for parallel assertions.
    time.sleep(0.01)
    state["web_findings"] = [
        f"{topic}: recent trend snapshot",
        f"{topic}: top 3 competitor strategies",
    ]
    state["web_error"] = ""
    return state


def recovery_node(state: ResearchState) -> ResearchState:
    """Recover from a failed web research step with fallback findings."""
    attempts = state.get("recovery_attempts", 0) + 1
    state["recovery_attempts"] = attempts

    if state.get("recovery_failure_mode") == "fail_once" and attempts == 1:
        state["web_error"] = "recovery temporary failure"
        state["recovered"] = False
        state["web_findings"] = []
        return state

    state["recovered"] = True
    error_msg = state.get("web_error", "unknown failure")
    state["web_findings"] = [
        "Fallback: use internal knowledge base snapshot",
        f"Fallback reason: {error_msg}",
    ]
    state["web_error"] = ""
    return state


def data_research_node(state: ResearchState) -> ResearchState:
    """Simulate structured data analysis."""
    topic = state["topic"]
    time.sleep(0.01)
    state["data_findings"] = [
        f"{topic}: baseline conversion estimate",
        f"{topic}: risk-adjusted confidence interval",
    ]
    return state


def synthesize_node(state: ResearchState) -> ResearchState:
    """Merge findings into a draft report."""
    findings = state.get("web_findings", []) + state.get("data_findings", [])
    state["report"] = "\n".join(
        [
            f"Topic: {state['topic']}",
            "Summary:",
            *[f"- {item}" for item in findings],
        ]
    )
    return state


def critique_node(state: ResearchState) -> ResearchState:
    """Apply quality checks and approve/reject output."""
    report = state.get("report", "")
    constraints = state.get("constraints", [])
    has_findings = "- " in report
    quality_score = 0.9 if has_findings else 0.2
    if constraints:
        quality_score -= 0.05

    state["quality_score"] = max(0.0, min(1.0, quality_score))
    state["approved"] = state["quality_score"] >= 0.75
    return state


def route_after_planning(state: ResearchState) -> str:
    """Route to fast path or deep path based on planner output."""
    return "deep" if state.get("route") == "deep" else "fast"


def route_after_web_research(state: ResearchState) -> str:
    """Route failed research through recovery before continuing."""
    return "recover" if state.get("web_error") else "continue"


def route_after_recovery(state: ResearchState) -> str:
    """Retry recovery once for transient failures, otherwise continue."""
    if state.get("web_error") and state.get("recovery_attempts", 0) < 2:
        return "retry"
    return "continue"


def build_research_crew_graph():
    """Build a research graph with branching and a parallel fan-out stage."""
    if StateGraph is None or END is None:
        raise RuntimeError("langgraph is required for the research crew example")

    graph = StateGraph(ResearchState)
    graph.add_node("planner", planner_node)
    graph.add_node("web_research", web_research_node)
    graph.add_node("recovery", recovery_node)
    graph.add_node("data_research", data_research_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("critique", critique_node)

    graph.add_conditional_edges(
        "planner",
        route_after_planning,
        {
            "deep": "web_research",
            "fast": "synthesize",
        },
    )

    # Deep route with failure-aware recovery.
    graph.add_conditional_edges(
        "web_research",
        route_after_web_research,
        {
            "continue": "data_research",
            "recover": "recovery",
        },
    )
    graph.add_conditional_edges(
        "recovery",
        route_after_recovery,
        {
            "retry": "recovery",
            "continue": "data_research",
        },
    )
    graph.add_edge("data_research", "synthesize")

    # Fast route goes straight to synthesis.
    graph.add_edge("synthesize", "critique")
    graph.add_edge("critique", END)

    graph.set_entry_point("planner")
    return graph


def default_research_input(topic: str = "pricing strategy") -> dict[str, Any]:
    """Provide a stable input payload for tests and demos."""
    return {
        "topic": topic,
        "depth": "deep",
        "constraints": ["cite assumptions", "flag uncertainty"],
        "route": "",
        "web_findings": [],
        "data_findings": [],
        "web_failure_mode": "",
        "web_error": "",
        "recovery_failure_mode": "",
        "recovery_attempts": 0,
        "recovered": False,
        "report": "",
        "quality_score": 0.0,
        "approved": False,
    }
