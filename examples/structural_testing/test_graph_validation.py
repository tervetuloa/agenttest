"""
Structural testing catches bugs BEFORE execution.

Killer demo:
1. Show graph with structural bug
2. Run assert_graph_valid() -> FAILS immediately
3. "This would have cost $47K. synkt caught it in 0.1 seconds. Zero LLM calls."
4. Fix graph, test passes
5. "This is what LangSmith CAN'T do."
"""

from __future__ import annotations

import pytest

pytest.importorskip("networkx")

from synkt.analysis.graph_topology import GraphTopologyAnalyzer
from synkt.assertions.structural import assert_graph_valid


class SimpleGraph:
    """Minimal graph stand-in for demo purposes."""

    def __init__(self, nodes, edges, conditional_edges=None):
        self.nodes = nodes
        self.edges = edges
        self.conditional_edges = conditional_edges or {}


# ─── The broken graph ────────────────────────────────────────────────


def broken_graph_with_unbounded_cycle():
    """
    This graph has an infinite loop:
    agent_a -> agent_b -> agent_a with no exit condition.

    LangSmith would show you this AFTER it costs $$.
    synkt catches it BEFORE any execution.
    """
    return SimpleGraph(
        nodes={"agent_a": lambda s: s, "agent_b": lambda s: s},
        edges=[("agent_a", "agent_b"), ("agent_b", "agent_a")],
    )


def test_structural_validation_catches_loop():
    """
    synkt catches the infinite loop at GRAPH DEFINITION time.
    Zero LLM calls. Zero cost. Instant feedback.
    """
    graph = broken_graph_with_unbounded_cycle()

    with pytest.raises(AssertionError, match="Unbounded cycles"):
        assert_graph_valid(graph)


# ─── The fixed graph ─────────────────────────────────────────────────


def fixed_graph_with_exit_condition():
    """The corrected version with proper termination."""
    return SimpleGraph(
        nodes={
            "__start__": None,
            "agent_a": lambda s: s,
            "agent_b": lambda s: s,
            "supervisor": lambda s: s,
            "__end__": None,
        },
        edges=[("__start__", "agent_a")],
        conditional_edges={
            "agent_a": {"done": "supervisor", "delegate": "agent_b"},
            "agent_b": {"done": "supervisor"},
            "supervisor": {"continue": "agent_a", "finish": "__end__"},
        },
    )


def test_fixed_graph_passes_validation():
    """The corrected graph passes structural validation."""
    graph = fixed_graph_with_exit_condition()
    assert_graph_valid(graph)  # Should not raise


# ─── Direct analyzer usage ───────────────────────────────────────────


def test_analyzer_report_on_broken_graph():
    """Show the full analysis report for the broken graph."""
    graph = broken_graph_with_unbounded_cycle()
    analyzer = GraphTopologyAnalyzer(graph)
    report = analyzer.analyze()

    assert len(report["unbounded_cycles"]) > 0
    assert "agent_a" in report["unbounded_cycles"][0]
    assert "agent_b" in report["unbounded_cycles"][0]
