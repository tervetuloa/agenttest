"""Tests for graph topology analysis and structural assertions."""

from __future__ import annotations

import pytest

pytest.importorskip("networkx")

from synkt.analysis.graph_topology import GraphTopologyAnalyzer
from synkt.assertions.structural import assert_graph_valid


# ─── Helper graph builders ────────────────────────────────────────────


class SimpleGraph:
    """Minimal graph double for testing."""

    def __init__(
        self,
        nodes: dict,
        edges: list[tuple[str, str]] | None = None,
        conditional_edges: dict | None = None,
    ):
        self.nodes = nodes
        self.edges = edges or []
        self.conditional_edges = conditional_edges or {}


def _make_graph(
    node_names: list[str],
    edges: list[tuple[str, str]],
    conditional_edges: dict | None = None,
) -> SimpleGraph:
    return SimpleGraph(
        nodes={n: None for n in node_names},
        edges=edges,
        conditional_edges=conditional_edges or {},
    )


# ─── Dead-end detection ──────────────────────────────────────────────


def test_find_dead_end_nodes() -> None:
    graph = _make_graph(
        ["__start__", "agent_a", "agent_b", "__end__"],
        [("__start__", "agent_a"), ("agent_a", "agent_b")],
        # agent_b has no outgoing edges → dead end
    )
    analyzer = GraphTopologyAnalyzer(graph)
    assert analyzer.find_dead_end_nodes() == ["agent_b"]


def test_no_dead_ends_when_all_connected() -> None:
    graph = _make_graph(
        ["__start__", "agent_a", "__end__"],
        [("__start__", "agent_a"), ("agent_a", "__end__")],
    )
    analyzer = GraphTopologyAnalyzer(graph)
    assert analyzer.find_dead_end_nodes() == []


# ─── Unreachable node detection ──────────────────────────────────────


def test_find_unreachable_nodes() -> None:
    graph = _make_graph(
        ["__start__", "agent_a", "orphan", "__end__"],
        [("__start__", "agent_a"), ("agent_a", "__end__")],
        # orphan is never connected
    )
    analyzer = GraphTopologyAnalyzer(graph)
    assert analyzer.find_unreachable_nodes() == ["orphan"]


def test_no_unreachable_when_all_connected() -> None:
    graph = _make_graph(
        ["__start__", "agent_a", "agent_b", "__end__"],
        [("__start__", "agent_a"), ("agent_a", "agent_b"), ("agent_b", "__end__")],
    )
    analyzer = GraphTopologyAnalyzer(graph)
    assert analyzer.find_unreachable_nodes() == []


# ─── Unbounded cycle detection ───────────────────────────────────────


def test_detect_unbounded_cycle() -> None:
    graph = _make_graph(
        ["agent_a", "agent_b"],
        [("agent_a", "agent_b"), ("agent_b", "agent_a")],
    )
    analyzer = GraphTopologyAnalyzer(graph)
    cycles = analyzer.detect_unbounded_cycles()
    assert len(cycles) == 1
    assert set(cycles[0]) == {"agent_a", "agent_b"}


def test_bounded_cycle_not_flagged() -> None:
    graph = _make_graph(
        ["agent_a", "agent_b", "__end__"],
        [("agent_a", "agent_b"), ("agent_b", "agent_a"), ("agent_b", "__end__")],
        # agent_b can exit the cycle → bounded
    )
    analyzer = GraphTopologyAnalyzer(graph)
    assert analyzer.detect_unbounded_cycles() == []


# ─── Missing END paths ──────────────────────────────────────────────


def test_find_missing_end_paths() -> None:
    graph = _make_graph(
        ["__start__", "agent_a", "agent_b", "__end__"],
        [("__start__", "agent_a"), ("agent_a", "agent_b")],
        # agent_b has no path to __end__
    )
    analyzer = GraphTopologyAnalyzer(graph)
    assert "agent_b" in analyzer.find_missing_end_paths()


def test_no_missing_end_when_all_reach_end() -> None:
    graph = _make_graph(
        ["__start__", "agent_a", "agent_b", "__end__"],
        [
            ("__start__", "agent_a"),
            ("agent_a", "agent_b"),
            ("agent_b", "__end__"),
        ],
    )
    analyzer = GraphTopologyAnalyzer(graph)
    assert analyzer.find_missing_end_paths() == []


# ─── Full analysis ───────────────────────────────────────────────────


def test_analyze_returns_all_keys() -> None:
    graph = _make_graph(
        ["__start__", "agent_a", "__end__"],
        [("__start__", "agent_a"), ("agent_a", "__end__")],
    )
    analyzer = GraphTopologyAnalyzer(graph)
    report = analyzer.analyze()
    assert "dead_end_nodes" in report
    assert "unreachable_nodes" in report
    assert "unbounded_cycles" in report
    assert "missing_end_paths" in report


def test_has_structural_issues_true() -> None:
    graph = _make_graph(
        ["agent_a", "agent_b"],
        [("agent_a", "agent_b"), ("agent_b", "agent_a")],
    )
    analyzer = GraphTopologyAnalyzer(graph)
    assert analyzer.has_structural_issues() is True


def test_has_structural_issues_false() -> None:
    graph = _make_graph(
        ["__start__", "agent_a", "__end__"],
        [("__start__", "agent_a"), ("agent_a", "__end__")],
    )
    analyzer = GraphTopologyAnalyzer(graph)
    assert analyzer.has_structural_issues() is False


# ─── from_dict constructor ───────────────────────────────────────────


def test_from_dict() -> None:
    analyzer = GraphTopologyAnalyzer.from_dict(
        nodes=["__start__", "a", "b", "__end__"],
        edges=[("__start__", "a"), ("a", "b"), ("b", "__end__")],
    )
    assert analyzer.has_structural_issues() is False


def test_from_dict_with_conditional_edges() -> None:
    analyzer = GraphTopologyAnalyzer.from_dict(
        nodes=["__start__", "router", "fast", "deep", "__end__"],
        edges=[("__start__", "router")],
        conditional_edges={
            "router": {"simple": "fast", "complex": "deep"},
            "fast": {"done": "__end__"},
            "deep": {"done": "__end__"},
        },
    )
    assert analyzer.has_structural_issues() is False


# ─── assert_graph_valid ──────────────────────────────────────────────


def test_assert_graph_valid_passes() -> None:
    graph = _make_graph(
        ["__start__", "agent_a", "__end__"],
        [("__start__", "agent_a"), ("agent_a", "__end__")],
    )
    assert_graph_valid(graph)  # Should not raise


def test_assert_graph_valid_catches_unbounded_cycle() -> None:
    graph = _make_graph(
        ["agent_a", "agent_b"],
        [("agent_a", "agent_b"), ("agent_b", "agent_a")],
    )
    with pytest.raises(AssertionError, match="Unbounded cycles"):
        assert_graph_valid(graph)


def test_assert_graph_valid_catches_dead_end() -> None:
    graph = _make_graph(
        ["__start__", "agent_a", "agent_b", "__end__"],
        [("__start__", "agent_a"), ("agent_a", "agent_b")],
    )
    with pytest.raises(AssertionError, match="Dead-end nodes"):
        assert_graph_valid(graph)


def test_assert_graph_valid_catches_unreachable() -> None:
    graph = _make_graph(
        ["__start__", "agent_a", "orphan", "__end__"],
        [("__start__", "agent_a"), ("agent_a", "__end__")],
    )
    with pytest.raises(AssertionError, match="Unreachable nodes"):
        assert_graph_valid(graph)


def test_assert_graph_valid_allow_cycles() -> None:
    # Graph with unbounded cycle but otherwise valid structure
    graph = _make_graph(
        ["__start__", "agent_a", "agent_b", "__end__"],
        [
            ("__start__", "agent_a"),
            ("agent_a", "agent_b"),
            ("agent_b", "agent_a"),
            # No path to __end__ → triggers missing_end_paths
        ],
    )
    # Without allow_cycles: fails on unbounded cycles
    with pytest.raises(AssertionError, match="Unbounded cycles"):
        assert_graph_valid(graph)

    # With allow_cycles=True: still fails but NOT on cycles
    with pytest.raises(AssertionError) as exc_info:
        assert_graph_valid(graph, allow_cycles=True)
    assert "Unbounded cycles" not in str(exc_info.value)
    assert "no path to END" in str(exc_info.value)


def test_assert_graph_valid_catches_missing_end_path() -> None:
    graph = _make_graph(
        ["__start__", "agent_a", "agent_b", "__end__"],
        [
            ("__start__", "agent_a"),
            ("agent_a", "agent_b"),
            ("agent_a", "__end__"),
        ],
        # agent_b has no path to __end__
    )
    with pytest.raises(AssertionError, match="no path to END"):
        assert_graph_valid(graph)
