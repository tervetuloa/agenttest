"""
Structural analysis for LangGraph StateGraph instances.

Detects dead-end nodes, unreachable states, and unbounded cycles
BEFORE any LLM calls — pure graph analysis.
"""

from __future__ import annotations

from typing import Any

try:
    import networkx as nx
except ImportError:
    raise ImportError(
        "networkx is required for graph topology analysis. "
        "Install it with: pip install synkt[analysis]"
    )


class GraphTopologyAnalyzer:
    """
    Analyze a LangGraph StateGraph for structural issues.

    Runs BEFORE any LLM calls — pure graph analysis.

    Accepts any graph-like object with:
    - ``.nodes``: dict of node_name -> callable/spec
    - ``.edges``: iterable of (source, target) tuples
    - ``.conditional_edges`` (optional): dict of node -> {condition: target}

    Alternatively, pass ``nodes`` and ``edges`` dicts directly via
    ``from_dict()``.
    """

    def __init__(self, graph: Any) -> None:
        self.graph = graph
        self.nx_graph = self._to_networkx()

    @classmethod
    def from_dict(
        cls,
        nodes: list[str],
        edges: list[tuple[str, str]],
        conditional_edges: dict[str, dict[str, str]] | None = None,
    ) -> GraphTopologyAnalyzer:
        """Build analyzer from raw node/edge data without a framework graph object."""

        class _DictGraph:
            pass

        g = _DictGraph()
        g.nodes = {n: None for n in nodes}  # type: ignore[attr-defined]
        g.edges = edges  # type: ignore[attr-defined]
        g.conditional_edges = conditional_edges or {}  # type: ignore[attr-defined]
        return cls(g)

    def _to_networkx(self) -> nx.DiGraph:
        """Convert the wrapped graph to a NetworkX DiGraph."""
        G = nx.DiGraph()

        # Add nodes
        for node in self.graph.nodes:
            G.add_node(node)

        # Add direct edges
        if hasattr(self.graph, "edges"):
            edges = self.graph.edges
            if isinstance(edges, dict):
                for src, tgt in edges.items():
                    G.add_edge(src, tgt)
            else:
                for edge in edges:
                    if isinstance(edge, tuple) and len(edge) >= 2:
                        G.add_edge(edge[0], edge[1])

        # Add conditional edges
        if hasattr(self.graph, "conditional_edges"):
            cond_edges = self.graph.conditional_edges
            if isinstance(cond_edges, dict):
                for node, conditions in cond_edges.items():
                    if isinstance(conditions, dict):
                        for condition, target in conditions.items():
                            G.add_edge(node, target, condition=condition)

        return G

    def find_dead_end_nodes(self) -> list[str]:
        """
        Find nodes with no outgoing edges (excluding __end__).

        These are nodes where execution gets stuck.
        """
        dead_ends = []
        for node in self.nx_graph.nodes():
            if node in ("__end__", "END"):
                continue
            if self.nx_graph.out_degree(node) == 0:
                dead_ends.append(node)
        return sorted(dead_ends)

    def find_unreachable_nodes(self) -> list[str]:
        """
        Find nodes that can never be reached from __start__ (or the first node).

        These represent dead code in the graph.
        """
        start_candidates = ["__start__", "START"]
        start_node = None
        for s in start_candidates:
            if s in self.nx_graph:
                start_node = s
                break

        if start_node is None:
            # Use nodes with in-degree 0 as potential starts
            roots = [n for n in self.nx_graph.nodes() if self.nx_graph.in_degree(n) == 0]
            if not roots:
                return []
            # Union of all reachable from any root
            reachable: set[str] = set()
            for r in roots:
                reachable |= nx.descendants(self.nx_graph, r)
                reachable.add(r)
        else:
            reachable = nx.descendants(self.nx_graph, start_node)
            reachable.add(start_node)

        all_nodes = set(self.nx_graph.nodes())
        skip = {"__end__", "END", "__start__", "START"}
        unreachable = all_nodes - reachable - skip

        return sorted(unreachable)

    def detect_unbounded_cycles(self) -> list[list[str]]:
        """
        Detect cycles that have NO exit condition.

        A cycle is *bounded* if at least one node in it has an edge leading
        outside the cycle. Unbounded cycles will loop forever.
        """
        try:
            cycles = list(nx.simple_cycles(self.nx_graph))
        except Exception:
            return []

        unbounded: list[list[str]] = []

        for cycle in cycles:
            cycle_set = set(cycle)
            has_exit = False
            for node in cycle:
                for successor in self.nx_graph.successors(node):
                    if successor not in cycle_set:
                        has_exit = True
                        break
                if has_exit:
                    break

            if not has_exit:
                unbounded.append(sorted(cycle))

        return unbounded

    def find_missing_end_paths(self) -> list[str]:
        """
        Find nodes that have NO path to __end__.

        These nodes can execute but the workflow can never complete through them.
        """
        end_candidates = ["__end__", "END"]
        end_node = None
        for e in end_candidates:
            if e in self.nx_graph:
                end_node = e
                break

        if end_node is None:
            return []

        reversed_graph = self.nx_graph.reverse()
        can_reach_end = nx.descendants(reversed_graph, end_node)
        can_reach_end.add(end_node)

        skip = {"__start__", "START", "__end__", "END"}
        all_nodes = set(self.nx_graph.nodes()) - skip
        missing = all_nodes - can_reach_end

        return sorted(missing)

    def analyze(self) -> dict[str, list[str] | list[list[str]]]:
        """
        Run all structural analyses.

        Returns a report dict with keys:
        - ``dead_end_nodes``: nodes with no outgoing edges
        - ``unreachable_nodes``: nodes unreachable from start
        - ``unbounded_cycles``: cycles with no exit
        - ``missing_end_paths``: nodes that can't reach END
        """
        return {
            "dead_end_nodes": self.find_dead_end_nodes(),
            "unreachable_nodes": self.find_unreachable_nodes(),
            "unbounded_cycles": self.detect_unbounded_cycles(),
            "missing_end_paths": self.find_missing_end_paths(),
        }

    def has_structural_issues(self) -> bool:
        """Quick check: does this graph have ANY structural issues?"""
        report = self.analyze()
        return any(bool(v) for v in report.values())
