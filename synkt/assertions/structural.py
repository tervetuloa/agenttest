"""
Structural assertions that run BEFORE execution.

These catch bugs at graph definition time, not runtime.
"""

from __future__ import annotations

from typing import Any


def assert_graph_valid(graph: Any, *, allow_cycles: bool = False) -> None:
    """
    Assert that a graph is structurally valid.

    Catches:
    - Dead-end nodes (nodes with no exit)
    - Unreachable nodes (dead code)
    - Unbounded cycles (infinite loops waiting to happen)
    - Nodes with no path to END

    This runs BEFORE any LLM calls — pure graph analysis.

    Args:
        graph: A LangGraph StateGraph, or any object with ``.nodes``,
               ``.edges``, and optionally ``.conditional_edges`` attributes.
        allow_cycles: If ``True``, skip the unbounded cycle check.

    Raises:
        AssertionError: Detailed message about structural issues found.
    """
    from synkt.analysis.graph_topology import GraphTopologyAnalyzer

    analyzer = GraphTopologyAnalyzer(graph)
    report = analyzer.analyze()

    errors: list[str] = []

    if report["dead_end_nodes"]:
        errors.append(
            f"Dead-end nodes detected (no outgoing edges): {report['dead_end_nodes']}\n"
            f"These nodes will cause execution to hang."
        )

    if report["unreachable_nodes"]:
        errors.append(
            f"Unreachable nodes detected: {report['unreachable_nodes']}\n"
            f"These nodes can never execute (dead code)."
        )

    if not allow_cycles and report["unbounded_cycles"]:
        errors.append(
            f"Unbounded cycles detected: {report['unbounded_cycles']}\n"
            f"These cycles have NO exit condition and will loop infinitely."
        )

    if report["missing_end_paths"]:
        errors.append(
            f"Nodes with no path to END: {report['missing_end_paths']}\n"
            f"These nodes can execute but never complete the workflow."
        )

    if errors:
        error_msg = "\n\n".join(errors)
        raise AssertionError(f"Graph structural validation failed:\n\n{error_msg}")
