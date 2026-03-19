from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agenttest.interceptors.base import BaseInterceptor
from agenttest.trace.storage import get_current_trace


class LangGraphInterceptor(BaseInterceptor):
    """
    Wrapper for LangGraph StateGraph that captures node transitions.

    The wrapped object is expected to expose:
    - graph.nodes: mapping of node_name -> callable
    - graph.compile().invoke(...): invocation API
    """

    def __init__(self, graph: Any):
        self.graph = graph
        self._original_nodes: dict[str, Callable[..., Any]] = {}
        self._previous_node_name = "start"
        self._wrap_nodes()

    def _wrap_nodes(self) -> None:
        """Wrap each node runnable/function to capture transitions."""
        for node_name, node_obj in self.graph.nodes.items():
            # LangGraph StateGraph stores a StateNodeSpec with a `.runnable` attribute.
            if hasattr(node_obj, "runnable"):
                original = node_obj.runnable
                self._original_nodes[node_name] = original
                node_obj.runnable = self._create_wrapper(node_name, original)
            else:
                # Fallback for simpler graph doubles used in local tests.
                self._original_nodes[node_name] = node_obj
                self.graph.nodes[node_name] = self._create_wrapper(node_name, node_obj)

    def _create_wrapper(self, node_name: str, original_func: Callable[..., Any]) -> Callable[..., Any]:
        """Create a wrapper that logs node handoffs to trace."""

        def wrapper(state: Any) -> Any:
            trace = get_current_trace()
            prev_node = self._previous_node_name

            trace.add_message(
                from_agent=prev_node,
                to_agent=node_name,
                content={"state": state},
            )

            if callable(original_func):
                result = original_func(state)
            elif hasattr(original_func, "invoke"):
                result = original_func.invoke(state)
            else:
                raise TypeError(
                    f"Unsupported node runnable type for '{node_name}': {type(original_func)!r}"
                )

            self._previous_node_name = node_name

            return result

        return wrapper

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        """Invoke the graph (same API as StateGraph.invoke)."""
        self._previous_node_name = "start"
        return self.graph.compile().invoke(*args, **kwargs)
