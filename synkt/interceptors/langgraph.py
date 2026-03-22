from __future__ import annotations

from collections.abc import Callable
from typing import Any

from synkt.interceptors.base import BaseInterceptor
from synkt.mocking._registry import get_mock_registry
from synkt.trace.collector import TraceCollector, get_collector, reset_collector
from synkt.trace.storage import get_current_trace


class LangGraphInterceptor(BaseInterceptor):
    """
    Wrapper for LangGraph StateGraph that captures node transitions.

    The wrapped object is expected to expose:
    - graph.nodes: mapping of node_name -> callable
    - graph.compile().invoke(...): invocation API
    """

    def __init__(self, graph: Any, *, live: bool = False, server_url: str = "http://localhost:8000"):
        self.graph = graph
        self._original_nodes: dict[str, Callable[..., Any]] = {}
        self._previous_node_name = "start"
        self._live = live
        self._collector: TraceCollector | None = None
        if live:
            reset_collector()
            self._collector = get_collector(server_url)
        self._wrap_tools()
        self._wrap_nodes()

    def _wrap_tools(self) -> None:
        """Wrap tool callables so active mocks can short-circuit execution."""
        for _, node_obj in self.graph.nodes.items():
            if not hasattr(node_obj, "runnable"):
                continue

            runnable = node_obj.runnable
            if not hasattr(runnable, "tools_by_name"):
                continue

            for tool_name, tool_obj in runnable.tools_by_name.items():
                runnable.tools_by_name[tool_name] = self._create_tool_wrapper(tool_name, tool_obj)

    def _create_tool_wrapper(self, tool_name: str, original_tool: Any) -> Callable[..., Any]:
        """Create wrapper that checks registry for mock behavior before calling real tool."""

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            registry = get_mock_registry()
            mock_config = registry.get(tool_name)

            if mock_config is not None:
                side_effect = mock_config.get("side_effect")
                if side_effect is not None:
                    return side_effect(*args, **kwargs)
                return mock_config.get("return_value")

            if callable(original_tool):
                return original_tool(*args, **kwargs)
            if hasattr(original_tool, "invoke"):
                return original_tool.invoke(*args, **kwargs)

            raise TypeError(f"Unsupported tool type for '{tool_name}': {type(original_tool)!r}")

        return wrapper

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

            # Stream to live UI if enabled
            if self._collector is not None:
                self._collector.record_handoff(prev_node, node_name)
                self._collector.record_agent_start(node_name)

            if callable(original_func):
                result = original_func(state)
            elif hasattr(original_func, "invoke"):
                result = original_func.invoke(state)
            else:
                raise TypeError(
                    f"Unsupported node runnable type for '{node_name}': {type(original_func)!r}"
                )

            self._previous_node_name = node_name

            # Mark agent complete in live stream
            if self._collector is not None:
                self._collector.record_agent_complete(node_name)

            return result

        return wrapper

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        """Invoke the graph (same API as StateGraph.invoke)."""
        self._previous_node_name = "start"
        return self.graph.compile().invoke(*args, **kwargs)

