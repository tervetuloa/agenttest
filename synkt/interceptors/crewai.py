"""CrewAI interceptor for capturing agent task execution."""

from __future__ import annotations

from typing import Any

from synkt.interceptors.base import BaseInterceptor
from synkt.mocking._registry import get_mock_registry
from synkt.trace.storage import get_current_trace


class CrewAIInterceptor(BaseInterceptor):
    """
    Wrapper for CrewAI Crew that captures task execution and agent handoffs.

    The wrapped object is expected to be a CrewAI Crew instance with:
    - crew.tasks: list of Task objects (each with .agent and .description)
    - crew.kickoff(...): execution method
    """

    def __init__(self, crew: Any):
        self.crew = crew
        self._previous_agent = "crew"
        self._wrap_tools()
        self._wrap_tasks()

    def _wrap_tools(self) -> None:
        """Wrap tools on each agent so active mocks can short-circuit execution."""
        for agent in self.crew.agents:
            if not hasattr(agent, "tools") or not agent.tools:
                continue
            for i, tool in enumerate(agent.tools):
                tool_name = getattr(tool, "name", None)
                if tool_name is None:
                    continue
                agent.tools[i] = self._create_tool_wrapper(tool_name, tool)

    def _wrap_tasks(self) -> None:
        """Wrap each task's execute method to record handoffs."""
        for task in self.crew.tasks:
            # CrewAI exposes execute_sync on Task in recent versions;
            # fall back to _execute if neither exists (older builds).
            attr = (
                "execute_sync"
                if hasattr(task, "execute_sync")
                else "_execute"
                if hasattr(task, "_execute")
                else None
            )
            if attr is None:
                continue

            original = getattr(task, attr)
            setattr(task, attr, self._make_task_wrapper(task, original))

    def _make_task_wrapper(self, task: Any, original: Any):  # noqa: ANN202
        """Return a wrapper that logs a handoff then calls the original."""
        interceptor = self

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            trace = get_current_trace()
            agent_name = getattr(task.agent, "role", "unknown") if task.agent else "unknown"

            trace.add_message(
                from_agent=interceptor._previous_agent,
                to_agent=agent_name,
                content={"task": task.description},
            )
            interceptor._previous_agent = agent_name

            return original(*args, **kwargs)

        return wrapper

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the crew (delegates to ``Crew.kickoff``)."""
        self._previous_agent = "crew"
        return self.crew.kickoff(*args, **kwargs)

    @staticmethod
    def _create_tool_wrapper(tool_name: str, original_tool: Any):  # noqa: ANN205
        """Create wrapper that checks mock registry before calling real tool."""

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

        # Preserve the name so downstream code can still identify the tool.
        wrapper.name = tool_name  # type: ignore[attr-defined]
        return wrapper

