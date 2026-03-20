"""AutoGen interceptor for capturing multi-agent conversations."""

from __future__ import annotations

from typing import Any

from synkt.interceptors.base import BaseInterceptor
from synkt.mocking._registry import get_mock_registry
from synkt.trace.storage import get_current_trace


class AutoGenInterceptor(BaseInterceptor):
    """
    Wrapper for AutoGen GroupChatManager that captures agent conversations.

    The wrapped object is expected to have:
    - manager.groupchat.agents: list of agent objects with a ``name`` attribute
    - Each agent exposes a ``send(message, recipient, ...)`` method
    """

    def __init__(self, manager: Any):
        self.manager = manager
        self.groupchat = manager.groupchat
        self._wrap_tools()
        self._wrap_agents()

    def _wrap_agents(self) -> None:
        """Wrap each agent's ``send`` method to record handoffs."""
        for agent in self.groupchat.agents:
            if not hasattr(agent, "send"):
                continue

            original_send = agent.send
            agent.send = self._make_send_wrapper(agent, original_send)

    @staticmethod
    def _make_send_wrapper(agent: Any, original_send: Any):  # noqa: ANN205
        """Return a wrapper around ``agent.send`` that logs the message."""

        def wrapper(message: Any, recipient: Any, *args: Any, **kwargs: Any) -> Any:
            trace = get_current_trace()
            from_name = getattr(agent, "name", "unknown")
            to_name = getattr(recipient, "name", "unknown")

            # message may be a string or a dict; normalise for the trace.
            if isinstance(message, str):
                content = {"message": message}
            elif isinstance(message, dict):
                content = message
            else:
                content = {"message": str(message)}

            trace.add_message(
                from_agent=from_name,
                to_agent=to_name,
                content=content,
            )

            return original_send(message, recipient, *args, **kwargs)

        return wrapper

    def invoke(self, message: str, *args: Any, **kwargs: Any) -> Any:
        """
        Start a conversation via the first agent in the group chat.

        Delegates to ``initiator.initiate_chat(manager, message=...)``.
        """
        initiator = self.groupchat.agents[0]
        return initiator.initiate_chat(self.manager, message=message, *args, **kwargs)

    def _wrap_tools(self) -> None:
        """Wrap registered functions/tools on each agent so mocks can intercept."""
        for agent in self.groupchat.agents:
            # AutoGen stores registered functions in _function_map (dict[str, callable]).
            func_map = getattr(agent, "_function_map", None)
            if not func_map:
                continue
            for tool_name, tool_fn in list(func_map.items()):
                func_map[tool_name] = self._create_tool_wrapper(tool_name, tool_fn)

    @staticmethod
    def _create_tool_wrapper(tool_name: str, original_fn: Any):  # noqa: ANN205
        """Create wrapper that checks mock registry before calling real function."""

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            registry = get_mock_registry()
            mock_config = registry.get(tool_name)

            if mock_config is not None:
                side_effect = mock_config.get("side_effect")
                if side_effect is not None:
                    return side_effect(*args, **kwargs)
                return mock_config.get("return_value")

            return original_fn(*args, **kwargs)

        return wrapper
