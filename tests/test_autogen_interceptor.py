"""Tests for AutoGen interceptor using fake doubles."""

from __future__ import annotations

from typing import Any

from synkt.interceptors.autogen import AutoGenInterceptor
from synkt.mocking.mock_tool import mock_tool
from synkt.trace.storage import get_current_trace


# ---------------------------------------------------------------------------
# Fake doubles that mirror the AutoGen interface without importing autogen
# ---------------------------------------------------------------------------

class FakeAgent:
    def __init__(self, name: str) -> None:
        self.name = name
        self.sent: list[tuple[Any, str]] = []

    def send(self, message: Any, recipient: Any, *args: Any, **kwargs: Any) -> None:
        self.sent.append((message, getattr(recipient, "name", "?")))

    def initiate_chat(self, manager: Any, *, message: str, **kwargs: Any) -> dict[str, Any]:
        # Simulate: initiator sends message to manager, manager forwards to next agent
        self.send(message, manager)
        # Manager would normally pick next speaker; simulate one round
        next_agent = [a for a in manager.groupchat.agents if a is not self][0]
        next_agent.send(f"reply to: {message}", self)
        return {"chat_history": [message, f"reply to: {message}"]}


class FakeGroupChat:
    def __init__(self, agents: list[FakeAgent]) -> None:
        self.agents = agents
        self.messages: list[dict[str, Any]] = []


class FakeGroupChatManager:
    def __init__(self, groupchat: FakeGroupChat) -> None:
        self.groupchat = groupchat
        self.name = "manager"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_autogen_interceptor_records_send_handoffs() -> None:
    user = FakeAgent("user")
    assistant = FakeAgent("assistant")

    groupchat = FakeGroupChat([user, assistant])
    manager = FakeGroupChatManager(groupchat)

    intercepted = AutoGenInterceptor(manager)
    result = intercepted.invoke("Hello")

    trace = get_current_trace()
    # user -> manager (initiate_chat calls send)
    # assistant -> user   (simulated reply)
    assert len(trace.messages) == 2
    assert trace.messages[0].from_agent == "user"
    assert trace.messages[0].to_agent == "manager"
    assert trace.messages[1].from_agent == "assistant"
    assert trace.messages[1].to_agent == "user"
    assert result["chat_history"][0] == "Hello"


def test_autogen_interceptor_multiple_agents() -> None:
    user = FakeAgent("user")
    coder = FakeAgent("coder")
    critic = FakeAgent("critic")

    groupchat = FakeGroupChat([user, coder, critic])
    manager = FakeGroupChatManager(groupchat)

    intercepted = AutoGenInterceptor(manager)
    assert intercepted.groupchat.agents == [user, coder, critic]


def test_autogen_interceptor_dict_message() -> None:
    """Dict-style messages are stored as-is in trace content."""
    user = FakeAgent("user")
    other = FakeAgent("other")

    groupchat = FakeGroupChat([user, other])
    manager = FakeGroupChatManager(groupchat)

    intercepted = AutoGenInterceptor(manager)

    # Directly call wrapped send with a dict message
    user.send({"role": "user", "content": "hi"}, other)

    trace = get_current_trace()
    last = trace.messages[-1]
    assert last.content == {"role": "user", "content": "hi"}


def test_autogen_interceptor_mock_tool() -> None:
    """mock_tool works with AutoGen interceptor function map."""
    real_called = False

    def real_search(query: str) -> str:
        nonlocal real_called
        real_called = True
        return "real results"

    user = FakeAgent("user")
    user._function_map = {"web_search": real_search}
    assistant = FakeAgent("assistant")

    groupchat = FakeGroupChat([user, assistant])
    manager = FakeGroupChatManager(groupchat)

    intercepted = AutoGenInterceptor(manager)

    with mock_tool("web_search", return_value="mocked results"):
        result = user._function_map["web_search"]("test query")
        assert result == "mocked results"

    assert not real_called

    # Outside mock context, real function runs
    result = user._function_map["web_search"]("test query")
    assert result == "real results"
    assert real_called
