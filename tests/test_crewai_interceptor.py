"""Tests for CrewAI interceptor using fake doubles."""

from __future__ import annotations

from typing import Any

from synkt.interceptors.crewai import CrewAIInterceptor
from synkt.mocking.mock_tool import mock_tool
from synkt.trace.storage import get_current_trace


# ---------------------------------------------------------------------------
# Fake doubles that mirror the CrewAI interface without importing crewai
# ---------------------------------------------------------------------------

class FakeAgent:
    def __init__(self, role: str) -> None:
        self.role = role
        self.tools: list[Any] = []


class FakeTask:
    def __init__(self, description: str, agent: FakeAgent) -> None:
        self.description = description
        self.agent = agent
        self.result: str = ""

    def execute_sync(self, *args: Any, **kwargs: Any) -> str:
        self.result = f"done:{self.description}"
        # If agent has tools, call them as part of task execution
        for tool in self.agent.tools:
            if callable(tool):
                tool()
        return self.result


class FakeCrew:
    def __init__(self, agents: list[FakeAgent], tasks: list[FakeTask]) -> None:
        self.agents = agents
        self.tasks = tasks

    def kickoff(self, **kwargs: Any) -> dict[str, Any]:
        results = []
        for task in self.tasks:
            results.append(task.execute_sync())
        return {"results": results, **kwargs}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_crewai_interceptor_records_task_handoffs() -> None:
    researcher = FakeAgent("researcher")
    writer = FakeAgent("writer")

    task1 = FakeTask("research AI safety", researcher)
    task2 = FakeTask("write report", writer)

    crew = FakeCrew([researcher, writer], [task1, task2])
    intercepted = CrewAIInterceptor(crew)

    result = intercepted.invoke(topic="AI")

    trace = get_current_trace()
    assert len(trace.messages) == 2
    assert trace.messages[0].from_agent == "crew"
    assert trace.messages[0].to_agent == "researcher"
    assert trace.messages[1].from_agent == "researcher"
    assert trace.messages[1].to_agent == "writer"
    assert result["topic"] == "AI"


def test_crewai_interceptor_single_agent() -> None:
    analyst = FakeAgent("analyst")
    task = FakeTask("analyze data", analyst)

    crew = FakeCrew([analyst], [task])
    intercepted = CrewAIInterceptor(crew)

    result = intercepted.invoke()

    trace = get_current_trace()
    assert len(trace.messages) == 1
    assert trace.messages[0].from_agent == "crew"
    assert trace.messages[0].to_agent == "analyst"
    assert "done:analyze data" in result["results"]


def test_crewai_interceptor_resets_on_reinvoke() -> None:
    """Previous-agent tracking resets between invocations."""
    agent = FakeAgent("worker")
    task = FakeTask("work", agent)
    crew = FakeCrew([agent], [task])
    intercepted = CrewAIInterceptor(crew)

    intercepted.invoke()
    intercepted.invoke()

    trace = get_current_trace()
    # Both invocations should start from "crew"
    assert trace.messages[0].from_agent == "crew"
    assert trace.messages[1].from_agent == "crew"


def test_crewai_interceptor_mock_tool() -> None:
    """mock_tool works with CrewAI interceptor tools."""

    class FakeTool:
        def __init__(self, name: str) -> None:
            self.name = name
            self.called = False

        def __call__(self, *args: Any, **kwargs: Any) -> str:
            self.called = True
            return "real result"

    tool = FakeTool("get_weather")
    agent = FakeAgent("researcher")
    agent.tools = [tool]
    task = FakeTask("look up weather", agent)
    crew = FakeCrew([agent], [task])
    intercepted = CrewAIInterceptor(crew)

    with mock_tool("get_weather", return_value="sunny"):
        intercepted.invoke()

    # The wrapped tool should have returned the mock — the real tool never fires
    assert not tool.called
