from __future__ import annotations

import pytest

from agenttest.trace.models import AgentTrace
from agenttest.trace.storage import clear_current_trace, set_current_trace


@pytest.fixture(autouse=True)
def agenttest_trace() -> AgentTrace:
    """Automatically create and clean up trace for each test."""
    trace = AgentTrace()
    set_current_trace(trace)
    yield trace
    clear_current_trace()


def pytest_configure(config: pytest.Config) -> None:
    """Register agenttest markers."""
    config.addinivalue_line(
        "markers",
        "agenttest: mark test as an agent coordination test",
    )
