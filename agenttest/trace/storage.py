from __future__ import annotations

from contextvars import ContextVar
from typing import Optional

from agenttest.trace.models import AgentTrace


_current_trace: ContextVar[Optional[AgentTrace]] = ContextVar("_current_trace", default=None)


def get_current_trace() -> AgentTrace:
    """Get the trace for the current test."""
    trace = _current_trace.get()
    if trace is None:
        raise RuntimeError("No active trace. Are you inside a test?")
    return trace


def set_current_trace(trace: AgentTrace) -> None:
    """Set the trace for the current test."""
    _current_trace.set(trace)


def clear_current_trace() -> None:
    """Clear the trace (called after test)."""
    _current_trace.set(None)
