"""Trace collector that sends real-time updates to the synkt server.

Called by interceptors during test execution to stream trace data.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Optional
from urllib.error import URLError
from urllib.request import Request, urlopen


@dataclass
class AgentState:
    """Current state of a single agent."""

    name: str
    status: str  # idle, active, complete, error
    cost: float = 0.0
    tokens: int = 0
    type: str = "executor"


@dataclass
class HandoffMessage:
    """A message (handoff) between agents."""

    from_agent: str
    to_agent: str
    timestamp: float = 0.0
    content: str = ""


@dataclass
class TraceSnapshot:
    """Current state of the test execution."""

    agents: list[AgentState] = field(default_factory=list)
    messages: list[HandoffMessage] = field(default_factory=list)
    total_cost: float = 0.0
    total_tokens: int = 0
    latency_ms: int = 0
    timestamp: float = 0.0
    loop_detected: bool = False
    loop_agents: list[str] = field(default_factory=list)


class TraceCollector:
    """Collects trace data and sends it to the SSE server in real-time."""

    def __init__(self, server_url: str = "http://localhost:8000") -> None:
        self.server_url = server_url
        self.agents: dict[str, AgentState] = {}
        self.messages: list[HandoffMessage] = []
        self.total_cost = 0.0
        self.total_tokens = 0
        self.start_time = datetime.now().timestamp()
        self.loop_detected = False
        self.loop_agents: list[str] = []

    def record_agent_start(self, name: str, agent_type: str = "executor") -> None:
        """Agent started executing."""
        self.agents[name] = AgentState(
            name=name,
            status="active",
            type=agent_type,
        )
        self._send_update()

    def record_agent_complete(
        self, name: str, cost: float = 0.0, tokens: int = 0
    ) -> None:
        """Agent finished successfully."""
        if name in self.agents:
            self.agents[name].status = "complete"
            self.agents[name].cost = cost
            self.agents[name].tokens = tokens
            self.total_cost += cost
            self.total_tokens += tokens
        self._send_update()

    def record_agent_error(self, name: str, error: str = "") -> None:
        """Agent encountered an error."""
        if name in self.agents:
            self.agents[name].status = "error"
        self._send_update()

    def record_handoff(
        self, from_agent: str, to_agent: str, content: str = ""
    ) -> None:
        """Agent handed off to another agent."""
        msg = HandoffMessage(
            from_agent=from_agent,
            to_agent=to_agent,
            timestamp=datetime.now().timestamp(),
            content=content[:200],
        )
        self.messages.append(msg)
        self._send_update()

    def record_loop(self, agents: list[str]) -> None:
        """Record that a loop was detected."""
        self.loop_detected = True
        self.loop_agents = agents
        self._send_update()

    def _send_update(self) -> None:
        """Send current state to server (synchronous, fire-and-forget)."""
        snapshot = TraceSnapshot(
            agents=list(self.agents.values()),
            messages=self.messages,
            total_cost=self.total_cost,
            total_tokens=self.total_tokens,
            latency_ms=int(
                (datetime.now().timestamp() - self.start_time) * 1000
            ),
            timestamp=datetime.now().timestamp(),
            loop_detected=self.loop_detected,
            loop_agents=self.loop_agents,
        )

        try:
            data = json.dumps(asdict(snapshot)).encode("utf-8")
            req = Request(
                f"{self.server_url}/trace",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urlopen(req, timeout=2)
        except (URLError, OSError):
            # Fail silently if server isn't running
            pass


_collector: Optional[TraceCollector] = None


def get_collector(server_url: str = "http://localhost:8000") -> TraceCollector:
    """Get or create the global trace collector."""
    global _collector
    if _collector is None:
        _collector = TraceCollector(server_url)
    return _collector


def reset_collector() -> None:
    """Reset the global collector (for testing)."""
    global _collector
    _collector = None
