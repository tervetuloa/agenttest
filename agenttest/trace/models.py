from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class AgentMessage(BaseModel):
    """A message sent from one agent to another."""

    from_agent: str
    to_agent: str
    content: dict[str, Any]
    timestamp: datetime


class ToolCall(BaseModel):
    """A tool called by an agent."""

    agent: str
    tool_name: str
    args: dict[str, Any]
    result: Optional[Any] = None
    timestamp: datetime
    duration_ms: float


class AgentTrace(BaseModel):
    """Complete trace of multi-agent execution."""

    messages: list[AgentMessage] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    total_cost: float = 0.0
    duration_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)

    def add_message(self, from_agent: str, to_agent: str, content: dict[str, Any]) -> None:
        """Add a message to the trace."""
        self.messages.append(
            AgentMessage(
                from_agent=from_agent,
                to_agent=to_agent,
                content=content,
                timestamp=datetime.now(),
            )
        )

    def add_tool_call(
        self,
        agent: str,
        tool_name: str,
        args: dict[str, Any],
        result: Any = None,
        duration_ms: float = 0.0,
    ) -> None:
        """Add a tool call to the trace."""
        self.tool_calls.append(
            ToolCall(
                agent=agent,
                tool_name=tool_name,
                args=args,
                result=result,
                timestamp=datetime.now(),
                duration_ms=duration_ms,
            )
        )
