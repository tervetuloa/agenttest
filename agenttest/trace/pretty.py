from __future__ import annotations

from typing import Any

from agenttest.trace.models import AgentTrace
from agenttest.trace.storage import get_current_trace


def _shorten(value: Any, max_len: int = 120) -> str:
    text = repr(value)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def format_trace(
    trace: AgentTrace | None = None,
    *,
    include_content: bool = False,
    include_tools: bool = True,
    include_summary: bool = True,
) -> str:
    """Return a human-readable timeline for an AgentTrace.

    Args:
        trace: Explicit trace. If omitted, uses the active trace from context.
        include_content: Include message payload snippets.
        include_tools: Include tool-call timeline entries.
        include_summary: Include final summary counts and cost.
    """
    active_trace = trace or get_current_trace()
    lines: list[str] = []

    lines.append("Agent Trace Timeline")
    lines.append("--------------------")

    if not active_trace.messages and (not include_tools or not active_trace.tool_calls):
        lines.append("(no events captured)")
    else:
        for idx, msg in enumerate(active_trace.messages, start=1):
            ts = msg.timestamp.strftime("%H:%M:%S.%f")[:-3]
            base = f"{idx:02d}. [{ts}] {msg.from_agent} -> {msg.to_agent}"
            if include_content:
                base += f" | content={_shorten(msg.content)}"
            lines.append(base)

        if include_tools and active_trace.tool_calls:
            start_idx = len(active_trace.messages) + 1
            for offset, tool in enumerate(active_trace.tool_calls):
                idx = start_idx + offset
                ts = tool.timestamp.strftime("%H:%M:%S.%f")[:-3]
                result_label = "ok" if tool.result is not None else "no-result"
                lines.append(
                    f"{idx:02d}. [{ts}] {tool.agent} => tool:{tool.tool_name} "
                    f"args={_shorten(tool.args, max_len=80)} "
                    f"duration={tool.duration_ms:.2f}ms {result_label}"
                )

    if include_summary:
        lines.append("")
        lines.append("Summary")
        lines.append("-------")
        lines.append(f"messages: {len(active_trace.messages)}")
        lines.append(f"tool_calls: {len(active_trace.tool_calls)}")
        lines.append(f"duration_ms: {active_trace.duration_ms:.2f}")
        lines.append(f"total_cost: ${active_trace.total_cost:.4f}")

    return "\n".join(lines)


def print_trace(
    trace: AgentTrace | None = None,
    *,
    include_content: bool = False,
    include_tools: bool = True,
    include_summary: bool = True,
) -> None:
    """Print a formatted trace timeline to stdout."""
    print(
        format_trace(
            trace,
            include_content=include_content,
            include_tools=include_tools,
            include_summary=include_summary,
        )
    )
