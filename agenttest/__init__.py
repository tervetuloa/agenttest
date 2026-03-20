"""agenttest - Testing framework for multi-agent LLM systems."""

__version__ = "0.1.0"

from agenttest.assertions.coordination import assert_handoff, assert_parallel_execution
from agenttest.assertions.system import assert_cost_under, assert_no_loop
from agenttest.assertions.tools import assert_no_tool_called, assert_tool_called
from agenttest.interceptors.langgraph import LangGraphInterceptor
from agenttest.trace.models import AgentMessage, AgentTrace, ToolCall
from agenttest.trace.pretty import format_trace, print_trace
from agenttest.trace.storage import get_current_trace

__all__ = [
	"assert_handoff",
	"assert_parallel_execution",
	"assert_tool_called",
	"assert_no_tool_called",
	"assert_no_loop",
	"assert_cost_under",
	"LangGraphInterceptor",
	"AgentTrace",
	"AgentMessage",
	"ToolCall",
	"get_current_trace",
	"format_trace",
	"print_trace",
]
