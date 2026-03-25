"""synkt - Testing framework for multi-agent LLM systems."""

__version__ = "0.2.0"

from synkt.assertions.coordination import assert_handoff, assert_parallel_execution
from synkt.assertions.structural import assert_graph_valid
from synkt.assertions.system import assert_cost_under, assert_no_loop
from synkt.assertions.tools import assert_no_tool_called, assert_tool_called
from synkt.interceptors.autogen import AutoGenInterceptor
from synkt.interceptors.crewai import CrewAIInterceptor
from synkt.interceptors.langgraph import LangGraphInterceptor
from synkt.mocking import mock_tool
from synkt.trace.models import AgentMessage, AgentTrace, ToolCall
from synkt.trace.pretty import format_trace, print_trace
from synkt.trace.storage import get_current_trace

__all__ = [
	"assert_graph_valid",
	"assert_handoff",
	"assert_parallel_execution",
	"assert_tool_called",
	"assert_no_tool_called",
	"assert_no_loop",
	"assert_cost_under",
	"AutoGenInterceptor",
	"CrewAIInterceptor",
	"LangGraphInterceptor",
	"AgentTrace",
	"AgentMessage",
	"ToolCall",
	"get_current_trace",
	"format_trace",
	"print_trace",
	"mock_tool",
]

