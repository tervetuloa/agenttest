# synkt

**pytest for multi-agent systems.**
Test agent handoffs, prevent infinite loops, and catch coordination bugs before deployment.

## The Problem

**73% of multi-agent AI systems fail in production.** A single undetected infinite loop between two agents can burn through thousands of dollars in API costs before anyone notices.

Current tools like LangSmith show you what happened. **synkt prevents it from happening.**

## Quick Start

```bash
pip install synkt

# Run tests
pytest tests/ -v

# Watch live visualization
python -m synkt.server   # starts on http://localhost:8000
cd synkt-ui && npm run dev  # dashboard on http://localhost:3000
```

## Works With

- LangGraph
- CrewAI
- AutoGen

## What Makes synkt Different

| Feature | synkt | LangSmith | Braintrust |
|---------|-------|-----------|------------|
| Tests coordination | Yes | No | No |
| Detects infinite loops | Yes | No | No |
| Real-time graph viz | Yes | No | No |
| 100% open-source | Yes | No | No |
| Free forever | Yes | 5K traces | 1M spans |

## Usage

### Test Agent Handoffs

```python
from synkt import assert_handoff, assert_no_loop
from synkt.interceptors.langgraph import LangGraphInterceptor

def test_refund_flow():
    graph = build_customer_service_graph()
    traced = LangGraphInterceptor(graph)

    traced.invoke({"input": "refund for order #12345"})

    assert_handoff("triage", "refunds")
    assert_no_loop(max_iterations=5)
```

### Detect Infinite Loops

```python
def test_detects_infinite_loop():
    graph = build_looping_graph()
    traced = LangGraphInterceptor(graph)

    traced.invoke({"messages": []})

    # Catches the loop — this is the point!
    with pytest.raises(AssertionError, match="Possible infinite loop"):
        assert_no_loop(max_iterations=2)
```

### Live Visualization

Stream test execution to the real-time dashboard:

```python
# Enable live streaming to the UI
traced = LangGraphInterceptor(graph, live=True)
traced.invoke({"messages": []})
```

### Mock Tool Responses

```python
from synkt import mock_tool

def test_weather_agent():
    with mock_tool("get_weather", return_value="sunny and 72F"):
        result = agent.invoke({"city": "San Francisco"})
        assert "sunny" in result
```

### Framework Support

```python
# LangGraph
from synkt import LangGraphInterceptor
traced = LangGraphInterceptor(graph)
result = traced.invoke({"input": "hello"})

# CrewAI
from synkt import CrewAIInterceptor
traced = CrewAIInterceptor(crew)
result = traced.invoke(inputs={"topic": "AI"})

# AutoGen
from synkt import AutoGenInterceptor
traced = AutoGenInterceptor(manager)
result = traced.invoke("Start conversation")
```

## Examples

- [Customer Service](examples/customer_service/) — Sequential agent flow
- [Research Crew](examples/research_crew/) — Parallel agent execution
- [Infinite Loop Detection](examples/infinite_loop_demo/) — Catching coordination bugs

## Architecture

```
synkt/                  # Python testing framework
├── assertions/         # assert_handoff, assert_no_loop, assert_cost_under
├── interceptors/       # LangGraph, CrewAI, AutoGen wrappers
├── mocking/            # mock_tool, mock_agent
├── trace/              # AgentTrace, collector, storage
└── server.py           # FastAPI + SSE for live streaming

synkt-ui/               # Next.js real-time dashboard
├── app/                # Dashboard page
├── components/         # Graph canvas, timeline, cost panel
└── hooks/              # useTraceStream (SSE client)
```

## Installation (Development)

```bash
# Python backend
pip install -e ".[dev,langgraph,server]"

# UI dashboard
cd synkt-ui && npm install
```

## Contributing

PRs are welcome. A good starting point:

1. Add or update tests first.
2. Keep error messages specific and useful.
3. Keep APIs typed and easy to read.

```bash
pytest -q
```

## License

MIT
