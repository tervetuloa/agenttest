# agenttest

A small testing framework for multi-agent systems.

```python
from agenttest import assert_handoff, assert_no_loop
from agenttest.interceptors.langgraph import LangGraphInterceptor

from examples.customer_service.system import build_customer_service_graph


def test_refund_flow():
    graph = build_customer_service_graph()
    test_graph = LangGraphInterceptor(graph)

    result = test_graph.invoke({"input": "refund for order #12345"})

    assert_handoff("triage", "refunds")
    assert_no_loop(max_iterations=5)
    assert "12345" in result["resolution"]
```

## Why agenttest?

Most eval tools check final output quality. That is useful, but multi-agent bugs usually happen in the middle:

- Agent A hands off to the wrong agent
- A tool gets called with the wrong payload
- A flow starts looping and burns tokens
- Parallel steps stop being parallel after a refactor

`agenttest` is for testing those coordination paths directly.

## Installation

```bash
pip install agenttest
```

For local development in this repo:

```bash
pip install -e .
pip install -e ".[dev,langgraph]"
```

## Quick Start

1. Build your agent graph/system as normal.
2. Wrap it with an interceptor (for now: LangGraph).
3. Run it in a test.
4. Assert on handoffs, loops, tools, and cost.

```python
from agenttest import assert_handoff, assert_no_loop
from agenttest.interceptors.langgraph import LangGraphInterceptor


def test_my_flow():
    graph = build_graph()
    tested = LangGraphInterceptor(graph)

    tested.invoke({"input": "help me with my refund"})

    assert_handoff("triage", "refunds")
    assert_no_loop(max_iterations=5)
```

## Examples

See real working examples:
- [Customer Service](examples/customer_service/) - Sequential agent flow
- [Research Crew](examples/research_crew/) - Parallel agent execution

## Documentation

- Design doc: [docs/DESIGN.md](docs/DESIGN.md)

## Features

- Test agent handoffs
- Prevent infinite loops
- Validate tool calls
- Mock agents for isolation
- Works with LangGraph today, with CrewAI and AutoGen adapters planned

## Contributing

PRs are welcome. A good starting point:

1. Add or update tests first.
2. Keep error messages specific and useful.
3. Keep APIs typed and easy to read.

Run locally before opening a PR:

```bash
pytest -q
```

## License

MIT
