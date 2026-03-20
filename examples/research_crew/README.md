# Research Crew Example

Concrete multi-agent research example with a branch + fan-out stage.

What is included:
- `system.py`: LangGraph workflow with planner, research, synthesis, and critique nodes
- `test_crew.py`: Tests for deep/fast routes plus failure injection recovery assertions

Failure injection modes:
- `web_failure_mode="timeout"`: simulates a timeout in `web_research`
- `web_failure_mode="error"`: simulates an error in `web_research`
- `recovery_failure_mode="fail_once"`: makes `recovery` fail once, then succeed on retry

Failures are recovered by a `recovery` agent before synthesis continues.
The cascading fault test covers a `recovery -> recovery` retry handoff.

Run only this example:

```bash
python -m pytest examples/research_crew/test_crew.py -vv
```

Print the timeline while tests run:

```python
from agenttest import print_trace

# ... after tested.invoke(...)
print_trace(include_content=True)
```

Then run with:

```bash
python -m pytest examples/research_crew/test_crew.py -s -vv
```
