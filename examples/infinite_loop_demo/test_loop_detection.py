"""Demo: synkt catches an infinite loop before it costs $47K.

Run with:
    pytest examples/infinite_loop_demo/test_loop_detection.py -v

For live visualization add ``live=True`` to the interceptor and start
the synkt server first:
    python -m synkt.server
"""

from __future__ import annotations

import pytest

from examples.infinite_loop_demo.system import build_looping_graph
from synkt import LangGraphInterceptor, assert_no_loop


def test_detects_infinite_loop() -> None:
    """synkt catches the A↔B loop before it burns tokens."""
    graph = build_looping_graph()
    traced = LangGraphInterceptor(graph)

    traced.invoke({"messages": []})

    # max_revisits=2 means any agent appearing >2 times is a loop.
    # The assertion *should* raise — that's the point: it catches the bug.
    with pytest.raises(AssertionError, match="Possible infinite loop detected"):
        assert_no_loop(max_iterations=2)


def test_detects_infinite_loop_live() -> None:
    """Same test but streams to the live dashboard.

    Start the server first:  python -m synkt.server
    Then open http://localhost:3000 to watch the graph update.
    """
    graph = build_looping_graph()
    traced = LangGraphInterceptor(graph, live=True)

    traced.invoke({"messages": []})

    with pytest.raises(AssertionError, match="Possible infinite loop detected"):
        assert_no_loop(max_iterations=2)
