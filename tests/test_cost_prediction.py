"""Tests for enhanced cost prediction."""

from __future__ import annotations

import pytest

from synkt.assertions.system import CostPredictor, assert_cost_under
from synkt.trace.models import AgentTrace
from synkt.trace.storage import set_current_trace


def test_cost_predictor_no_data() -> None:
    p = CostPredictor(budget=10.0)
    assert p.predict_total_cost() == 0.0
    assert p.will_exceed_budget() is False


def test_cost_predictor_constant_rate() -> None:
    p = CostPredictor(budget=10.0)
    for _ in range(5):
        p.record_step(0.5)
    # 50 steps * 0.5 per step = 25.0 predicted (with 3+ points, uses trend)
    predicted = p.predict_total_cost(max_steps=50)
    assert predicted > 10.0
    assert p.will_exceed_budget(max_steps=50) is True


def test_cost_predictor_under_budget() -> None:
    p = CostPredictor(budget=100.0)
    for _ in range(5):
        p.record_step(0.01)
    assert p.will_exceed_budget(max_steps=50) is False


def test_assert_cost_under_predict_mode() -> None:
    trace = AgentTrace(total_cost=3.0)
    # Add enough messages to trigger prediction
    for _ in range(10):
        trace.add_message("a", "b", {})
    set_current_trace(trace)

    # Current cost is 3.0, under threshold 5.0
    # But predicted cost for 50 steps = (3.0/10)*50 = 15.0 > 5.0
    with pytest.raises(AssertionError, match="trajectory will exceed budget"):
        assert_cost_under(5.0, predict=True, max_steps=50)


def test_assert_cost_under_predict_passes() -> None:
    trace = AgentTrace(total_cost=0.01)
    for _ in range(10):
        trace.add_message("a", "b", {})
    set_current_trace(trace)

    # Very low cost, prediction should be fine
    assert_cost_under(5.0, predict=True, max_steps=50)
