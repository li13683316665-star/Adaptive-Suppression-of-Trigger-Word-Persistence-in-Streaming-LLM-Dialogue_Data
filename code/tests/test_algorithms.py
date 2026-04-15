"""Tests for reduction algorithms."""

from __future__ import annotations

import pytest

from src.reduction.algorithms import apply_reduction


def test_baseline_noop():
    msgs = [{"role": "user", "content": "hi"}]
    out = apply_reduction(method="baseline", messages=msgs)
    assert out["messages"] == msgs


def test_context_reset_truncates():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "old"},
        {"role": "user", "content": "new"},
    ]
    out = apply_reduction(method="context_reset", messages=msgs)
    assert len(out["messages"]) == 2
    assert out["messages"][-1]["content"] == "new"


def test_guardrail_prefix():
    msgs = [{"role": "user", "content": "hi"}]
    out = apply_reduction(method="targeted_guardrail", messages=msgs, trigger_family=["foo"])
    assert out["messages"][0]["role"] == "system"
    assert "foo" in out["messages"][0]["content"]


def test_unknown_method():
    with pytest.raises(ValueError, match="Unknown reduction"):
        apply_reduction(method="not_a_method", messages=[])
