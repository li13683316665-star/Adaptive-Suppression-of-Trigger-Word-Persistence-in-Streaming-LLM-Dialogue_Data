"""Tests for AdaptiveSuppressionController."""

from __future__ import annotations

import src.bias.streaming_detector as sd
from src.bias.streaming_detector import TokenRisk
from src.reduction.adaptive_controller import AdaptiveSuppressionController


def _risk(token: str, r: float) -> TokenRisk:
    return TokenRisk(token=token, risk=r, signals={}, stats=sd._TokenStats(token))


def test_escalation_disabled_never_escalates_beyond_tier_mapping():
    ctrl = AdaptiveSuppressionController(
        {"escalate_after": 999, "deescalate_after": 999, "risk_threshold": 0.2}
    )
    # High synthetic risk -> tier should map to context_reset without escalation ladder
    risks = [_risk("x", 0.9)]
    d = ctrl.decide(risks)
    assert d.method in ("baseline", "targeted_guardrail", "repetition_penalty", "context_reset")


def test_reset_controller():
    ctrl = AdaptiveSuppressionController()
    ctrl.decide([_risk("a", 0.5)])
    ctrl.reset()
    assert ctrl.history == []
