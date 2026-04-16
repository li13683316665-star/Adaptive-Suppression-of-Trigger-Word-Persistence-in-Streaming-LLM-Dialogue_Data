"""Tests for StreamingTriggerDetector."""

from __future__ import annotations

from src.bias.streaming_detector import StreamingTriggerDetector, std_weights_leave_one_out


def test_std_weights_leave_one_out_renormalizes():
    w = std_weights_leave_one_out("f_freq")
    assert abs(sum(w.values()) - 1.0) < 1e-6
    assert w["w_freq"] == 0.0


def test_std_weights_full_sum():
    w = std_weights_leave_one_out(None)
    assert abs(sum(w.values()) - 1.0) < 1e-6


def test_detector_observes_turns():
    d = StreamingTriggerDetector()
    d.observe_turn("hello location place here", "user", "dialogue")
    scores = d.get_risk_scores(top_k=5)
    assert isinstance(scores, list)
