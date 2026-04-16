"""Tests for bias metrics."""

from __future__ import annotations

from src.bias.metrics import compute_bias_metrics


def test_pctp_after_correction_simple():
    # Turn 0: trigger in assistant output; turn 1: suppression; turn 2+: post-correction
    gens = [
        "I will mention location here.",
        "okay",
        "still location in reply",
        "clean reply about weather",
    ]
    m = compute_bias_metrics(
        generations=gens,
        reference_keywords=["location"],
        keyword_family=["location"],
        suppression_turn_indices={1},
        expected_topics=["weather"],
    )
    assert m["pctp"] is not None
    assert m["pctp"] > 0.0
    assert m["contamination_aware_pctp"] is not None


def test_contamination_aware_only_counts_off_task():
    gens = [
        "location here",
        "okay",
        "the weather is nice",  # on-task, no family hit counted as contamination
    ]
    m = compute_bias_metrics(
        generations=gens,
        reference_keywords=["location"],
        keyword_family=["location"],
        suppression_turn_indices={1},
        expected_topics=["weather"],
    )
    assert m["contamination_aware_pctp"] == 0.0
