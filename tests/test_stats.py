"""Tests for src.utils.stats."""

from __future__ import annotations

import pytest

from src.utils.stats import (
    bonferroni_alpha,
    bootstrap_confidence_interval,
    cohen_d_interpretation,
    holm_adjusted_p_values,
    paired_effect_size,
    paired_permutation_test,
    summarize_distribution,
)


def test_bootstrap_ci_known_mean():
    vals = [1.0] * 10
    ci = bootstrap_confidence_interval(vals, n_resamples=500, seed=0)
    assert ci["low"] == pytest.approx(1.0, abs=0.01)
    assert ci["high"] == pytest.approx(1.0, abs=0.01)


def test_summarize_distribution_empty():
    s = summarize_distribution([])
    assert s["count"] == 0


def test_paired_permutation_identical():
    a = [1.0, 2.0, 3.0]
    b = [1.0, 2.0, 3.0]
    out = paired_permutation_test(a, b, n_resamples=200, seed=1)
    assert out["p_value"] == 1.0


def test_paired_effect_size_zero_diff():
    assert paired_effect_size([1, 2], [1, 2]) == 0.0


def test_bonferroni_alpha():
    assert bonferroni_alpha(0.05, 5) == pytest.approx(0.01)


def test_holm_order():
    p = [0.01, 0.04, 0.03]
    adj = holm_adjusted_p_values(p)
    assert len(adj) == 3
    assert all(0 <= x <= 1 for x in adj)


def test_cohen_d_interpretation():
    assert cohen_d_interpretation(0.1) == "negligible"
    assert cohen_d_interpretation(0.4) == "small"
    assert cohen_d_interpretation(0.6) == "medium"
    assert cohen_d_interpretation(1.0) == "large"
