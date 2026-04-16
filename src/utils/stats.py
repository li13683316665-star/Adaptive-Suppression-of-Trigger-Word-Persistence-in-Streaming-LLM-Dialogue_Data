"""Small statistical helpers for repeated-run experiment reporting."""

from __future__ import annotations

import math
import random
from statistics import mean, median, pstdev


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * q
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return float(sorted_values[lower_index])
    lower = sorted_values[lower_index]
    upper = sorted_values[upper_index]
    fraction = position - lower_index
    return float(lower + (upper - lower) * fraction)


def bootstrap_confidence_interval(
    values: list[float],
    *,
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 42,
) -> dict[str, float]:
    cleaned = [float(value) for value in values]
    if not cleaned:
        return {"low": 0.0, "high": 0.0}
    if len(cleaned) == 1:
        value = round(cleaned[0], 4)
        return {"low": value, "high": value}

    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(max(100, n_resamples)):
        draw = [cleaned[rng.randrange(len(cleaned))] for _ in range(len(cleaned))]
        samples.append(mean(draw))
    samples.sort()
    alpha = max(0.0, min(1.0, 1.0 - confidence))
    low = _quantile(samples, alpha / 2.0)
    high = _quantile(samples, 1.0 - alpha / 2.0)
    return {"low": round(low, 4), "high": round(high, 4)}


def summarize_distribution(values: list[float]) -> dict[str, float | int | list[float] | dict[str, float]]:
    cleaned = [float(value) for value in values]
    if not cleaned:
        return {
            "count": 0,
            "values": [],
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "ci95": {"low": 0.0, "high": 0.0},
        }
    return {
        "count": len(cleaned),
        "values": [round(value, 4) for value in cleaned],
        "mean": round(mean(cleaned), 4),
        "median": round(median(cleaned), 4),
        "std": round(pstdev(cleaned), 4),
        "min": round(min(cleaned), 4),
        "max": round(max(cleaned), 4),
        "ci95": bootstrap_confidence_interval(cleaned),
    }


def paired_effect_size(values_a: list[float], values_b: list[float]) -> float:
    paired = [(float(a), float(b)) for a, b in zip(values_a, values_b)]
    if not paired:
        return 0.0
    diffs = [b - a for a, b in paired]
    diff_std = pstdev(diffs)
    if diff_std == 0:
        return 0.0
    return round(mean(diffs) / diff_std, 4)


def paired_permutation_test(
    values_a: list[float],
    values_b: list[float],
    *,
    n_resamples: int = 5000,
    seed: int = 42,
) -> dict[str, float | int]:
    paired = [(float(a), float(b)) for a, b in zip(values_a, values_b)]
    if not paired:
        return {"count": 0, "observed_mean_diff": 0.0, "p_value": 1.0}
    diffs = [b - a for a, b in paired]
    observed = abs(mean(diffs))
    rng = random.Random(seed)
    extreme = 0
    for _ in range(max(200, n_resamples)):
        signed = [diff if rng.random() < 0.5 else -diff for diff in diffs]
        if abs(mean(signed)) >= observed:
            extreme += 1
    p_value = (extreme + 1) / (max(200, n_resamples) + 1)
    return {
        "count": len(paired),
        "observed_mean_diff": round(mean(diffs), 4),
        "p_value": round(p_value, 6),
    }


def bonferroni_alpha(alpha: float, m: int) -> float:
    """Per-test significance level for Bonferroni control of FWER across *m* tests."""
    if m <= 0:
        return alpha
    return max(0.0, min(1.0, alpha / m))


def holm_adjusted_p_values(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (step-up cumulative max on sorted p)."""
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    sorted_p = [p_values[i] for i in order]
    adj_sorted: list[float] = []
    for i in range(m):
        adj_sorted.append(min(1.0, (m - i) * sorted_p[i]))
    for i in range(1, m):
        adj_sorted[i] = max(adj_sorted[i], adj_sorted[i - 1])
    out = [0.0] * m
    for i, oi in enumerate(order):
        out[oi] = adj_sorted[i]
    return out


def cohen_d_interpretation(dz: float) -> str:
    """Rough Cohen's d_z label for paired standardized mean difference."""
    a = abs(dz)
    if a < 0.2:
        return "negligible"
    if a < 0.5:
        return "small"
    if a < 0.8:
        return "medium"
    return "large"
