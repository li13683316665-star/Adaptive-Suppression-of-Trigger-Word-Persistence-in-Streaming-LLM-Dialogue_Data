"""Shared quality-retention scoring helpers for method evaluation scripts."""

from __future__ import annotations

import re
from statistics import mean
from typing import Any

GENERIC_COLLAPSE_PHRASES = {
    "i don't know",
    "not sure",
    "cannot answer",
    "can't answer",
    "unknown",
}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def normalize_exact_text(text: str) -> str:
    normalized = normalize_text(text)
    return re.sub(r"^[^\w]+|[^\w]+$", "", normalized)


def count_sentence_like_units(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    pieces = [
        piece.strip()
        for piece in re.split(r"[.!?]+", stripped)
        if piece.strip() and re.search(r"\w", piece)
    ]
    return len(pieces) if pieces else 1


def count_bullets(text: str) -> int:
    count = 0
    for line in text.splitlines():
        if re.match(r"^\s*(?:[-*]|\d+\.)\s+", line):
            count += 1
    return count


def count_phrase_hits(text: str, phrases: list[str]) -> int:
    normalized = normalize_text(text)
    return sum(1 for phrase in phrases if normalize_text(phrase) in normalized)


def format_success(case: dict[str, Any], response: str) -> float:
    checks: list[float] = []
    exact_target = case.get("exact_target")
    if exact_target is not None:
        checks.append(
            1.0
            if normalize_exact_text(response) == normalize_exact_text(str(exact_target))
            else 0.0
        )

    max_sentences = case.get("max_sentences")
    if max_sentences is not None:
        sentence_count = count_sentence_like_units(response)
        checks.append(1.0 if 1 <= sentence_count <= int(max_sentences) else 0.0)

    bullet_count = case.get("bullet_count")
    if bullet_count is not None:
        checks.append(1.0 if count_bullets(response) == int(bullet_count) else 0.0)

    return round(mean(checks), 4) if checks else 1.0


def score_by_phrases(text: str, phrases: list[str], min_hits: int) -> float:
    if not phrases:
        return 1.0
    hits = count_phrase_hits(text, phrases)
    return round(min(1.0, hits / max(1, min_hits)), 4)


def score_quality_case(case: dict[str, Any], response: str) -> dict[str, Any]:
    format_score = format_success(case, response)
    relevance_score = score_by_phrases(
        response,
        case.get("required_phrases", []),
        int(case.get("min_relevance_hits", max(1, len(case.get("required_phrases", []))))),
    )

    context_phrases = case.get("context_phrases", [])
    context_score = None
    if context_phrases:
        context_score = score_by_phrases(
            response,
            context_phrases,
            int(case.get("min_context_hits", max(1, len(context_phrases)))),
        )

    weighted_components = [
        (format_score, 0.4),
        (relevance_score, 0.3),
    ]
    if context_score is not None:
        weighted_components.append((context_score, 0.3))
    weight_total = sum(weight for _, weight in weighted_components)
    quality_retention_score = round(
        sum(value * weight for value, weight in weighted_components) / weight_total,
        4,
    )

    normalized = normalize_text(response)
    over_suppression_flag = 0
    if any(phrase in normalized for phrase in GENERIC_COLLAPSE_PHRASES):
        over_suppression_flag = 1
    elif context_score is not None and context_score == 0 and relevance_score < 1.0:
        over_suppression_flag = 1
    elif format_score == 0 and relevance_score == 0:
        over_suppression_flag = 1

    return {
        "format_success": format_score,
        "relevance_score": relevance_score,
        "context_retention_score": context_score,
        "quality_retention_score": quality_retention_score,
        "over_suppression_flag": over_suppression_flag,
        "bullet_count_observed": count_bullets(response),
        "sentence_count_observed": count_sentence_like_units(response),
    }


def aggregate_quality_results(method_results: list[dict[str, Any]]) -> dict[str, float]:
    context_scores = [
        result["score"]["context_retention_score"]
        for result in method_results
        if result["score"]["context_retention_score"] is not None
    ]
    return {
        "avg_quality_retention_score": round(
            mean(result["score"]["quality_retention_score"] for result in method_results), 4
        ),
        "avg_format_success": round(
            mean(result["score"]["format_success"] for result in method_results), 4
        ),
        "avg_relevance_score": round(
            mean(result["score"]["relevance_score"] for result in method_results), 4
        ),
        "avg_context_retention_score": round(mean(context_scores), 4)
        if context_scores
        else 0.0,
        "over_suppression_rate": round(
            mean(result["score"]["over_suppression_flag"] for result in method_results), 4
        ),
    }
