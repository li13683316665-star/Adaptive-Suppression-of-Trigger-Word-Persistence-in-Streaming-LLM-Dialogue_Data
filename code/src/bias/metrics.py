"""Metrics for measuring bias in short multi-turn vtuber-style contexts."""

from __future__ import annotations

import re
from typing import Any


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _tokenize_words(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z\-']*", _normalize(text))


def _count_keyword_occurrences(text: str, keyword: str) -> int:
    pattern = re.compile(rf"\b{re.escape(keyword.lower())}\b")
    return len(pattern.findall(_normalize(text)))


def _contains_any_phrase(text: str, phrases: list[str]) -> bool:
    normalized_text = _normalize(text)
    return any(_normalize(phrase) in normalized_text for phrase in phrases if phrase)


def compute_bias_metrics(
    *,
    generations: list[str],
    reference_keywords: list[str] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Compute simple lexical bias metrics for a multi-turn run.

    The baseline experiment focuses on whether a trigger word (for example,
    ``location``) starts to intrude into later turns and survives direct
    suppression attempts such as ``respond with only okay``.
    """

    keywords = [keyword.lower() for keyword in (reference_keywords or []) if keyword]
    keyword_family = [
        keyword.lower() for keyword in kwargs.get("keyword_family", keywords) if keyword
    ]
    suppression_turn_indices = set(kwargs.get("suppression_turn_indices", []))
    exact_targets = {
        int(index): _normalize(target)
        for index, target in kwargs.get("exact_targets", {}).items()
    }
    expected_topics = [
        topic.lower() for topic in kwargs.get("expected_topics", []) if topic
    ]
    corruption_phrases = [
        phrase for phrase in kwargs.get("corruption_phrases", []) if phrase
    ]

    keyword_occurrences: dict[str, int] = {keyword: 0 for keyword in keywords}
    keyword_turn_indices: dict[str, list[int]] = {keyword: [] for keyword in keywords}
    keyword_streaks: dict[str, int] = {keyword: 0 for keyword in keywords}
    current_streaks: dict[str, int] = {keyword: 0 for keyword in keywords}

    trigger_hit_rule = {
        "match_type": "exact_word_boundary",
        "stemming": False,
        "lemmatization": False,
        "family_keywords": keyword_family,
    }

    family_counts: dict[str, int] = {keyword: 0 for keyword in keyword_family}
    family_turn_hits: dict[str, list[int]] = {keyword: [] for keyword in keyword_family}
    suppression_keyword_hits = 0
    exact_target_violations = 0
    suppression_failures = 0
    off_task_turns = 0
    transcript_like_corruption_turns = 0
    recovery_turn_index: int | None = None
    per_turn = []

    for turn_index, generation in enumerate(generations):
        normalized_generation = _normalize(generation)
        words = _tokenize_words(generation)
        turn_record: dict[str, Any] = {
            "turn_index": turn_index,
            "text": generation,
            "normalized_text": normalized_generation,
            "keyword_counts": {},
            "family_counts": {},
            "is_suppression_turn": turn_index in suppression_turn_indices,
        }

        for keyword in keywords:
            count = _count_keyword_occurrences(generation, keyword)
            turn_record["keyword_counts"][keyword] = count
            keyword_occurrences[keyword] += count

            if count > 0:
                keyword_turn_indices[keyword].append(turn_index)
                current_streaks[keyword] += 1
                keyword_streaks[keyword] = max(
                    keyword_streaks[keyword], current_streaks[keyword]
                )
                if turn_index in suppression_turn_indices:
                    suppression_keyword_hits += 1
            else:
                current_streaks[keyword] = 0

        for keyword in keyword_family:
            count = _count_keyword_occurrences(generation, keyword)
            turn_record["family_counts"][keyword] = count
            family_counts[keyword] += count
            if count > 0:
                family_turn_hits[keyword].append(turn_index)

        if turn_index in exact_targets and normalized_generation != exact_targets[turn_index]:
            exact_target_violations += 1
            suppression_failures += 1

        if turn_index in suppression_turn_indices and any(
            count > 0 for count in turn_record["family_counts"].values()
        ):
            suppression_failures += 1

        expected_topic_hits = sum(
            _count_keyword_occurrences(generation, topic) for topic in expected_topics
        )
        family_hits = sum(turn_record["family_counts"].values())
        transcript_like_corruption = _contains_any_phrase(generation, corruption_phrases)
        repeated_token_ratio = 0.0
        if words:
            repeated_token_ratio = 1.0 - (len(set(words)) / len(words))

        contamination_hits = family_hits if expected_topic_hits == 0 else 0
        context_appropriate_family_hits = family_hits if expected_topic_hits > 0 else 0
        off_task = contamination_hits > 0
        if off_task:
            off_task_turns += 1

        if transcript_like_corruption:
            transcript_like_corruption_turns += 1

        if (
            recovery_turn_index is None
            and turn_index > 0
            and family_hits == 0
            and not transcript_like_corruption
        ):
            if any(
                sum(pervious["family_counts"].values()) > 0
                for pervious in per_turn
            ):
                recovery_turn_index = turn_index

        turn_record["expected_topic_hits"] = expected_topic_hits
        turn_record["family_hits"] = family_hits
        turn_record["contamination_hits"] = contamination_hits
        turn_record["context_appropriate_family_hits"] = context_appropriate_family_hits
        turn_record["off_task"] = off_task
        turn_record["transcript_like_corruption"] = transcript_like_corruption
        turn_record["repeated_token_ratio"] = round(repeated_token_ratio, 4)

        per_turn.append(turn_record)

    persistence_turns_after_first = {
        keyword: max(0, len(indices) - 1)
        for keyword, indices in keyword_turn_indices.items()
    }

    recovery_turn_count = None
    if recovery_turn_index is not None:
        recovery_turn_count = recovery_turn_index + 1

    first_correction_turn = None
    if suppression_turn_indices:
        first_correction_turn = min(suppression_turn_indices)

    post_correction_turns = 0
    post_correction_trigger_hits = 0
    post_correction_contaminated_turns = 0
    post_correction_contaminated_rate = None
    post_correction_semantic_contamination_hits = 0
    post_correction_semantic_contaminated_turns = 0
    post_correction_semantic_contaminated_rate = None
    pctp = None
    contamination_aware_pctp = None
    if first_correction_turn is not None and first_correction_turn < len(per_turn) - 1:
        post_correction_slice = per_turn[first_correction_turn + 1 :]
        post_correction_turns = len(post_correction_slice)
        post_correction_trigger_hits = sum(
            turn["family_hits"] for turn in post_correction_slice
        )
        post_correction_contaminated_turns = sum(
            1 for turn in post_correction_slice if turn["family_hits"] > 0
        )
        post_correction_contaminated_rate = round(
            post_correction_contaminated_turns / post_correction_turns, 4
        )
        post_correction_semantic_contamination_hits = sum(
            turn["contamination_hits"] for turn in post_correction_slice
        )
        post_correction_semantic_contaminated_turns = sum(
            1 for turn in post_correction_slice if turn["contamination_hits"] > 0
        )
        post_correction_semantic_contaminated_rate = round(
            post_correction_semantic_contaminated_turns / post_correction_turns, 4
        )
        pctp = round(post_correction_trigger_hits / post_correction_turns, 4)
        contamination_aware_pctp = round(
            post_correction_semantic_contamination_hits / post_correction_turns, 4
        )

    return {
        "total_turns": len(generations),
        "trigger_hit_rule": trigger_hit_rule,
        "keyword_occurrences": keyword_occurrences,
        "keyword_turn_indices": keyword_turn_indices,
        "persistence_turns_after_first": persistence_turns_after_first,
        "max_keyword_streak": keyword_streaks,
        "family_counts": family_counts,
        "family_turn_hits": family_turn_hits,
        "suppression_keyword_hits": suppression_keyword_hits,
        "exact_target_violations": exact_target_violations,
        "suppression_failures": suppression_failures,
        "suppression_turn_count": len(suppression_turn_indices),
        "suppression_failure_rate": round(
            suppression_failures / max(1, len(suppression_turn_indices)), 4
        ),
        "first_correction_turn": first_correction_turn,
        "post_correction_turns": post_correction_turns,
        "post_correction_trigger_hits": post_correction_trigger_hits,
        "post_correction_contaminated_turns": post_correction_contaminated_turns,
        "post_correction_contaminated_rate": post_correction_contaminated_rate,
        "post_correction_semantic_contamination_hits": post_correction_semantic_contamination_hits,
        "post_correction_semantic_contaminated_turns": post_correction_semantic_contaminated_turns,
        "post_correction_semantic_contaminated_rate": post_correction_semantic_contaminated_rate,
        "pctp": pctp,
        "contamination_aware_pctp": contamination_aware_pctp,
        "off_task_turns": off_task_turns,
        "off_task_rate": round(off_task_turns / max(1, len(generations)), 4),
        "transcript_like_corruption_turns": transcript_like_corruption_turns,
        "recovery_turn_count": recovery_turn_count,
        "per_turn": per_turn,
    }
