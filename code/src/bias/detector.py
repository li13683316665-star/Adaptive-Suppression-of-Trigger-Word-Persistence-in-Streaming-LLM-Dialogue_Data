"""Automatic discovery of tokens that correlate with attention / output bias."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any

STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "is",
    "it",
    "you",
    "i",
    "we",
    "that",
    "this",
    "be",
    "are",
    "was",
    "but",
    "just",
    "what",
    "your",
}


def _tokenize(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z\-']*", text.lower())
        if len(token) > 2 and token not in STOPWORDS
    ]


def detect_biased_tokens(
    *,
    model: Any,
    tokenizer: Any,
    contexts: list[str],
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Identify suspicious tokens from text-only experiment traces.

    This first-pass detector is black-box friendly: it looks for tokens that
    appear unusually often in model outputs, survive suppression prompts, or
    cluster inside off-task drift responses.
    """
    del model, tokenizer

    responses = kwargs.get("responses", [])
    suppression_turn_indices = set(kwargs.get("suppression_turn_indices", []))
    off_task_turn_indices = set(kwargs.get("off_task_turn_indices", []))
    top_k = int(kwargs.get("top_k", 10))

    context_counter: Counter[str] = Counter()
    response_counter: Counter[str] = Counter()
    suppression_counter: Counter[str] = Counter()
    off_task_counter: Counter[str] = Counter()
    evidence: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for context_index, context in enumerate(contexts):
        for token in _tokenize(context):
            context_counter[token] += 1
            evidence[token].append({"source": "context", "index": context_index, "text": context})

    for response_index, response in enumerate(responses):
        tokens = _tokenize(response)
        for token in tokens:
            response_counter[token] += 1
            evidence[token].append(
                {"source": "response", "index": response_index, "text": response}
            )
            if response_index in suppression_turn_indices:
                suppression_counter[token] += 1
            if response_index in off_task_turn_indices:
                off_task_counter[token] += 1

    scored_tokens = []
    for token, response_count in response_counter.items():
        context_count = context_counter[token]
        score = (
            response_count * 2
            + suppression_counter[token] * 3
            + off_task_counter[token] * 2
            - min(context_count, response_count)
        )
        scored_tokens.append(
            {
                "token": token,
                "score": score,
                "response_count": response_count,
                "context_count": context_count,
                "suppression_survival": suppression_counter[token],
                "off_task_count": off_task_counter[token],
                "sample_evidence": evidence[token][:3],
            }
        )

    scored_tokens.sort(
        key=lambda item: (
            item["score"],
            item["suppression_survival"],
            item["off_task_count"],
            item["response_count"],
        ),
        reverse=True,
    )

    return {
        "top_tokens": scored_tokens[:top_k],
        "all_scored_tokens": scored_tokens,
    }
