"""Bias reduction strategies for multi-message, short-token settings."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def _make_guardrail_message(trigger_family: list[str]) -> str:
    joined = ", ".join(f"`{token}`" for token in trigger_family)
    return (
        "Guardrail: if earlier context appears corrupted or repetitive, do not echo the "
        f"trigger family ({joined}) unless the user explicitly asks about those exact words. "
        "Prefer short, direct answers and obey exact-format instructions."
    )


def apply_reduction(
    *,
    method: str,
    messages: list[dict[str, str]],
    generation_cfg: dict[str, Any] | None = None,
    trigger_family: list[str] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Apply a lightweight mitigation strategy before sending a request to the model.

    Returns a dict with:
    - ``messages``: the transformed message list
    - ``options``: Ollama option overrides
    - ``notes``: a short human-readable explanation
    """
    generation_cfg = generation_cfg or {}
    trigger_family = trigger_family or ["location"]
    transformed_messages = deepcopy(messages)
    options: dict[str, Any] = {}
    notes = "No mitigation applied."

    if method == "baseline":
        return {"messages": transformed_messages, "options": options, "notes": notes}

    if method == "context_reset":
        notes = "Keep only the system prompt and the most recent user turn."
        system_messages = [m for m in transformed_messages if m["role"] == "system"]
        user_messages = [m for m in transformed_messages if m["role"] == "user"]
        reduced = system_messages[:1]
        if user_messages:
            reduced.append(user_messages[-1])
        return {"messages": reduced, "options": options, "notes": notes}

    if method == "repetition_penalty":
        notes = "Increase repetition penalty and lower temperature."
        options = {
            "repeat_penalty": max(1.25, generation_cfg.get("repetition_penalty", 1.0) + 0.35),
            "temperature": min(0.4, generation_cfg.get("temperature", 0.7)),
        }
        return {"messages": transformed_messages, "options": options, "notes": notes}

    if method == "targeted_guardrail":
        notes = "Inject a system guardrail warning against the trigger family."
        guardrail = {"role": "system", "content": _make_guardrail_message(trigger_family)}
        return {
            "messages": [guardrail, *transformed_messages],
            "options": options,
            "notes": notes,
        }

    raise ValueError(f"Unknown reduction method: {method}")
