"""Resolve chat completion backend (local Ollama vs OpenAI-compatible API)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def get_chat_fn(backend: str) -> Callable[..., str]:
    """Return ``ollama_chat`` or ``openai_chat`` based on ``backend``."""
    b = (backend or "ollama").strip().lower()
    if b == "openai":
        from src.model.openai_client import openai_chat

        return openai_chat
    from src.model.ollama_client import ollama_chat

    return ollama_chat
