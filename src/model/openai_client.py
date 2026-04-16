"""OpenAI-compatible chat client for DeepSeek / OpenAI / other hosted APIs.

Drop-in alternative to ``ollama_chat()`` with the same return type (``str``).
All secrets come from environment variables — never hard-code API keys.

Environment variables:
    OPENAI_API_KEY   (required) Bearer token for the API.
    OPENAI_BASE_URL  (optional) Default ``https://api.deepseek.com``.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from typing import Any

LOGGER = logging.getLogger(__name__)

DEFAULT_OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
_MAX_ATTEMPTS = max(1, int(os.getenv("OPENAI_CHAT_ATTEMPTS", "4")))
_RETRY_BACKOFF_SEC = float(os.getenv("OPENAI_RETRY_BACKOFF_SEC", "1.0"))

_TRANSIENT_HTTP_CODES = {429, 500, 502, 503, 504}


def _get_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it before running:\n"
            "  set OPENAI_API_KEY=sk-..."
        )
    return key


def openai_chat(
    *,
    host: str = "",
    model: str,
    messages: list[dict[str, str]],
    generation_cfg: dict[str, Any] | None = None,
    options_override: dict[str, Any] | None = None,
) -> str:
    """Send a single non-streaming chat completion request.

    Parameters match ``ollama_chat`` so callers can swap backends with minimal
    changes.  ``host`` maps to the base URL (falls back to ``OPENAI_BASE_URL``
    env / ``https://api.deepseek.com``).
    """
    generation_cfg = generation_cfg or {}
    base = (host or DEFAULT_OPENAI_BASE_URL).rstrip("/")
    url = f"{base}/chat/completions"
    api_key = _get_api_key()

    ovr = dict(options_override or {})
    temperature = float(ovr.get("temperature", generation_cfg.get("temperature", 0.7)))
    top_p = float(ovr.get("top_p", generation_cfg.get("top_p", 0.9)))
    max_tokens = int(ovr.get("num_predict", generation_cfg.get("max_new_tokens", 256)))
    rep = float(ovr.get("repeat_penalty", generation_cfg.get("repetition_penalty", 1.0)))
    frequency_penalty = min(max(rep - 1.0, -2.0), 2.0)
    if "frequency_penalty" in ovr:
        frequency_penalty = float(ovr["frequency_penalty"])

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "frequency_penalty": frequency_penalty,
    }

    payload_bytes = json.dumps(payload).encode("utf-8")
    body: dict[str, Any] | None = None

    for attempt in range(_MAX_ATTEMPTS):
        req = urllib.request.Request(
            url,
            data=payload_bytes,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            code = exc.code
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            LOGGER.warning(
                "openai_chat HTTP %d (attempt %d/%d): %s",
                code, attempt + 1, _MAX_ATTEMPTS, detail,
            )
            if code in _TRANSIENT_HTTP_CODES and attempt < _MAX_ATTEMPTS - 1:
                sleep = _RETRY_BACKOFF_SEC * (2 ** attempt)
                LOGGER.info("Retrying in %.1fs …", sleep)
                time.sleep(sleep)
                continue
            raise RuntimeError(
                f"OpenAI-compatible API returned HTTP {code}: {detail}"
            ) from exc
        except (urllib.error.URLError, OSError) as exc:
            LOGGER.warning(
                "openai_chat network error (attempt %d/%d): %s",
                attempt + 1, _MAX_ATTEMPTS, exc,
            )
            if attempt < _MAX_ATTEMPTS - 1:
                sleep = _RETRY_BACKOFF_SEC * (2 ** attempt)
                time.sleep(sleep)
                continue
            raise RuntimeError(
                "Could not reach the OpenAI-compatible API. Check OPENAI_BASE_URL and network."
            ) from exc

    if body is None:
        raise RuntimeError("OpenAI-compatible chat failed after retries.")

    try:
        return body["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            f"Unexpected response shape: {json.dumps(body, ensure_ascii=False)[:600]}"
        ) from exc
