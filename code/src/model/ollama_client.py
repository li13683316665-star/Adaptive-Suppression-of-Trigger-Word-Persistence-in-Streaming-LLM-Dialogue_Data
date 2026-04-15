"""Small Ollama client helpers for local chat-based experiments."""

from __future__ import annotations

import errno
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
_OLLAMA_CHAT_ATTEMPTS = max(1, int(os.getenv("OLLAMA_CHAT_ATTEMPTS", "4")))
_RETRY_BACKOFF_SEC = float(os.getenv("OLLAMA_RETRY_BACKOFF_SEC", "0.5"))

# region agent log
_AGENT_LOG = Path(__file__).resolve().parents[2] / "debug-a09cfa.log"


def agent_debug_log(
    message: str,
    *,
    hypothesis_id: str,
    data: dict[str, Any],
    location: str = "ollama_client.py",
) -> None:
    try:
        line = (
            json.dumps(
                {
                    "sessionId": "a09cfa",
                    "hypothesisId": hypothesis_id,
                    "location": location,
                    "message": message,
                    "data": data,
                    "timestamp": int(time.time() * 1000),
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        with open(_AGENT_LOG, "a", encoding="utf-8") as f:
            f.write(line)
    except OSError:
        pass


# endregion


def _transient_os_err(exc: OSError) -> bool:
    if isinstance(exc, (BrokenPipeError, ConnectionAbortedError, ConnectionResetError)):
        return True
    if getattr(exc, "winerror", None) == 10054:
        return True
    en = getattr(exc, "errno", None)
    return en in (errno.ECONNRESET, errno.EPIPE, errno.ECONNABORTED)


def _transient_url_err(exc: urllib.error.URLError) -> bool:
    r = exc.reason
    if isinstance(r, OSError):
        return _transient_os_err(r)
    return False


def ollama_chat(
    *,
    host: str = DEFAULT_OLLAMA_HOST,
    model: str,
    messages: list[dict[str, str]],
    generation_cfg: dict[str, Any] | None = None,
    options_override: dict[str, Any] | None = None,
) -> str:
    """Send a single non-streaming chat request to a local Ollama server.

    Thinking-capable models (e.g. Qwen 3, Gemma 4 in Ollama) default to ``think: false`` so
    token budget is not consumed by a hidden reasoning trace, which would otherwise
    leave ``message.content`` empty under small ``num_predict`` limits and break
    downstream metrics. Pass ``generation_cfg={"think": true}`` to enable.
    """
    generation_cfg = generation_cfg or {}
    options = {
        "temperature": generation_cfg.get("temperature", 0.7),
        "top_p": generation_cfg.get("top_p", 0.9),
        "repeat_penalty": generation_cfg.get("repetition_penalty", 1.0),
        "num_predict": generation_cfg.get("max_new_tokens", 256),
    }
    options.update(options_override or {})

    # Qwen3 / Gemma4 / other thinking models: without this, reasoning can fill the budget and
    # return empty assistant text (see Ollama "thinking" capability docs).
    think = generation_cfg.get("think", False)

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": options,
        "think": think,
    }
    payload_bytes = json.dumps(payload).encode("utf-8")
    body: dict[str, Any] | None = None
    chat_url = f"{host.rstrip('/')}/api/chat"

    for attempt in range(_OLLAMA_CHAT_ATTEMPTS):
        # region agent log
        agent_debug_log(
            "ollama_chat before urlopen",
            hypothesis_id="H2",
            data={
                "model": model,
                "host": host,
                "payload_bytes": len(payload_bytes),
                "num_messages": len(messages),
                "num_predict": options.get("num_predict"),
                "attempt": attempt,
            },
        )
        # endregion
        request = urllib.request.Request(
            url=chat_url,
            data=payload_bytes,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                body = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.URLError as exc:
            # region agent log
            agent_debug_log(
                "ollama_chat URLError",
                hypothesis_id="H3",
                data={
                    "reason": str(exc.reason) if getattr(exc, "reason", None) else None,
                    "model": model,
                    "attempt": attempt,
                },
            )
            # endregion
            if attempt < _OLLAMA_CHAT_ATTEMPTS - 1 and _transient_url_err(exc):
                # region agent log
                agent_debug_log(
                    "ollama_chat retry after URLError",
                    hypothesis_id="H1",
                    data={
                        "attempt": attempt,
                        "runId": "post-fix",
                        "sleep_sec": _RETRY_BACKOFF_SEC * (2**attempt),
                    },
                )
                # endregion
                time.sleep(_RETRY_BACKOFF_SEC * (2**attempt))
                continue
            raise RuntimeError(
                "Could not reach the Ollama server. Make sure Ollama is running locally."
            ) from exc
        except OSError as exc:
            # region agent log
            agent_debug_log(
                "ollama_chat OSError (incl. ConnectionResetError)",
                hypothesis_id="H1",
                data={
                    "exc_type": type(exc).__name__,
                    "errno": getattr(exc, "errno", None),
                    "winerror": getattr(exc, "winerror", None),
                    "model": model,
                    "payload_bytes": len(payload_bytes),
                    "attempt": attempt,
                },
            )
            # endregion
            if attempt < _OLLAMA_CHAT_ATTEMPTS - 1 and _transient_os_err(exc):
                # region agent log
                agent_debug_log(
                    "ollama_chat retry after OSError",
                    hypothesis_id="H1",
                    data={
                        "attempt": attempt,
                        "runId": "post-fix",
                        "sleep_sec": _RETRY_BACKOFF_SEC * (2**attempt),
                    },
                )
                # endregion
                time.sleep(_RETRY_BACKOFF_SEC * (2**attempt))
                continue
            raise
        except Exception as exc:
            # region agent log
            agent_debug_log(
                "ollama_chat unexpected exception",
                hypothesis_id="H5",
                data={"exc_type": type(exc).__name__, "repr": repr(exc)[:400], "model": model},
            )
            # endregion
            raise

    if body is None:
        raise RuntimeError("Ollama chat failed after retries.")

    return body["message"]["content"].strip()
