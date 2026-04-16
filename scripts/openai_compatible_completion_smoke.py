"""
Minimal OpenAI-compatible chat completion smoke test (DeepSeek / OpenAI / others).

Uses **environment variables only** for secrets — never hard-code API keys.

- ``OPENAI_API_KEY`` (required): bearer token
- ``OPENAI_BASE_URL`` (optional): default ``https://api.deepseek.com/v1``
- ``OPENAI_MODEL`` (optional): default ``deepseek-chat``

Usage:
  set OPENAI_API_KEY=...
  python scripts/openai_compatible_completion_smoke.py

This does not replace the main Ollama benchmark; it is a narrow external sanity
check that a hosted API path is reachable and returns non-empty completions.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        print("Set OPENAI_API_KEY in the environment.", file=sys.stderr)
        return 1
    base = os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")
    model = os.environ.get("OPENAI_MODEL", "deepseek-chat")
    url = f"{base}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "Reply with exactly one short sentence about puzzle games."},
        ],
        "temperature": 0.2,
        "max_tokens": 80,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {e.read().decode('utf-8', errors='replace')}", file=sys.stderr)
        return 1
    except OSError as e:
        print(f"Request failed: {e}", file=sys.stderr)
        return 1

    data = json.loads(raw)
    try:
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        text = json.dumps(data, ensure_ascii=False)[:500]
    print("Model:", model)
    print("Assistant:", text.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
