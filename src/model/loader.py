"""Load LLMs and tokenizers via Hugging Face Transformers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML config; defaults to ``configs/default.yaml``."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with cfg_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model_and_tokenizer(
    config: dict[str, Any] | None = None,
    config_path: str | Path | None = None,
):
    """
    Load model and tokenizer from Hugging Face.

    Uses quantization settings from config when ``quantization.enabled`` is true.
    Implement full loading logic when experiments begin (transformers, bitsandbytes).
    """
    raise NotImplementedError(
        "Wire up transformers.AutoModelForCausalLM and AutoTokenizer in experiments."
    )
