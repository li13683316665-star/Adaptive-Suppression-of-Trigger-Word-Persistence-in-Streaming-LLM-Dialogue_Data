"""Logging, paths, and small plotting helpers."""

from __future__ import annotations

import logging
import os
from pathlib import Path


def get_project_root() -> Path:
    """Repository root (parent of ``src``)."""
    return Path(__file__).resolve().parents[2]


def get_results_dir() -> Path:
    """Directory for experiment JSON/MD artifacts (cross-model, calibration, paper exports).

    Override with env ``RSE_RESULTS_DIR`` (path relative to project root, or absolute).
    Default: ``<repo>/data_new/results`` so fresh benchmark runs stay separate from legacy
    ``data/results``. Use ``set RSE_RESULTS_DIR=data/results`` (Windows) to read/write the old tree.
    """
    root = get_project_root()
    override = os.environ.get("RSE_RESULTS_DIR", "").strip()
    if override:
        p = Path(override)
        return (p if p.is_absolute() else root / p).resolve()
    return (root / "data_new" / "results").resolve()


def setup_logging(level: str = "INFO") -> None:
    """Configure basic logging for experiments."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
