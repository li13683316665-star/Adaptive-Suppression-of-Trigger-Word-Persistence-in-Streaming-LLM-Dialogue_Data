"""
Flatten ``data/results/paper_*.json`` (or any list/dict JSON) to CSV for manual tables.

Uses pandas ``json_normalize`` so nested keys become ``pctp_mean``, ``pctp_ci95_low``, etc.

Usage (from repo root)::

    python scripts/export_paper_json_to_csv.py data/results/paper_crossmodel_baseline_three_small_models_smoke.json
    python scripts/export_paper_json_to_csv.py data/results/paper_crossmodel_compare_three_small_models_smoke.json -o out/compare.csv

Also works if your shell is in ``scripts/`` (paths are resolved against the project root next to ``scripts/``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_json_path(p: Path) -> Path:
    """Try cwd-relative path, then project-root-relative (fixes running from ``scripts/``)."""
    if p.is_file():
        return p.resolve()
    root = _repo_root()
    cand = (root / p).resolve()
    if cand.is_file():
        return cand
    # Re-raise a clear error
    tried = [str(p.resolve()), str(cand)]
    raise FileNotFoundError(
        "JSON not found. Tried:\n  " + "\n  ".join(tried) + "\n"
        "Run from the project root, or pass a full path to the file."
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export paper JSON to flattened CSV.")
    p.add_argument(
        "json_path",
        type=Path,
        help="Path to JSON (typically paper_crossmodel_*.json or paper_method_*.json).",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. Default: same stem as input with .csv",
    )
    p.add_argument(
        "--sep",
        default="_",
        help="Nested key separator for column names (pandas json_normalize).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        import pandas as pd
    except ImportError as e:
        sys.stderr.write("Requires pandas: pip install pandas\n")
        raise SystemExit(1) from e

    src = _resolve_json_path(args.json_path)
    text = src.read_text(encoding="utf-8")
    data = json.loads(text)

    if isinstance(data, list):
        df = pd.json_normalize(data, sep=args.sep)
    elif isinstance(data, dict):
        # Single object or wrapper — try common shapes
        if "rows" in data and isinstance(data["rows"], list):
            df = pd.json_normalize(data["rows"], sep=args.sep)
        elif "tests_per_model" in data and "rows" in data:
            # e.g. paper_method_paired_adjusted_*
            df = pd.json_normalize(data["rows"], sep=args.sep)
        else:
            df = pd.json_normalize([data], sep=args.sep)
    else:
        raise SystemExit(f"Unsupported JSON root type: {type(data).__name__}")

    if args.output:
        out = args.output
        if not out.is_absolute():
            out = (_repo_root() / out).resolve()
    else:
        out = src.with_suffix(".csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    abs_out = out.resolve()
    print(f"OK: wrote {abs_out}")
    print(f"     rows={len(df)}, columns={len(df.columns)}")
    print(f"     source JSON: {src}")


if __name__ == "__main__":
    main()
