"""
Audit adaptive-vs-baseline paired rows in ``paper_method_paired*.json``.

Verifies that each comparison has equal repeat counts for baseline and adaptive
vectors (paired permutation tests assume aligned indices). Prints a short summary
and optional list of mismatches.

Usage:
  python scripts/audit_adaptive_static_pairing.py \\
    --paired data_new/results/paper_method_paired_minicpm_v45_incr_full_eswa_20260415_051850_suite.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit paired baseline vs adaptive counts.")
    ap.add_argument(
        "--paired",
        type=Path,
        default=ROOT
        / "data_new"
        / "results"
        / "paper_method_paired_minicpm_v45_incr_full_eswa_20260415_051850_suite.json",
    )
    ap.add_argument("--max-mismatches", type=int, default=20)
    args = ap.parse_args()
    path = args.paired.resolve()
    rows = json.loads(path.read_text(encoding="utf-8"))

    mismatches: list[str] = []
    ok = 0
    by_model: dict[str, int] = defaultdict(int)

    for row in rows:
        comp = row.get("comparison") or {}
        b = comp.get("baseline") or {}
        a = comp.get("adaptive") or {}
        nb = int(b.get("count") or 0)
        na = int(a.get("count") or 0)
        model = str(row.get("model", ""))
        if nb == na and nb > 0:
            ok += 1
            by_model[model] += 1
        else:
            key = f"{model}|{row.get('family')}|{row.get('case_id')}|baseline={nb}|adaptive={na}"
            mismatches.append(key)

    print(f"File: {path.name}")
    print(f"Rows with equal positive counts: {ok} / {len(rows)}")
    print("Per-model paired rows:", dict(by_model))
    if mismatches:
        print(f"Mismatches (showing up to {args.max_mismatches}):")
        for m in mismatches[: args.max_mismatches]:
            print("  ", m)
        if len(mismatches) > args.max_mismatches:
            print(f"  ... and {len(mismatches) - args.max_mismatches} more")
    else:
        print("No count mismatches.")


if __name__ == "__main__":
    main()
