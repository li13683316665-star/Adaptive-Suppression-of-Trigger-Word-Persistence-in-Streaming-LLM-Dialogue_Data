"""
Audit ``paired_comparisons`` in raw ``crossmodel_adaptive_*.json`` payloads.

Validates that each row has equal repeat counts for baseline vs adaptive (and
that ``paired_test.count`` matches when present). For manuscript statistics,
``paper_method_paired_*.json`` rows are the canonical export; this script is a
descriptive cross-check on upstream ASC evaluation JSON.

Usage:
  python scripts/audit_crossmodel_adaptive_pairing.py \\
    --glob \"data_new/results/crossmodel_adaptive_deepseek_chat_r*.json\" \\
    --out-md data_new/results/crossmodel_adaptive_pairing_audit_deepseek.md
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _audit_file(path: Path) -> tuple[int, int, list[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    pairs = data.get("paired_comparisons") or []
    ok = 0
    bad = 0
    issues: list[str] = []
    for i, row in enumerate(pairs):
        comp = row.get("comparison") or {}
        b = comp.get("baseline") or {}
        a = comp.get("adaptive") or {}
        nb = int(b.get("count") or 0)
        na = int(a.get("count") or 0)
        pt = comp.get("paired_test") or {}
        ptc = int(pt.get("count") or 0)
        key = f"{path.name} row {i} {row.get('family')}|{row.get('case_id')}|{row.get('metric')}"
        if nb != na:
            bad += 1
            issues.append(f"{key}: baseline_count={nb} adaptive_count={na}")
            continue
        if ptc and ptc != nb:
            bad += 1
            issues.append(f"{key}: paired_test.count={ptc} vs baseline.count={nb}")
            continue
        ok += 1
    return ok, bad, issues


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit crossmodel_adaptive paired rows.")
    ap.add_argument(
        "--glob",
        dest="glob_pat",
        default=str(ROOT / "data_new" / "results" / "crossmodel_adaptive_deepseek_chat_r*.json"),
        help="Glob relative to repo root (quoted on shells that expand *).",
    )
    ap.add_argument(
        "--out-md",
        type=Path,
        default=ROOT / "data_new" / "results" / "crossmodel_adaptive_pairing_audit.md",
    )
    ap.add_argument("--max-issues", type=int, default=40)
    args = ap.parse_args()

    pattern = (ROOT / args.glob_pat).as_posix() if not Path(args.glob_pat).is_absolute() else args.glob_pat
    paths = sorted(Path(p) for p in glob.glob(pattern))
    if not paths:
        print(f"No files matched: {pattern}")
        raise SystemExit(1)

    total_ok = 0
    total_bad = 0
    lines = [
        "# Cross-model adaptive pairing audit (raw JSON)",
        "",
        f"- Pattern: `{args.glob_pat}`",
        f"- Files: {len(paths)}",
        "Per file: rows where baseline and adaptive counts match, and ``paired_test.count`` "
        "matches baseline when present, are counted as **ok**. Unequal counts or mismatched "
        "paired_test counts are **bad**.",
        "",
    ]

    for p in paths:
        ok, bad, issues = _audit_file(p)
        total_ok += ok
        total_bad += bad
        lines.append(f"## `{p.name}`")
        lines.append("")
        lines.append(f"- ok rows: {ok}")
        lines.append(f"- bad rows: {bad}")
        lines.append("")
        if issues:
            show = issues[: args.max_issues]
            for s in show:
                lines.append(f"- {s}")
            if len(issues) > len(show):
                lines.append(f"- … {len(issues) - len(show)} more …")
        lines.append("")

    lines.insert(4, f"- **Aggregate** ok rows: {total_ok}; bad rows: {total_bad}")
    lines.insert(5, "")

    out = args.out_md.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out.relative_to(ROOT)}")
    print(f"Aggregate: ok={total_ok} bad={total_bad} across {len(paths)} file(s)")
    if total_bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
