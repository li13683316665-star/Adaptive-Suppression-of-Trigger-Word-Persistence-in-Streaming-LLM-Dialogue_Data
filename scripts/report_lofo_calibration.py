"""
Leave-one-family-out (LOFO) calibration report from a calibration sweep JSON.

For each calibration model and each held-out development family F (location,
emotion, color), we use only the *other two* families' recommended difficulty
tiers to pick a conservative merged tier (max rank among the two), then report
the held-out family's mean PCTP at that tier (``mean_pctp`` in the sweep file).

This matches the manuscript's robustness narrative: thresholds are not
re-estimated on the held-out family; we only audit how a tier chosen from
two-families-only relates to the third family's observed means.

Usage:
  python scripts/report_lofo_calibration.py \\
    --calibration data_new/results/calibration_sweep_20260411_092539.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TIER_RANK = {"d1": 1, "d2": 2, "d3": 3}


def _tier_from_rank(r: int) -> str:
    for k, v in TIER_RANK.items():
        if v == r:
            return k
    return "d2"


def _max_tier(rec_a: str, rec_b: str) -> str:
    ra = TIER_RANK.get(rec_a, 2)
    rb = TIER_RANK.get(rec_b, 2)
    return _tier_from_rank(max(ra, rb))


def main() -> None:
    ap = argparse.ArgumentParser(description="LOFO calibration table from sweep JSON.")
    ap.add_argument(
        "--calibration",
        type=Path,
        default=ROOT / "data_new" / "results" / "calibration_sweep_20260411_092539.json",
    )
    args = ap.parse_args()
    path = args.calibration.resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    results = data.get("results", {})
    families = list(data.get("families", ["location", "emotion", "color"]))

    lines = [
        "# Leave-one-family-out calibration audit",
        "",
        f"- Source: `{path.relative_to(ROOT)}`",
        f"- `run_group`: `{data.get('run_group', '')}`",
        "",
        "For each model, when holding out family **F**, we merge the two other families' "
        "`recommended` tiers by taking the **more difficult** tier (higher d1<d2<d3 rank), "
        "then read the held-out family's `mean_pctp` at that tier.",
        "",
        "| model | held_out | tier_from_other_two | mean_pctp_at_tier (held_out) |",
        "|---|---|---|---|",
    ]

    for model, fam_block in sorted(results.items()):
        if not isinstance(fam_block, dict):
            continue
        for held in families:
            others = [f for f in families if f != held]
            if len(others) != 2:
                continue
            fa, fb = others
            block_a = fam_block.get(fa) or {}
            block_b = fam_block.get(fb) or {}
            block_h = fam_block.get(held) or {}
            rec_a = str(block_a.get("recommended", "d2"))
            rec_b = str(block_b.get("recommended", "d2"))
            merged = _max_tier(rec_a, rec_b)
            tier_block = block_h.get(merged) or {}
            mean_p = tier_block.get("mean_pctp", "")
            lines.append(
                f"| `{model}` | `{held}` | `{merged}` (from `{fa}`={rec_a}, `{fb}`={rec_b}) | {mean_p} |"
            )

    out = ROOT / "data_new" / "results" / "lofo_calibration_report.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
