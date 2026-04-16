"""Quick summary table of DeepSeek sanity benchmark results."""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data_new" / "results"


def main() -> None:
    files = sorted(RESULTS.glob("deepseek_sanity_*_deepseek_chat_r*_*.json"))
    if not files:
        print("No deepseek_sanity result files found.")
        return

    rows: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        for r in d.get("results", []):
            family = r.get("trigger_word", "?")
            scenario = r.get("scenario_type", r.get("id", "?"))
            pctp = r["metrics"].get("pctp")
            if pctp is not None:
                rows[family][scenario].append(pctp)

    print(f"{'family':<12} {'scenario':<25} {'n':>3} {'mean_pctp':>10} {'min':>6} {'max':>6}")
    print("-" * 70)
    for family in sorted(rows):
        for scenario in sorted(rows[family]):
            vals = rows[family][scenario]
            n = len(vals)
            mean = sum(vals) / n if n else 0
            lo = min(vals) if vals else 0
            hi = max(vals) if vals else 0
            print(f"{family:<12} {scenario:<25} {n:>3} {mean:>10.4f} {lo:>6.4f} {hi:>6.4f}")


if __name__ == "__main__":
    main()
