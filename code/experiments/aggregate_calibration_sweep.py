"""
Aggregate difficulty calibration into a single ``calibration_sweep_<timestamp>.json``.

Consumes legacy ``calib_*.json`` outputs (from partial or completed sweeps),
recomputes per-cell mean PCTP and recommended difficulty tiers, and matches the
schema of ``experiments/09_calibration_sweep.py``. Use when a full sweep was
interrupted or when merging historical calibration files.

Execution: ``python experiments/aggregate_calibration_sweep.py`` (optional
``--threshold`` for the acceptance floor on mean PCTP).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.helpers import get_results_dir  # noqa: E402

# Must match 09_calibration_sweep.py
NON_CONTROL_SCENARIOS = {"dialogue_triggered", "environment_triggered", "correction_recovery"}
DIFFICULTY_ORDER = ["d1", "d2", "d3"]
DEFAULT_FAMILIES = ["location", "emotion", "color"]

FILENAME_RE = re.compile(
    r"^calib_(location|emotion|color)_(d[123])_(.+)_r(\d+)_\d{8}_\d{6}\.json$"
)


def _mean_pctp_from_payload(payload: dict) -> float:
    pctp_values: list[float] = []
    for result in payload.get("results", []):
        if result.get("scenario_type") not in NON_CONTROL_SCENARIOS:
            continue
        metrics = result.get("metrics", {})
        pctp = metrics.get("pctp")
        if pctp is not None:
            pctp_values.append(float(pctp))
    if not pctp_values:
        return 0.0
    return sum(pctp_values) / len(pctp_values)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate calib_* JSONs into calibration_sweep_*.json")
    p.add_argument("--threshold", type=float, default=0.05)
    p.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Defaults to get_results_dir() (see RSE_RESULTS_DIR; normally data_new/results).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    results_dir = args.results_dir or get_results_dir()
    # runs[model][family][difficulty][rep_index] = mean_pctp
    runs: dict[str, dict[str, dict[str, dict[int, float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))
    )

    for path in sorted(results_dir.glob("calib_*.json")):
        m = FILENAME_RE.match(path.name)
        if not m:
            continue
        family, difficulty, _slug, rep_s = m.groups()
        rep = int(rep_s)
        payload = json.loads(path.read_text(encoding="utf-8"))
        model = payload.get("model")
        if not model:
            continue
        mean_p = _mean_pctp_from_payload(payload)
        runs[model][family][difficulty][rep] = mean_p

    models = sorted(runs.keys())
    calibration_results: dict[str, dict] = {}

    for model in models:
        calibration_results[model] = {}
        for family in DEFAULT_FAMILIES:
            calibration_results[model][family] = {}
            recommended: str | None = None
            for difficulty in DIFFICULTY_ORDER:
                cell = runs[model][family].get(difficulty, {})
                if not cell:
                    calibration_results[model][family][difficulty] = {
                        "runs": [],
                        "mean_pctp": None,
                        "meets_threshold": False,
                    }
                    continue
                sorted_reps = sorted(cell.keys())
                pctp_per_repeat = [cell[r] for r in sorted_reps]
                overall_mean = sum(pctp_per_repeat) / len(pctp_per_repeat)
                meets = overall_mean >= args.threshold
                calibration_results[model][family][difficulty] = {
                    "runs": [round(x, 4) for x in pctp_per_repeat],
                    "mean_pctp": round(overall_mean, 4),
                    "meets_threshold": meets,
                }
                if recommended is None and meets:
                    recommended = difficulty
            calibration_results[model][family]["recommended"] = recommended or "d3_insufficient"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "created_at": timestamp,
        "run_group": f"aggregated_from_existing_{timestamp}",
        "threshold": args.threshold,
        "repeats_per_cell": "variable_from_filenames",
        "models": models,
        "families": DEFAULT_FAMILIES,
        "results": calibration_results,
        "note": "Built by aggregate_calibration_sweep.py from existing calib_*.json files.",
    }
    out = results_dir / f"calibration_sweep_{timestamp}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out}")
    print("\n=== CALIBRATION SUMMARY (aggregated) ===")
    print(f"{'Model':<22} {'Family':<12} {'d1':>8} {'d2':>8} {'d3':>8} {'Recommended':>14}")
    print("-" * 76)
    for model in models:
        for family in DEFAULT_FAMILIES:
            family_data = calibration_results[model].get(family, {})
            row = [model[:21], family[:11]]
            for d in DIFFICULTY_ORDER:
                cell = family_data.get(d, {})
                val = cell.get("mean_pctp")
                row.append(f"{val:.4f}" if val is not None else "  n/a")
            row.append(str(family_data.get("recommended", "?")))
            print(f"{row[0]:<22} {row[1]:<12} {row[2]:>8} {row[3]:>8} {row[4]:>8} {row[5]:>14}")


if __name__ == "__main__":
    main()
