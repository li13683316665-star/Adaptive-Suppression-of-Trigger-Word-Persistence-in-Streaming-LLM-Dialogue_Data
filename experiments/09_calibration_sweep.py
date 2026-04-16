"""
Difficulty calibration sweep: estimate effective prompt tier per model and family.

For each (model, family, difficulty) cell, runs repeated baseline trials and
aggregates mean PCTP over non-control scenarios; recommends the lowest tier
whose mean exceeds a configurable threshold. Outputs ``calib_*.json`` and a
final ``calibration_sweep_*.json`` report. See CLI for ``--repeats`` and
``--threshold``. Execution: ``python experiments/09_calibration_sweep.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.helpers import get_project_root, get_results_dir, setup_logging  # noqa: E402

LOGGER = logging.getLogger("calibration_sweep")

DEFAULT_MODELS = [
    "qwen3.5:4b",
    "gemma4:e4b",
    "openbmb/minicpm-v4.5:8b",
    "ministral-3:8b",
]

# d1 = original files, d2/d3 = new difficulty-tiered files
DIFFICULTY_PROMPT_FILES: dict[str, dict[str, str]] = {
    "location": {
        "d1": "data/prompts/location_bias_cases.json",
        "d2": "data/prompts/location_bias_cases_d2.json",
        "d3": "data/prompts/location_bias_cases_d3.json",
    },
    "emotion": {
        "d1": "data/prompts/emotion_bias_cases.json",
        "d2": "data/prompts/emotion_bias_cases_d2.json",
        "d3": "data/prompts/emotion_bias_cases_d3.json",
    },
    "color": {
        "d1": "data/prompts/color_bias_cases.json",
        "d2": "data/prompts/color_bias_cases_d2.json",
        "d3": "data/prompts/color_bias_cases_d3.json",
    },
}

DIFFICULTY_ORDER = ["d1", "d2", "d3"]
NON_CONTROL_SCENARIOS = {"dialogue_triggered", "environment_triggered", "correction_recovery"}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run difficulty calibration sweep.")
    p.add_argument(
        "--models",
        nargs="*",
        default=DEFAULT_MODELS,
        help="Ollama model tags to calibrate.",
    )
    p.add_argument(
        "--families",
        nargs="*",
        default=list(DIFFICULTY_PROMPT_FILES.keys()),
        help="Trigger families to include.",
    )
    p.add_argument(
        "--difficulties",
        nargs="*",
        default=DIFFICULTY_ORDER,
        choices=DIFFICULTY_ORDER,
        help="Difficulty tiers to sweep through.",
    )
    p.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of baseline repetitions per (model, family, difficulty) cell.",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Minimum mean PCTP required to accept a difficulty tier.",
    )
    p.add_argument(
        "--host",
        default="http://127.0.0.1:11434",
        help="Ollama host URL.",
    )
    p.add_argument(
        "--run-group",
        default="",
        help="Optional run group label.",
    )
    return p.parse_args()


def _mean_pctp_from_file(result_file: Path) -> float:
    """Return mean PCTP across non-control scenarios; returns 0.0 if no valid cases."""
    payload = json.loads(result_file.read_text(encoding="utf-8"))
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


def _run_baseline(
    *,
    project_root: Path,
    results_dir: Path,
    model: str,
    host: str,
    prompt_file: str,
    output_stem: str,
    run_group: str,
    repeat_index: int,
) -> Path | None:
    """Invoke 01_baseline_bias.py and return the path of the newly created JSON file."""
    before = {p.name: p for p in results_dir.glob(f"{output_stem}_*.json")}
    cmd = [
        sys.executable,
        "experiments/01_baseline_bias.py",
        "--model", model,
        "--host", host,
        "--prompt-file", prompt_file,
        "--output-stem", output_stem,
        "--run-group", run_group,
        "--repeat-index", str(repeat_index),
    ]
    LOGGER.info("Running: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, cwd=project_root, check=True)
    except subprocess.CalledProcessError as exc:
        LOGGER.error("Subprocess failed: %s", exc)
        return None
    after = {p.name: p for p in results_dir.glob(f"{output_stem}_*.json")}
    new_files = sorted(set(after.keys()) - set(before.keys()))
    if not new_files:
        LOGGER.warning("No new JSON file created for stem %s", output_stem)
        return None
    return after[new_files[-1]]


def main() -> None:
    args = _parse_args()
    setup_logging("INFO")
    project_root = get_project_root()
    results_dir = get_results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_group = args.run_group or f"calibration_sweep_{timestamp}"

    # calibration_results[model][family] = {
    #   difficulty -> {"runs": [...pctp...], "mean_pctp": float},
    #   "recommended": "d1"|"d2"|"d3"|"none"
    # }
    calibration_results: dict[str, dict] = {}

    for model in args.models:
        model_slug = model.replace(":", "_").replace(".", "_")
        calibration_results[model] = {}

        for family in args.families:
            calibration_results[model][family] = {}
            recommended: str | None = None

            for difficulty in args.difficulties:
                prompt_file = DIFFICULTY_PROMPT_FILES[family].get(difficulty)
                if prompt_file is None:
                    LOGGER.warning("No prompt file for %s/%s, skipping.", family, difficulty)
                    continue

                pctp_per_repeat: list[float] = []

                for rep in range(1, args.repeats + 1):
                    stem = f"calib_{family}_{difficulty}_{model_slug}_r{rep:02d}"
                    out_file = _run_baseline(
                        project_root=project_root,
                        results_dir=results_dir,
                        model=model,
                        host=args.host,
                        prompt_file=prompt_file,
                        output_stem=stem,
                        run_group=run_group,
                        repeat_index=rep,
                    )
                    if out_file is not None:
                        mean_pctp = _mean_pctp_from_file(out_file)
                        pctp_per_repeat.append(mean_pctp)
                        LOGGER.info(
                            "  %s / %s / %s / rep %d -> mean_pctp=%.4f",
                            model, family, difficulty, rep, mean_pctp,
                        )

                if pctp_per_repeat:
                    overall_mean = sum(pctp_per_repeat) / len(pctp_per_repeat)
                else:
                    overall_mean = 0.0

                calibration_results[model][family][difficulty] = {
                    "runs": pctp_per_repeat,
                    "mean_pctp": round(overall_mean, 4),
                    "meets_threshold": overall_mean >= args.threshold,
                }

                if recommended is None and overall_mean >= args.threshold:
                    recommended = difficulty
                    LOGGER.info(
                        ">>> %s / %s: minimum effective difficulty = %s (mean_pctp=%.4f)",
                        model, family, difficulty, overall_mean,
                    )

            calibration_results[model][family]["recommended"] = recommended or "d3_insufficient"

    # Write calibration report
    report = {
        "created_at": timestamp,
        "run_group": run_group,
        "threshold": args.threshold,
        "repeats_per_cell": args.repeats,
        "models": args.models,
        "families": args.families,
        "results": calibration_results,
    }
    report_path = results_dir / f"calibration_sweep_{timestamp}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("Calibration report saved to %s", report_path)

    # Print summary table
    print("\n=== CALIBRATION SUMMARY ===")
    print(f"{'Model':<22} {'Family':<12} {'d1':>8} {'d2':>8} {'d3':>8} {'Recommended':>14}")
    print("-" * 76)
    for model in args.models:
        for family in args.families:
            family_data = calibration_results[model].get(family, {})
            row = [model[:21], family[:11]]
            for d in DIFFICULTY_ORDER:
                cell = family_data.get(d, {})
                val = cell.get("mean_pctp")
                row.append(f"{val:.4f}" if val is not None else "  n/a")
            row.append(family_data.get("recommended", "?"))
            print(f"{row[0]:<22} {row[1]:<12} {row[2]:>8} {row[3]:>8} {row[4]:>8} {row[5]:>14}")
    print()


if __name__ == "__main__":
    main()
