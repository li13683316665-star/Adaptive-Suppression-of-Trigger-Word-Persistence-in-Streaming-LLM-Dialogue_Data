"""
Method-layer runner: STD detector and ASC adaptive evaluation at calibrated tiers.

Consumes a calibration report to set per-(model, family) prompt difficulty, then
invokes ``07_detector_eval.py`` and ``08_adaptive_eval.py``. Typically run after
``09_calibration_sweep.py``. CLI supports skipping adaptive runs or overriding
difficulty. Execution: ``python experiments/10_run_method_layer.py``.
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

LOGGER = logging.getLogger("run_method_layer")

DEFAULT_MODELS = [
    "qwen3.5:4b",
    "gemma4:e4b",
    "openbmb/minicpm-v4.5:8b",
    "ministral-3:8b",
]
DEFAULT_FAMILIES = ["location", "emotion", "color"]
DIFFICULTY_ORDER = ["d1", "d2", "d3"]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run method layer with calibrated difficulties.")
    p.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    p.add_argument("--families", nargs="*", default=DEFAULT_FAMILIES)
    p.add_argument("--repeat", type=int, default=5, help="Repeats per condition.")
    p.add_argument("--host", default="http://127.0.0.1:11434")
    p.add_argument("--run-group", default="", help="Run group label.")
    p.add_argument(
        "--override-difficulty",
        default="",
        choices=["", "d1", "d2", "d3"],
        help="Force this difficulty for all models (ignores calibration report).",
    )
    p.add_argument("--skip-detector", action="store_true", help="Skip detector evaluation.")
    p.add_argument("--skip-adaptive", action="store_true", help="Skip adaptive evaluation.")
    p.add_argument("--skip-quality", action="store_true", help="Skip quality section in adaptive eval.")
    return p.parse_args()


def _load_latest_calibration(results_dir: Path) -> dict | None:
    reports = sorted(results_dir.glob("calibration_sweep_*.json"), key=lambda p: p.stat().st_mtime)
    if not reports:
        return None
    return json.loads(reports[-1].read_text(encoding="utf-8"))


def _get_difficulty(
    calibration: dict | None,
    model: str,
    family: str,
    override: str,
) -> str:
    if override:
        return override
    if calibration is None:
        LOGGER.warning("No calibration report found; defaulting to d1 for %s/%s", model, family)
        return "d1"
    rec = (
        calibration.get("results", {})
        .get(model, {})
        .get(family, {})
        .get("recommended", "d1")
    )
    if rec == "d3_insufficient":
        LOGGER.warning(
            "No difficulty met threshold for %s/%s; using d3 (best available)", model, family
        )
        return "d3"
    return rec or "d1"


def _run_cmd_capture(
    project_root: Path,
    results_dir: Path,
    cmd: list[str],
    glob_pattern: str,
) -> tuple[bool, list[str]]:
    """Run command, return (success, list_of_new_json_files_created)."""
    before = {p.name for p in results_dir.glob(glob_pattern)}
    LOGGER.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=project_root)
    after = {p.name for p in results_dir.glob(glob_pattern)}
    created = sorted(after - before)
    return result.returncode == 0, created


def main() -> None:
    args = _parse_args()
    setup_logging("INFO")
    project_root = get_project_root()
    results_dir = get_results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_group = args.run_group or f"method_layer_{timestamp}"

    calibration = _load_latest_calibration(results_dir)
    if calibration:
        LOGGER.info("Loaded calibration report: %s runs", len(calibration.get("results", {})))
    else:
        LOGGER.warning("No calibration report found. Using d1 for all or --override-difficulty value.")

    # crossmodel manifest for use by 04_build_paper_artifacts.py
    cm_manifest: dict = {
        "run_group": run_group,
        "created_at": timestamp,
        "models": args.models,
        "methods": [],
        "baseline_repeats": 0,
        "compare_repeats": 0,
        "quality_repeats": 0,
        "ablation_repeats": 0,
        "detector_repeats": args.repeat,
        "adaptive_repeats": args.repeat,
        "families": args.families,
        "artifacts": [],
    }
    run_log: list[dict] = []

    for model in args.models:
        model_slug = model.replace(":", "_").replace(".", "_")

        # Group families by difficulty so we can batch them per difficulty
        difficulty_to_families: dict[str, list[str]] = {}
        for family in args.families:
            diff = _get_difficulty(calibration, model, family, args.override_difficulty)
            difficulty_to_families.setdefault(diff, []).append(family)

        for difficulty, families in sorted(difficulty_to_families.items()):
            LOGGER.info(
                "Model=%s  difficulty=%s  families=%s", model, difficulty, families
            )

            if not args.skip_detector:
                stem = f"crossmodel_detector_{model_slug}_{difficulty}_methodlayer"
                cmd = [
                    sys.executable,
                    "experiments/07_detector_eval.py",
                    "--model", model,
                    "--host", args.host,
                    "--families", *families,
                    "--repeat", str(args.repeat),
                    "--prompt-difficulty", difficulty,
                    "--output-stem", stem,
                    "--run-group", run_group,
                ]
                success, created_files = _run_cmd_capture(
                    project_root, results_dir, cmd, "crossmodel_detector_*.json"
                )
                artifact = {
                    "kind": "detector",
                    "model": model,
                    "families": families,
                    "prompt_difficulty": difficulty,
                    "files": created_files,
                }
                cm_manifest["artifacts"].append(artifact)
                run_log.append({**artifact, "success": success})

            if not args.skip_adaptive:
                stem = f"crossmodel_adaptive_{model_slug}_{difficulty}_methodlayer"
                cmd = [
                    sys.executable,
                    "experiments/08_adaptive_eval.py",
                    "--model", model,
                    "--host", args.host,
                    "--families", *families,
                    "--repeat", str(args.repeat),
                    "--prompt-difficulty", difficulty,
                    "--output-stem", stem,
                    "--run-group", run_group,
                ]
                if args.skip_quality:
                    cmd.append("--skip-quality")
                success, created_files = _run_cmd_capture(
                    project_root, results_dir, cmd, "crossmodel_adaptive_*.json"
                )
                artifact = {
                    "kind": "adaptive",
                    "model": model,
                    "families": families,
                    "prompt_difficulty": difficulty,
                    "files": created_files,
                }
                cm_manifest["artifacts"].append(artifact)
                run_log.append({**artifact, "success": success})

    # Save crossmodel manifest (readable by 04_build_paper_artifacts.py)
    cm_manifest_path = results_dir / f"crossmodel_manifest_{run_group}.json"
    cm_manifest_path.write_text(json.dumps(cm_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("Crossmodel manifest saved to %s", cm_manifest_path)

    # Save detailed run log
    report = {
        "created_at": timestamp,
        "run_group": run_group,
        "manifest_path": str(cm_manifest_path),
        "run_log": run_log,
        "calibration_used": str(
            sorted(results_dir.glob("calibration_sweep_*.json"),
                   key=lambda p: p.stat().st_mtime)[-1]
        ) if calibration else None,
    }
    report_path = results_dir / f"method_layer_run_{timestamp}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("Method layer run log saved to %s", report_path)

    n_ok = sum(1 for e in run_log if e.get("success"))
    n_total = len(run_log)
    print(f"\nMethod layer run complete: {n_ok}/{n_total} sub-runs succeeded.")
    print(f"Run group: {run_group}")
    print(f"Crossmodel manifest: {cm_manifest_path}")


if __name__ == "__main__":
    main()
