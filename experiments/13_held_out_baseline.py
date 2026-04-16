"""
Held-out trigger families: baseline-only generalization benchmark.

Runs the baseline script over families excluded from calibration (e.g.\
animal, weather) without re-tuning STD weights, to assess external validity.
Produces a manifest JSON under ``data/results/``. Execution:
``python experiments/13_held_out_baseline.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.helpers import get_project_root, get_results_dir, setup_logging  # noqa: E402

LOGGER = logging.getLogger("held_out_baseline")

DEFAULT_MODELS = [
    "qwen3.5:4b",
    "gemma4:e4b",
    "openbmb/minicpm-v4.5:8b",
    "ministral-3:8b",
]
DEFAULT_FAMILIES = ["animal", "weather", "food", "music", "shape"]

FAMILY_PROMPTS: dict[str, str] = {
    "animal": "data/prompts/animal_bias_cases.json",
    "weather": "data/prompts/weather_bias_cases.json",
    "food": "data/prompts/food_bias_cases.json",
    "music": "data/prompts/music_bias_cases.json",
    "shape": "data/prompts/shape_bias_cases.json",
}


def _slugify_model(model: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Held-out family baseline runs.")
    p.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    p.add_argument("--families", nargs="*", default=DEFAULT_FAMILIES)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--host", default="http://127.0.0.1:11434")
    p.add_argument("--run-group", default="", dest="run_group")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip when heldout_baseline_<stem>_*.json already exists for that output stem.",
    )
    return p.parse_args()


def _run_and_capture(
    *,
    project_root: Path,
    results_dir: Path,
    command: list[str],
    result_glob: str,
    output_stem: str,
    resume: bool,
) -> list[str]:
    existing_json = sorted(path.name for path in results_dir.glob(f"{output_stem}_*.json"))
    if resume and existing_json:
        LOGGER.info(
            "Resume: skipping %s (found %d existing JSON file(s))",
            output_stem,
            len(existing_json),
        )
        return existing_json
    before = {path.name for path in results_dir.glob(result_glob)}
    LOGGER.info("Running: %s", " ".join(command))
    subprocess.run(command, cwd=project_root, check=True)
    after = {path.name for path in results_dir.glob(result_glob)}
    return sorted(after - before)


def main() -> None:
    args = _parse_args()
    setup_logging("INFO")
    project_root = get_project_root()
    results_dir = get_results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_group = args.run_group or f"held_out_{ts}"

    manifest: dict[str, object] = {
        "run_group": run_group,
        "created_at": ts,
        "models": args.models,
        "families": args.families,
        "repeats": args.repeats,
        "artifacts": [],
    }

    for model in args.models:
        slug = _slugify_model(model)
        for family in args.families:
            rel = FAMILY_PROMPTS.get(family)
            if not rel:
                LOGGER.warning("Unknown family %s", family)
                continue
            pf = project_root / rel
            if not pf.exists():
                LOGGER.error("Missing prompt file: %s", pf)
                continue
            fam_slug = family.replace("-", "_")
            for r in range(1, args.repeats + 1):
                stem = f"heldout_baseline_{fam_slug}_{slug}_r{r:02d}"
                created = _run_and_capture(
                    project_root=project_root,
                    results_dir=results_dir,
                    result_glob="heldout_baseline_*.json",
                    output_stem=stem,
                    resume=args.resume,
                    command=[
                        sys.executable,
                        "experiments/01_baseline_bias.py",
                        "--model",
                        model,
                        "--host",
                        args.host,
                        "--prompt-file",
                        rel,
                        "--output-stem",
                        stem,
                        "--run-group",
                        run_group,
                        "--repeat-index",
                        str(r),
                    ],
                )
                manifest["artifacts"].append(
                    {
                        "kind": "held_out_baseline",
                        "model": model,
                        "family": family,
                        "repeat_index": r,
                        "prompt_file": rel,
                        "files": created,
                    }
                )

    out = results_dir / f"held_out_manifest_{run_group}.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("Wrote manifest %s", out)


if __name__ == "__main__":
    main()
