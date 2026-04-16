"""
DeepSeek API sanity-check benchmark for the ESWA submission.

Runs the same baseline prompt suites used by the local Ollama quartet against
a hosted DeepSeek model via the OpenAI-compatible API. This serves as the
"external stronger-model sanity check" called for in the manuscript.

All API credentials come from **environment variables only**:
  OPENAI_API_KEY   — bearer token (required)
  OPENAI_BASE_URL  — default https://api.deepseek.com

Usage (from repo root):
  set OPENAI_API_KEY=sk-...
  python experiments/15_deepseek_sanity_benchmark.py
  python experiments/15_deepseek_sanity_benchmark.py --families location emotion color --repeats 5
  python experiments/15_deepseek_sanity_benchmark.py --families animal weather food music shape --repeats 3
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

LOGGER = logging.getLogger("deepseek_sanity")

DEFAULT_MODEL = "deepseek-chat"
DEFAULT_BASE_URL = "https://api.deepseek.com"

FAMILY_PROMPTS: dict[str, str] = {
    "location": "data/prompts/location_bias_cases.json",
    "emotion": "data/prompts/emotion_bias_cases.json",
    "color": "data/prompts/color_bias_cases.json",
    "animal": "data/prompts/animal_bias_cases.json",
    "weather": "data/prompts/weather_bias_cases.json",
    "food": "data/prompts/food_bias_cases.json",
    "music": "data/prompts/music_bias_cases.json",
    "shape": "data/prompts/shape_bias_cases.json",
}

DEFAULT_DEV_FAMILIES = ["location", "emotion", "color"]
DEFAULT_HELDOUT_FAMILIES = ["animal", "weather"]


def _slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run baseline persistence benchmark against DeepSeek API.",
    )
    p.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI-compatible model name.")
    p.add_argument("--host", default=DEFAULT_BASE_URL, help="API base URL.")
    p.add_argument(
        "--families",
        nargs="*",
        default=DEFAULT_DEV_FAMILIES + DEFAULT_HELDOUT_FAMILIES,
        help="Trigger families to benchmark.",
    )
    p.add_argument("--repeats", type=int, default=3, help="Repeats per family.")
    p.add_argument("--run-group", default="", dest="run_group")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip cells that already have result JSON.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    setup_logging("INFO")
    project_root = get_project_root()
    results_dir = get_results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_group = args.run_group or f"deepseek_sanity_{ts}"
    model_slug = _slugify(args.model)

    manifest: dict[str, object] = {
        "run_group": run_group,
        "created_at": ts,
        "backend": "openai",
        "model": args.model,
        "host": args.host,
        "families": args.families,
        "repeats": args.repeats,
        "artifacts": [],
    }

    for family in args.families:
        rel = FAMILY_PROMPTS.get(family)
        if not rel:
            LOGGER.warning("Unknown family %s — skipping", family)
            continue
        pf = project_root / rel
        if not pf.exists():
            LOGGER.error("Missing prompt file: %s", pf)
            continue
        fam_slug = family.replace("-", "_")
        for r in range(1, args.repeats + 1):
            stem = f"deepseek_sanity_{fam_slug}_{model_slug}_r{r:02d}"
            existing = sorted(p.name for p in results_dir.glob(f"{stem}_*.json"))
            if args.resume and existing:
                LOGGER.info("Resume: skipping %s (found %d file(s))", stem, len(existing))
                manifest["artifacts"].append({
                    "kind": "deepseek_sanity",
                    "model": args.model,
                    "family": family,
                    "repeat_index": r,
                    "prompt_file": rel,
                    "files": existing,
                    "skipped": True,
                })
                continue

            before = {p.name for p in results_dir.glob(f"{stem}_*.json")}
            cmd = [
                sys.executable,
                "experiments/01_baseline_bias.py",
                "--backend",
                "openai",
                "--model", args.model,
                "--host", args.host,
                "--prompt-file", rel,
                "--output-stem", stem,
                "--run-group", run_group,
                "--repeat-index", str(r),
            ]
            LOGGER.info("Running: %s", " ".join(cmd))
            try:
                subprocess.run(cmd, cwd=project_root, check=True)
            except subprocess.CalledProcessError:
                LOGGER.error("Failed: %s repeat %d", family, r)
                continue
            after = {p.name for p in results_dir.glob(f"{stem}_*.json")}
            created = sorted(after - before)
            manifest["artifacts"].append({
                "kind": "deepseek_sanity",
                "model": args.model,
                "family": family,
                "repeat_index": r,
                "prompt_file": rel,
                "files": created,
            })

    out = results_dir / f"deepseek_sanity_manifest_{run_group}.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("Wrote manifest %s", out)
    print(f"\nManifest: {out}")
    print(f"Artifacts: {len(manifest['artifacts'])} cells")


if __name__ == "__main__":
    main()
