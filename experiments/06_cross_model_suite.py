"""
Cross-model benchmark orchestration (baseline, compare, quality, ablations, methods).

Runs repeated trials per model and trigger family, invokes baseline and
comparison scripts, optional detector/adaptive layers, and records a manifest
JSON for downstream artifact generation. Execution:
``python experiments/06_cross_model_suite.py``.
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

LOGGER = logging.getLogger("cross_model_suite")

DEFAULT_MODELS = [
    "qwen3.5:4b",
    "gemma4:e4b",
    "openbmb/minicpm-v4.5:8b",
    "ministral-3:8b",
]
DEFAULT_METHODS = [
    "baseline",
    "context_reset",
    "targeted_guardrail",
    "repetition_penalty",
]
DEFAULT_FAMILIES = [
    "location",
    "emotion",
    "color",
]
DEFAULT_RUN_GROUP_PREFIX = "cross_model_expansion"

# Prompt files for each trigger family (baseline + compare)
FAMILY_PROMPT_MAP: dict[str, str] = {
    "location": "data/prompts/location_bias_cases.json",
    "emotion": "data/prompts/emotion_bias_cases.json",
    "color": "data/prompts/color_bias_cases.json",
}

FAMILY_PROMPT_MAP_TIERED: dict[str, dict[str, str]] = {
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the expanded cross-model experiment suite.")
    parser.add_argument(
        "--models",
        nargs="*",
        default=DEFAULT_MODELS,
        help="Ollama model tags to evaluate.",
    )
    parser.add_argument(
        "--host",
        default="http://127.0.0.1:11434",
        help="Base URL for the local Ollama server.",
    )
    parser.add_argument(
        "--baseline-repeats",
        type=int,
        default=10,
        help="Number of baseline repetitions per model on the main benchmark.",
    )
    parser.add_argument(
        "--compare-repeats",
        type=int,
        default=5,
        help="Number of mitigation comparison repetitions per model.",
    )
    parser.add_argument(
        "--quality-repeats",
        type=int,
        default=3,
        help="Number of quality-retention repetitions per model.",
    )
    parser.add_argument(
        "--ablation-repeats",
        type=int,
        default=5,
        help="Number of baseline repetitions per model on each ablation prompt file.",
    )
    parser.add_argument(
        "--detector-repeats",
        type=int,
        default=10,
        help="Number of detector-evaluation repetitions per model.",
    )
    parser.add_argument(
        "--adaptive-repeats",
        type=int,
        default=10,
        help="Number of adaptive-evaluation repetitions per model.",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=DEFAULT_METHODS,
        help="Mitigation methods passed to the comparison and quality scripts.",
    )
    parser.add_argument(
        "--families",
        nargs="*",
        default=DEFAULT_FAMILIES,
        help="Trigger families passed to baseline/compare/detector/adaptive evaluation.",
    )
    parser.add_argument(
        "--prompt-difficulty",
        default="d1",
        choices=["d1", "d2", "d3"],
        help="Prompt difficulty tier used for baseline and compare runs (d1=original).",
    )
    parser.add_argument(
        "--ablation-only",
        action="store_true",
        help="Only run density/channel ablation baselines (skip benchmark, quality, detector, adaptive).",
    )
    parser.add_argument(
        "--skip-ablation",
        action="store_true",
        help="Skip the density/channel ablation prompt files.",
    )
    parser.add_argument(
        "--skip-methods",
        action="store_true",
        help="Skip detector/adaptive method experiments.",
    )
    parser.add_argument(
        "--skip-quality",
        action="store_true",
        help="Skip quality-retention runs (use when only filling benchmark families, e.g. extra trigger families).",
    )
    parser.add_argument(
        "--run-group",
        default="",
        help="Optional explicit run group label. Defaults to a timestamped cross-model label.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip a sub-run when a JSON result already exists for the same --output-stem "
            "(any timestamp). Use after a failure to continue without re-doing finished repeats."
        ),
    )
    parser.add_argument(
        "--backend",
        choices=["ollama", "openai"],
        default="ollama",
        help="Chat backend for all subprocess scripts (Ollama or OpenAI-compatible e.g. DeepSeek).",
    )
    ns = parser.parse_args()
    if getattr(ns, "ablation_only", False) and ns.skip_ablation:
        parser.error("--ablation-only cannot be combined with --skip-ablation")
    return ns


def _slugify_model(model: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")


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
    LOGGER.info("Running command: %s", " ".join(command))
    subprocess.run(command, cwd=project_root, check=True)
    after = {path.name for path in results_dir.glob(result_glob)}
    created = sorted(after - before)
    return created


def _merge_manifest_artifacts(
    previous: dict[str, object], current: dict[str, object]
) -> dict[str, object]:
    """Merge artifact lists when reusing an existing run_group.

    This prevents an ablation-only or resume-only rerun from overwriting the older
    baseline/compare/quality/method artifacts that were already recorded under the
    same manifest file.
    """
    merged = dict(previous)
    current_artifacts = list(current.get("artifacts", []))
    prior_artifacts = list(previous.get("artifacts", []))

    seen_keys: set[tuple[str, str, int, str, tuple[str, ...]]] = set()
    merged_artifacts: list[dict[str, object]] = []
    for artifact in prior_artifacts + current_artifacts:
        if not isinstance(artifact, dict):
            continue
        key = (
            str(artifact.get("kind", "")),
            str(artifact.get("model", "")),
            int(artifact.get("repeat_index", 0) or 0),
            str(artifact.get("prompt_file", "")),
            tuple(str(x) for x in artifact.get("files", []) or []),
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        merged_artifacts.append(dict(artifact))

    merged["artifacts"] = merged_artifacts
    for field in (
        "backend",
        "methods",
        "baseline_repeats",
        "compare_repeats",
        "quality_repeats",
        "ablation_repeats",
        "detector_repeats",
        "adaptive_repeats",
        "families",
        "prompt_difficulty",
    ):
        merged[field] = current.get(field, merged.get(field))

    models = []
    for source in (previous.get("models", []), current.get("models", [])):
        for model in source:
            if model not in models:
                models.append(model)
    merged["models"] = models
    merged["run_group"] = current.get("run_group", previous.get("run_group"))
    merged["created_at"] = current.get("created_at", previous.get("created_at"))
    return merged


def main() -> None:
    args = _parse_args()
    setup_logging("INFO")
    project_root = get_project_root()
    results_dir = get_results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_group = args.run_group or f"{DEFAULT_RUN_GROUP_PREFIX}_{timestamp}"

    # Resolve prompt files for each family at the requested difficulty tier
    def _family_prompt(family: str) -> str:
        tiered = FAMILY_PROMPT_MAP_TIERED.get(family, {})
        return tiered.get(args.prompt_difficulty) or FAMILY_PROMPT_MAP.get(family, "")

    benchmark_families = args.families  # baseline + compare run for all specified families
    quality_prompt = "data/prompts/quality_retention_cases.json"
    ablation_prompts = [
        ("density", "data/prompts/location_ablation_density_cases.json"),
        ("channel", "data/prompts/location_ablation_channel_cases.json"),
    ]

    manifest: dict[str, object] = {
        "run_group": run_group,
        "created_at": timestamp,
        "backend": args.backend,
        "models": args.models,
        "methods": args.methods,
        "baseline_repeats": args.baseline_repeats,
        "compare_repeats": args.compare_repeats,
        "quality_repeats": args.quality_repeats,
        "ablation_repeats": args.ablation_repeats,
        "detector_repeats": args.detector_repeats,
        "adaptive_repeats": args.adaptive_repeats,
        "families": args.families,
        "prompt_difficulty": args.prompt_difficulty,
        "artifacts": [],
    }

    for model in args.models:
        model_slug = _slugify_model(model)

        if not args.ablation_only:
            for family in benchmark_families:
                family_prompt = _family_prompt(family)
                family_slug = family.replace("-", "_")

                for repeat_index in range(1, args.baseline_repeats + 1):
                    stem = f"crossmodel_baseline_{family_slug}_{model_slug}_r{repeat_index:02d}"
                    created = _run_and_capture(
                        project_root=project_root,
                        results_dir=results_dir,
                        result_glob="crossmodel_baseline_*.json",
                        output_stem=stem,
                        resume=args.resume,
                        command=[
                            sys.executable,
                            "experiments/01_baseline_bias.py",
                            "--backend",
                            args.backend,
                            "--model",
                            model,
                            "--host",
                            args.host,
                            "--prompt-file",
                            family_prompt,
                            "--output-stem",
                            stem,
                            "--run-group",
                            run_group,
                            "--repeat-index",
                            str(repeat_index),
                        ],
                    )
                    manifest["artifacts"].append(
                        {
                            "kind": "baseline",
                            "model": model,
                            "family": family,
                            "prompt_difficulty": args.prompt_difficulty,
                            "repeat_index": repeat_index,
                            "prompt_file": family_prompt,
                            "files": created,
                        }
                    )

                for repeat_index in range(1, args.compare_repeats + 1):
                    stem = f"crossmodel_compare_{family_slug}_{model_slug}_r{repeat_index:02d}"
                    created = _run_and_capture(
                        project_root=project_root,
                        results_dir=results_dir,
                        result_glob="crossmodel_compare_*.json",
                        output_stem=stem,
                        resume=args.resume,
                        command=[
                            sys.executable,
                            "experiments/02_algorithm_compare.py",
                            "--backend",
                            args.backend,
                            "--model",
                            model,
                            "--host",
                            args.host,
                            "--prompt-file",
                            family_prompt,
                            "--output-stem",
                            stem,
                            "--run-group",
                            run_group,
                            "--repeat-index",
                            str(repeat_index),
                            "--methods",
                            *args.methods,
                        ],
                    )
                    manifest["artifacts"].append(
                        {
                            "kind": "compare",
                            "model": model,
                            "family": family,
                            "prompt_difficulty": args.prompt_difficulty,
                            "repeat_index": repeat_index,
                            "prompt_file": family_prompt,
                            "files": created,
                        }
                    )

            if not args.skip_quality:
                for repeat_index in range(1, args.quality_repeats + 1):
                    qstem = f"crossmodel_quality_{model_slug}_r{repeat_index:02d}"
                    created = _run_and_capture(
                        project_root=project_root,
                        results_dir=results_dir,
                        result_glob="crossmodel_quality_*.json",
                        output_stem=qstem,
                        resume=args.resume,
                        command=[
                            sys.executable,
                            "experiments/05_quality_retention.py",
                            "--backend",
                            args.backend,
                            "--model",
                            model,
                            "--host",
                            args.host,
                            "--prompt-file",
                            quality_prompt,
                            "--output-stem",
                            qstem,
                            "--run-group",
                            run_group,
                            "--repeat-index",
                            str(repeat_index),
                            "--methods",
                            *args.methods,
                        ],
                    )
                    manifest["artifacts"].append(
                        {
                            "kind": "quality",
                            "model": model,
                            "repeat_index": repeat_index,
                            "prompt_file": quality_prompt,
                            "files": created,
                        }
                    )

            if not args.skip_methods:
                for repeat_index in range(1, args.detector_repeats + 1):
                    dstem = f"crossmodel_detector_{model_slug}_r{repeat_index:02d}"
                    created = _run_and_capture(
                        project_root=project_root,
                        results_dir=results_dir,
                        result_glob="crossmodel_detector_*.json",
                        output_stem=dstem,
                        resume=args.resume,
                        command=[
                            sys.executable,
                            "experiments/07_detector_eval.py",
                            "--backend",
                            args.backend,
                            "--model",
                            model,
                            "--host",
                            args.host,
                            "--output-stem",
                            dstem,
                            "--run-group",
                            run_group,
                            "--repeat",
                            "5",
                            "--families",
                            *args.families,
                            "--prompt-difficulty",
                            args.prompt_difficulty,
                        ],
                    )
                    manifest["artifacts"].append(
                        {
                            "kind": "detector",
                            "model": model,
                            "repeat_index": repeat_index,
                            "families": args.families,
                            "files": created,
                        }
                    )

                for repeat_index in range(1, args.adaptive_repeats + 1):
                    astem = f"crossmodel_adaptive_{model_slug}_r{repeat_index:02d}"
                    created = _run_and_capture(
                        project_root=project_root,
                        results_dir=results_dir,
                        result_glob="crossmodel_adaptive_*.json",
                        output_stem=astem,
                        resume=args.resume,
                        command=[
                            sys.executable,
                            "experiments/08_adaptive_eval.py",
                            "--backend",
                            args.backend,
                            "--model",
                            model,
                            "--host",
                            args.host,
                            "--output-stem",
                            astem,
                            "--run-group",
                            run_group,
                            "--repeat",
                            "5",
                            "--families",
                            *args.families,
                            "--prompt-difficulty",
                            args.prompt_difficulty,
                        ],
                    )
                    manifest["artifacts"].append(
                        {
                            "kind": "adaptive",
                            "model": model,
                            "repeat_index": repeat_index,
                            "families": args.families,
                            "files": created,
                        }
                    )

        if args.skip_ablation:
            continue

        for ablation_name, prompt_file in ablation_prompts:
            for repeat_index in range(1, args.ablation_repeats + 1):
                abstem = f"crossmodel_ablation_{ablation_name}_{model_slug}_r{repeat_index:02d}"
                created = _run_and_capture(
                    project_root=project_root,
                    results_dir=results_dir,
                    result_glob="crossmodel_ablation_*.json",
                    output_stem=abstem,
                    resume=args.resume,
                    command=[
                        sys.executable,
                        "experiments/01_baseline_bias.py",
                        "--backend",
                        args.backend,
                        "--model",
                        model,
                        "--host",
                        args.host,
                        "--prompt-file",
                        prompt_file,
                        "--output-stem",
                        abstem,
                        "--run-group",
                        run_group,
                        "--repeat-index",
                        str(repeat_index),
                    ],
                )
                manifest["artifacts"].append(
                    {
                        "kind": f"ablation_{ablation_name}",
                        "model": model,
                        "repeat_index": repeat_index,
                        "prompt_file": prompt_file,
                        "files": created,
                    }
                )

    manifest_path = results_dir / f"crossmodel_manifest_{run_group}.json"
    if manifest_path.exists():
        try:
            previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest = _merge_manifest_artifacts(previous_manifest, manifest)
        except json.JSONDecodeError:
            LOGGER.warning("Existing manifest is not valid JSON; overwriting: %s", manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("Saved run manifest to %s", manifest_path)


if __name__ == "__main__":
    main()
