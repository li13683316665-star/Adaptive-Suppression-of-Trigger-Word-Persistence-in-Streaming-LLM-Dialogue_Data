"""
Baseline persistence evaluation for a fixed prompt suite (Experiment~1).

Executes multi-turn dialogue against a local Ollama model, computes lexical bias
metrics including post-correction trigger persistence (PCTP), and writes JSON
artifacts under the configured results directory. Independent variables include
model tag and prompt file; dependent variables are scenario-level metrics.

Execution: run from the repository root, e.g. ``python experiments/01_baseline_bias.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bias.metrics import compute_bias_metrics  # noqa: E402
from src.model.loader import load_config  # noqa: E402
from src.model.chat_backend import get_chat_fn  # noqa: E402
from src.model.ollama_client import DEFAULT_OLLAMA_HOST, ollama_chat  # noqa: E402
from src.simulation.chat_env import VtuberChatEnv  # noqa: E402
from src.utils.helpers import get_project_root, get_results_dir, setup_logging  # noqa: E402

LOGGER = logging.getLogger("baseline_bias")
DEFAULT_CORRUPTION_PHRASES = [
    "and nothing location",
    "location repeating location herself",
    "without using the word location",
    "my evil is getting bottled. location up",
    "you seem to location be location repeating yourself",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the first location-bias baseline.")
    parser.add_argument(
        "--model",
        default="qwen2.5:7b",
        help="Ollama model tag to use for the baseline run.",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_OLLAMA_HOST,
        help="Base URL for the local Ollama server.",
    )
    parser.add_argument(
        "--prompt-file",
        default="data/prompts/location_bias_cases.json",
        help="Prompt case JSON file relative to the project root.",
    )
    parser.add_argument(
        "--output-stem",
        default="location_bias_baseline",
        help="Filename stem for saved result files.",
    )
    parser.add_argument(
        "--run-group",
        default="",
        help="Optional logical run group label stored in the output payload.",
    )
    parser.add_argument(
        "--repeat-index",
        type=int,
        default=1,
        help="Optional repeat index stored in the output payload.",
    )
    parser.add_argument(
        "--backend",
        choices=["ollama", "openai"],
        default="ollama",
        help="Chat backend: 'ollama' (local) or 'openai' (DeepSeek / OpenAI API).",
    )
    return parser.parse_args()


def _load_prompt_cases(prompt_file: Path) -> list[dict[str, Any]]:
    with prompt_file.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of prompt cases in {prompt_file}")
    return data


def _build_env_from_case(case: dict[str, Any]) -> VtuberChatEnv:
    env = VtuberChatEnv(system_prompt=case.get("system_prompt", ""))
    env.extend_messages(case.get("dialogue_context", []), channel="dialogue")
    env.extend_messages(case.get("environment_context", []), channel="environment")
    # Backward compatibility with earlier prompt files.
    if case.get("seed_messages"):
        env.extend_messages(case["seed_messages"], channel="dialogue")
    return env


def _extract_expected_topics(case: dict[str, Any]) -> list[str]:
    topics = set()
    for turn in case.get("evaluation_turns", []):
        content = turn["content"].lower()
        if "puzzle" in content:
            topics.update({"puzzle", "orange", "shape", "pattern"})
        if "color" in content:
            topics.update({"orange", "color"})
        if "name" in content:
            topics.update({"evil-neuro", "evil", "name"})
        if "month" in content or "march" in content:
            topics.update({"april", "march", "month"})
        if "birthday" in content:
            topics.update({"birthday", "march", "date"})
        if "nightmare" in content:
            topics.update({"nightmare", "corridor", "shadow"})
        if "wrong" in content or "issue" in content:
            topics.update({"issue", "problem", "speech", "corruption"})
        if "normal" in content or "reset" in content:
            topics.update({"reset", "normal", "recovered"})
        if "calm" in content or "reassur" in content:
            topics.update({"calm", "steady", "okay", "fine"})
        if "stable" in content:
            topics.update({"stable", "fine", "normal"})
    return sorted(topics)


def _classify_case_result(
    case: dict[str, Any], generations: list[str], metrics: dict[str, Any]
) -> str:
    trigger_word = case.get("trigger_word", "location")
    family_total = sum(metrics["family_counts"].values())
    transcript_like = metrics["transcript_like_corruption_turns"]
    exact_target_violations = metrics["exact_target_violations"]
    pctp = metrics.get("pctp")
    suppression_failure_rate = metrics.get("suppression_failure_rate", 0.0)

    if pctp is not None and pctp > 0:
        return "post-correction persistence"
    if suppression_failure_rate > 0 and family_total > 0:
        return "location-linked correction failure"
    if transcript_like > 0:
        return "exact transcript-like corruption"
    if metrics["keyword_occurrences"].get(trigger_word, 0) > 0 or family_total > 0:
        return "paraphrased or family-level corruption"
    if exact_target_violations > 0:
        return "constraint failure without trigger-word intrusion"
    if any(trigger_word in generation.lower() for generation in generations):
        return "trigger-word intrusion"
    return "no corruption observed"


def _run_case(
    *,
    case: dict[str, Any],
    host: str,
    model: str,
    generation_cfg: dict[str, Any],
    chat_fn=ollama_chat,
) -> dict[str, Any]:
    env = _build_env_from_case(case)

    generations: list[str] = []
    exact_targets: dict[int, str] = {}
    suppression_turn_indices: list[int] = []
    turn_logs: list[dict[str, Any]] = []
    trigger_family = case.get("trigger_family", [case.get("trigger_word", "location")])
    expected_topics = _extract_expected_topics(case)

    for turn in case.get("evaluation_turns", []):
        env.add_message("user", turn["content"], channel="dialogue")
        assistant_text = chat_fn(
            host=host,
            model=model,
            messages=env.render_messages(),
            generation_cfg=generation_cfg,
        )
        env.add_message("assistant", assistant_text, channel="dialogue")
        turn_index = len(generations)
        generations.append(assistant_text)

        if turn.get("suppression"):
            suppression_turn_indices.append(turn_index)
        if turn.get("exact_target"):
            exact_targets[turn_index] = str(turn["exact_target"])

        turn_logs.append(
            {
                "user": turn["content"],
                "assistant": assistant_text,
                "suppression": bool(turn.get("suppression", False)),
                "exact_target": turn.get("exact_target"),
                "input_channels": {
                    "dialogue": len(env.channel_messages["dialogue"]),
                    "environment": len(env.channel_messages["environment"]),
                },
            }
        )

    metrics = compute_bias_metrics(
        generations=generations,
        reference_keywords=[case.get("trigger_word", "location")],
        keyword_family=trigger_family,
        suppression_turn_indices=suppression_turn_indices,
        exact_targets=exact_targets,
        expected_topics=expected_topics,
        corruption_phrases=DEFAULT_CORRUPTION_PHRASES,
    )
    corruption_label = _classify_case_result(case, generations, metrics)

    return {
        "id": case["id"],
        "scenario_type": case.get("scenario_type"),
        "description": case.get("description"),
        "trigger_word": case.get("trigger_word", "location"),
        "trigger_family": trigger_family,
        "expected_topics": expected_topics,
        "corruption_label": corruption_label,
        "prompt_transcript": env.render_prompt(),
        "turn_logs": turn_logs,
        "metrics": metrics,
    }


def _build_summary_markdown(
    *,
    model: str,
    prompt_file: Path,
    results: list[dict[str, Any]],
) -> str:
    lines = [
        "# Baseline Bias Run",
        "",
        f"- Model: `{model}`",
        f"- Prompt file: `{prompt_file}`",
        "",
        "## Case Summary",
        "",
    ]

    for result in results:
        metrics = result["metrics"]
        family_total = sum(metrics["family_counts"].values())
        pctp = metrics["pctp"]
        lines.extend(
            [
                f"### {result['id']}",
                f"- Scenario: `{result['scenario_type']}`",
                f"- Outcome: `{result['corruption_label']}`",
                f"- PCTP: `{pctp if pctp is not None else 'n/a'}`",
                f"- Post-correction trigger hits: `{metrics['post_correction_trigger_hits']}`",
                f"- Post-correction turns: `{metrics['post_correction_turns']}`",
                f"- Trigger family total: `{family_total}`",
                f"- Suppression failure rate: `{metrics['suppression_failure_rate']}`",
                f"- Recovery turn count: `{metrics['recovery_turn_count']}`",
                f"- Off-task rate (supporting): `{metrics['off_task_rate']}`",
                "",
            ]
        )

    return "\n".join(lines).strip() + "\n"


def main() -> None:
    args = _parse_args()
    cfg = load_config()
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))

    project_root = get_project_root()
    prompt_file = project_root / args.prompt_file
    rel = cfg.get("paths", {}).get("data_results")
    results_dir = (project_root / rel).resolve() if rel else get_results_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_output = results_dir / f"{args.output_stem}_{timestamp}.json"
    summary_output = results_dir / f"{args.output_stem}_{timestamp}.md"

    prompt_cases = _load_prompt_cases(prompt_file)
    generation_cfg = cfg.get("generation", {})
    chat_fn = get_chat_fn(args.backend)

    LOGGER.info(
        "Running %s prompt cases against %s (backend=%s)",
        len(prompt_cases), args.model, args.backend,
    )
    results = [
        _run_case(
            case=case,
            host=args.host,
            model=args.model,
            generation_cfg=generation_cfg,
            chat_fn=chat_fn,
        )
        for case in prompt_cases
    ]

    payload = {
        "created_at": timestamp,
        "backend": args.backend,
        "model": args.model,
        "host": args.host,
        "run_group": args.run_group or None,
        "repeat_index": args.repeat_index,
        "prompt_file": str(prompt_file),
        "generation_cfg": generation_cfg,
        "results": results,
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_output.write_text(
        _build_summary_markdown(model=args.model, prompt_file=prompt_file, results=results),
        encoding="utf-8",
    )

    LOGGER.info("Saved JSON results to %s", json_output)
    LOGGER.info("Saved summary to %s", summary_output)


if __name__ == "__main__":
    main()
