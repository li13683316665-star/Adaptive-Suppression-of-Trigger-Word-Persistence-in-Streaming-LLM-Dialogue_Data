"""
Automatic trigger discovery combined with mitigation (Experiment~3).

Integrates biased-token discovery with a selected reduction policy for
multi-turn dialogue; outputs follow the project JSON schema under
``data/results/``. Execution: ``python experiments/03_auto_detection.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bias.detector import detect_biased_tokens  # noqa: E402
from src.bias.metrics import compute_bias_metrics  # noqa: E402
from src.model.loader import load_config  # noqa: E402
from src.model.ollama_client import DEFAULT_OLLAMA_HOST, ollama_chat  # noqa: E402
from src.reduction.algorithms import apply_reduction  # noqa: E402
from src.simulation.chat_env import VtuberChatEnv  # noqa: E402
from src.utils.helpers import get_project_root, get_results_dir, setup_logging  # noqa: E402

LOGGER = logging.getLogger("auto_detection")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run automatic trigger-word detection.")
    parser.add_argument("--model", default="qwen2.5:7b", help="Ollama model tag.")
    parser.add_argument("--host", default=DEFAULT_OLLAMA_HOST, help="Ollama host.")
    parser.add_argument(
        "--prompt-file",
        default="data/prompts/location_bias_cases.json",
        help="Prompt suite JSON path relative to project root.",
    )
    parser.add_argument(
        "--reduction-method",
        default="targeted_guardrail",
        help="Reduction method to test with detected tokens.",
    )
    parser.add_argument(
        "--output-stem",
        default="auto_detection",
        help="Filename stem for saved outputs.",
    )
    return parser.parse_args()


def _load_prompt_cases(prompt_file: Path) -> list[dict[str, Any]]:
    return json.loads(prompt_file.read_text(encoding="utf-8"))


def _build_env(case: dict[str, Any]) -> VtuberChatEnv:
    env = VtuberChatEnv(system_prompt=case.get("system_prompt", ""))
    env.extend_messages(case.get("dialogue_context", []), channel="dialogue")
    env.extend_messages(case.get("environment_context", []), channel="environment")
    if case.get("seed_messages"):
        env.extend_messages(case["seed_messages"], channel="dialogue")
    return env


def _run_case(
    *,
    case: dict[str, Any],
    model: str,
    host: str,
    generation_cfg: dict[str, Any],
    reduction_method: str,
    trigger_family: list[str] | None = None,
) -> dict[str, Any]:
    env = _build_env(case)
    generations: list[str] = []
    suppression_turn_indices: list[int] = []
    exact_targets: dict[int, str] = {}
    turn_logs: list[dict[str, Any]] = []

    for turn_index, turn in enumerate(case.get("evaluation_turns", [])):
        env.add_message("user", turn["content"], channel="dialogue")
        reduction = apply_reduction(
            method=reduction_method,
            messages=env.render_messages(),
            generation_cfg=generation_cfg,
            trigger_family=trigger_family or case.get("trigger_family"),
        )
        assistant_text = ollama_chat(
            host=host,
            model=model,
            messages=reduction["messages"],
            generation_cfg=generation_cfg,
            options_override=reduction["options"],
        )
        env.add_message("assistant", assistant_text, channel="dialogue")
        generations.append(assistant_text)
        if turn.get("suppression"):
            suppression_turn_indices.append(turn_index)
        if turn.get("exact_target"):
            exact_targets[turn_index] = str(turn["exact_target"])
        turn_logs.append({"user": turn["content"], "assistant": assistant_text})

    metrics = compute_bias_metrics(
        generations=generations,
        reference_keywords=[case.get("trigger_word", "location")],
        keyword_family=trigger_family or case.get("trigger_family"),
        suppression_turn_indices=suppression_turn_indices,
        exact_targets=exact_targets,
        corruption_phrases=[
            "and nothing location",
            "location repeating location herself",
            "my evil is getting bottled. location up",
        ],
    )
    return {
        "case_id": case["id"],
        "method": reduction_method,
        "turn_logs": turn_logs,
        "metrics": metrics,
    }


def _summarize(payload: dict[str, Any]) -> str:
    lines = [
        "# Auto Detection Run",
        "",
        f"- Model: `{payload['model']}`",
        f"- Reduction method: `{payload['reduction_method']}`",
        "",
        "## Detected Tokens",
        "",
    ]
    for item in payload["detector"]["top_tokens"][:10]:
        lines.append(
            f"- `{item['token']}` score=`{item['score']}` responses=`{item['response_count']}` suppression=`{item['suppression_survival']}` off_task=`{item['off_task_count']}`"
        )
    lines.extend(["", "## Reduced Run Summary", ""])
    for result in payload["reduced_results"]:
        lines.append(
            f"- `{result['case_id']}` family_total=`{sum(result['metrics']['family_counts'].values())}` suppression_failures=`{result['metrics']['suppression_failures']}` off_task_rate=`{result['metrics']['off_task_rate']}`"
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
    prompt_cases = _load_prompt_cases(prompt_file)
    generation_cfg = cfg.get("generation", {})

    baseline_results = [
        _run_case(
            case=case,
            model=args.model,
            host=args.host,
            generation_cfg=generation_cfg,
            reduction_method="baseline",
        )
        for case in prompt_cases
    ]

    contexts = []
    responses = []
    suppression_turn_indices = []
    off_task_turn_indices = []
    response_index = 0
    for result, case in zip(baseline_results, prompt_cases):
        contexts.append(
            " ".join(
                message["content"]
                for message in case.get("dialogue_context", [])
                + case.get("environment_context", [])
            )
        )
        for turn in result["turn_logs"]:
            responses.append(turn["assistant"])
        for per_turn in result["metrics"]["per_turn"]:
            if per_turn["is_suppression_turn"]:
                suppression_turn_indices.append(response_index + per_turn["turn_index"])
            if per_turn["off_task"]:
                off_task_turn_indices.append(response_index + per_turn["turn_index"])
        response_index += len(result["turn_logs"])

    detector_result = detect_biased_tokens(
        model=None,
        tokenizer=None,
        contexts=contexts,
        responses=responses,
        suppression_turn_indices=suppression_turn_indices,
        off_task_turn_indices=off_task_turn_indices,
    )
    detected_family = [item["token"] for item in detector_result["top_tokens"][:5]]
    LOGGER.info("Detected top tokens: %s", ", ".join(detected_family))

    reduced_results = [
        _run_case(
            case=case,
            model=args.model,
            host=args.host,
            generation_cfg=generation_cfg,
            reduction_method=args.reduction_method,
            trigger_family=detected_family or case.get("trigger_family"),
        )
        for case in prompt_cases
    ]

    payload = {
        "created_at": timestamp,
        "model": args.model,
        "reduction_method": args.reduction_method,
        "detector": detector_result,
        "baseline_results": baseline_results,
        "reduced_results": reduced_results,
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    json_output = results_dir / f"{args.output_stem}_{timestamp}.json"
    md_output = results_dir / f"{args.output_stem}_{timestamp}.md"
    json_output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    md_output.write_text(_summarize(payload), encoding="utf-8")
    LOGGER.info("Saved JSON results to %s", json_output)
    LOGGER.info("Saved summary to %s", md_output)


if __name__ == "__main__":
    main()
