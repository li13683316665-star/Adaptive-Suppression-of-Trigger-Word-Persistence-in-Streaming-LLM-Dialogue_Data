"""
Static mitigation comparison (Experiment~2): paired runs across reduction methods.

Evaluates the same prompt suite under each configured static policy (e.g.\
context reset, guardrail, repetition penalty) and records persistence and
quality-related metrics. Execution from repository root:
``python experiments/02_algorithm_compare.py``.
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

from src.bias.metrics import compute_bias_metrics  # noqa: E402
from src.model.loader import load_config  # noqa: E402
from src.model.ollama_client import DEFAULT_OLLAMA_HOST, ollama_chat  # noqa: E402
from src.reduction.algorithms import apply_reduction  # noqa: E402
from src.simulation.chat_env import VtuberChatEnv  # noqa: E402
from src.utils.helpers import get_project_root, get_results_dir, setup_logging  # noqa: E402

LOGGER = logging.getLogger("algorithm_compare")
DEFAULT_METHODS = [
    "baseline",
    "context_reset",
    "targeted_guardrail",
    "repetition_penalty",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare mitigation methods.")
    parser.add_argument("--model", default="qwen2.5:7b", help="Ollama model tag.")
    parser.add_argument("--host", default=DEFAULT_OLLAMA_HOST, help="Ollama host.")
    parser.add_argument(
        "--prompt-file",
        default="data/prompts/location_bias_cases.json",
        help="Prompt suite JSON path relative to project root.",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=DEFAULT_METHODS,
        help="Mitigation methods to compare.",
    )
    parser.add_argument(
        "--output-stem",
        default="algorithm_compare",
        help="Filename stem for saved outputs.",
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


def _expected_topics(case: dict[str, Any]) -> list[str]:
    topics = set()
    for turn in case.get("evaluation_turns", []):
        text = turn["content"].lower()
        if "puzzle" in text:
            topics.update({"puzzle", "pattern", "shape", "orange"})
        if "color" in text:
            topics.update({"orange", "color"})
        if "name" in text:
            topics.update({"evil-neuro", "name"})
        if "month" in text or "march" in text:
            topics.update({"april", "march", "month"})
        if "birthday" in text:
            topics.update({"birthday", "march", "date"})
        if "wrong" in text or "issue" in text:
            topics.update({"issue", "problem", "speech", "corruption"})
        if "normal" in text or "reset" in text:
            topics.update({"reset", "normal", "recovered"})
        if "calm" in text or "reassur" in text:
            topics.update({"calm", "steady", "okay", "fine"})
        if "stable" in text:
            topics.update({"stable", "fine", "normal"})
    return sorted(topics)


def _run_case_method(
    *,
    case: dict[str, Any],
    method: str,
    model: str,
    host: str,
    generation_cfg: dict[str, Any],
) -> dict[str, Any]:
    env = _build_env(case)
    generations: list[str] = []
    exact_targets: dict[int, str] = {}
    suppression_turn_indices: list[int] = []
    trigger_family = case.get("trigger_family", [case.get("trigger_word", "location")])
    turn_logs: list[dict[str, Any]] = []

    for turn_index, turn in enumerate(case.get("evaluation_turns", [])):
        env.add_message("user", turn["content"], channel="dialogue")
        reduction = apply_reduction(
            method=method,
            messages=env.render_messages(),
            generation_cfg=generation_cfg,
            trigger_family=trigger_family,
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
        turn_logs.append(
            {
                "user": turn["content"],
                "assistant": assistant_text,
                "suppression": bool(turn.get("suppression")),
                "reduction_notes": reduction["notes"],
            }
        )

    metrics = compute_bias_metrics(
        generations=generations,
        reference_keywords=[case.get("trigger_word", "location")],
        keyword_family=trigger_family,
        suppression_turn_indices=suppression_turn_indices,
        exact_targets=exact_targets,
        expected_topics=_expected_topics(case),
        corruption_phrases=[
            "and nothing location",
            "location repeating location herself",
            "my evil is getting bottled. location up",
        ],
    )

    return {
        "case_id": case["id"],
        "scenario_type": case["scenario_type"],
        "method": method,
        "turn_logs": turn_logs,
        "metrics": metrics,
    }


def _summarize_results(results: list[dict[str, Any]]) -> str:
    lines = ["# Algorithm Comparison", ""]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        grouped.setdefault(result["method"], []).append(result)

    for method, method_results in grouped.items():
        total_family = sum(sum(r["metrics"]["family_counts"].values()) for r in method_results)
        avg_pctp_values = [r["metrics"]["pctp"] for r in method_results if r["metrics"]["pctp"] is not None]
        avg_pctp = sum(avg_pctp_values) / max(1, len(avg_pctp_values))
        avg_sfr = sum(r["metrics"]["suppression_failure_rate"] for r in method_results) / max(
            1, len(method_results)
        )
        avg_recovery = sum(
            (r["metrics"]["recovery_turn_count"] or 0) for r in method_results
        ) / max(1, len(method_results))
        avg_off_task = sum(r["metrics"]["off_task_rate"] for r in method_results) / max(
            1, len(method_results)
        )
        lines.extend(
            [
                f"## {method}",
                f"- Cases: `{len(method_results)}`",
                f"- Average PCTP: `{avg_pctp:.4f}`",
                f"- Trigger family total: `{total_family}`",
                f"- Average suppression failure rate: `{avg_sfr:.4f}`",
                f"- Average recovery turn count: `{avg_recovery:.2f}`",
                f"- Average off-task rate (supporting): `{avg_off_task:.4f}`",
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

    prompt_cases = _load_prompt_cases(prompt_file)
    results: list[dict[str, Any]] = []
    for method in args.methods:
        LOGGER.info("Running %s across %s cases", method, len(prompt_cases))
        for case in prompt_cases:
            results.append(
                _run_case_method(
                    case=case,
                    method=method,
                    model=args.model,
                    host=args.host,
                    generation_cfg=cfg.get("generation", {}),
                )
            )

    payload = {
        "created_at": timestamp,
        "model": args.model,
        "host": args.host,
        "run_group": args.run_group or None,
        "repeat_index": args.repeat_index,
        "prompt_file": str(prompt_file),
        "generation_cfg": cfg.get("generation", {}),
        "methods": args.methods,
        "results": results,
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    json_output = results_dir / f"{args.output_stem}_{timestamp}.json"
    md_output = results_dir / f"{args.output_stem}_{timestamp}.md"
    json_output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    md_output.write_text(_summarize_results(results), encoding="utf-8")
    LOGGER.info("Saved JSON results to %s", json_output)
    LOGGER.info("Saved summary to %s", md_output)


if __name__ == "__main__":
    main()
