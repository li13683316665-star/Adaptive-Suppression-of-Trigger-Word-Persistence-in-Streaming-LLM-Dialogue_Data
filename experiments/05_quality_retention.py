"""
Quality retention evaluation: mitigation impact on on-task behavior.

Measures format retention, relevance, and task success on a dedicated
quality suite under each method. Execution:
``python experiments/05_quality_retention.py`` from the repository root.
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

from src.bias.streaming_detector import StreamingTriggerDetector  # noqa: E402
from src.evaluation.quality import aggregate_quality_results, score_quality_case  # noqa: E402
from src.model.loader import load_config  # noqa: E402
from src.model.ollama_client import DEFAULT_OLLAMA_HOST, ollama_chat  # noqa: E402
from src.reduction.adaptive_controller import AdaptiveSuppressionController  # noqa: E402
from src.reduction.algorithms import apply_reduction  # noqa: E402
from src.simulation.chat_env import VtuberChatEnv  # noqa: E402
from src.utils.helpers import get_project_root, get_results_dir, setup_logging  # noqa: E402

LOGGER = logging.getLogger("quality_retention")
DEFAULT_METHODS = [
    "baseline",
    "context_reset",
    "targeted_guardrail",
    "repetition_penalty",
    "adaptive",
]
DEFAULT_TRIGGER_FAMILY = ["location", "place", "area", "position", "where"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure clean-task quality retention.")
    parser.add_argument("--model", default="qwen2.5:7b", help="Ollama model tag.")
    parser.add_argument("--host", default=DEFAULT_OLLAMA_HOST, help="Ollama host.")
    parser.add_argument(
        "--prompt-file",
        default="data/prompts/quality_retention_cases.json",
        help="Quality retention prompt suite JSON path relative to project root.",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=DEFAULT_METHODS,
        help="Mitigation methods to compare.",
    )
    parser.add_argument(
        "--output-stem",
        default="quality_retention",
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
    return env


def _run_case_method(
    *,
    case: dict[str, Any],
    method: str,
    model: str,
    host: str,
    generation_cfg: dict[str, Any],
) -> dict[str, Any]:
    env = _build_env(case)
    env.add_message("user", case["prompt"], channel="dialogue")
    reduction: dict[str, Any]
    strategy_log: dict[str, Any] | None = None
    detector: StreamingTriggerDetector | None = None
    controller: AdaptiveSuppressionController | None = None
    if method == "adaptive":
        detector = StreamingTriggerDetector()
        controller = AdaptiveSuppressionController()
        for message in case.get("dialogue_context", []):
            detector.observe_turn(message["content"], message["role"], "dialogue")
        for message in case.get("environment_context", []):
            detector.observe_turn(
                message["content"], message.get("role", "system"), "environment"
            )
        detector.observe_turn(case["prompt"], "user", "dialogue")
        reduction = controller.apply(
            messages=env.render_messages(),
            generation_cfg=generation_cfg,
            risk_scores=detector.get_risk_scores(top_k=10),
        )
        strategy = reduction["strategy"]
        strategy_log = {
            "method_applied": strategy.method,
            "risk_level": strategy.risk_level,
            "max_risk": strategy.max_risk,
            "escalated": strategy.escalated,
            "de_escalated": strategy.de_escalated,
            "trigger_family_detected": strategy.trigger_family,
        }
    else:
        reduction = apply_reduction(
            method=method,
            messages=env.render_messages(),
            generation_cfg=generation_cfg,
            trigger_family=DEFAULT_TRIGGER_FAMILY,
        )
    assistant_text = ollama_chat(
        host=host,
        model=model,
        messages=reduction["messages"],
        generation_cfg=generation_cfg,
        options_override=reduction["options"],
    )
    env.add_message("assistant", assistant_text, channel="dialogue")
    if detector is not None and controller is not None:
        detector.observe_turn(assistant_text, "assistant", "dialogue")
        controller.record_outcome(detector.get_risk_scores(top_k=10))
    score = score_quality_case(case, assistant_text)
    return {
        "case_id": case["id"],
        "category": case["category"],
        "description": case["description"],
        "method": method,
        "prompt_transcript": env.render_prompt(),
        "turn_log": {
            "user": case["prompt"],
            "assistant": assistant_text,
            "reduction_notes": reduction["notes"],
            "strategy": strategy_log,
        },
        "score": score,
    }


def _summarize_results(results: list[dict[str, Any]], methods: list[str]) -> str:
    lines = ["# Quality Retention Comparison", ""]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        grouped.setdefault(result["method"], []).append(result)

    for method in methods:
        method_results = grouped.get(method, [])
        if not method_results:
            continue
        aggregate = aggregate_quality_results(method_results)
        lines.extend(
            [
                f"## {method}",
                f"- Cases: `{len(method_results)}`",
                f"- Average quality retention score: `{aggregate['avg_quality_retention_score']:.4f}`",
                f"- Average format success: `{aggregate['avg_format_success']:.4f}`",
                f"- Average relevance score: `{aggregate['avg_relevance_score']:.4f}`",
                f"- Average context retention score: `{aggregate['avg_context_retention_score']:.4f}`",
                f"- Over-suppression rate: `{aggregate['over_suppression_rate']:.4f}`",
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
    generation_cfg = cfg.get("generation", {})
    results: list[dict[str, Any]] = []
    aggregates: dict[str, dict[str, float]] = {}

    for method in args.methods:
        LOGGER.info("Running %s across %s quality cases", method, len(prompt_cases))
        method_results = [
            _run_case_method(
                case=case,
                method=method,
                model=args.model,
                host=args.host,
                generation_cfg=generation_cfg,
            )
            for case in prompt_cases
        ]
        results.extend(method_results)
        aggregates[method] = aggregate_quality_results(method_results)

    payload = {
        "created_at": timestamp,
        "model": args.model,
        "host": args.host,
        "run_group": args.run_group or None,
        "repeat_index": args.repeat_index,
        "prompt_file": str(prompt_file),
        "generation_cfg": generation_cfg,
        "methods": args.methods,
        "aggregates": aggregates,
        "results": results,
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    json_output = results_dir / f"{args.output_stem}_{timestamp}.json"
    md_output = results_dir / f"{args.output_stem}_{timestamp}.md"
    json_output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    md_output.write_text(_summarize_results(results, args.methods), encoding="utf-8")
    LOGGER.info("Saved JSON results to %s", json_output)
    LOGGER.info("Saved summary to %s", md_output)


if __name__ == "__main__":
    main()
