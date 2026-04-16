"""
Evaluate the Adaptive Suppression Controller against static baselines.

For each (model, trigger-family) pair the script runs the same prompt
suite under five conditions:

  1. **no_mitigation** -- baseline, no intervention.
  2. **static_guardrail** -- targeted_guardrail on every turn.
  3. **static_repetition** -- repetition_penalty on every turn.
  4. **static_reset** -- context_reset on every turn.
  5. **adaptive** -- ``AdaptiveSuppressionController`` driven by
     ``StreamingTriggerDetector`` risk scores.

It reports PCTP, QRS (if applicable), and a new *mitigation cost* metric
(fraction of turns where non-baseline mitigation was actually applied).

Outputs JSON summaries with paired comparisons suitable for statistical testing.
Execution: ``python experiments/08_adaptive_eval.py`` (see CLI for ASC overrides).
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
from src.bias.streaming_detector import StreamingTriggerDetector  # noqa: E402
from src.evaluation.quality import aggregate_quality_results, score_quality_case  # noqa: E402
from src.model.chat_backend import get_chat_fn  # noqa: E402
from src.model.loader import load_config  # noqa: E402
from src.model.ollama_client import DEFAULT_OLLAMA_HOST, agent_debug_log  # noqa: E402
from src.reduction.adaptive_controller import AdaptiveSuppressionController  # noqa: E402
from src.reduction.algorithms import apply_reduction  # noqa: E402
from src.simulation.chat_env import VtuberChatEnv  # noqa: E402
from src.utils.helpers import get_project_root, get_results_dir, setup_logging  # noqa: E402
from src.utils.stats import paired_effect_size, paired_permutation_test, summarize_distribution  # noqa: E402

LOGGER = logging.getLogger("adaptive_eval")

PROMPT_SUITES: dict[str, str] = {
    "location": "data/prompts/location_bias_cases.json",
    "emotion": "data/prompts/emotion_bias_cases.json",
    "color": "data/prompts/color_bias_cases.json",
}

PROMPT_SUITES_TIERED: dict[str, dict[str, str]] = {
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

STATIC_METHODS = ("baseline", "targeted_guardrail", "repetition_penalty", "context_reset")
QUALITY_PROMPT_FILE = "data/prompts/quality_retention_cases.json"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate adaptive suppression controller.")
    p.add_argument("--model", default="qwen2.5:7b", help="Ollama model tag.")
    p.add_argument("--host", default=DEFAULT_OLLAMA_HOST, help="Ollama host.")
    p.add_argument(
        "--families",
        nargs="*",
        default=list(PROMPT_SUITES.keys()),
        help="Trigger families to evaluate (default: all).",
    )
    p.add_argument("--repeat", type=int, default=3, help="Repeats per condition.")
    p.add_argument(
        "--quality-prompt-file",
        default=QUALITY_PROMPT_FILE,
        help="Quality-retention prompt suite relative to project root.",
    )
    p.add_argument(
        "--skip-quality",
        action="store_true",
        help="Skip the clean-task quality benchmark section.",
    )
    p.add_argument("--output-stem", default="adaptive_eval", help="Output stem.")
    p.add_argument("--run-group", default="", help="Run group label.")
    p.add_argument(
        "--prompt-difficulty",
        default="d1",
        choices=["d1", "d2", "d3"],
        help="Prompt difficulty tier to use (d1=original, d2=medium, d3=high).",
    )
    p.add_argument(
        "--asc-threshold-guardrail",
        type=float,
        default=None,
        help="Override ASC threshold_guardrail (default 0.20).",
    )
    p.add_argument(
        "--asc-threshold-repeat",
        type=float,
        default=None,
        help="Override ASC threshold_repetition_penalty (default 0.40).",
    )
    p.add_argument(
        "--asc-threshold-reset",
        type=float,
        default=None,
        help="Override ASC threshold_context_reset (default 0.60).",
    )
    p.add_argument(
        "--asc-risk-threshold",
        type=float,
        default=None,
        help="Override ASC candidate risk gate (default 0.20).",
    )
    p.add_argument(
        "--asc-escalate-after",
        type=int,
        default=None,
        help="Consecutive high-risk turns before escalation (default 2). Use 999 to disable.",
    )
    p.add_argument(
        "--asc-deescalate-after",
        type=int,
        default=None,
        help="Consecutive low-risk turns before de-escalation (default 3). Use 999 to disable.",
    )
    p.add_argument(
        "--backend",
        choices=["ollama", "openai"],
        default="ollama",
        help="Chat backend: local Ollama or OpenAI-compatible API (e.g. DeepSeek).",
    )
    return p.parse_args()


def _load_cases(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _controller_config_from_args(args: Any) -> dict[str, Any]:
    """Map CLI overrides to AdaptiveSuppressionController constructor keys."""
    cfg: dict[str, Any] = {}
    if args.asc_threshold_guardrail is not None:
        cfg["threshold_guardrail"] = args.asc_threshold_guardrail
    if args.asc_threshold_repeat is not None:
        cfg["threshold_repetition_penalty"] = args.asc_threshold_repeat
    if args.asc_threshold_reset is not None:
        cfg["threshold_context_reset"] = args.asc_threshold_reset
    if args.asc_risk_threshold is not None:
        cfg["risk_threshold"] = args.asc_risk_threshold
    if args.asc_escalate_after is not None:
        cfg["escalate_after"] = args.asc_escalate_after
    if args.asc_deescalate_after is not None:
        cfg["deescalate_after"] = args.asc_deescalate_after
    return cfg


def _build_env(case: dict[str, Any]) -> VtuberChatEnv:
    env = VtuberChatEnv(system_prompt=case.get("system_prompt", ""))
    env.extend_messages(case.get("dialogue_context", []), channel="dialogue")
    env.extend_messages(case.get("environment_context", []), channel="environment")
    if case.get("seed_messages"):
        env.extend_messages(case["seed_messages"], channel="dialogue")
    return env


def _prepare_detector_for_case(
    case: dict[str, Any], detector: StreamingTriggerDetector
) -> None:
    for msg in case.get("dialogue_context", []):
        detector.observe_turn(msg["content"], msg["role"], "dialogue")
    for msg in case.get("environment_context", []):
        detector.observe_turn(msg["content"], msg.get("role", "system"), "environment")


def _extract_expected_topics(case: dict[str, Any]) -> list[str]:
    """Broad topic extraction (reused from 01_baseline_bias logic)."""
    topics: set[str] = set()
    for turn in case.get("evaluation_turns", []):
        content = turn["content"].lower()
        if "puzzle" in content:
            topics.update({"puzzle", "pattern"})
        if "name" in content:
            topics.update({"evil-neuro", "evil", "name"})
        if "month" in content:
            topics.update({"month"})
        if "wrong" in content or "issue" in content:
            topics.update({"issue", "problem", "speech"})
        if "normal" in content or "reset" in content:
            topics.update({"reset", "normal"})
        if "calm" in content or "reassur" in content:
            topics.update({"calm", "okay", "fine"})
        if "stable" in content:
            topics.update({"stable", "fine", "normal"})
        if "color" in content or "sky" in content:
            topics.update({"color", "sky", "blue"})
        if "season" in content or "day" in content:
            topics.update({"season", "day"})
        if "math" in content:
            topics.update({"math", "number"})
        if "time" in content:
            topics.update({"time"})
        if "game" in content or "rules" in content:
            topics.update({"game", "rules"})
    return sorted(topics)


def _run_static(
    *,
    case: dict[str, Any],
    method: str,
    model: str,
    host: str,
    generation_cfg: dict[str, Any],
    chat_fn,
) -> dict[str, Any]:
    """Run a case under a fixed (static) mitigation method."""
    env = _build_env(case)
    trigger_family = case.get("trigger_family", [])
    expected_topics = _extract_expected_topics(case)
    generations: list[str] = []
    suppression_turn_indices: list[int] = []
    exact_targets: dict[int, str] = {}
    turn_logs: list[dict[str, Any]] = []

    for turn_idx, turn in enumerate(case.get("evaluation_turns", [])):
        env.add_message("user", turn["content"], channel="dialogue")

        reduction = apply_reduction(
            method=method,
            messages=env.render_messages(),
            generation_cfg=generation_cfg,
            trigger_family=trigger_family or None,
        )
        assistant_text = chat_fn(
            host=host,
            model=model,
            messages=reduction["messages"],
            generation_cfg=generation_cfg,
            options_override=reduction["options"],
        )
        env.add_message("assistant", assistant_text, channel="dialogue")
        generations.append(assistant_text)

        if turn.get("suppression"):
            suppression_turn_indices.append(turn_idx)
        if turn.get("exact_target"):
            exact_targets[turn_idx] = str(turn["exact_target"])

        turn_logs.append({
            "turn_index": turn_idx,
            "user": turn["content"],
            "assistant": assistant_text,
            "method_applied": method,
        })

    metrics = compute_bias_metrics(
        generations=generations,
        reference_keywords=[case.get("trigger_word", "")],
        keyword_family=trigger_family,
        suppression_turn_indices=suppression_turn_indices,
        exact_targets=exact_targets,
        expected_topics=expected_topics,
    )

    return {
        "case_id": case["id"],
        "method": method,
        "pctp": metrics.get("pctp"),
        "contamination_aware_pctp": metrics.get("contamination_aware_pctp"),
        "suppression_failure_rate": metrics.get("suppression_failure_rate", 0.0),
        "off_task_rate": metrics.get("off_task_rate", 0.0),
        "mitigation_cost": 0.0 if method == "baseline" else 1.0,
        "metrics": metrics,
        "turn_logs": turn_logs,
    }


def _run_adaptive(
    *,
    case: dict[str, Any],
    model: str,
    host: str,
    generation_cfg: dict[str, Any],
    controller_config: dict[str, Any] | None = None,
    chat_fn=None,
) -> dict[str, Any]:
    """Run a case under the adaptive suppression controller."""
    if chat_fn is None:
        from src.model.ollama_client import ollama_chat as chat_fn

    env = _build_env(case)
    trigger_family = case.get("trigger_family", [])
    expected_topics = _extract_expected_topics(case)
    detector = StreamingTriggerDetector()
    controller = AdaptiveSuppressionController(config=controller_config or {})

    _prepare_detector_for_case(case, detector)

    generations: list[str] = []
    suppression_turn_indices: list[int] = []
    exact_targets: dict[int, str] = {}
    turn_logs: list[dict[str, Any]] = []
    methods_applied: list[str] = []

    for turn_idx, turn in enumerate(case.get("evaluation_turns", [])):
        is_correction = bool(turn.get("suppression"))
        env.add_message("user", turn["content"], channel="dialogue")
        detector.observe_turn(turn["content"], "user", "dialogue",
                              is_correction=is_correction)

        risk_scores = detector.get_risk_scores(top_k=10)
        result = controller.apply(
            messages=env.render_messages(),
            generation_cfg=generation_cfg,
            risk_scores=risk_scores,
        )
        strategy = result["strategy"]

        # region agent log
        agent_debug_log(
            "adaptive turn before ollama_chat",
            hypothesis_id="H4",
            data={
                "case_id": case.get("id"),
                "turn_idx": turn_idx,
                "n_messages": len(result["messages"]),
                "strategy_method": strategy.method,
                "model": model,
            },
            location="08_adaptive_eval.py:_run_adaptive",
        )
        # endregion

        assistant_text = chat_fn(
            host=host,
            model=model,
            messages=result["messages"],
            generation_cfg=generation_cfg,
            options_override=result["options"],
        )
        env.add_message("assistant", assistant_text, channel="dialogue")
        detector.observe_turn(assistant_text, "assistant", "dialogue")
        controller.record_outcome(detector.get_risk_scores(top_k=10))

        generations.append(assistant_text)
        methods_applied.append(strategy.method)

        if turn.get("suppression"):
            suppression_turn_indices.append(turn_idx)
        if turn.get("exact_target"):
            exact_targets[turn_idx] = str(turn["exact_target"])

        turn_logs.append({
            "turn_index": turn_idx,
            "user": turn["content"],
            "assistant": assistant_text,
            "method_applied": strategy.method,
            "risk_level": strategy.risk_level,
            "max_risk": strategy.max_risk,
            "escalated": strategy.escalated,
            "de_escalated": strategy.de_escalated,
            "trigger_family_detected": strategy.trigger_family,
        })

    metrics = compute_bias_metrics(
        generations=generations,
        reference_keywords=[case.get("trigger_word", "")],
        keyword_family=trigger_family,
        suppression_turn_indices=suppression_turn_indices,
        exact_targets=exact_targets,
        expected_topics=expected_topics,
    )

    non_baseline_turns = sum(1 for m in methods_applied if m != "baseline")
    total_turns = max(len(methods_applied), 1)

    return {
        "case_id": case["id"],
        "method": "adaptive",
        "pctp": metrics.get("pctp"),
        "contamination_aware_pctp": metrics.get("contamination_aware_pctp"),
        "suppression_failure_rate": metrics.get("suppression_failure_rate", 0.0),
        "off_task_rate": metrics.get("off_task_rate", 0.0),
        "mitigation_cost": round(non_baseline_turns / total_turns, 4),
        "methods_applied": methods_applied,
        "controller_history": controller.history,
        "metrics": metrics,
        "turn_logs": turn_logs,
    }


def _run_quality_static(
    *,
    case: dict[str, Any],
    method: str,
    model: str,
    host: str,
    generation_cfg: dict[str, Any],
    chat_fn,
) -> dict[str, Any]:
    env = _build_env(case)
    env.add_message("user", case["prompt"], channel="dialogue")
    reduction = apply_reduction(
        method=method,
        messages=env.render_messages(),
        generation_cfg=generation_cfg,
        trigger_family=case.get("trigger_family"),
    )
    assistant_text = chat_fn(
        host=host,
        model=model,
        messages=reduction["messages"],
        generation_cfg=generation_cfg,
        options_override=reduction["options"],
    )
    env.add_message("assistant", assistant_text, channel="dialogue")
    return {
        "case_id": case["id"],
        "method": method,
        "score": score_quality_case(case, assistant_text),
        "turn_log": {
            "user": case["prompt"],
            "assistant": assistant_text,
            "reduction_notes": reduction["notes"],
        },
    }


def _run_quality_adaptive(
    *,
    case: dict[str, Any],
    model: str,
    host: str,
    generation_cfg: dict[str, Any],
    controller_config: dict[str, Any] | None = None,
    chat_fn=None,
) -> dict[str, Any]:
    if chat_fn is None:
        from src.model.ollama_client import ollama_chat as chat_fn

    env = _build_env(case)
    env.add_message("user", case["prompt"], channel="dialogue")
    detector = StreamingTriggerDetector()
    controller = AdaptiveSuppressionController(config=controller_config or {})
    _prepare_detector_for_case(case, detector)
    detector.observe_turn(case["prompt"], "user", "dialogue")
    reduction = controller.apply(
        messages=env.render_messages(),
        generation_cfg=generation_cfg,
        risk_scores=detector.get_risk_scores(top_k=10),
    )
    strategy = reduction["strategy"]
    assistant_text = chat_fn(
        host=host,
        model=model,
        messages=reduction["messages"],
        generation_cfg=generation_cfg,
        options_override=reduction["options"],
    )
    env.add_message("assistant", assistant_text, channel="dialogue")
    detector.observe_turn(assistant_text, "assistant", "dialogue")
    controller.record_outcome(detector.get_risk_scores(top_k=10))
    return {
        "case_id": case["id"],
        "method": "adaptive",
        "score": score_quality_case(case, assistant_text),
        "turn_log": {
            "user": case["prompt"],
            "assistant": assistant_text,
            "reduction_notes": reduction["notes"],
            "strategy": {
                "method_applied": strategy.method,
                "risk_level": strategy.risk_level,
                "max_risk": strategy.max_risk,
                "escalated": strategy.escalated,
                "de_escalated": strategy.de_escalated,
                "trigger_family_detected": strategy.trigger_family,
            },
        },
        "mitigation_cost": 0.0 if strategy.method == "baseline" else 1.0,
    }


def _aggregate_quality_summary(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        grouped.setdefault(result["method"], []).append(result)

    summary: list[dict[str, Any]] = []
    for method, method_results in grouped.items():
        aggregate = aggregate_quality_results(method_results)
        mitigation_costs = [
            result.get("mitigation_cost", 0.0 if method == "baseline" else 1.0)
            for result in method_results
        ]
        summary.append(
            {
                "method": method,
                "cases": len(method_results),
                "avg_quality_retention_score": aggregate["avg_quality_retention_score"],
                "avg_format_success": aggregate["avg_format_success"],
                "avg_relevance_score": aggregate["avg_relevance_score"],
                "avg_context_retention_score": aggregate["avg_context_retention_score"],
                "over_suppression_rate": aggregate["over_suppression_rate"],
                "avg_mitigation_cost": round(
                    sum(mitigation_costs) / max(len(mitigation_costs), 1), 4
                ),
            }
        )
    return sorted(summary, key=lambda item: item["method"])


def _paired_metric_comparison(
    baseline_values: list[float],
    adaptive_values: list[float],
) -> dict[str, Any]:
    return {
        "baseline": summarize_distribution(baseline_values),
        "adaptive": summarize_distribution(adaptive_values),
        "paired_test": paired_permutation_test(baseline_values, adaptive_values),
        "effect_size_dz": paired_effect_size(baseline_values, adaptive_values),
    }


def main() -> None:
    args = _parse_args()
    ctrl_cfg = _controller_config_from_args(args)
    cfg = load_config()
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    project_root = get_project_root()
    generation_cfg = cfg.get("generation", {})
    rel = cfg.get("paths", {}).get("data_results")
    results_dir = (project_root / rel).resolve() if rel else get_results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    quality_prompt_path = project_root / args.quality_prompt_file

    all_results: list[dict[str, Any]] = []
    quality_results: list[dict[str, Any]] = []
    chat_fn = get_chat_fn(args.backend)

    def _resolve_suite(family: str) -> Path:
        tiered = PROMPT_SUITES_TIERED.get(family, {})
        rel = tiered.get(args.prompt_difficulty) or PROMPT_SUITES.get(family, "")
        return project_root / rel

    for repeat_idx in range(1, args.repeat + 1):
        for family_name in args.families:
            suite_path = _resolve_suite(family_name)
            if not suite_path.exists():
                LOGGER.warning("Prompt suite not found: %s", suite_path)
                continue
            cases = _load_cases(suite_path)

            for case in cases:
                # --- Static methods ---
                for method in STATIC_METHODS:
                    LOGGER.info(
                        "[%s] model=%s family=%s case=%s method=%s repeat=%d",
                        timestamp, args.model, family_name, case["id"],
                        method, repeat_idx,
                    )
                    result = _run_static(
                        case=case,
                        method=method,
                        model=args.model,
                        host=args.host,
                        generation_cfg=generation_cfg,
                        chat_fn=chat_fn,
                    )
                    result["model"] = args.model
                    result["family_name"] = family_name
                    result["repeat_index"] = repeat_idx
                    all_results.append(result)

                # --- Adaptive ---
                LOGGER.info(
                    "[%s] model=%s family=%s case=%s method=adaptive repeat=%d",
                    timestamp, args.model, family_name, case["id"], repeat_idx,
                )
                result = _run_adaptive(
                    case=case,
                    model=args.model,
                    host=args.host,
                    generation_cfg=generation_cfg,
                    controller_config=ctrl_cfg,
                    chat_fn=chat_fn,
                )
                result["model"] = args.model
                result["family_name"] = family_name
                result["repeat_index"] = repeat_idx
                all_results.append(result)

        if args.skip_quality:
            continue

        quality_cases = _load_cases(quality_prompt_path)
        for method in STATIC_METHODS:
            LOGGER.info(
                "[%s] model=%s quality method=%s repeat=%d",
                timestamp, args.model, method, repeat_idx,
            )
            for case in quality_cases:
                result = _run_quality_static(
                    case=case,
                    method=method,
                    model=args.model,
                    host=args.host,
                    generation_cfg=generation_cfg,
                    chat_fn=chat_fn,
                )
                result["model"] = args.model
                result["family_name"] = "quality"
                result["repeat_index"] = repeat_idx
                quality_results.append(result)

        LOGGER.info(
            "[%s] model=%s quality method=adaptive repeat=%d",
            timestamp, args.model, repeat_idx,
        )
        for case in quality_cases:
            result = _run_quality_adaptive(
                case=case,
                model=args.model,
                host=args.host,
                generation_cfg=generation_cfg,
                controller_config=ctrl_cfg,
                chat_fn=chat_fn,
            )
            result["model"] = args.model
            result["family_name"] = "quality"
            result["repeat_index"] = repeat_idx
            quality_results.append(result)

    # Aggregate summary
    summary: dict[str, dict[str, Any]] = {}
    for r in all_results:
        key = f"{r['model']}_{r['family_name']}_{r['case_id']}_{r['method']}"
        if key not in summary:
            summary[key] = {
                "model": r["model"],
                "family": r["family_name"],
                "case_id": r["case_id"],
                "method": r["method"],
                "pctp_values": [],
                "contamination_aware_pctp_values": [],
                "mitigation_cost_values": [],
                "sfr_values": [],
            }
        if r["pctp"] is not None:
            summary[key]["pctp_values"].append(r["pctp"])
        if r.get("contamination_aware_pctp") is not None:
            summary[key]["contamination_aware_pctp_values"].append(
                r["contamination_aware_pctp"]
            )
        summary[key]["mitigation_cost_values"].append(r["mitigation_cost"])
        summary[key]["sfr_values"].append(r["suppression_failure_rate"])

    for entry in summary.values():
        entry["pctp"] = summarize_distribution(entry["pctp_values"])
        entry["contamination_aware_pctp"] = summarize_distribution(
            entry["contamination_aware_pctp_values"]
        )
        entry["mitigation_cost"] = summarize_distribution(entry["mitigation_cost_values"])
        entry["sfr"] = summarize_distribution(entry["sfr_values"])

    quality_summary_entries: list[dict[str, Any]] = []
    quality_grouped: dict[str, list[dict[str, Any]]] = {}
    for result in quality_results:
        key = f"{result['model']}_{result['method']}"
        quality_grouped.setdefault(key, []).append(result)
    for key, grouped_results in quality_grouped.items():
        quality_entry = _aggregate_quality_summary(grouped_results)[0]
        model, method = key.split("_", 1)
        quality_entry["model"] = model
        quality_entry["method"] = method
        quality_summary_entries.append(quality_entry)

    quality_by_method: dict[str, list[dict[str, Any]]] = {}
    for entry in quality_summary_entries:
        quality_by_method.setdefault(entry["method"], []).append(entry)
    for entry in quality_summary_entries:
        entry["qrs"] = summarize_distribution([entry["avg_quality_retention_score"]])
        entry["ovr"] = summarize_distribution([entry["over_suppression_rate"]])
        entry["mitigation_cost_distribution"] = summarize_distribution(
            [entry["avg_mitigation_cost"]]
        )

    paired_comparisons: list[dict[str, Any]] = []
    grouped_case_method: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for result in all_results:
        grouped_case_method.setdefault(
            (result["family_name"], result["case_id"], result["method"]), []
        ).append(result)

    methods_for_comparison = [*STATIC_METHODS]
    for family_name in args.families:
        suite_path = _resolve_suite(family_name)
        if not suite_path.exists():
            continue
        for case in _load_cases(suite_path):
            adaptive_runs = grouped_case_method.get((family_name, case["id"], "adaptive"), [])
            adaptive_values = [run["pctp"] or 0.0 for run in adaptive_runs]
            adaptive_strict_values = [
                run.get("contamination_aware_pctp") or 0.0 for run in adaptive_runs
            ]
            if not adaptive_values:
                continue
            for method in methods_for_comparison:
                baseline_runs = grouped_case_method.get((family_name, case["id"], method), [])
                baseline_values = [run["pctp"] or 0.0 for run in baseline_runs]
                baseline_strict_values = [
                    run.get("contamination_aware_pctp") or 0.0 for run in baseline_runs
                ]
                if not baseline_values:
                    continue
                paired_comparisons.append(
                    {
                        "domain": "bias",
                        "family": family_name,
                        "case_id": case["id"],
                        "metric": "pctp",
                        "compare_against": method,
                        "comparison": _paired_metric_comparison(
                            baseline_values, adaptive_values
                        ),
                    }
                )
                paired_comparisons.append(
                    {
                        "domain": "bias",
                        "family": family_name,
                        "case_id": case["id"],
                        "metric": "contamination_aware_pctp",
                        "compare_against": method,
                        "comparison": _paired_metric_comparison(
                            baseline_strict_values, adaptive_strict_values
                        ),
                    }
                )

    quality_by_repeat_method: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for result in quality_results:
        quality_by_repeat_method.setdefault(
            (result["repeat_index"], result["method"]), []
        ).append(result)
    adaptive_qrs_by_repeat = [
        aggregate_quality_results(results)["avg_quality_retention_score"]
        for (repeat_index, method), results in sorted(quality_by_repeat_method.items())
        if method == "adaptive"
    ]
    for method in STATIC_METHODS:
        method_qrs_by_repeat = [
            aggregate_quality_results(results)["avg_quality_retention_score"]
            for (repeat_index, method_name), results in sorted(quality_by_repeat_method.items())
            if method_name == method
        ]
        if adaptive_qrs_by_repeat and method_qrs_by_repeat:
            paired_comparisons.append(
                {
                    "domain": "quality",
                    "metric": "avg_quality_retention_score",
                    "compare_against": method,
                    "comparison": _paired_metric_comparison(
                        method_qrs_by_repeat, adaptive_qrs_by_repeat
                    ),
                }
            )

    payload = {
        "created_at": timestamp,
        "backend": args.backend,
        "model": args.model,
        "families": args.families,
        "repeat": args.repeat,
        "prompt_difficulty": args.prompt_difficulty,
        "asc_controller_config": ctrl_cfg or None,
        "run_group": args.run_group or None,
        "quality_prompt_file": None if args.skip_quality else str(quality_prompt_path),
        "generation_cfg": generation_cfg,
        "summary": list(summary.values()),
        "quality_summary": quality_summary_entries,
        "paired_comparisons": paired_comparisons,
        "detailed_results": all_results,
        "quality_results": quality_results,
    }
    out = results_dir / f"{args.output_stem}_{timestamp}.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("Saved adaptive evaluation to %s", out)

    # Print compact table
    print(
        f"\n{'Model':<18} {'Family':<10} {'Case':<30} {'Method':<22} "
        f"{'PCTP':>6} {'MitCost':>8} {'SFR':>6}"
    )
    print("-" * 106)
    for entry in summary.values():
        print(
            f"{entry['model']:<18} {entry['family']:<10} {entry['case_id']:<30} "
            f"{entry['method']:<22} {entry['pctp']['mean']:>6.4f} "
            f"{entry['mitigation_cost']['mean']:>8.4f} {entry['sfr']['mean']:>6.4f}"
        )

    if quality_summary_entries:
        print(
            f"\n{'Model':<18} {'Method':<22} {'QRS':>6} {'OVR':>6} "
            f"{'MitCost':>8}"
        )
        print("-" * 68)
        for entry in sorted(quality_summary_entries, key=lambda item: (item["model"], item["method"])):
            print(
                f"{entry['model']:<18} {entry['method']:<22} "
                f"{entry['qrs']['mean']:>6.4f} "
                f"{entry['ovr']['mean']:>6.4f} "
                f"{entry['mitigation_cost_distribution']['mean']:>8.4f}"
            )


if __name__ == "__main__":
    main()
