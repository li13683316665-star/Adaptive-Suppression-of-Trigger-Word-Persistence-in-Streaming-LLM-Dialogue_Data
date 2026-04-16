"""
Evaluate the Streaming Trigger Detector across multiple trigger families
and models.

For each (model, trigger-family) pair the script:
  1. Runs a baseline conversation (no mitigation).
  2. Feeds every turn to ``StreamingTriggerDetector`` in real time.
  3. After all turns, checks whether the detector's top-ranked tokens
     overlap with the ground-truth trigger family.
  4. Reports precision@k, recall@k, and detection latency (earliest
     turn at which a ground-truth token crosses the risk threshold).

Execution: ``python experiments/07_detector_eval.py`` (see CLI for model,
repeat count, and optional STD weight ablations).
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

from src.bias.streaming_detector import (  # noqa: E402
    StreamingTriggerDetector,
    std_weights_leave_one_out,
)
from src.model.loader import load_config  # noqa: E402
from src.model.ollama_client import DEFAULT_OLLAMA_HOST, ollama_chat  # noqa: E402
from src.simulation.chat_env import VtuberChatEnv  # noqa: E402
from src.utils.helpers import get_project_root, get_results_dir, setup_logging  # noqa: E402
from src.utils.stats import summarize_distribution  # noqa: E402

LOGGER = logging.getLogger("detector_eval")

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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate streaming trigger detector.")
    p.add_argument("--model", default="qwen2.5:7b", help="Ollama model tag.")
    p.add_argument("--host", default=DEFAULT_OLLAMA_HOST, help="Ollama host.")
    p.add_argument(
        "--families",
        nargs="*",
        default=list(PROMPT_SUITES.keys()),
        help="Trigger families to evaluate (default: all).",
    )
    p.add_argument("--repeat", type=int, default=3, help="Repeats per family/case.")
    p.add_argument(
        "--risk-threshold", type=float, default=0.3,
        help="Risk threshold for considering a token detected.",
    )
    p.add_argument("--top-k", type=int, default=10, help="Top-k for precision/recall.")
    p.add_argument("--output-stem", default="detector_eval", help="Output stem.")
    p.add_argument("--run-group", default="", help="Run group label.")
    p.add_argument(
        "--prompt-difficulty",
        default="d1",
        choices=["d1", "d2", "d3"],
        help="Prompt difficulty tier to use (d1=original, d2=medium, d3=high).",
    )
    p.add_argument(
        "--std-drop-signal",
        default="",
        choices=["", "f_freq", "f_surv", "f_leak", "f_pers", "f_lag"],
        help="Leave-one-signal-out ablation: zero this signal's weight and renormalize.",
    )
    return p.parse_args()


def _load_cases(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_env(case: dict[str, Any]) -> VtuberChatEnv:
    env = VtuberChatEnv(system_prompt=case.get("system_prompt", ""))
    env.extend_messages(case.get("dialogue_context", []), channel="dialogue")
    env.extend_messages(case.get("environment_context", []), channel="environment")
    if case.get("seed_messages"):
        env.extend_messages(case["seed_messages"], channel="dialogue")
    return env


def _precision_at_k(detected: list[str], ground_truth: set[str], k: int) -> float:
    top = detected[:k]
    if not top:
        return 0.0
    return sum(1 for t in top if t in ground_truth) / len(top)


def _recall_at_k(detected: list[str], ground_truth: set[str], k: int) -> float:
    if not ground_truth:
        return 0.0
    top = detected[:k]
    return sum(1 for t in top if t in ground_truth) / len(ground_truth)


def _run_case(
    *,
    case: dict[str, Any],
    model: str,
    host: str,
    generation_cfg: dict[str, Any],
    risk_threshold: float,
    top_k: int,
    std_drop_signal: str = "",
) -> dict[str, Any]:
    """Run one case and return detector evaluation metrics."""
    env = _build_env(case)
    wcfg = std_weights_leave_one_out(std_drop_signal or None)
    detector = StreamingTriggerDetector(config=wcfg)
    ground_truth = set(case.get("trigger_family", []))
    turn_logs: list[dict[str, Any]] = []

    # Feed dialogue context to the detector so it knows the vocabulary
    for msg in case.get("dialogue_context", []):
        detector.observe_turn(msg["content"], msg["role"], "dialogue")
    for msg in case.get("environment_context", []):
        detector.observe_turn(msg["content"], msg.get("role", "system"), "environment")

    detection_latency: int | None = None

    for turn_idx, turn in enumerate(case.get("evaluation_turns", [])):
        is_correction = bool(turn.get("suppression"))
        env.add_message("user", turn["content"], channel="dialogue")
        detector.observe_turn(turn["content"], "user", "dialogue",
                              is_correction=is_correction)

        assistant_text = ollama_chat(
            host=host,
            model=model,
            messages=env.render_messages(),
            generation_cfg=generation_cfg,
        )
        env.add_message("assistant", assistant_text, channel="dialogue")
        detector.observe_turn(assistant_text, "assistant", "dialogue")

        risks = detector.get_risk_scores(top_k=top_k)
        detected_now = [r.token for r in risks if r.risk >= risk_threshold]

        if detection_latency is None and any(t in ground_truth for t in detected_now):
            detection_latency = turn_idx

        turn_logs.append({
            "turn_index": turn_idx,
            "user": turn["content"],
            "assistant": assistant_text,
            "top_risks": [
                {"token": r.token, "risk": r.risk, "signals": r.signals}
                for r in risks[:5]
            ],
            "detected_family": detected_now,
        })

    # Final evaluation
    final_risks = detector.get_risk_scores(top_k=top_k)
    final_detected = [r.token for r in final_risks]

    prec = _precision_at_k(final_detected, ground_truth, top_k)
    rec = _recall_at_k(final_detected, ground_truth, top_k)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)

    return {
        "case_id": case["id"],
        "scenario_type": case.get("scenario_type"),
        "ground_truth_family": sorted(ground_truth),
        "final_detected_top_k": [
            {"token": r.token, "risk": r.risk} for r in final_risks
        ],
        "precision_at_k": round(prec, 4),
        "recall_at_k": round(rec, 4),
        "f1_at_k": round(f1, 4),
        "detection_latency": detection_latency,
        "top_k": top_k,
        "risk_threshold": risk_threshold,
        "std_drop_signal": std_drop_signal or None,
        "std_weights": wcfg,
        "turn_logs": turn_logs,
    }


def main() -> None:
    args = _parse_args()
    cfg = load_config()
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    project_root = get_project_root()
    generation_cfg = cfg.get("generation", {})
    rel = cfg.get("paths", {}).get("data_results")
    results_dir = (project_root / rel).resolve() if rel else get_results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_results: list[dict[str, Any]] = []

    for family_name in args.families:
        tiered = PROMPT_SUITES_TIERED.get(family_name, {})
        suite_rel = tiered.get(args.prompt_difficulty) or PROMPT_SUITES.get(family_name)
        if suite_rel is None:
            LOGGER.warning("No prompt suite known for family %s", family_name)
            continue
        suite_path = project_root / suite_rel
        if not suite_path.exists():
            LOGGER.warning("Prompt suite not found: %s", suite_path)
            continue
        cases = _load_cases(suite_path)

        for repeat_idx in range(1, args.repeat + 1):
            for case in cases:
                LOGGER.info(
                    "[%s] model=%s family=%s case=%s repeat=%d",
                    timestamp, args.model, family_name, case["id"], repeat_idx,
                )
                result = _run_case(
                    case=case,
                    model=args.model,
                    host=args.host,
                    generation_cfg=generation_cfg,
                    risk_threshold=args.risk_threshold,
                    top_k=args.top_k,
                    std_drop_signal=args.std_drop_signal,
                )
                result["model"] = args.model
                result["family_name"] = family_name
                result["repeat_index"] = repeat_idx
                all_results.append(result)

    # Aggregate summary
    summary: dict[str, Any] = {}
    for r in all_results:
        key = f"{r['model']}_{r['family_name']}_{r['case_id']}"
        if key not in summary:
            summary[key] = {
                "model": r["model"],
                "family": r["family_name"],
                "case_id": r["case_id"],
                "precision_values": [],
                "recall_values": [],
                "f1_values": [],
                "latency_values": [],
            }
        summary[key]["precision_values"].append(r["precision_at_k"])
        summary[key]["recall_values"].append(r["recall_at_k"])
        summary[key]["f1_values"].append(r["f1_at_k"])
        if r["detection_latency"] is not None:
            summary[key]["latency_values"].append(r["detection_latency"])

    for entry in summary.values():
        for metric in ("precision", "recall", "f1"):
            vals = entry[f"{metric}_values"]
            entry[metric] = summarize_distribution(vals)
        lat = entry["latency_values"]
        entry["latency"] = summarize_distribution(lat) if lat else None

    payload = {
        "created_at": timestamp,
        "model": args.model,
        "families": args.families,
        "repeat": args.repeat,
        "prompt_difficulty": args.prompt_difficulty,
        "risk_threshold": args.risk_threshold,
        "top_k": args.top_k,
        "run_group": args.run_group or None,
        "generation_cfg": generation_cfg,
        "summary": list(summary.values()),
        "detailed_results": all_results,
    }
    out = results_dir / f"{args.output_stem}_{timestamp}.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("Saved detector evaluation to %s", out)

    # Print compact table
    print(f"\n{'Model':<18} {'Family':<10} {'Case':<30} {'P@k':>6} {'R@k':>6} {'F1':>6} {'Lat':>5}")
    print("-" * 85)
    for entry in summary.values():
        print(
            f"{entry['model']:<18} {entry['family']:<10} {entry['case_id']:<30} "
            f"{entry['precision']['mean']:>6.3f} {entry['recall']['mean']:>6.3f} "
            f"{entry['f1']['mean']:>6.3f} "
            f"{str(entry['latency']['mean'] if entry['latency'] else 'n/a'):>5}"
        )


if __name__ == "__main__":
    main()
