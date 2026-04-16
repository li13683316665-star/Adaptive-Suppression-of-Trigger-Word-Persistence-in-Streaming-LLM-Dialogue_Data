"""
Figure and table export for the manuscript (cross-model benchmark layer).

Loads ``crossmodel_manifest_*.json`` payloads from the configured results directory
(``get_results_dir()`` / ``RSE_RESULTS_DIR``; default ``data_new/results``), optionally
merges manifests, aggregates metrics with bootstrap confidence intervals, and
writes publication PNGs to ``Docs/Paper/figures/`` plus companion JSON tables.
After merges, payloads are **always** filtered to ``paper.figure_models`` in ``configs/default.yaml``
(default: quartet + ``deepseek-chat``). CLI flags include ``--merge-all-manifests``, ``--figure-models``,
optional ``--figure-all-models``, and ``--manifest`` to pin a specific run group.

Execution: ``python experiments/04_build_paper_artifacts.py`` from the repository root.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import warnings
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model.loader import load_config  # noqa: E402
from src.utils.helpers import get_results_dir  # noqa: E402
from src.utils.stats import (
    bonferroni_alpha,
    cohen_d_interpretation,
    holm_adjusted_p_values,
    summarize_distribution,
)

RESULTS_DIR = get_results_dir()
FIGURES_DIR = ROOT / "Docs" / "Paper" / "figures"
# Mutable output directory for PNGs (``main()`` may point this at a supplemental folder).
_active_figures_dir: Path = FIGURES_DIR


def _fig_out(name: str) -> Path:
    return _active_figures_dir / name

# Fallback if configs/default.yaml omits paper.figure_models.
_FALLBACK_PAPER_FIGURE_MODELS = (
    "qwen3.5:4b",
    "gemma4:e4b",
    "openbmb/minicpm-v4.5:8b",
    "ministral-3:8b",
    "deepseek-chat",
)


def _artifact_kinds(manifest: dict[str, Any]) -> set[str]:
    return {str(a.get("kind")) for a in manifest.get("artifacts", [])}


def _manifest_has_full_benchmark_coverage(manifest: dict[str, Any]) -> bool:
    kinds = _artifact_kinds(manifest)
    has_ablation = any(k.startswith("ablation_") for k in kinds)
    return (
        "baseline" in kinds
        and "compare" in kinds
        and "quality" in kinds
        and has_ablation
    )


def _manifest_has_core_crossmodel_layers(manifest: dict[str, Any]) -> bool:
    """Baseline + compare + quality present (hosted/OpenAI runs often omit ablation JSON files)."""
    kinds = _artifact_kinds(manifest)
    return "baseline" in kinds and "compare" in kinds and "quality" in kinds


def _manifest_eligible_for_model_merge(manifest: dict[str, Any]) -> bool:
    """Use full benchmark definition when possible; otherwise allow core layers so DeepSeek et al. merge."""
    return _manifest_has_full_benchmark_coverage(manifest) or _manifest_has_core_crossmodel_layers(
        manifest
    )


def _find_manifest_with_benchmark_coverage(exclude_run_group: str | None) -> Path | None:
    candidates = sorted(
        RESULTS_DIR.glob("crossmodel_manifest_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if exclude_run_group and loaded.get("run_group") == exclude_run_group:
            continue
        if _manifest_has_full_benchmark_coverage(loaded):
            return path
    return None


def _models_in_payload_entries(payload_entries: list[dict[str, Any]]) -> set[str]:
    """Distinct ``artifact.model`` tags present in loaded result JSON payloads."""
    return {
        str(e.get("artifact", {}).get("model", ""))
        for e in payload_entries
        if str(e.get("artifact", {}).get("model", ""))
    }


def _merge_extra_benchmark_manifests_for_paper_models(
    payload_entries: list[dict[str, Any]],
    models: list[str],
    *,
    merged_run_groups: set[str],
    default_figure_models: list[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Merge manifests until every ``paper.figure_models`` tag has at least one payload.

    The newest ``crossmodel_manifest_*.json`` is often a *single-model* incremental suite that still
    lists ``baseline``/``compare``/``quality``/``ablation`` kinds, so
    ``_manifest_has_full_benchmark_coverage`` is true and the legacy one-shot merge never pulls the
    other quartet manifests. This pass walks older manifests (by mtime) and merges manifests that
    introduce a missing configured model. Hosted ``deepseek-chat`` manifests often have no
    ``ablation_*`` artifacts; those still qualify if ``baseline``+``compare``+``quality`` are present.
    """
    desired = set(default_figure_models)
    candidates = sorted(
        RESULTS_DIR.glob("crossmodel_manifest_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        if desired.issubset(_models_in_payload_entries(payload_entries)):
            break
        loaded = _load_json(path)
        rg = str(loaded.get("run_group") or "")
        if rg in merged_run_groups:
            continue
        if not _manifest_eligible_for_model_merge(loaded):
            continue
        extra_entries = _load_payloads_from_manifest(loaded)
        new_models = _models_in_payload_entries(extra_entries) - _models_in_payload_entries(
            payload_entries
        )
        if not new_models:
            continue
        missing = desired - _models_in_payload_entries(payload_entries)
        if not (new_models & missing):
            continue
        payload_entries = _merge_payload_entries_unique(payload_entries, extra_entries)
        merged_run_groups.add(rg)
        if not _manifest_has_full_benchmark_coverage(loaded):
            print(
                f"[04] merged manifest without ablation layers (e.g. hosted API): {rg}",
                flush=True,
            )
        for m in loaded.get("models", []):
            if m not in models:
                models.append(m)
    return payload_entries, models


def _merge_payload_entries_unique(
    primary: list[dict[str, Any]],
    extra: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    seen = {entry["path"] for entry in primary}
    merged = list(primary)
    for entry in extra:
        if entry["path"] not in seen:
            merged.append(entry)
            seen.add(entry["path"])
    return merged


def _payload_has_adaptive_entries(payload_entries: list[dict[str, Any]]) -> bool:
    return any(str(e["artifact"].get("kind")) == "adaptive" for e in payload_entries)


def _find_manifest_with_adaptive_artifacts(exclude_run_group: str | None) -> Path | None:
    candidates = sorted(
        RESULTS_DIR.glob("crossmodel_manifest_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if exclude_run_group and loaded.get("run_group") == exclude_run_group:
            continue
        if "adaptive" in _artifact_kinds(loaded):
            return path
    return None

SCENARIO_ORDER = [
    "control_clean",
    "dialogue_triggered",
    "environment_triggered",
    "correction_recovery",
]
METHOD_ORDER = [
    "baseline",
    "context_reset",
    "targeted_guardrail",
    "repetition_penalty",
    "adaptive",
]
ABLATION_DENSITY_ORDER = ["density_low", "density_medium", "density_high"]
ABLATION_CHANNEL_ORDER = [
    "channel_dialogue_only",
    "channel_environment_only",
    "channel_mixed",
]

# Consistent model colors for cross-model figures (colorblind-friendly)
_MODEL_PALETTE = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#F0E442",
    "#56B4E9",
]


def _model_color(model: str, index: int) -> str:
    return _MODEL_PALETTE[index % len(_MODEL_PALETTE)]


def _benchmark_methods_from_manifest(manifest: dict[str, Any]) -> list[str]:
    """Methods actually run in the benchmark layer (omit adaptive unless listed)."""
    listed = manifest.get("methods")
    if not listed:
        return [m for m in METHOD_ORDER if m != "adaptive"]
    allowed = set(listed)
    return [m for m in METHOD_ORDER if m in allowed]


_SINGLETON_EPS = 1e-6


def _singleton_column_flags_and_note(
    models: list[str],
    col_labels: list[str],
    values_per_model: list[list[float]],
    missing_per_model: list[list[bool]],
    *,
    enabled: bool = False,
) -> tuple[list[bool], str]:
    """
    Flag columns where exactly one model has a non-zero value (and runs exist).
    Returns footnote text listing column→model to preempt “scenario fitted to one model” concerns.
    Disabled by default so figures show raw bars without diagnostic shading; pass enabled=True
    (CLI: --show-singleton-signal) to restore the legacy annotation.
    """
    if not enabled:
        n = len(col_labels)
        return ([False] * n if n else []), ""
    if not models or not values_per_model or not col_labels:
        return [], ""
    n_models, n_cols = len(models), len(col_labels)
    if len(values_per_model) != n_models or any(len(v) != n_cols for v in values_per_model):
        return [], ""
    flags: list[bool] = []
    chunks: list[str] = []
    for j in range(n_cols):
        nz = [
            models[i]
            for i in range(n_models)
            if not missing_per_model[i][j] and values_per_model[i][j] > _SINGLETON_EPS
        ]
        one = len(nz) == 1 and n_models >= 2
        flags.append(one)
        if one:
            chunks.append(f"{col_labels[j]} → {nz[0]}")
    note = ""
    if chunks:
        note = (
            "† Singleton signal: " + "; ".join(chunks) + ". "
            "Same prompt suite for every model — not tuned per model."
        )
    return flags, note


def _shade_singleton_groups(ax: Any, x_positions: list[int], flags: list[bool]) -> None:
    """Light highlight behind groups where only one model is non-zero."""
    for j, flag in enumerate(flags):
        if not flag:
            continue
        ax.axvspan(
            float(x_positions[j]) - 0.5,
            float(x_positions[j]) + 0.5,
            facecolor="#FFF3CD",
            alpha=0.45,
            zorder=0,
            linewidth=0,
        )


def _apply_figure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.35,
            "grid.linestyle": "--",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )
    for style in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot"):
        try:
            plt.style.use(style)
            break
        except OSError:
            continue


# Upper y-axis limit for bar charts: max(non-missing bar height) + pad.
Y_AXIS_VALUE_PAD = 0.25


def _bar_chart_ylim_top(
    values_per_model: list[list[float]],
    missing_per_model: list[list[bool]],
    *,
    pad: float = Y_AXIS_VALUE_PAD,
) -> float:
    """Upper y limit = max plotted value among non-missing cells + pad."""
    max_v = 0.0
    for vals, miss in zip(values_per_model, missing_per_model):
        for v, m in zip(vals, miss):
            if not m:
                max_v = max(max_v, float(v))
    return max_v + pad


def _lock_bar_ylim(axes: Any, ylim_top: float) -> None:
    """Apply y limits and disable autoscale. Call after tight_layout / subplots_adjust."""
    seq = axes if isinstance(axes, list) else [axes]
    for ax in seq:
        ax.set_autoscaley_on(False)
        ax.set_ylim(0, ylim_top)


def _annotate_bar_tops(
    ax: Any,
    bars: Any,
    values: list[float],
    missing: list[bool],
    *,
    fmt: str = "{:.2f}",
    pad: float = 0.02,
) -> None:
    for bar, val, miss in zip(bars, values, missing):
        if miss:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                0.02,
                "n/a",
                ha="center",
                va="bottom",
                fontsize=7,
                color="0.35",
                rotation=90,
                clip_on=False,
            )
            continue
        h = bar.get_height()
        # Measured zero is still data — label so it is not confused with missing runs
        if h <= 1e-9:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                0.02,
                fmt.format(0.0),
                ha="center",
                va="bottom",
                fontsize=7,
                color="0.35",
                clip_on=False,
            )
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + pad,
            fmt.format(val),
            ha="center",
            va="bottom",
            fontsize=7,
            color="0.2",
            clip_on=False,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paper artifacts from cross-model runs.")
    parser.add_argument(
        "--manifest",
        default="",
        help="Optional explicit manifest file path. Defaults to the latest crossmodel manifest.",
    )
    parser.add_argument(
        "--no-merge-benchmark",
        action="store_true",
        help="Do not merge a full benchmark manifest when the primary manifest is method-only.",
    )
    parser.add_argument(
        "--no-merge-adaptive",
        action="store_true",
        help="Do not merge adaptive/detector artifacts from another manifest when the primary has no adaptive payloads.",
    )
    parser.add_argument(
        "--merge-all-manifests",
        action="store_true",
        help="Merge baseline/compare/detector/adaptive data from ALL available crossmodel manifests.",
    )
    parser.add_argument(
        "--figure-models",
        nargs="*",
        default=None,
        help=(
            "Subset of model tags for figure panels only. Must match manifest strings. "
            "If omitted (or passed with no tags), uses paper.figure_models from configs/default.yaml "
            "(default: quartet + deepseek-chat)."
        ),
    )
    parser.add_argument(
        "--figure-all-models",
        action="store_true",
        help=(
            "Draw figures for every model present in the manifest (ignore default paper figure-models list). "
            "Use when the manifest includes extra pilot models you want in the panels."
        ),
    )
    parser.add_argument(
        "--show-singleton-signal",
        action="store_true",
        help=(
            "Highlight columns where only one model is non-zero (yellow band, †, footnote). "
            "Default is off so plots show full numeric data without this diagnostic overlay."
        ),
    )
    parser.add_argument(
        "--exclude-models",
        nargs="*",
        default=None,
        help=(
            "After merging manifests, drop payloads whose artifact model tag is in this list."
        ),
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help=(
            "Only write paper_*.json tables under the results directory; do not emit or copy PNGs "
            "under Docs/Paper/figures/. Use for supplemental manifests so the main submission "
            "figure freeze is not overwritten."
        ),
    )
    parser.add_argument(
        "--figures-output-dir",
        default="",
        help=(
            "Directory for PNG output (project-root-relative or absolute). "
            "Default: Docs/Paper/figures."
        ),
    )
    parser.add_argument(
        "--extra-manifest",
        action="append",
        default=[],
        metavar="PATH",
        help=(
            "Additional crossmodel_manifest_*.json to merge into the primary manifest "
            "(e.g. hosted DeepSeek runs alongside the quartet freeze)."
        ),
    )
    parser.add_argument(
        "--no-manuscript-figure-aliases",
        action="store_true",
        help="Skip copying crossmodel_*.png to Figure_2.png..Figure_6.png in the output directory.",
    )
    return parser.parse_args()


def _filter_payloads_to_paper_models(
    payload_entries: list[dict[str, Any]],
    allowed_models: list[str],
) -> list[dict[str, Any]]:
    """Drop artifact payloads whose model tag is not in the paper allowlist."""
    allowed = set(allowed_models)
    return [
        e
        for e in payload_entries
        if str(e.get("artifact", {}).get("model", "")) in allowed
    ]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_manifest() -> Path:
    manifests = sorted(
        RESULTS_DIR.glob("crossmodel_manifest_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not manifests:
        raise FileNotFoundError(f"No cross-model manifest found in {RESULTS_DIR}.")
    return manifests[0]


def _round(value: float | None) -> float:
    return round(float(value or 0.0), 4)


def _stats(values: list[float]) -> dict[str, float | int | list[float] | dict[str, float]]:
    return summarize_distribution(values)


def _load_payloads_from_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for artifact in manifest.get("artifacts", []):
        for filename in artifact.get("files", []):
            path = RESULTS_DIR / filename
            if not path.exists():
                warnings.warn(
                    f"Manifest lists a result file that is not on disk (skipped): {filename}",
                    UserWarning,
                    stacklevel=2,
                )
                continue
            norm_path = str(path.resolve())
            if norm_path in seen_paths:
                continue
            payloads.append(
                {
                    "artifact": artifact,
                    "path": norm_path,
                    "payload": _load_json(path),
                }
            )
            seen_paths.add(norm_path)
    return payloads


def _infer_family_from_artifact(artifact: dict[str, Any]) -> str:
    """Infer trigger family from artifact metadata or prompt file path."""
    if artifact.get("family"):
        return str(artifact["family"])
    prompt_file = str(artifact.get("prompt_file", ""))
    for fam in ("emotion", "color", "location"):
        if fam in prompt_file:
            return fam
    return "location"


def _infer_family_for_entry(entry: dict[str, Any]) -> str:
    """Prefer payload evidence over manifest metadata when family tags drift."""
    payload = entry.get("payload", {})
    prompt_file = str(payload.get("prompt_file", ""))
    for fam in ("emotion", "color", "location"):
        if fam in prompt_file:
            return fam

    results = payload.get("results", [])
    if results:
        first = results[0]
        result_id = str(first.get("id") or first.get("case_id") or "")
        for prefix in ("emotion_", "color_", "location_"):
            if result_id.startswith(prefix):
                return prefix[:-1]

        trigger_word = str(first.get("trigger_word", "")).strip().lower()
        if trigger_word in {"emotion", "color", "location"}:
            return trigger_word

    return _infer_family_from_artifact(entry["artifact"])


def _normalize_scenario_id(result_id: str) -> str:
    """Strip family prefix from scenario IDs (e.g. 'emotion_control_clean' → 'control_clean')."""
    for prefix in ("emotion_", "color_", "location_"):
        if result_id.startswith(prefix):
            return result_id[len(prefix):]
    return result_id


def _build_baseline_rows(payload_entries: list[dict[str, Any]], models: list[str]) -> list[dict[str, Any]]:
    # Key: (model, family, normalized_scenario_id)
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for entry in payload_entries:
        artifact = entry["artifact"]
        if artifact["kind"] != "baseline":
            continue
        model = artifact["model"]
        family = _infer_family_for_entry(entry)
        for result in entry["payload"]["results"]:
            norm_id = _normalize_scenario_id(result["id"])
            grouped[(model, family, norm_id)].append(result)

    # Collect all (family, model) pairs found in data, preserving family order
    found_families: list[str] = []
    for (model, family, _) in grouped:
        if family not in found_families:
            found_families.append(family)
    family_order = [f for f in ("location", "emotion", "color") if f in found_families] + \
                   [f for f in found_families if f not in ("location", "emotion", "color")]

    rows: list[dict[str, Any]] = []
    for family in family_order:
        for model in models:
            for scenario in SCENARIO_ORDER:
                runs = grouped.get((model, family, scenario), [])
                pctp_values = [run["metrics"]["pctp"] or 0.0 for run in runs]
                contaminated_rate_values = [
                    run["metrics"]["post_correction_contaminated_rate"] or 0.0 for run in runs
                ]
                family_values = [sum(run["metrics"]["family_counts"].values()) for run in runs]
                sfr_values = [run["metrics"]["suppression_failure_rate"] for run in runs]
                rtc_values = [(run["metrics"]["recovery_turn_count"] or 0) for run in runs]
                rows.append(
                    {
                        "model": model,
                        "family": family,
                        "scenario": scenario,
                        "runs": len(runs),
                        "pctp": _stats(pctp_values),
                        "post_correction_contaminated_rate": _stats(contaminated_rate_values),
                        "trigger_family_total": _stats(family_values),
                        "sfr": _stats(sfr_values),
                        "rtc": _stats(rtc_values),
                    }
                )
    return rows


def _build_compare_rows(payload_entries: list[dict[str, Any]], models: list[str]) -> list[dict[str, Any]]:
    # Key: (model, family, method)
    per_run_grouped: dict[tuple[str, str, str], list[dict[str, float]]] = defaultdict(list)
    found_families: list[str] = []
    for entry in payload_entries:
        artifact = entry["artifact"]
        if artifact["kind"] != "compare":
            continue
        model = artifact["model"]
        family = _infer_family_for_entry(entry)
        if family not in found_families:
            found_families.append(family)
        grouped_by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for result in entry["payload"]["results"]:
            grouped_by_method[result["method"]].append(result)

        for method, method_results in grouped_by_method.items():
            pctp_values = [result["metrics"]["pctp"] or 0.0 for result in method_results]
            environment_result = next(
                (
                    result
                    for result in method_results
                    if _normalize_scenario_id(result.get("case_id", "")) == "environment_triggered"
                ),
                None,
            )
            per_run_grouped[(model, family, method)].append(
                {
                    "avg_pctp": mean(pctp_values) if pctp_values else 0.0,
                    "avg_sfr": mean(
                        result["metrics"]["suppression_failure_rate"] for result in method_results
                    ),
                    "avg_rtc": mean(
                        (result["metrics"]["recovery_turn_count"] or 0)
                        for result in method_results
                    ),
                    "avg_odr": mean(result["metrics"]["off_task_rate"] for result in method_results),
                    "environment_pctp": (
                        (environment_result["metrics"]["pctp"] or 0.0)
                        if environment_result
                        else 0.0
                    ),
                }
            )

    family_order = [f for f in ("location", "emotion", "color") if f in found_families] + \
                   [f for f in found_families if f not in ("location", "emotion", "color")]

    rows: list[dict[str, Any]] = []
    for family in family_order:
        for model in models:
            for method in METHOD_ORDER:
                runs = per_run_grouped.get((model, family, method), [])
                rows.append(
                    {
                        "model": model,
                        "family": family,
                        "method": method,
                        "runs": len(runs),
                        "avg_pctp": _stats([run["avg_pctp"] for run in runs]),
                        "avg_sfr": _stats([run["avg_sfr"] for run in runs]),
                        "avg_rtc": _stats([run["avg_rtc"] for run in runs]),
                        "avg_odr": _stats([run["avg_odr"] for run in runs]),
                        "environment_pctp": _stats([run["environment_pctp"] for run in runs]),
                    }
                )
    return rows


def _build_quality_rows(payload_entries: list[dict[str, Any]], models: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    for entry in payload_entries:
        artifact = entry["artifact"]
        if artifact["kind"] != "quality":
            continue
        model = artifact["model"]
        for method, aggregate in entry["payload"]["aggregates"].items():
            grouped[(model, method)].append(aggregate)

    rows: list[dict[str, Any]] = []
    for model in models:
        for method in METHOD_ORDER:
            runs = grouped.get((model, method), [])
            rows.append(
                {
                    "model": model,
                    "method": method,
                    "runs": len(runs),
                    "avg_quality_retention_score": _stats(
                        [run["avg_quality_retention_score"] for run in runs]
                    ),
                    "avg_format_success": _stats([run["avg_format_success"] for run in runs]),
                    "avg_relevance_score": _stats([run["avg_relevance_score"] for run in runs]),
                    "avg_context_retention_score": _stats(
                        [run["avg_context_retention_score"] for run in runs]
                    ),
                    "over_suppression_rate": _stats(
                        [run["over_suppression_rate"] for run in runs]
                    ),
                }
            )
    return rows


def _build_detector_rows(payload_entries: list[dict[str, Any]], models: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for entry in payload_entries:
        artifact = entry["artifact"]
        if artifact["kind"] != "detector":
            continue
        model = artifact["model"]
        for result in entry["payload"].get("detailed_results", []):
            grouped[(model, result["family_name"], result["case_id"])].append(result)

    rows: list[dict[str, Any]] = []
    for model in models:
        for family in sorted(
            {key[1] for key in grouped if key[0] == model}
        ):
            for case_id in sorted(
                {key[2] for key in grouped if key[0] == model and key[1] == family}
            ):
                runs = grouped.get((model, family, case_id), [])
                rows.append(
                    {
                        "model": model,
                        "family": family,
                        "case_id": case_id,
                        "runs": len(runs),
                        "precision": _stats([run["precision_at_k"] for run in runs]),
                        "recall": _stats([run["recall_at_k"] for run in runs]),
                        "f1": _stats([run["f1_at_k"] for run in runs]),
                        "latency": _stats(
                            [
                                float(run["detection_latency"])
                                for run in runs
                                if run["detection_latency"] is not None
                            ]
                        ),
                    }
                )
    return rows


def _build_adaptive_rows(payload_entries: list[dict[str, Any]], models: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for entry in payload_entries:
        artifact = entry["artifact"]
        if artifact["kind"] != "adaptive":
            continue
        model = artifact["model"]
        for result in entry["payload"].get("detailed_results", []):
            grouped[(model, result["family_name"], result["case_id"], result["method"])].append(
                result
            )

    rows: list[dict[str, Any]] = []
    for model in models:
        for family in sorted(
            {key[1] for key in grouped if key[0] == model}
        ):
            for case_id in sorted(
                {key[2] for key in grouped if key[0] == model and key[1] == family}
            ):
                for method in METHOD_ORDER:
                    runs = grouped.get((model, family, case_id, method), [])
                    rows.append(
                        {
                            "model": model,
                            "family": family,
                            "case_id": case_id,
                            "method": method,
                            "runs": len(runs),
                            "pctp": _stats([run["pctp"] or 0.0 for run in runs]),
                            "contamination_aware_pctp": _stats(
                                [
                                    run.get("contamination_aware_pctp") or 0.0
                                    for run in runs
                                ]
                            ),
                            "mitigation_cost": _stats(
                                [run["mitigation_cost"] for run in runs]
                            ),
                            "sfr": _stats(
                                [run["suppression_failure_rate"] for run in runs]
                            ),
                        }
                    )
    return rows


def _build_method_quality_rows(
    payload_entries: list[dict[str, Any]], models: list[str]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for entry in payload_entries:
        artifact = entry["artifact"]
        if artifact["kind"] != "adaptive":
            continue
        model = artifact["model"]
        for quality_entry in entry["payload"].get("quality_summary", []):
            grouped[(model, quality_entry["method"])].append(quality_entry)

    rows: list[dict[str, Any]] = []
    for model in models:
        for method in METHOD_ORDER:
            runs = grouped.get((model, method), [])
            rows.append(
                {
                    "model": model,
                    "method": method,
                    "runs": len(runs),
                    "qrs": _stats(
                        [run["avg_quality_retention_score"] for run in runs]
                    ),
                    "ovr": _stats([run["over_suppression_rate"] for run in runs]),
                    "mitigation_cost": _stats(
                        [run["avg_mitigation_cost"] for run in runs]
                    ),
                }
            )
    return rows


def _resolve_figure_models(
    all_models: list[str],
    figure_models: list[str] | None,
    *,
    default_figure_models: list[str],
    use_all_manifest_models: bool,
) -> list[str]:
    """Choose which models appear in PNG panels (tables still use full manifest model list upstream)."""
    if use_all_manifest_models:
        return list(all_models)
    if figure_models:
        allowed = set(all_models)
        ordered = [m for m in figure_models if m in allowed]
        return ordered if ordered else list(all_models)
    ordered = [m for m in default_figure_models if m in set(all_models)]
    return ordered if ordered else list(all_models)


def _build_paired_comparison_rows(payload_entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in payload_entries:
        artifact = entry["artifact"]
        if artifact["kind"] != "adaptive":
            continue
        model = artifact["model"]
        for comparison in entry["payload"].get("paired_comparisons", []):
            rows.append({"model": model, **comparison})
    return rows


def _enrich_paired_comparisons_for_multiple_testing(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Per model: Holm-adjusted p-values and Bonferroni alpha; Cohen's d_z labels."""
    by_model: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        by_model[str(row.get("model", ""))].append(i)
    enriched_rows: list[dict[str, Any]] = [deepcopy(r) for r in rows]
    for model, indices in by_model.items():
        if not indices:
            continue
        pvals = [
            float(
                enriched_rows[i]
                .get("comparison", {})
                .get("paired_test", {})
                .get("p_value", 1.0)
            )
            for i in indices
        ]
        holm = holm_adjusted_p_values(pvals)
        m = len(pvals)
        abonf = bonferroni_alpha(0.05, m)
        for j, i in enumerate(indices):
            comp = enriched_rows[i].setdefault("comparison", {})
            dz = float(comp.get("effect_size_dz", 0.0))
            comp["effect_size_interpretation"] = cohen_d_interpretation(dz)
            pt = dict(comp.get("paired_test", {}))
            pt["p_value_holm"] = round(holm[j], 6)
            pt["alpha_bonferroni_per_test"] = round(abonf, 6)
            pt["m_comparisons"] = m
            comp["paired_test"] = pt
    return {
        "models": sorted(by_model.keys()),
        "tests_per_model": {m: len(ix) for m, ix in sorted(by_model.items())},
        "rows": enriched_rows,
    }


def _build_ablation_rows(payload_entries: list[dict[str, Any]], models: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for entry in payload_entries:
        artifact = entry["artifact"]
        kind = artifact["kind"]
        if not str(kind).startswith("ablation_"):
            continue
        ablation_group = str(kind).replace("ablation_", "", 1)
        model = artifact["model"]
        for result in entry["payload"]["results"]:
            grouped[(model, ablation_group, result["id"])].append(result)

    rows: list[dict[str, Any]] = []
    for model in models:
        for ablation_group, ordered_cases in [
            ("density", ABLATION_DENSITY_ORDER),
            ("channel", ABLATION_CHANNEL_ORDER),
        ]:
            for case_id in ordered_cases:
                runs = grouped.get((model, ablation_group, case_id), [])
                rows.append(
                    {
                        "model": model,
                        "ablation_group": ablation_group,
                        "case_id": case_id,
                        "runs": len(runs),
                        "pctp": _stats([run["metrics"]["pctp"] or 0.0 for run in runs]),
                        "trigger_family_total": _stats(
                            [sum(run["metrics"]["family_counts"].values()) for run in runs]
                        ),
                        "sfr": _stats(
                            [run["metrics"]["suppression_failure_rate"] for run in runs]
                        ),
                    }
                )
    return rows


def _build_distribution_summary(
    manifest: dict[str, Any],
    baseline_rows: list[dict[str, Any]],
    compare_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "run_group": manifest["run_group"],
        "models": manifest["models"],
        "baseline_repeats": manifest["baseline_repeats"],
        "compare_repeats": manifest["compare_repeats"],
        "quality_repeats": manifest["quality_repeats"],
        "ablation_repeats": manifest["ablation_repeats"],
        "environment_triggered_baseline": {},
        "environment_triggered_compare": {},
    }

    for model in manifest["models"]:
        baseline_row = next(
            (
                row
                for row in baseline_rows
                if row["model"] == model
                and row["scenario"] == "environment_triggered"
                and row.get("family", "location") == "location"
            ),
            None,
        )
        summary["environment_triggered_baseline"][model] = (
            {
                "pctp": baseline_row["pctp"],
                "post_correction_contaminated_rate": baseline_row[
                    "post_correction_contaminated_rate"
                ],
                "trigger_family_total": baseline_row["trigger_family_total"],
            }
            if baseline_row
            else {}
        )

        method_entries = {
            row["method"]: row
            for row in compare_rows
            if row["model"] == model and row.get("family", "location") == "location"
        }
        summary["environment_triggered_compare"][model] = {
            method: row["environment_pctp"] for method, row in method_entries.items()
        }
    return summary


def _plot_baseline_panel(
    ax: Any,
    rows: list[dict[str, Any]],
    models: list[str],
    family: str,
    *,
    highlight_singleton: bool = False,
) -> tuple[list[bool], str, float]:
    """Draw one family's baseline PCTP bars on ax; return (singleton_flags, singleton_note, panel_max)."""
    x_positions = list(range(len(SCENARIO_ORDER)))
    width = 0.8 / max(1, len(models))
    col_labels = list(SCENARIO_ORDER)
    values_per_model: list[list[float]] = []
    missing_per_model: list[list[bool]] = []
    for model in models:
        model_rows = [r for r in rows if r["model"] == model and r.get("family", "location") == family]
        values: list[float] = []
        missing: list[bool] = []
        for scenario in SCENARIO_ORDER:
            row = next((r for r in model_rows if r["scenario"] == scenario), None)
            if row is None or row.get("runs", 0) == 0:
                values.append(0.0)
                missing.append(True)
            else:
                values.append(float(row["pctp"]["mean"]))
                missing.append(False)
        values_per_model.append(values)
        missing_per_model.append(missing)
    sing_flags, sing_note = _singleton_column_flags_and_note(
        models, col_labels, values_per_model, missing_per_model, enabled=highlight_singleton
    )
    _shade_singleton_groups(ax, x_positions, sing_flags)
    for index, model in enumerate(models):
        values = values_per_model[index]
        missing = missing_per_model[index]
        offsets = [
            position - 0.4 + width / 2 + index * width for position in x_positions
        ]
        color = _model_color(model, index)
        bars = ax.bar(
            offsets,
            values,
            width=width,
            label=model,
            color=color,
            edgecolor="0.25",
            linewidth=0.4,
            zorder=3,
        )
        for bar, miss in zip(bars, missing):
            if miss:
                bar.set_hatch("///")
                bar.set_facecolor("#E8E8E8")
                bar.set_edgecolor("0.45")
        _annotate_bar_tops(ax, bars, values, missing, pad=0.03)
    base_lbl = [s.replace("_", "\n") for s in SCENARIO_ORDER]
    tick_lbl = [b + ("\n†" if f else "") for b, f in zip(base_lbl, sing_flags)]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(tick_lbl, fontsize=8)
    panel_max = 0.0
    for vals, miss in zip(values_per_model, missing_per_model):
        for v, m in zip(vals, miss):
            if not m:
                panel_max = max(panel_max, float(v))
    ax.set_ylabel("Mean PCTP")
    ax.set_title(f"[{family}] baseline PCTP by scenario", fontsize=10)
    return sing_flags, sing_note, panel_max


def _build_baseline_figure(
    rows: list[dict[str, Any]],
    models: list[str],
    *,
    highlight_singleton: bool = False,
) -> None:
    _apply_figure_style()
    families_in_data = sorted(
        {r.get("family", "location") for r in rows},
        key=lambda f: ("location", "emotion", "color").index(f) if f in ("location", "emotion", "color") else 99,
    )
    if not families_in_data:
        families_in_data = ["location"]

    n_fam = len(families_in_data)
    # Stacked vertical panels (read top→bottom: location → emotion → color) — easier in narrow columns.
    fig_w = max(10.0, 2.1 * len(SCENARIO_ORDER) + 1.5)
    fig_h = max(5.4, 3.85 * n_fam + 1.55)
    fig, axes = plt.subplots(n_fam, 1, figsize=(fig_w, fig_h), sharey=True)
    if n_fam == 1:
        axes = [axes]
    else:
        axes = list(axes)

    all_sing_notes: list[str] = []
    panel_maxes: list[float] = []
    for ax, family in zip(axes, families_in_data):
        _, note, pmax = _plot_baseline_panel(
            ax, rows, models, family, highlight_singleton=highlight_singleton
        )
        panel_maxes.append(pmax)
        if note:
            all_sing_notes.append(note)
    ylim_top = max(panel_maxes) + Y_AXIS_VALUE_PAD if panel_maxes else Y_AXIS_VALUE_PAD
    axes[-1].legend(
        framealpha=0.95,
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=min(3, len(models)),
    )

    fig.text(
        0.5,
        0.032,
        "Note: 0.00 means no measured persistence (data present); hatched = no runs for that cell.",
        ha="center",
        fontsize=8,
        color="0.35",
    )
    if all_sing_notes:
        combined = " | ".join(all_sing_notes)
        fig.text(0.5, 0.011, combined[:200], ha="center", fontsize=7, color="#8B4513")

    fig.tight_layout(rect=(0, 0.09, 1, 0.98))
    _lock_bar_ylim(axes, ylim_top)
    out_path = _fig_out("crossmodel_baseline_pctp.png")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Save per-family individual figures for LaTeX use
    for family in families_in_data:
        fam_fig, fam_ax = plt.subplots(figsize=(10, 5.6))
        _apply_figure_style()
        _, note, pmax = _plot_baseline_panel(
            fam_ax, rows, models, family, highlight_singleton=highlight_singleton
        )
        fam_lim = pmax + Y_AXIS_VALUE_PAD
        fam_ax.legend(
            framealpha=0.95,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.10),
            ncol=min(3, len(models)),
        )
        fam_fig.text(
            0.5, 0.034,
            "Note: 0.00 means no measured persistence; hatched = no runs.",
            ha="center", fontsize=8, color="0.35",
        )
        if note:
            fam_fig.text(0.5, 0.012, note[:200], ha="center", fontsize=7, color="#8B4513")
        fam_fig.tight_layout(rect=(0, 0.11, 1, 0.97))
        _lock_bar_ylim(fam_ax, fam_lim)
        fam_fig.savefig(
            _fig_out(f"crossmodel_baseline_pctp_{family}.png"), dpi=220, bbox_inches="tight"
        )
        plt.close(fam_fig)


def _plot_compare_panel(
    ax: Any,
    rows: list[dict[str, Any]],
    models: list[str],
    method_order: list[str],
    family: str,
    *,
    highlight_singleton: bool = False,
) -> tuple[list[bool], str, float]:
    """Draw one family's compare (environment PCTP by method) bars on ax.

    Returns ``(singleton flags, footnote text, max observed non-missing value)`` for y-axis scaling.
    """
    n_met = len(method_order)
    x_positions = list(range(n_met))
    width = 0.8 / max(1, len(models))
    col_labels = list(method_order)
    values_per_model: list[list[float]] = []
    missing_per_model: list[list[bool]] = []
    for model in models:
        model_rows = [r for r in rows if r["model"] == model and r.get("family", "location") == family]
        values: list[float] = []
        missing: list[bool] = []
        for method in method_order:
            row = next((r for r in model_rows if r["method"] == method), None)
            if row is None or row.get("runs", 0) == 0:
                values.append(0.0)
                missing.append(True)
            else:
                values.append(float(row["environment_pctp"]["mean"]))
                missing.append(False)
        values_per_model.append(values)
        missing_per_model.append(missing)
    sing_flags, sing_note = _singleton_column_flags_and_note(
        models, col_labels, values_per_model, missing_per_model, enabled=highlight_singleton
    )
    _shade_singleton_groups(ax, x_positions, sing_flags)
    for index, model in enumerate(models):
        values = values_per_model[index]
        missing = missing_per_model[index]
        offsets = [
            position - 0.4 + width / 2 + index * width for position in x_positions
        ]
        color = _model_color(model, index)
        bars = ax.bar(
            offsets,
            values,
            width=width,
            label=model,
            color=color,
            edgecolor="0.25",
            linewidth=0.4,
            zorder=3,
        )
        for bar, miss in zip(bars, missing):
            if miss:
                bar.set_hatch("///")
                bar.set_facecolor("#E8E8E8")
                bar.set_edgecolor("0.45")
        _annotate_bar_tops(ax, bars, values, missing)
    base_lbl = [m.replace("_", "\n") for m in method_order]
    tick_lbl = [b + ("\n†" if f else "") for b, f in zip(base_lbl, sing_flags)]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(tick_lbl, fontsize=8)
    max_observed = 0.0
    for vals, miss in zip(values_per_model, missing_per_model):
        for v, m in zip(vals, miss):
            if not m:
                max_observed = max(max_observed, float(v))
    ax.set_ylabel("Mean env_triggered PCTP")
    ax.set_title(f"[{family}] mitigation comparison", fontsize=10)
    return sing_flags, sing_note, max_observed


def _build_compare_figure(
    rows: list[dict[str, Any]],
    models: list[str],
    method_order: list[str],
    *,
    highlight_singleton: bool = False,
) -> None:
    _apply_figure_style()
    families_in_data = sorted(
        {r.get("family", "location") for r in rows},
        key=lambda f: ("location", "emotion", "color").index(f) if f in ("location", "emotion", "color") else 99,
    )
    if not families_in_data:
        families_in_data = ["location"]

    n_fam = len(families_in_data)
    n_met = len(method_order)
    fig_w = max(8.0, 1.9 * n_met + 3)
    fig_h = max(5.4, 3.95 * n_fam + 1.55)
    fig, axes = plt.subplots(n_fam, 1, figsize=(fig_w, fig_h), sharey=True)
    if n_fam == 1:
        axes = [axes]
    else:
        axes = list(axes)

    all_sing_notes: list[str] = []
    panel_maxes: list[float] = []
    for ax, family in zip(axes, families_in_data):
        _, note, pmax = _plot_compare_panel(
            ax, rows, models, method_order, family, highlight_singleton=highlight_singleton
        )
        panel_maxes.append(pmax)
        if note:
            all_sing_notes.append(note)
    ylim_top = max(panel_maxes) + Y_AXIS_VALUE_PAD if panel_maxes else Y_AXIS_VALUE_PAD
    axes[-1].legend(
        framealpha=0.95,
        ncol=min(3, len(models)),
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
    )

    fig.text(
        0.5,
        0.032,
        "Note: 0.00 for context_reset is a measured outcome (strong suppression), not missing data.",
        ha="center",
        fontsize=8,
        color="0.35",
    )
    if all_sing_notes:
        combined = " | ".join(all_sing_notes)
        fig.text(0.5, 0.011, combined[:200], ha="center", fontsize=7, color="#8B4513")

    fig.tight_layout(rect=(0, 0.09, 1, 0.98))
    _lock_bar_ylim(axes, ylim_top)
    fig.savefig(_fig_out("crossmodel_compare_environment_pctp.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Per-family individual figures for LaTeX use
    for family in families_in_data:
        _apply_figure_style()
        fam_w = max(8.0, 1.9 * n_met + 3)
        # ~40% shorter than prior tall single-panel exports (emotion/color).
        fam_h = max(4.0, (max(6.8, 5.4 + 0.2 * len(models))) * 0.6)
        fam_fig, fam_ax = plt.subplots(figsize=(fam_w, fam_h))
        _, note, pmax = _plot_compare_panel(
            fam_ax, rows, models, method_order, family, highlight_singleton=highlight_singleton
        )
        fam_lim = pmax + Y_AXIS_VALUE_PAD
        fam_ax.legend(
            framealpha=0.95,
            ncol=min(3, len(models)),
            loc="upper center",
            bbox_to_anchor=(0.5, -0.10),
        )
        fam_fig.text(
            0.5, 0.042,
            "Note: 0.00 for context_reset is a measured outcome, not missing data.",
            ha="center", fontsize=8, color="0.35",
        )
        if note:
            fam_fig.text(0.5, 0.018, note[:200], ha="center", fontsize=7, color="#8B4513")
        fam_fig.subplots_adjust(left=0.1, right=0.98, top=0.92, bottom=0.24)
        _lock_bar_ylim(fam_ax, fam_lim)
        fam_fig.savefig(
            _fig_out(f"crossmodel_compare_environment_pctp_{family}.png"),
            dpi=220, bbox_inches="tight",
        )
        plt.close(fam_fig)


def _build_quality_figure(
    rows: list[dict[str, Any]],
    models: list[str],
    method_order: list[str],
) -> None:
    """Heatmap: rows = models, columns = methods — easier to compare than N separate bar charts."""
    _apply_figure_style()
    n_m, n_met = len(models), len(method_order)
    mat = np.full((n_m, n_met), np.nan, dtype=float)
    mask = np.zeros((n_m, n_met), dtype=bool)
    for mi, model in enumerate(models):
        model_rows = [row for row in rows if row["model"] == model]
        for mj, method in enumerate(method_order):
            row = next((r for r in model_rows if r["method"] == method), None)
            if row is None or row.get("runs", 0) == 0:
                mask[mi, mj] = True
                continue
            mat[mi, mj] = float(row["avg_quality_retention_score"]["mean"])
    # QRS is defined on [0, 1]; keep color scale fixed so panels stay comparable (~0.96 vs ceiling 1.0).
    fig, ax = plt.subplots(figsize=(max(8.0, 1.2 * n_met + 4), max(3.8, 0.55 * n_m + 2.5)))
    try:
        cmap = plt.colormaps["viridis"].copy()
    except (AttributeError, KeyError):
        cmap = plt.cm.get_cmap("viridis").copy()
    cmap.set_bad("#DDDDDD")
    im = ax.imshow(mat, aspect="auto", vmin=0.0, vmax=1.0, cmap=cmap)
    ax.set_xticks(np.arange(n_met))
    ax.set_yticks(np.arange(n_m))
    ax.set_xticklabels([m.replace("_", "\n") for m in method_order])
    ax.set_yticklabels(models)
    for i in range(n_m):
        for j in range(n_met):
            if mask[i, j]:
                ax.text(j, i, "n/a", ha="center", va="center", color="0.35", fontsize=9)
            else:
                ax.text(
                    j,
                    i,
                    f"{mat[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if mat[i, j] < 0.45 else "0.15",
                    fontsize=9,
                )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean QRS")
    ax.set_title("Quality Retention (QRS) — model × mitigation")
    ax.set_xlabel("Mitigation")
    ax.set_ylabel("Model")
    fig.tight_layout()
    fig.savefig(_fig_out("crossmodel_quality_qrs.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def _build_detector_figure(rows: list[dict[str, Any]], models: list[str]) -> None:
    families = sorted({row["family"] for row in rows})
    if not families:
        return
    _apply_figure_style()

    def _metric_for_family(model: str, family: str, key: str) -> tuple[float, bool]:
        family_rows = [row for row in rows if row["model"] == model and row["family"] == family]
        if not family_rows or all(r.get("runs", 0) == 0 for r in family_rows):
            return 0.0, True
        usable = [r for r in family_rows if r.get("runs", 0) > 0]
        if not usable:
            return 0.0, True
        return float(mean(r[key]["mean"] for r in usable)), False

    metrics = [
        ("f1", "F1"),
        ("precision", "Precision"),
        ("recall", "Recall"),
    ]
    # Vertical stack: F1 / Precision / Recall — avoids cramped horizontal triples in manuscript width.
    fig_h = max(10.5, 3.35 * len(metrics) + 1.8)
    fig_w = max(9.5, 2.0 * len(families) + 4.5)
    fig, axes = plt.subplots(len(metrics), 1, figsize=(fig_w, fig_h), sharey=True)
    axes = [axes] if len(metrics) == 1 else list(axes)
    x_positions = list(range(len(families)))
    width = 0.8 / max(1, len(models))
    global_max = 0.0
    for ax, (field, short_title) in zip(axes, metrics):
        for index, model in enumerate(models):
            values: list[float] = []
            missing: list[bool] = []
            for family in families:
                v, miss = _metric_for_family(model, family, field)
                if not miss:
                    global_max = max(global_max, float(v))
                values.append(v)
                missing.append(miss)
            offsets = [
                position - 0.4 + width / 2 + index * width for position in x_positions
            ]
            color = _model_color(model, index)
            bars = ax.bar(
                offsets,
                values,
                width=width,
                label=model,
                color=color,
                edgecolor="0.25",
                linewidth=0.35,
            )
            for bar, miss in zip(bars, missing):
                if miss:
                    bar.set_hatch("///")
                    bar.set_facecolor("#E8E8E8")
                    bar.set_edgecolor("0.45")
            _annotate_bar_tops(ax, bars, values, missing, pad=0.025)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(families, rotation=0, ha="center")
        ax.set_title(short_title, fontsize=10, pad=8)
        ax.set_ylabel("Score")
    ylim_top = global_max + Y_AXIS_VALUE_PAD
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Streaming trigger detector", y=0.98, fontsize=11)
    fig.text(
        0.5,
        0.945,
        r"F1, precision, and recall computed at $k$ · by trigger family",
        ha="center",
        fontsize=9,
        color="0.35",
    )
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        bbox_transform=fig.transFigure,
        ncol=min(4, len(models)),
        framealpha=0.95,
        fontsize=9,
    )
    fig.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.10, hspace=0.36)
    _lock_bar_ylim(axes, ylim_top)
    fig.savefig(_fig_out("method_detector_f1.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def _build_tradeoff_figure(
    adaptive_rows: list[dict[str, Any]],
    method_quality_rows: list[dict[str, Any]],
    models: list[str],
) -> None:
    """Grouped bars per model: QRS, mean CA-PCTP, mean mitigation cost (adaptive policy)."""
    _apply_figure_style()
    n_models = len(models)
    if n_models == 0:
        return

    metric_colors = ["#4472C4", "#ED7D31", "#70AD47"]
    metric_labels = ["Mean QRS", "Mean CA-PCTP", "Mean cost"]
    x = np.arange(n_models, dtype=float)
    width = 0.22
    fig, ax = plt.subplots(figsize=(max(8.0, 1.15 * n_models + 2), 5.8))

    heights_all: list[list[float]] = [[], [], []]
    missing_all: list[list[bool]] = [[], [], []]

    for model in models:
        qrs_row = next(
            (
                row
                for row in method_quality_rows
                if row["model"] == model and row["method"] == "adaptive"
            ),
            None,
        )
        adaptive_family_rows = [
            row
            for row in adaptive_rows
            if row["model"] == model and row["method"] == "adaptive" and row.get("runs", 0) > 0
        ]
        cost_rows = [
            row
            for row in adaptive_rows
            if row["model"] == model and row["method"] == "adaptive" and row.get("runs", 0) > 0
        ]

        q_ok = qrs_row is not None and int(qrs_row.get("runs", 0) or 0) > 0
        ca_ok = bool(adaptive_family_rows)
        co_ok = bool(cost_rows)

        qrs_val = float(qrs_row["qrs"]["mean"]) if q_ok else None
        ca_val = (
            mean(row["contamination_aware_pctp"]["mean"] for row in adaptive_family_rows)
            if ca_ok
            else None
        )
        cost_val = mean(row["mitigation_cost"]["mean"] for row in cost_rows) if co_ok else None

        heights_all[0].append(qrs_val if qrs_val is not None else 0.0)
        heights_all[1].append(ca_val if ca_val is not None else 0.0)
        heights_all[2].append(cost_val if cost_val is not None else 0.0)
        missing_all[0].append(not q_ok)
        missing_all[1].append(not ca_ok)
        missing_all[2].append(not co_ok)

    any_data = any(not m for sub in missing_all for m in sub)
    if not any_data:
        ax.text(
            0.5,
            0.55,
            "Adaptive evaluation not completed",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=13,
            fontweight="600",
            color="0.25",
        )
        ax.text(
            0.5,
            0.38,
            "No method-layer adaptive runs in the current manifest.",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=9.5,
            color="0.45",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=0)
        fig.subplots_adjust(left=0.1, right=0.97, top=0.9, bottom=0.18)
        fig.savefig(_fig_out("method_adaptive_tradeoff.png"), dpi=220, bbox_inches="tight")
        plt.close(fig)
        return

    for j in range(3):
        offsets = x + (j - 1) * width
        hs = heights_all[j]
        miss = missing_all[j]
        bars = ax.bar(
            offsets,
            hs,
            width,
            label=metric_labels[j],
            color=metric_colors[j],
            edgecolor="0.25",
            linewidth=0.4,
            zorder=3,
        )
        for bar, m in zip(bars, miss):
            if m:
                bar.set_hatch("///")
                bar.set_facecolor("#E8E8E8")
                bar.set_edgecolor("0.45")
        _annotate_bar_tops(ax, bars, hs, miss)

    max_h = 0.0
    for j in range(3):
        for i in range(n_models):
            if not missing_all[j][i]:
                max_h = max(max_h, float(heights_all[j][i]))
    ylim_trade = max_h + Y_AXIS_VALUE_PAD
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(":", "\n") for m in models], fontsize=8)
    ax.set_ylabel("Mean value (QRS / CA-PCTP / cost in [0,1])")
    ax.set_title("Adaptive policy: quality vs contamination vs cost (grouped by model)")
    ax.grid(True, alpha=0.35, axis="y")
    ax.legend(framealpha=0.95, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9)
    fig.text(
        0.5,
        0.02,
        "Hatched bars: no runs or missing aggregate for that metric. Cost: mean mitigation cost over adaptive cells.",
        ha="center",
        fontsize=8,
        color="0.35",
    )
    fig.subplots_adjust(left=0.1, right=0.90, top=0.92, bottom=0.14)
    _lock_bar_ylim(ax, ylim_trade)
    fig.savefig(_fig_out("method_adaptive_tradeoff.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def _build_ablation_figure(
    rows: list[dict[str, Any]],
    models: list[str],
    *,
    highlight_singleton: bool = False,
) -> None:
    _apply_figure_style()
    fig, ax = plt.subplots(figsize=(10, 5.4))
    x_positions = list(range(len(ABLATION_DENSITY_ORDER)))
    width = 0.8 / max(1, len(models))
    col_labels = [case_id.replace("density_", "") for case_id in ABLATION_DENSITY_ORDER]
    values_per_model: list[list[float]] = []
    missing_per_model: list[list[bool]] = []
    for model in models:
        model_rows = [
            row
            for row in rows
            if row["model"] == model and row["ablation_group"] == "density"
        ]
        values: list[float] = []
        missing: list[bool] = []
        for case_id in ABLATION_DENSITY_ORDER:
            row = next((r for r in model_rows if r["case_id"] == case_id), None)
            if row is None or row.get("runs", 0) == 0:
                values.append(0.0)
                missing.append(True)
            else:
                values.append(float(row["pctp"]["mean"]))
                missing.append(False)
        values_per_model.append(values)
        missing_per_model.append(missing)
    sing_flags, sing_note = _singleton_column_flags_and_note(
        models, col_labels, values_per_model, missing_per_model, enabled=highlight_singleton
    )
    _shade_singleton_groups(ax, x_positions, sing_flags)
    for index, model in enumerate(models):
        values = values_per_model[index]
        missing = missing_per_model[index]
        offsets = [
            position - 0.4 + width / 2 + index * width for position in x_positions
        ]
        color = _model_color(model, index)
        bars = ax.bar(
            offsets,
            values,
            width=width,
            label=model,
            color=color,
            edgecolor="0.25",
            linewidth=0.4,
            zorder=3,
        )
        for bar, miss in zip(bars, missing):
            if miss:
                bar.set_hatch("///")
                bar.set_facecolor("#E8E8E8")
                bar.set_edgecolor("0.45")
        _annotate_bar_tops(ax, bars, values, missing)
    base_lbl = col_labels
    tick_lbl = [b + ("\n†" if f else "") for b, f in zip(base_lbl, sing_flags)]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(tick_lbl)
    ylim_ab = _bar_chart_ylim_top(values_per_model, missing_per_model)
    ax.set_ylabel("Mean PCTP")
    ax.set_title("Density ablation — mean PCTP (location slice)")
    ax.legend(framealpha=0.95)
    fig.text(
        0.5,
        0.028,
        "Note: 0.00 means no measured persistence on this slice (data present), not a missing run.",
        ha="center",
        fontsize=8,
        color="0.35",
    )
    if sing_note:
        fig.text(0.5, 0.008, sing_note, ha="center", fontsize=7.5, color="#8B4513")
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    _lock_bar_ylim(ax, ylim_ab)
    fig.savefig(_fig_out("Figure_7.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_manuscript_figure_aliases(out_dir: Path) -> None:
    """Copy pipeline PNGs to Figure_2..Figure_6 (content order; Figure_1.png is motivation, Figure_7 ablation)."""
    pairs = [
        ("crossmodel_baseline_pctp.png", "Figure_2.png"),
        ("crossmodel_compare_environment_pctp.png", "Figure_3.png"),
        ("crossmodel_quality_qrs.png", "Figure_4.png"),
        ("method_detector_f1.png", "Figure_5.png"),
        ("method_adaptive_tradeoff.png", "Figure_6.png"),
    ]
    for src_name, dst_name in pairs:
        src = out_dir / src_name
        dst = out_dir / dst_name
        if src.is_file():
            shutil.copyfile(src, dst)


def _debug_log_manuscript_figure_assets(fig_dir: Path) -> None:
    # region agent log
    log_path = ROOT / "debug-6367b5.log"
    ts = int(time.time() * 1000)
    for n in range(1, 8):
        name = f"Figure_{n}.png"
        payload = {
            "sessionId": "6367b5",
            "hypothesisId": "H1" if n in (2, 3) else "H2",
            "location": "04_build_paper_artifacts:_debug_log_manuscript_figure_assets",
            "message": "figure file presence after build",
            "data": {
                "filename": name,
                "in_figures_dir": (fig_dir / name).is_file(),
            },
            "timestamp": ts,
            "runId": "post-mirror",
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    # endregion agent log


def main() -> None:
    global _active_figures_dir
    args = _parse_args()
    cfg = load_config()
    paper_cfg = cfg.get("paper") or {}
    default_figure_models: list[str] = list(
        paper_cfg.get("figure_models") or list(_FALLBACK_PAPER_FIGURE_MODELS)
    )

    fod = (args.figures_output_dir or "").strip()
    if fod:
        _fo = Path(fod)
        _active_figures_dir = _fo.resolve() if _fo.is_absolute() else (ROOT / _fo).resolve()
    else:
        _active_figures_dir = FIGURES_DIR

    if not args.skip_figures:
        _active_figures_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.manifest) if args.manifest else _latest_manifest()
    manifest = _load_json(manifest_path)
    payload_entries = _load_payloads_from_manifest(manifest)
    models = manifest.get("models", [])
    merged_run_groups: set[str] = {str(manifest.get("run_group") or "")} - {""}

    if getattr(args, "merge_all_manifests", False):
        # Merge data from every available manifest
        for extra_path in sorted(
            RESULTS_DIR.glob("crossmodel_manifest_*.json"),
            key=lambda p: p.stat().st_mtime,
        ):
            if extra_path == manifest_path:
                continue
            extra_m = _load_json(extra_path)
            extra_entries = _load_payloads_from_manifest(extra_m)
            payload_entries = _merge_payload_entries_unique(payload_entries, extra_entries)
            # Merge model list
            for m in extra_m.get("models", []):
                if m not in models:
                    models.append(m)
    else:
        if not args.no_merge_benchmark and not _manifest_has_full_benchmark_coverage(manifest):
            merge_benchmark_path = _find_manifest_with_benchmark_coverage(manifest.get("run_group"))
            if merge_benchmark_path is not None:
                merge_manifest = _load_json(merge_benchmark_path)
                extra_entries = _load_payloads_from_manifest(merge_manifest)
                payload_entries = _merge_payload_entries_unique(payload_entries, extra_entries)
                for m in merge_manifest.get("models", []):
                    if m not in models:
                        models.append(m)
                rg = str(merge_manifest.get("run_group") or "")
                if rg:
                    merged_run_groups.add(rg)

        if not args.no_merge_benchmark:
            payload_entries, models = _merge_extra_benchmark_manifests_for_paper_models(
                payload_entries,
                models,
                merged_run_groups=merged_run_groups,
                default_figure_models=default_figure_models,
            )
            still_missing = set(default_figure_models) - _models_in_payload_entries(
                payload_entries
            )
            if still_missing:
                print(
                    "[04] WARN: some paper.figure_models have no result files on disk after merge: "
                    f"{sorted(still_missing)}",
                    flush=True,
                )
            # region agent log
            _dbg = ROOT / "debug-6367b5.log"
            try:
                _pl = {
                    "sessionId": "6367b5",
                    "hypothesisId": "H3",
                    "location": "04:merge_extra_benchmark",
                    "message": "models after quartet merge",
                    "data": {
                        "payload_models": sorted(_models_in_payload_entries(payload_entries)),
                        "desired": list(default_figure_models),
                        "still_missing": sorted(still_missing),
                    },
                    "timestamp": int(time.time() * 1000),
                    "runId": "post-quartet-merge",
                }
                with open(_dbg, "a", encoding="utf-8") as _df:
                    _df.write(json.dumps(_pl, ensure_ascii=False) + "\n")
            except OSError:
                pass
            # endregion agent log

        if not args.no_merge_adaptive and not _payload_has_adaptive_entries(payload_entries):
            merge_adaptive_path = _find_manifest_with_adaptive_artifacts(manifest.get("run_group"))
            if merge_adaptive_path is not None:
                merge_adaptive_manifest = _load_json(merge_adaptive_path)
                extra_adaptive = _load_payloads_from_manifest(merge_adaptive_manifest)
                payload_entries = _merge_payload_entries_unique(payload_entries, extra_adaptive)

    for raw_extra in args.extra_manifest or []:
        extra_path = Path(raw_extra)
        if not extra_path.is_absolute():
            extra_path = ROOT / extra_path
        if not extra_path.is_file():
            raise FileNotFoundError(f"--extra-manifest not found: {extra_path}")
        extra_m = _load_json(extra_path)
        extra_entries = _load_payloads_from_manifest(extra_m)
        payload_entries = _merge_payload_entries_unique(payload_entries, extra_entries)
        for m in extra_m.get("models", []):
            if m not in models:
                models.append(m)

    # Incremental primary manifests may list only one model tag even after benchmark merges.
    # Union in every model tag still present in payload entries so --figure-models can resolve.
    _seen_payload_models: list[str] = []
    for e in payload_entries:
        m = str(e.get("artifact", {}).get("model", ""))
        if m and m not in _seen_payload_models:
            _seen_payload_models.append(m)
    for m in _seen_payload_models:
        if m not in models:
            models.append(m)

    payload_entries = _filter_payloads_to_paper_models(
        payload_entries, default_figure_models
    )
    models = list(default_figure_models)
    if getattr(args, "exclude_models", None):
        excluded = {str(m) for m in args.exclude_models}
        payload_entries = [
            e
            for e in payload_entries
            if str(e.get("artifact", {}).get("model", "")) not in excluded
        ]
        models = [m for m in models if str(m) not in excluded]

    baseline_rows = _build_baseline_rows(payload_entries, models)
    compare_rows = _build_compare_rows(payload_entries, models)
    quality_rows = _build_quality_rows(payload_entries, models)
    detector_rows = _build_detector_rows(payload_entries, models)
    adaptive_rows = _build_adaptive_rows(payload_entries, models)
    method_quality_rows = _build_method_quality_rows(payload_entries, models)
    paired_comparison_rows = _build_paired_comparison_rows(payload_entries)
    paired_adjusted = _enrich_paired_comparisons_for_multiple_testing(paired_comparison_rows)
    ablation_rows = _build_ablation_rows(payload_entries, models)
    distribution_summary = _build_distribution_summary(manifest, baseline_rows, compare_rows)

    models_fig = _resolve_figure_models(
        models,
        args.figure_models,
        default_figure_models=default_figure_models,
        use_all_manifest_models=args.figure_all_models,
    )
    print(
        f"[04] Results: {RESULTS_DIR}\n"
        f"[04] Figure models ({len(models_fig)}): {', '.join(models_fig)}",
        flush=True,
    )
    bench_methods = _benchmark_methods_from_manifest(manifest)
    if not args.skip_figures:
        if baseline_rows:
            _build_baseline_figure(
                baseline_rows, models_fig, highlight_singleton=args.show_singleton_signal
            )
        if compare_rows:
            _build_compare_figure(
                compare_rows,
                models_fig,
                bench_methods,
                highlight_singleton=args.show_singleton_signal,
            )
        if quality_rows:
            _build_quality_figure(quality_rows, models_fig, bench_methods)
        if detector_rows:
            _build_detector_figure(detector_rows, models_fig)
        if adaptive_rows and method_quality_rows:
            _build_tradeoff_figure(adaptive_rows, method_quality_rows, models_fig)
        if ablation_rows:
            _build_ablation_figure(
                ablation_rows, models_fig, highlight_singleton=args.show_singleton_signal
            )

        # Legacy figure names referenced in older docs / handouts (same pixels as crossmodel_*)
        for src, dst in (
            ("crossmodel_baseline_pctp.png", "baseline_pctp_by_scenario.png"),
            ("crossmodel_compare_environment_pctp.png", "mitigation_pctp_comparison.png"),
        ):
            p_src = _active_figures_dir / src
            if p_src.exists():
                shutil.copyfile(p_src, _active_figures_dir / dst)
        if not getattr(args, "no_manuscript_figure_aliases", False):
            _write_manuscript_figure_aliases(_active_figures_dir)
            _debug_log_manuscript_figure_assets(_active_figures_dir)
            print(
                f"[04] Manuscript Figure_2..7 (content order) under {_active_figures_dir}",
                flush=True,
            )
    else:
        print(
            f"[04] --skip-figures: skipping PNG generation (would use {_active_figures_dir})",
            flush=True,
        )

    run_group = manifest["run_group"]
    (RESULTS_DIR / f"paper_crossmodel_baseline_{run_group}.json").write_text(
        json.dumps(baseline_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (RESULTS_DIR / f"paper_crossmodel_compare_{run_group}.json").write_text(
        json.dumps(compare_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (RESULTS_DIR / f"paper_crossmodel_quality_{run_group}.json").write_text(
        json.dumps(quality_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (RESULTS_DIR / f"paper_method_detector_{run_group}.json").write_text(
        json.dumps(detector_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (RESULTS_DIR / f"paper_method_adaptive_{run_group}.json").write_text(
        json.dumps(adaptive_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (RESULTS_DIR / f"paper_method_quality_{run_group}.json").write_text(
        json.dumps(method_quality_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (RESULTS_DIR / f"paper_method_paired_{run_group}.json").write_text(
        json.dumps(paired_comparison_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (RESULTS_DIR / f"paper_method_paired_adjusted_{run_group}.json").write_text(
        json.dumps(paired_adjusted, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (RESULTS_DIR / f"paper_crossmodel_ablation_{run_group}.json").write_text(
        json.dumps(ablation_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (RESULTS_DIR / f"paper_crossmodel_distribution_{run_group}.json").write_text(
        json.dumps(distribution_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
