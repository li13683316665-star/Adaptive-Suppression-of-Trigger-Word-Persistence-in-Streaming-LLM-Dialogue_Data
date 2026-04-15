"""Adaptive suppression controller that dynamically selects mitigation
strategies based on the streaming trigger detector's risk output.

The controller implements a tiered escalation policy:

    risk < low_threshold      -> baseline (no mitigation)
    low  <= risk < mid        -> targeted_guardrail
    mid  <= risk < high       -> repetition_penalty
    risk >= high              -> context_reset

It also tracks mitigation history so it can escalate when a lighter
intervention fails and de-escalate once risk drops for several
consecutive turns.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from src.bias.streaming_detector import TokenRisk
from src.reduction.algorithms import apply_reduction

# ---------------------------------------------------------------------------
# Public data classes
# ---------------------------------------------------------------------------

TIERS = ("baseline", "targeted_guardrail", "repetition_penalty", "context_reset")
TIER_ORDER = {name: idx for idx, name in enumerate(TIERS)}


@dataclass
class SuppressionDecision:
    """The controller's output for a single turn."""

    method: str
    trigger_family: list[str]
    risk_level: str          # "low" | "medium" | "high" | "critical"
    max_risk: float
    escalated: bool = False
    de_escalated: bool = False


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


class AdaptiveSuppressionController:
    """Picks the cheapest mitigation that keeps trigger risk below threshold.

    Usage::

        ctrl = AdaptiveSuppressionController()
        decision = ctrl.decide(detector.get_risk_scores())
        result = ctrl.apply(messages, generation_cfg, detector.get_risk_scores())
        # ... send result["messages"] to the model, get response ...
        ctrl.record_outcome(detector.get_risk_scores())
    """

    DEFAULT_THRESHOLDS: dict[str, float] = {
        "guardrail": 0.20,
        "repetition_penalty": 0.40,
        "context_reset": 0.60,
    }

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self._thresholds = {
            k: cfg.get(f"threshold_{k}", v)
            for k, v in self.DEFAULT_THRESHOLDS.items()
        }
        self._escalate_after = int(cfg.get("escalate_after", 2))
        self._deescalate_after = int(cfg.get("deescalate_after", 3))
        self._risk_threshold = float(cfg.get("risk_threshold", 0.20))

        # History tracking
        self._consecutive_above: int = 0
        self._consecutive_below: int = 0
        self._current_tier: str = "baseline"
        self._history: list[dict[str, Any]] = []

    # ---- public API -------------------------------------------------------

    def decide(self, risk_scores: list[TokenRisk]) -> SuppressionDecision:
        """Given detector risk scores, choose the mitigation strategy."""
        max_risk = max((r.risk for r in risk_scores), default=0.0)
        trigger_tokens = [
            r.token for r in risk_scores if r.risk >= self._risk_threshold
        ]

        raw_tier = self._tier_for_risk(max_risk)
        escalated = False
        de_escalated = False

        # Escalation: if risk stays above threshold after current tier,
        # bump up one level
        if max_risk >= self._risk_threshold:
            self._consecutive_above += 1
            self._consecutive_below = 0
            if (
                self._consecutive_above >= self._escalate_after
                and TIER_ORDER.get(raw_tier, 0) <= TIER_ORDER.get(self._current_tier, 0)
            ):
                next_idx = min(
                    TIER_ORDER[self._current_tier] + 1, len(TIERS) - 1
                )
                raw_tier = TIERS[next_idx]
                escalated = True
        else:
            self._consecutive_below += 1
            self._consecutive_above = 0
            # De-escalation: if risk stays below threshold for N turns,
            # step down one level
            if (
                self._consecutive_below >= self._deescalate_after
                and TIER_ORDER.get(self._current_tier, 0) > 0
            ):
                prev_idx = max(TIER_ORDER[self._current_tier] - 1, 0)
                raw_tier = TIERS[prev_idx]
                de_escalated = True
                self._consecutive_below = 0

        self._current_tier = raw_tier

        risk_level = self._risk_label(max_risk)
        decision = SuppressionDecision(
            method=raw_tier,
            trigger_family=trigger_tokens,
            risk_level=risk_level,
            max_risk=round(max_risk, 4),
            escalated=escalated,
            de_escalated=de_escalated,
        )
        self._history.append({
            "turn": len(self._history),
            "decision": decision.method,
            "max_risk": decision.max_risk,
            "escalated": escalated,
            "de_escalated": de_escalated,
        })
        return decision

    def apply(
        self,
        messages: list[dict[str, str]],
        generation_cfg: dict[str, Any] | None,
        risk_scores: list[TokenRisk],
    ) -> dict[str, Any]:
        """Decide and apply mitigation in one call.

        Returns the same dict shape as ``apply_reduction()`` with an
        additional ``"strategy"`` key containing the ``SuppressionDecision``.
        """
        decision = self.decide(risk_scores)
        result = apply_reduction(
            method=decision.method,
            messages=deepcopy(messages),
            generation_cfg=generation_cfg,
            trigger_family=decision.trigger_family or None,
        )
        result["strategy"] = decision
        return result

    def record_outcome(self, post_turn_risk_scores: list[TokenRisk]) -> None:
        """Inform the controller of the risk state AFTER the model replied.

        This allows the history tracker to judge whether the last intervention
        was effective and adjust escalation counters accordingly.
        """
        max_risk = max((r.risk for r in post_turn_risk_scores), default=0.0)
        if self._history:
            self._history[-1]["post_risk"] = round(max_risk, 4)

    @property
    def history(self) -> list[dict[str, Any]]:
        return list(self._history)

    def reset(self) -> None:
        """Clear all state."""
        self._consecutive_above = 0
        self._consecutive_below = 0
        self._current_tier = "baseline"
        self._history.clear()

    # ---- internals --------------------------------------------------------

    def _tier_for_risk(self, risk: float) -> str:
        if risk >= self._thresholds["context_reset"]:
            return "context_reset"
        if risk >= self._thresholds["repetition_penalty"]:
            return "repetition_penalty"
        if risk >= self._thresholds["guardrail"]:
            return "targeted_guardrail"
        return "baseline"

    @staticmethod
    def _risk_label(risk: float) -> str:
        if risk >= 0.6:
            return "critical"
        if risk >= 0.4:
            return "high"
        if risk >= 0.2:
            return "medium"
        return "low"
