"""Online streaming trigger-word detector for multi-turn conversation traces.

Unlike the batch detector in ``detector.py``, this module processes turns
one at a time and maintains rolling per-token statistics so that risk
scores are available at every point in the conversation.  The detector is
model-agnostic and black-box friendly: it needs only the surface text of
each turn, not internal attention weights or logits.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

STOPWORDS = frozenset(
    {
        "the", "a", "an", "and", "or", "to", "of", "in", "on", "for",
        "with", "is", "it", "you", "i", "we", "that", "this", "be",
        "are", "was", "but", "just", "what", "your", "not", "do", "if",
        "so", "can", "has", "have", "had", "my", "me", "no", "yes",
        "will", "would", "could", "should", "all", "about", "been",
        "from", "they", "them", "their", "there", "here", "also",
        "its", "get", "got", "than", "then", "very", "more", "how",
        "did", "does", "don", "let", "may", "might", "any", "some",
        "out", "too", "now", "much", "well", "our", "way", "own",
        "say", "said", "one", "two", "new", "like",
    }
)


def _tokenize(text: str) -> list[str]:
    """Extract lowercase alphabetic tokens, dropping stopwords and short words."""
    return [
        tok
        for tok in re.findall(r"[a-zA-Z][a-zA-Z\-']*", text.lower())
        if len(tok) > 2 and tok not in STOPWORDS
    ]


# ---------------------------------------------------------------------------
# Per-token statistics accumulator
# ---------------------------------------------------------------------------


@dataclass
class _TokenStats:
    """Running statistics for a single token across the conversation."""

    token: str

    # Raw counts by role
    context_appearances: int = 0      # user / system turns
    response_appearances: int = 0     # assistant turns

    # Channel-level tracking
    env_appearances: int = 0          # appeared in environment channel
    response_after_env: int = 0       # appeared in reply when env contained it

    # Correction-related
    post_correction_appearances: int = 0
    corrections_seen: int = 0

    # Persistence tracking
    current_streak: int = 0
    max_streak: int = 0
    last_seen_turn: int = -1

    # Recovery tracking
    _correction_pending: bool = field(default=False, repr=False)
    _turns_since_last_correction: int = field(default=0, repr=False)
    _recovery_lags: list[int] = field(default_factory=list, repr=False)

    # Turn-level record for evidence
    turn_indices: list[int] = field(default_factory=list)

    def record_appearance(
        self,
        turn_index: int,
        role: str,
        channel: str,
        is_post_correction: bool,
        env_tokens: frozenset[str],
    ) -> None:
        self.turn_indices.append(turn_index)

        if role == "assistant":
            self.response_appearances += 1
            if self.token in env_tokens:
                self.response_after_env += 1
        else:
            self.context_appearances += 1

        if channel == "environment":
            self.env_appearances += 1

        if is_post_correction:
            self.post_correction_appearances += 1

        # Streak
        if turn_index == self.last_seen_turn + 1:
            self.current_streak += 1
        else:
            self.current_streak = 1
        self.max_streak = max(self.max_streak, self.current_streak)
        self.last_seen_turn = turn_index

        # Recovery: if we were waiting for this token to disappear, record lag
        if self._correction_pending:
            self._turns_since_last_correction += 1

    def record_absence(self, turn_index: int) -> None:
        """Called when this token does NOT appear in the current turn."""
        _ = turn_index
        if self._correction_pending:
            self._recovery_lags.append(self._turns_since_last_correction)
            self._correction_pending = False
            self._turns_since_last_correction = 0

    def mark_correction(self) -> None:
        """Signal that a correction turn just happened."""
        self.corrections_seen += 1
        self._correction_pending = True
        self._turns_since_last_correction = 0

    @property
    def mean_recovery_lag(self) -> float:
        if not self._recovery_lags:
            return 0.0
        return sum(self._recovery_lags) / len(self._recovery_lags)


# ---------------------------------------------------------------------------
# Public data class returned by get_risk_scores()
# ---------------------------------------------------------------------------


@dataclass
class TokenRisk:
    """Risk assessment for a single token."""

    token: str
    risk: float
    signals: dict[str, float]
    stats: _TokenStats


# ---------------------------------------------------------------------------
# Main detector
# ---------------------------------------------------------------------------


class StreamingTriggerDetector:
    """Online detector that observes conversation turns and identifies tokens
    exhibiting trigger-like behavior patterns.

    Usage::

        detector = StreamingTriggerDetector()
        for turn in conversation:
            detector.observe_turn(turn.text, turn.role, turn.channel,
                                  is_correction=turn.is_correction)
            risks = detector.get_risk_scores(top_k=5)
            family = detector.get_detected_family(threshold=0.3)
    """

    DEFAULT_WEIGHTS: dict[str, float] = {
        "w_freq": 0.20,
        "w_surv": 0.30,
        "w_leak": 0.15,
        "w_pers": 0.20,
        "w_lag": 0.15,
    }

    SIGNAL_TO_WEIGHT_KEY: dict[str, str] = {
        "f_freq": "w_freq",
        "f_surv": "w_surv",
        "f_leak": "w_leak",
        "f_pers": "w_pers",
        "f_lag": "w_lag",
    }

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self._weights = {
            k: cfg.get(k, v) for k, v in self.DEFAULT_WEIGHTS.items()
        }
        self._min_appearances = int(cfg.get("min_appearances", 2))
        self._token_stats: dict[str, _TokenStats] = {}
        self._turn_count: int = 0
        self._correction_turn_count: int = 0
        self._env_tokens: frozenset[str] = frozenset()
        self._all_env_tokens: set[str] = set()

    # ---- public API -------------------------------------------------------

    def observe_turn(
        self,
        text: str,
        role: str,
        channel: str,
        *,
        is_correction: bool = False,
    ) -> None:
        """Process one conversation turn and update internal state."""
        tokens = _tokenize(text)
        unique_tokens = set(tokens)
        turn_idx = self._turn_count
        self._turn_count += 1

        is_post_correction = self._correction_turn_count > 0 and not is_correction

        # Track environment vocabulary
        if channel == "environment":
            self._all_env_tokens.update(unique_tokens)
            self._env_tokens = frozenset(self._all_env_tokens)

        # If this is a correction turn, mark all existing stats
        if is_correction:
            self._correction_turn_count += 1
            for stats in self._token_stats.values():
                stats.mark_correction()

        # Update stats for every token seen in this turn
        for tok in unique_tokens:
            if tok not in self._token_stats:
                self._token_stats[tok] = _TokenStats(token=tok)
            self._token_stats[tok].record_appearance(
                turn_index=turn_idx,
                role=role,
                channel=channel,
                is_post_correction=is_post_correction,
                env_tokens=self._env_tokens,
            )

        # Record absence for tokens NOT in this turn (only for assistant turns,
        # which is where persistence matters)
        if role == "assistant":
            for tok, stats in self._token_stats.items():
                if tok not in unique_tokens:
                    stats.record_absence(turn_idx)

    def get_risk_scores(self, top_k: int = 10) -> list[TokenRisk]:
        """Return the current top-k tokens ranked by trigger risk."""
        scored: list[TokenRisk] = []
        for stats in self._token_stats.values():
            total = stats.context_appearances + stats.response_appearances
            if total < self._min_appearances:
                continue
            signals = self._compute_signals(stats)
            risk = self._aggregate_risk(signals)
            scored.append(TokenRisk(
                token=stats.token,
                risk=round(risk, 4),
                signals={k: round(v, 4) for k, v in signals.items()},
                stats=stats,
            ))
        scored.sort(key=lambda r: r.risk, reverse=True)
        return scored[:top_k]

    def get_detected_family(self, threshold: float = 0.3) -> list[str]:
        """Return tokens whose risk exceeds *threshold*, sorted by risk."""
        return [
            r.token
            for r in self.get_risk_scores(top_k=len(self._token_stats))
            if r.risk >= threshold
        ]

    def reset(self) -> None:
        """Clear all accumulated state."""
        self._token_stats.clear()
        self._turn_count = 0
        self._correction_turn_count = 0
        self._env_tokens = frozenset()
        self._all_env_tokens.clear()

    # ---- internals --------------------------------------------------------

    def _compute_signals(self, stats: _TokenStats) -> dict[str, float]:
        """Compute the five raw signal values for one token."""

        # 1. Frequency anomaly: response_freq / context_freq
        ctx = max(stats.context_appearances, 1)
        f_freq = min(stats.response_appearances / ctx, 5.0) / 5.0

        # 2. Post-correction survival
        corr = max(stats.corrections_seen, 1)
        f_surv = min(stats.post_correction_appearances / corr, 5.0) / 5.0

        # 3. Cross-channel leakage
        total_resp = max(stats.response_appearances, 1)
        f_leak = min(stats.response_after_env / total_resp, 1.0)

        # 4. Consecutive persistence
        window = max(self._turn_count, 1)
        f_pers = min(stats.max_streak / window, 1.0)

        # 5. Recovery lag (normalised by turn count)
        window_norm = max(self._turn_count, 1)
        f_lag = min(stats.mean_recovery_lag / window_norm, 1.0)

        return {
            "f_freq": f_freq,
            "f_surv": f_surv,
            "f_leak": f_leak,
            "f_pers": f_pers,
            "f_lag": f_lag,
        }

    def _aggregate_risk(self, signals: dict[str, float]) -> float:
        """Weighted sum of signals, clamped to [0, 1]."""
        risk = (
            self._weights["w_freq"] * signals["f_freq"]
            + self._weights["w_surv"] * signals["f_surv"]
            + self._weights["w_leak"] * signals["f_leak"]
            + self._weights["w_pers"] * signals["f_pers"]
            + self._weights["w_lag"] * signals["f_lag"]
        )
        return max(0.0, min(risk, 1.0))


def std_weights_leave_one_out(drop_signal: str | None) -> dict[str, float]:
    """Return STD weight dict with one signal zeroed and remaining weights renormalized.

    ``drop_signal`` uses the ``f_*`` names matching ``_compute_signals`` keys.
    Pass ``None`` or ``""`` for default (no drop).
    """
    w = dict(StreamingTriggerDetector.DEFAULT_WEIGHTS)
    if drop_signal and drop_signal in StreamingTriggerDetector.SIGNAL_TO_WEIGHT_KEY:
        w[StreamingTriggerDetector.SIGNAL_TO_WEIGHT_KEY[drop_signal]] = 0.0
    total = sum(w.values())
    if total <= 0:
        return dict(StreamingTriggerDetector.DEFAULT_WEIGHTS)
    return {k: v / total for k, v in w.items()}
