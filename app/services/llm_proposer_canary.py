"""LLM proposer canary (Phase B) — low-weight bandit arm with auto-disable.

The canary mechanism for promoting the LLM proposer from shadow toward
influence: the LLM becomes one more low-weight arm offered to the existing
``ContextualStrategyBandit``, and its health is tracked so it auto-disables on
repeated poor reward or safety warnings.

Boundaries (intentional):

* This module is the *mechanism only*. It is gated by
  ``LLM_PROPOSER_CANARY_ENABLED`` (default off) and is NOT wired into live
  candidate selection here — activation is the final promote step, gated on real
  shadow evidence (validity / overlap / beating a random baseline).
* Trust reuses the existing bandit's reward/credit convention; this layer only
  decides whether to *offer* the ``llm`` arm and when to retire it.
* Deterministic and pure; once disabled, the state is sticky.
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

__all__ = [
    "LLMCanaryState",
    "canary_arms",
    "should_offer_llm_arm",
    "update_canary_state",
]

_LLM_ARM = "llm"
_INITIAL_WEIGHT = 0.1
_EMA_ALPHA = 0.3
_POOR_REWARD_THRESHOLD = 0.0
_MAX_CONSECUTIVE_POOR = 3
_MAX_SAFETY_WARNINGS = 2
_REWARD_FLOOR = -0.2
_MIN_SAMPLES_FOR_FLOOR = 5


class LLMCanaryState(BaseModel):
    """Health/weight state of the LLM canary arm."""

    enabled: bool = True
    disabled_reason: str | None = None
    weight: float = Field(default=_INITIAL_WEIGHT, ge=0.0, le=1.0)
    n_uses: int = Field(default=0, ge=0)
    reward_ema: float = 0.0
    consecutive_poor: int = Field(default=0, ge=0)
    safety_warning_count: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("created_at")
    @classmethod
    def _created_at_must_be_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError("created_at must be timezone-aware")
        return value


def update_canary_state(
    state: LLMCanaryState,
    *,
    reward: float,
    safety_warning: bool = False,
    ema_alpha: float = _EMA_ALPHA,
    poor_reward_threshold: float = _POOR_REWARD_THRESHOLD,
    max_consecutive_poor: int = _MAX_CONSECUTIVE_POOR,
    max_safety_warnings: int = _MAX_SAFETY_WARNINGS,
    reward_floor: float = _REWARD_FLOOR,
    min_samples_for_floor: int = _MIN_SAMPLES_FOR_FLOOR,
    now: datetime | None = None,
) -> LLMCanaryState:
    """Update the canary from one observed reward; auto-disable when unhealthy.

    Returns a new state. Once disabled, the state is sticky (returned unchanged).
    """
    if not state.enabled:
        return state

    timestamp = now or datetime.now(UTC)
    n_uses = state.n_uses + 1
    reward_ema = reward if state.n_uses == 0 else (
        ema_alpha * reward + (1.0 - ema_alpha) * state.reward_ema
    )
    consecutive_poor = state.consecutive_poor + 1 if reward < poor_reward_threshold else 0
    safety_warning_count = state.safety_warning_count + (1 if safety_warning else 0)

    disabled_reason = _disabled_reason(
        safety_warning_count=safety_warning_count,
        consecutive_poor=consecutive_poor,
        reward_ema=reward_ema,
        n_uses=n_uses,
        max_safety_warnings=max_safety_warnings,
        max_consecutive_poor=max_consecutive_poor,
        reward_floor=reward_floor,
        min_samples_for_floor=min_samples_for_floor,
    )

    return state.model_copy(
        update={
            "enabled": disabled_reason is None,
            "disabled_reason": disabled_reason,
            "n_uses": n_uses,
            "reward_ema": round(reward_ema, 10),
            "consecutive_poor": consecutive_poor,
            "safety_warning_count": safety_warning_count,
            "updated_at": timestamp,
        }
    )


def should_offer_llm_arm(state: LLMCanaryState, *, canary_enabled: bool) -> bool:
    """Offer the LLM arm only when the canary flag is on and it is healthy."""
    return canary_enabled and state.enabled


def canary_arms(
    base_arms: tuple[str, ...],
    state: LLMCanaryState,
    *,
    canary_enabled: bool,
    arm_name: str = _LLM_ARM,
) -> tuple[tuple[str, ...], dict[str, float]]:
    """Return (arms, priors) with the low-weight LLM arm added when offered.

    Ready to pass to ``ContextualStrategyBandit.select``; returns the base arms
    unchanged when the canary is off or the arm is disabled.
    """
    if not should_offer_llm_arm(state, canary_enabled=canary_enabled):
        return base_arms, {}
    if arm_name in base_arms:
        return base_arms, {arm_name: state.weight}
    return (*base_arms, arm_name), {arm_name: state.weight}


def _disabled_reason(
    *,
    safety_warning_count: int,
    consecutive_poor: int,
    reward_ema: float,
    n_uses: int,
    max_safety_warnings: int,
    max_consecutive_poor: int,
    reward_floor: float,
    min_samples_for_floor: int,
) -> str | None:
    if safety_warning_count >= max_safety_warnings:
        return "safety_warnings"
    if consecutive_poor >= max_consecutive_poor:
        return "consecutive_poor_rewards"
    if n_uses >= min_samples_for_floor and reward_ema < reward_floor:
        return "reward_ema_below_floor"
    return None
