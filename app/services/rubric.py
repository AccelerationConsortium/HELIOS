"""Phase-aware, versioned evaluation rubric (Phase B / B1).

A rubric is a set of per-signal weight multipliers applied over the reward
*verifications* produced by ``verifiable_reward``. Because it operates on the
already-recorded verifications (not raw campaign state), a decision can be
re-scored under any rubric after the fact — the mechanism behind the evolving
evaluation loop and offline policy evaluation (B3/B4).

    verifications ──► rescore(·, rubric) ──► WeightedReward(process, outcome, total)

``v0.1_static`` is the identity rubric (all weights 1.0) and reproduces the
Phase A reward exactly, so introducing rubrics changes nothing until a caller
opts into a phase-aware version.

Design intent (growloop.md §1): what counts as a *good* decision depends on the
campaign's phase. Early/diagnostic phases value information and safety; the
optimization phase values objective gains; validation phases value validation.
The weights below encode that, keyed off ``CampaignMode``.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from app.services.campaign_mode import CampaignMode
from app.services.verifiable_reward import (
    OUTCOME_SIGNALS,
    PROCESS_SIGNALS,
    RewardVerification,
    round_component,
)

__all__ = [
    "Rubric",
    "WeightedReward",
    "STATIC_RUBRIC",
    "rescore",
    "rubric_for_mode",
    "get_rubric",
]


class Rubric(BaseModel):
    """Versioned per-signal weight multipliers over reward verifications."""

    version: str
    weights: dict[str, float] = Field(default_factory=dict)

    def weight(self, signal_name: str) -> float:
        """Multiplier for a signal; unspecified signals default to 1.0."""
        return self.weights.get(signal_name, 1.0)


class WeightedReward(BaseModel):
    """Result of applying a rubric to a set of verifications."""

    rubric_version: str
    process_reward: float
    outcome_reward: float
    total: float


def rescore(
    verifications: list[RewardVerification], rubric: Rubric
) -> WeightedReward:
    """Re-score verifications under *rubric*, returning the weighted split."""
    process = sum(
        v.score * rubric.weight(v.name)
        for v in verifications
        if v.name in PROCESS_SIGNALS
    )
    outcome = sum(
        v.score * rubric.weight(v.name)
        for v in verifications
        if v.name in OUTCOME_SIGNALS
    )
    process = round_component(process)
    outcome = round_component(outcome)
    return WeightedReward(
        rubric_version=rubric.version,
        process_reward=process,
        outcome_reward=outcome,
        total=round_component(process + outcome),
    )


# ---------------------------------------------------------------------------
# Rubric registry
# ---------------------------------------------------------------------------

# Identity rubric — reproduces Phase A exactly.
STATIC_RUBRIC = Rubric(version="v0.1_static", weights={})

# Campaign-aware rubric family (v0.2). Each mode emphasises the signals that
# matter most for that scientific activity. Only signals that deviate from 1.0
# are listed; everything else stays at identity weight.
_MODE_WEIGHTS: dict[CampaignMode, dict[str, float]] = {
    CampaignMode.BO_OPTIMIZATION: {"objective": 2.0, "proxy_gap": 1.5},
    CampaignMode.VALIDATION: {"validation": 2.0},
    CampaignMode.CALIBRATION: {"context": 1.5, "proxy_gap": 1.5},
    CampaignMode.FAILURE_DIAGNOSIS: {"failure": 2.0, "proxy_gap": 1.5},
    CampaignMode.LITERATURE_CONTEXT_SEEKING: {"context": 2.0},
    CampaignMode.HUMAN_OBSERVATION_REQUEST: {"safety": 1.5},
    CampaignMode.SAFETY_CONSTRAINT_TIGHTENING: {"safety": 2.0, "failure": 1.5},
    CampaignMode.STOP_RECOMMENDED: {"objective": 1.5, "validation": 1.5},
}


def rubric_for_mode(mode: CampaignMode) -> Rubric:
    """Return the campaign-aware (v0.2) rubric for a campaign mode."""
    return Rubric(
        version=f"v0.2_campaign_aware:{mode.value}",
        weights=dict(_MODE_WEIGHTS.get(mode, {})),
    )


def get_rubric(version: str) -> Rubric:
    """Look up a rubric by version string. Raises KeyError if unknown."""
    if version == STATIC_RUBRIC.version:
        return STATIC_RUBRIC
    prefix = "v0.2_campaign_aware:"
    if version.startswith(prefix):
        mode_value = version[len(prefix):]
        return rubric_for_mode(CampaignMode(mode_value))
    raise KeyError(version)
