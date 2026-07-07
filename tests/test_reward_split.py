"""T3: rubric_version + process/outcome split + per-signal verifications on the
two reward DTOs, and regression that the total reward is unchanged."""
from __future__ import annotations

from app.services.decision_outcome import (
    CampaignDecisionOutcome,
    CampaignDecisionRewardCalculator,
)
from app.services.loop_engineering import LoopOutcome, LoopRewardCalculator
from app.services.verifiable_reward import RUBRIC_VERSION_DEFAULT


def _campaign_outcome(**kw):
    base = dict(trace_id="t-1", campaign_id="c-1", round_index=0)
    base.update(kw)
    return CampaignDecisionOutcome(**base)


# --- campaign decision reward -------------------------------------------


def test_campaign_new_fields_present_with_defaults():
    reward = CampaignDecisionRewardCalculator().calculate(_campaign_outcome())
    assert reward.rubric_version == RUBRIC_VERSION_DEFAULT
    assert reward.verifications  # non-empty
    assert {v.name for v in reward.verifications} == {
        "execution", "failure", "safety", "objective", "proxy_gap",
        "validation", "context",
    }


def test_campaign_process_plus_outcome_equals_raw():
    reward = CampaignDecisionRewardCalculator().calculate(
        _campaign_outcome(
            execution_success=True,
            objective_delta=0.5,
            proxy_gap_delta=-0.4,
            validation_success=True,
            context_request_fulfilled=True,
        )
    )
    raw = reward.metadata["raw_reward"]
    assert round(reward.process_reward + reward.outcome_reward, 10) == round(raw, 10)
    # process = proxy_gap + context; outcome = execution + objective + validation
    assert reward.process_reward == round(
        reward.proxy_gap_reward + reward.context_reward, 10
    )


def test_campaign_reward_value_regression():
    # Known outcome: exec True (0.2) + objective 0.5 (0.15) + validation True (0.2)
    # = 0.55, no clamp. This must not change under the refactor.
    reward = CampaignDecisionRewardCalculator().calculate(
        _campaign_outcome(
            execution_success=True, objective_delta=0.5, validation_success=True
        )
    )
    assert reward.reward == 0.55


# --- loop reward ---------------------------------------------------------


def test_loop_new_fields_present():
    reward = LoopRewardCalculator().calculate(
        iteration_id="it-1", outcome=LoopOutcome(execution_success=True)
    )
    assert reward.rubric_version == RUBRIC_VERSION_DEFAULT
    assert {v.name for v in reward.verifications} == {
        "execution", "objective", "validation", "recovery", "failure", "safety",
    }


def test_loop_objective_not_positive_clamped():
    # Loop layer scales the raw delta: objective_delta=2.0 -> 2.0*0.3 = 0.6
    # (decision layer would clamp positive to 1.0 -> 0.3). This divergence must
    # survive the shared-core refactor.
    reward = LoopRewardCalculator().calculate(
        iteration_id="it-1", outcome=LoopOutcome(objective_delta=2.0)
    )
    assert reward.objective_reward == 0.6


def test_loop_process_plus_outcome_equals_raw():
    reward = LoopRewardCalculator().calculate(
        iteration_id="it-1",
        outcome=LoopOutcome(execution_success=True, objective_delta=0.3),
    )
    raw = reward.metadata["raw_reward"]
    assert round(reward.process_reward + reward.outcome_reward, 10) == round(raw, 10)
