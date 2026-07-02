from __future__ import annotations

from app.services.action_policy import ActionPolicyMatrix, ActionTransitionGuard
from app.services.strategy_models import (
    CampaignContext,
    CampaignIntent,
    CampaignSnapshot,
    FailureEvent,
    OptimizationMode,
)


def _snapshot(**kwargs):
    base = dict(
        round_number=1,
        max_rounds=10,
        n_observations=6,
        n_dimensions=2,
        has_categorical=False,
        has_log_scale=False,
        kpi_history=(0.1, 0.2, 0.3),
        available_backends={"lhs": True, "bomcp": True, "nexus_gp_bo": True},
    )
    base.update(kwargs)
    return CampaignSnapshot(**base)


def test_action_policy_matrix_maps_objective_failure_and_backend_priors():
    snap = _snapshot(
        campaign_context=CampaignContext(current_objective_level="performance"),
        failure_events=(
            FailureEvent(
                failure_type="constraint",
                reason="composition outside validated simplex",
                backend_name="nexus_gp_bo",
                penalize_backend=True,
            ),
        ),
    )

    decision = ActionPolicyMatrix().evaluate(
        snap,
        selected_intent=CampaignIntent.OPTIMIZE,
        selected_mode=OptimizationMode.EXPLOIT,
        selected_backend="nexus_gp_bo",
    )

    assert decision.intent_priors["optimize"] > 0
    assert decision.intent_priors["diagnose"] > 0
    assert decision.intent_priors["revise_space"] > 0
    assert decision.mode_priors["constrained_search"] > 0
    assert decision.mode_priors["revise_space"] > 0
    assert decision.backend_priors["bomcp"] > 0
    assert decision.backend_priors["nexus_gp_bo"] > 0
    assert any(e.source == "action_policy:objective" for e in decision.evidence)
    assert any(e.source == "action_policy:failure" for e in decision.evidence)
    assert any(e.source == "intent_backend_prior" for e in decision.evidence)


def test_action_policy_matrix_is_hypothesis_aware():
    snap = _snapshot(
        campaign_context=CampaignContext(
            current_objective_level="mechanism",
            domain_hypotheses=("phase segregation controls activity", "strain controls activity"),
        ),
        failure_events=(
            FailureEvent(
                failure_type="scientific_negative",
                reason="clean negative under active mechanism",
            ),
        ),
    )

    decision = ActionPolicyMatrix().evaluate(
        snap,
        selected_intent=CampaignIntent.VALIDATE,
        selected_mode=OptimizationMode.MECHANISM_VALIDATION,
        selected_backend="built_in",
    )

    assert decision.intent_priors["validate"] > 0
    assert decision.intent_priors["hypothesis_generate"] > 0
    assert decision.intent_priors["hypothesis_test"] > 0
    assert decision.mode_priors["mechanism_validation"] > 0
    assert decision.mode_priors["hypothesis_generate"] > 0
    assert decision.mode_priors["hypothesis_test"] > 0
    assert any(e.target == "hypothesis_update" for e in decision.evidence)
    assert any(e.target == "discriminative_validation" for e in decision.evidence)


def test_action_transition_guard_flags_unstable_without_blocking():
    guard = ActionTransitionGuard().evaluate(
        previous_intent="optimize",
        selected_intent="discover",
        evidence=(),
    )

    assert guard.allowed is False
    assert guard.unstable is True
    assert guard.evidence[0].effect == "penalize"
