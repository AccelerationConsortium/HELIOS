from __future__ import annotations

from app.services.decision_layer import CampaignDecisionLayer
from app.services.decision_outcome import (
    CampaignDecisionAccountingBuilder,
    CampaignDecisionOutcomeBuilder,
    CampaignDecisionRewardCalculator,
    build_campaign_decision_accounting,
    build_campaign_decision_outcome,
    calculate_campaign_decision_reward,
)
from app.services.decision_trace import CampaignDecisionTraceBuilder
from app.services.round_context import CampaignRoundContextBuilder


def _trace():
    context = CampaignRoundContextBuilder().build(
        campaign_id="campaign-1",
        round_index=2,
        strategy_selection_result={
            "campaign_intent": "optimize",
            "optimization_mode": "exploit",
            "candidate_generation_backend": "bo_mcp",
            "confidence": 0.75,
        },
    )
    plan = CampaignDecisionLayer().decide(context)
    return CampaignDecisionTraceBuilder().build(
        trace_id="trace-1",
        context=context,
        decision_plan=plan,
        actual_action="propose_candidates",
    )


def test_outcome_builder_copies_trace_identity():
    trace = _trace()

    outcome = CampaignDecisionOutcomeBuilder().build(trace=trace)

    assert outcome.trace_id == trace.trace_id
    assert outcome.campaign_id == trace.campaign_id
    assert outcome.round_index == trace.round_index


def test_positive_outcome_gives_positive_reward():
    outcome = build_campaign_decision_outcome(
        trace=_trace(),
        execution_success=True,
        objective_delta=0.5,
        proxy_gap_delta=-0.2,
        validation_success=True,
        context_request_fulfilled=True,
    )

    reward = calculate_campaign_decision_reward(outcome)

    assert reward.reward > 0.0
    assert reward.objective_reward == 0.15
    assert reward.proxy_gap_reward == 0.06
    assert reward.validation_reward == 0.2
    assert reward.context_reward == 0.1


def test_safety_incident_dominates_penalty():
    outcome = build_campaign_decision_outcome(
        trace=_trace(),
        execution_success=True,
        safety_incident_count=2,
    )

    reward = CampaignDecisionRewardCalculator().calculate(outcome)

    assert reward.safety_penalty == -1.0
    assert reward.reward <= 0.0


def test_failure_count_penalty():
    outcome = build_campaign_decision_outcome(trace=_trace(), failure_count=3)

    reward = calculate_campaign_decision_reward(outcome)

    assert reward.failure_penalty == -0.3


def test_proxy_gap_delta_semantics():
    improved = calculate_campaign_decision_reward(
        build_campaign_decision_outcome(trace=_trace(), proxy_gap_delta=-0.5)
    )
    worsened = calculate_campaign_decision_reward(
        build_campaign_decision_outcome(trace=_trace(), proxy_gap_delta=0.5)
    )

    assert improved.proxy_gap_reward == 0.15
    assert worsened.proxy_gap_reward == -0.15


def test_reward_is_clamped():
    positive = calculate_campaign_decision_reward(
        build_campaign_decision_outcome(
            trace=_trace(),
            execution_success=True,
            objective_delta=10.0,
            proxy_gap_delta=-10.0,
            validation_success=True,
            context_request_fulfilled=True,
        )
    )
    negative = calculate_campaign_decision_reward(
        build_campaign_decision_outcome(
            trace=_trace(),
            execution_success=False,
            failure_count=20,
            safety_incident_count=4,
            objective_delta=-10.0,
            proxy_gap_delta=10.0,
            validation_success=False,
        )
    )

    assert positive.reward <= 1.0
    assert positive.reward == 1.0
    assert positive.metadata["clamped"] is True
    assert negative.reward >= -1.0
    assert negative.reward == -1.0
    assert negative.metadata["clamped"] is True


def test_accounting_builder_binds_trace_outcome_and_reward():
    trace = _trace()
    outcome = build_campaign_decision_outcome(
        trace=trace,
        execution_success=True,
    )

    accounting = CampaignDecisionAccountingBuilder().build(
        trace=trace,
        outcome=outcome,
    )

    assert accounting.trace == trace
    assert accounting.outcome == outcome
    assert accounting.reward.trace_id == trace.trace_id


def test_json_serialization():
    trace = _trace()
    outcome = build_campaign_decision_outcome(
        trace=trace,
        execution_success=True,
    )
    reward = calculate_campaign_decision_reward(outcome)
    accounting = build_campaign_decision_accounting(trace=trace, outcome=outcome)

    dumped_outcome = outcome.model_dump(mode="json")
    dumped_reward = reward.model_dump(mode="json")
    dumped_accounting = accounting.model_dump(mode="json")
    outcome_json = outcome.model_dump_json()
    reward_json = reward.model_dump_json()
    accounting_json = accounting.model_dump_json()

    assert dumped_outcome["trace_id"] == "trace-1"
    assert dumped_outcome["created_at"].endswith("Z") or "+" in dumped_outcome["created_at"]
    assert dumped_reward["trace_id"] == "trace-1"
    assert dumped_accounting["trace"]["trace_id"] == "trace-1"
    assert '"trace_id":"trace-1"' in outcome_json
    assert '"trace_id":"trace-1"' in reward_json
    assert '"trace_id":"trace-1"' in accounting_json


def test_inputs_are_not_mutated():
    trace = _trace()
    metadata = {"source": {"name": "initial"}}

    outcome = build_campaign_decision_outcome(trace=trace, metadata=metadata)
    accounting = build_campaign_decision_accounting(
        trace=trace,
        outcome=outcome,
        metadata=metadata,
    )
    outcome.metadata["source"]["name"] = "changed"
    accounting.metadata["source"]["name"] = "changed"
    accounting.trace.context.strategy_selection_result["candidate_generation_backend"] = "changed"

    assert metadata == {"source": {"name": "initial"}}
    assert trace.context.strategy_selection_result["candidate_generation_backend"] == "bo_mcp"


def test_import_smoke():
    import app.services.decision_layer  # noqa: F401
    import app.services.decision_outcome  # noqa: F401
    import app.services.decision_trace  # noqa: F401
    import app.services.strategy_selector  # noqa: F401
