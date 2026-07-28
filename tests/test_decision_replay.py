from __future__ import annotations

from app.services.decision_models import (
    CampaignDecisionAction,
    CampaignDecisionPlan,
)
from app.services.decision_outcome import (
    CampaignDecisionAccountingBuilder,
    CampaignDecisionOutcomeBuilder,
)
from app.services.decision_replay import (
    CampaignDecisionReplayAnalyzer,
    summarize_campaign_decision_replay,
)
from app.services.decision_trace import CampaignDecisionTraceBuilder
from app.services.round_context import CampaignRoundContextBuilder


def _trace(
    *,
    trace_id: str,
    campaign_id: str = "campaign-1",
    round_index: int = 1,
    action: CampaignDecisionAction = CampaignDecisionAction.PROPOSE_CANDIDATES,
    actual_action: str | None = None,
):
    context = CampaignRoundContextBuilder().build(
        campaign_id=campaign_id,
        round_index=round_index,
        strategy_selection_result={"candidate_generation_backend": "bo_mcp"},
    )
    plan = CampaignDecisionPlan(action_type=action, rationale="test")
    return CampaignDecisionTraceBuilder().build(
        trace_id=trace_id,
        context=context,
        decision_plan=plan,
        actual_action=actual_action,
    )


def _accounting(
    *,
    trace_id: str,
    campaign_id: str = "campaign-1",
    round_index: int = 1,
    action: CampaignDecisionAction = CampaignDecisionAction.PROPOSE_CANDIDATES,
    actual_action: str | None = None,
    execution_success: bool | None = None,
    failure_count: int = 0,
    safety_incident_count: int = 0,
    objective_delta: float | None = None,
    proxy_gap_delta: float | None = None,
    validation_success: bool | None = None,
    recovery_attempted: bool = False,
    recovery_success: bool | None = None,
    context_request_fulfilled: bool | None = None,
    human_override: bool | None = None,
):
    trace = _trace(
        trace_id=trace_id,
        campaign_id=campaign_id,
        round_index=round_index,
        action=action,
        actual_action=actual_action,
    )
    outcome = CampaignDecisionOutcomeBuilder().build(
        trace=trace,
        execution_success=execution_success,
        failure_count=failure_count,
        safety_incident_count=safety_incident_count,
        objective_delta=objective_delta,
        proxy_gap_delta=proxy_gap_delta,
        validation_success=validation_success,
        recovery_attempted=recovery_attempted,
        recovery_success=recovery_success,
        context_request_fulfilled=context_request_fulfilled,
        human_override=human_override,
    )
    return CampaignDecisionAccountingBuilder().build(trace=trace, outcome=outcome)


def test_empty_replay_summary():
    summary = CampaignDecisionReplayAnalyzer().analyze([], replay_id="replay-empty")

    assert summary.replay_id == "replay-empty"
    assert summary.accounting_count == 0
    assert summary.average_reward == 0.0
    assert summary.route_change_rate == 0.0
    assert summary.action_distribution == {}
    assert "No decision accounting records" in summary.rationale


def test_aggregates_average_reward_and_counts():
    positive = _accounting(trace_id="positive", execution_success=True)
    negative = _accounting(trace_id="negative", execution_success=False)
    neutral = _accounting(trace_id="neutral")

    summary = summarize_campaign_decision_replay(
        [positive, negative, neutral],
        replay_id="replay-rewards",
    )

    assert summary.accounting_count == 3
    assert summary.average_reward == -0.0333333333
    assert summary.positive_reward_count == 1
    assert summary.negative_reward_count == 1
    assert summary.neutral_reward_count == 1


def test_action_distribution():
    summary = summarize_campaign_decision_replay(
        [
            _accounting(
                trace_id="propose",
                action=CampaignDecisionAction.PROPOSE_CANDIDATES,
            ),
            _accounting(
                trace_id="revise",
                action=CampaignDecisionAction.REVISE_OBJECTIVE,
            ),
            _accounting(
                trace_id="query",
                action=CampaignDecisionAction.QUERY_LITERATURE,
            ),
            _accounting(
                trace_id="query-2",
                action=CampaignDecisionAction.QUERY_LITERATURE,
            ),
        ],
        replay_id="replay-actions",
    )

    assert summary.action_distribution == {
        "propose_candidates": 1,
        "revise_objective": 1,
        "query_literature": 2,
    }


def test_route_change_rate():
    summary = summarize_campaign_decision_replay(
        [
            _accounting(
                trace_id="same",
                action=CampaignDecisionAction.PROPOSE_CANDIDATES,
                actual_action="propose_candidates",
            ),
            _accounting(
                trace_id="changed",
                action=CampaignDecisionAction.QUERY_LITERATURE,
                actual_action="propose_candidates",
            ),
        ],
        replay_id="replay-route-change",
    )

    assert summary.route_change_count == 1
    assert summary.route_change_rate == 0.5


def test_component_averages():
    summary = summarize_campaign_decision_replay(
        [
            _accounting(
                trace_id="components-1",
                safety_incident_count=1,
                failure_count=2,
                objective_delta=0.5,
                proxy_gap_delta=-0.2,
                validation_success=True,
                recovery_attempted=True,
                recovery_success=True,
                context_request_fulfilled=True,
            ),
            _accounting(
                trace_id="components-2",
                safety_incident_count=0,
                failure_count=0,
                objective_delta=-0.5,
                proxy_gap_delta=0.2,
                validation_success=False,
                recovery_attempted=True,
                recovery_success=False,
                context_request_fulfilled=False,
            ),
        ],
        replay_id="replay-components",
    )

    assert summary.average_safety_penalty == -0.25
    assert summary.average_failure_penalty == -0.1
    assert summary.average_objective_reward == 0.0
    assert summary.average_proxy_gap_reward == 0.0
    assert summary.average_validation_reward == 0.0
    assert summary.average_recovery_reward == 0.0
    assert summary.average_context_reward == 0.05
    assert summary.recovery_success_rate == 0.5


def test_optional_rates_ignore_none():
    summary = summarize_campaign_decision_replay(
        [
            _accounting(
                trace_id="rates-1",
                context_request_fulfilled=True,
                validation_success=True,
                human_override=True,
            ),
            _accounting(
                trace_id="rates-2",
                context_request_fulfilled=False,
                validation_success=False,
                human_override=False,
            ),
            _accounting(trace_id="rates-3"),
        ],
        replay_id="replay-rates",
    )

    assert summary.context_request_fulfillment_rate == 0.5
    assert summary.validation_success_rate == 0.5
    assert summary.human_override_rate == 0.5


def test_campaign_ids_and_round_range():
    summary = summarize_campaign_decision_replay(
        [
            _accounting(trace_id="c2-r3", campaign_id="campaign-2", round_index=3),
            _accounting(trace_id="c1-r1", campaign_id="campaign-1", round_index=1),
            _accounting(trace_id="c1-r5", campaign_id="campaign-1", round_index=5),
        ],
        replay_id="replay-campaigns",
    )

    assert summary.campaign_ids == ["campaign-1", "campaign-2"]
    assert summary.round_range == {"min_round": 1, "max_round": 5}


def test_json_serialization():
    summary = summarize_campaign_decision_replay(
        [_accounting(trace_id="json-1")],
        replay_id="replay-json",
    )

    dumped = summary.model_dump(mode="json")
    raw_json = summary.model_dump_json()

    assert dumped["replay_id"] == "replay-json"
    assert dumped["created_at"].endswith("Z") or "+" in dumped["created_at"]
    assert '"replay_id":"replay-json"' in raw_json


def test_inputs_are_not_mutated():
    metadata = {"source": {"name": "initial"}}

    summary = summarize_campaign_decision_replay(
        [_accounting(trace_id="immutable")],
        replay_id="replay-immutable",
        metadata=metadata,
    )
    summary.metadata["source"]["name"] = "changed"

    assert metadata == {"source": {"name": "initial"}}


def test_import_smoke():
    import app.services.decision_outcome  # noqa: F401
    import app.services.decision_replay  # noqa: F401
    import app.services.decision_trace  # noqa: F401
    import app.services.strategy_selector  # noqa: F401
