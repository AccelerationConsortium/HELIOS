from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.services.decision_models import (
    CampaignDecisionAction,
    CampaignDecisionPlan,
)
from app.services.decision_outcome import (
    CampaignDecisionAccountingBuilder,
    CampaignDecisionOutcomeBuilder,
)
from app.services.decision_trace import CampaignDecisionTraceBuilder
from app.services.round_context import CampaignRoundContextBuilder
from app.services.verifiable_decision_reward import (
    RUBRIC_VERSION,
    VerifiableDecisionRewardReport,
    VerifiableDecisionRewardVerifier,
    verify_campaign_decision_reward,
)

_FIXED_NOW = datetime(2026, 1, 1, tzinfo=UTC)

_PROCESS_NAMES = {"execution", "safety", "failure", "context", "proxy_gap", "validation"}
_OUTCOME_NAMES = {"objective"}


def _accounting(
    *,
    trace_id: str = "cdt-1",
    campaign_id: str = "campaign-1",
    round_index: int = 1,
    execution_success: bool | None = None,
    failure_count: int = 0,
    safety_incident_count: int = 0,
    objective_delta: float | None = None,
    proxy_gap_delta: float | None = None,
    validation_success: bool | None = None,
    context_request_fulfilled: bool | None = None,
):
    context = CampaignRoundContextBuilder().build(
        campaign_id=campaign_id,
        round_index=round_index,
    )
    plan = CampaignDecisionPlan(
        action_type=CampaignDecisionAction.PROPOSE_CANDIDATES,
        rationale="test",
    )
    trace = CampaignDecisionTraceBuilder().build(
        trace_id=trace_id,
        context=context,
        decision_plan=plan,
    )
    outcome = CampaignDecisionOutcomeBuilder().build(
        trace=trace,
        execution_success=execution_success,
        failure_count=failure_count,
        safety_incident_count=safety_incident_count,
        objective_delta=objective_delta,
        proxy_gap_delta=proxy_gap_delta,
        validation_success=validation_success,
        context_request_fulfilled=context_request_fulfilled,
    )
    return CampaignDecisionAccountingBuilder().build(trace=trace, outcome=outcome)


def _by_name(report: VerifiableDecisionRewardReport):
    return {verification.name: verification for verification in report.verifications}


def test_seven_verifications_with_expected_verifier_types() -> None:
    report = verify_campaign_decision_reward(_accounting(), now=_FIXED_NOW)
    by_name = _by_name(report)

    assert set(by_name) == _PROCESS_NAMES | _OUTCOME_NAMES
    assert len(report.verifications) == 7
    assert by_name["execution"].verifier_type == "state_transition"
    assert by_name["safety"].verifier_type == "safety_rule"
    assert by_name["failure"].verifier_type == "failure_rule"
    assert by_name["context"].verifier_type == "context_rule"
    assert by_name["proxy_gap"].verifier_type == "proxy_metric"
    assert by_name["validation"].verifier_type == "validation_rule"
    assert by_name["objective"].verifier_type == "outcome_metric"


def test_process_and_outcome_split_membership() -> None:
    report = verify_campaign_decision_reward(
        _accounting(
            execution_success=True,
            objective_delta=0.5,
            proxy_gap_delta=-0.2,
            validation_success=True,
            context_request_fulfilled=True,
        ),
        now=_FIXED_NOW,
    )
    by_name = _by_name(report)

    process_sum = round(sum(by_name[name].score for name in _PROCESS_NAMES), 10)
    outcome_sum = round(sum(by_name[name].score for name in _OUTCOME_NAMES), 10)
    assert report.process_reward == process_sum
    assert report.outcome_reward == outcome_sum
    # Objective is the only outcome component.
    assert report.outcome_reward == by_name["objective"].score


def test_decomposition_equals_raw_component_sum() -> None:
    report = verify_campaign_decision_reward(
        _accounting(
            execution_success=True,
            objective_delta=0.4,
            proxy_gap_delta=-0.1,
            validation_success=True,
        ),
        now=_FIXED_NOW,
    )
    assert report.raw_component_sum == round(
        report.process_reward + report.outcome_reward, 10
    )
    all_scores = round(sum(v.score for v in report.verifications), 10)
    assert report.raw_component_sum == all_scores


def test_total_reward_is_clamped_source_reward() -> None:
    accounting = _accounting(
        execution_success=True,
        objective_delta=1.0,
        proxy_gap_delta=-1.0,
        validation_success=True,
        context_request_fulfilled=True,
    )
    report = verify_campaign_decision_reward(accounting, now=_FIXED_NOW)

    assert report.total_reward == round(accounting.reward.reward, 10)
    # This outcome sums above 1.0 before clamping, so it must be clamped.
    assert report.total_reward == 1.0
    assert report.raw_component_sum > 1.0
    assert report.clamped is True


def test_not_clamped_when_within_range() -> None:
    report = verify_campaign_decision_reward(
        _accounting(execution_success=True, objective_delta=0.1),
        now=_FIXED_NOW,
    )
    assert report.clamped is False
    assert report.total_reward == report.raw_component_sum


def test_clean_outcome_passes_present_signals() -> None:
    report = verify_campaign_decision_reward(
        _accounting(
            execution_success=True,
            objective_delta=0.2,
            proxy_gap_delta=-0.2,
            validation_success=True,
            context_request_fulfilled=True,
        ),
        now=_FIXED_NOW,
    )
    by_name = _by_name(report)
    assert by_name["execution"].passed is True
    assert by_name["objective"].passed is True
    assert by_name["proxy_gap"].passed is True
    assert by_name["validation"].passed is True
    assert by_name["context"].passed is True
    # Counts default to zero -> always present and passing.
    assert by_name["safety"].passed is True
    assert by_name["failure"].passed is True
    for verification in report.verifications:
        assert verification.status == "verified"
        assert verification.evidence["signal_present"] is True


def test_safety_and_failure_incidents_fail() -> None:
    report = verify_campaign_decision_reward(
        _accounting(safety_incident_count=1, failure_count=2),
        now=_FIXED_NOW,
    )
    by_name = _by_name(report)
    assert by_name["safety"].passed is False
    assert by_name["safety"].score < 0.0
    assert by_name["failure"].passed is False
    assert by_name["failure"].score < 0.0
    # Counts are always present, never neutral.
    assert by_name["safety"].status == "verified"
    assert by_name["failure"].status == "verified"


def test_execution_and_validation_failure() -> None:
    report = verify_campaign_decision_reward(
        _accounting(execution_success=False, validation_success=False),
        now=_FIXED_NOW,
    )
    by_name = _by_name(report)
    assert by_name["execution"].passed is False
    assert by_name["validation"].passed is False


def test_widening_proxy_gap_and_regressing_objective_fail() -> None:
    report = verify_campaign_decision_reward(
        _accounting(objective_delta=-0.3, proxy_gap_delta=0.3),
        now=_FIXED_NOW,
    )
    by_name = _by_name(report)
    assert by_name["objective"].passed is False
    assert by_name["objective"].score < 0.0
    assert by_name["proxy_gap"].passed is False
    assert by_name["proxy_gap"].score < 0.0


def test_missing_signals_are_neutral_not_passed() -> None:
    # All optional signals left as None.
    report = verify_campaign_decision_reward(_accounting(), now=_FIXED_NOW)
    by_name = _by_name(report)
    for name in ("execution", "context", "proxy_gap", "validation", "objective"):
        verification = by_name[name]
        assert verification.status == "neutral_due_to_missing_signal"
        assert verification.passed is None
        assert verification.score == 0.0
        assert verification.evidence["signal_present"] is False
    # Neutral components contribute zero; totals collapse to zero.
    assert report.raw_component_sum == 0.0
    assert report.total_reward == 0.0
    assert report.clamped is False


def test_shadow_only_is_true() -> None:
    report = verify_campaign_decision_reward(_accounting(), now=_FIXED_NOW)
    assert report.shadow_only is True
    assert report.rubric_version == RUBRIC_VERSION


def test_serialization_round_trip() -> None:
    report = verify_campaign_decision_reward(
        _accounting(
            execution_success=True,
            objective_delta=0.2,
            proxy_gap_delta=-0.1,
            validation_success=True,
            context_request_fulfilled=False,
        ),
        now=_FIXED_NOW,
    )
    restored = VerifiableDecisionRewardReport.model_validate_json(
        report.model_dump_json()
    )
    assert restored == report


def test_determinism() -> None:
    accounting = _accounting(execution_success=True, objective_delta=0.3)
    first = verify_campaign_decision_reward(accounting, now=_FIXED_NOW)
    second = VerifiableDecisionRewardVerifier().verify(accounting, now=_FIXED_NOW)
    assert first == second
    assert first.decision_id == accounting.trace.trace_id


def test_created_at_must_be_timezone_aware() -> None:
    with pytest.raises(ValueError):
        VerifiableDecisionRewardReport(
            decision_id="d1",
            campaign_id="c1",
            round_index=0,
            total_reward=0.0,
            raw_component_sum=0.0,
            clamped=False,
            process_reward=0.0,
            outcome_reward=0.0,
            created_at=datetime(2026, 1, 1),  # naive
        )
