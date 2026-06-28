from __future__ import annotations

from app.services.decision_layer import CampaignDecisionLayer
from app.services.decision_models import (
    CampaignDecisionAction,
    CampaignDecisionEvidence,
    CampaignDecisionPlan,
)
from app.services.decision_trace import (
    CampaignDecisionTraceBuilder,
    build_campaign_decision_trace,
)
from app.services.round_context import CampaignRoundContextBuilder


def _context():
    return CampaignRoundContextBuilder().build(
        campaign_id="campaign-1",
        round_index=2,
        strategy_selection_result={
            "campaign_intent": "optimize",
            "optimization_mode": "exploit",
            "candidate_generation_backend": "bo_mcp",
            "confidence": 0.75,
        },
    )


def _plan(context=None):
    return CampaignDecisionLayer().decide(context or _context())


def test_minimal_trace_build_preserves_context_and_plan():
    context = _context()
    plan = _plan(context)

    trace = CampaignDecisionTraceBuilder().build(
        trace_id="trace-1",
        context=context,
        decision_plan=plan,
    )

    assert trace.trace_id == "trace-1"
    assert trace.campaign_id == "campaign-1"
    assert trace.round_index == 2
    assert trace.shadow_action == plan.action_type
    assert trace.decision_plan == plan
    assert trace.context == context
    assert trace.would_change_route is False


def test_trace_detects_route_change():
    trace = build_campaign_decision_trace(
        trace_id="trace-2",
        context=_context(),
        decision_plan=CampaignDecisionPlan(
            action_type=CampaignDecisionAction.PROPOSE_CANDIDATES,
            rationale="test",
        ),
        actual_action="recover_failure",
    )

    assert trace.would_change_route is True
    assert trace.comparison == {
        "actual_action": "recover_failure",
        "shadow_action": "propose_candidates",
        "would_change_route": True,
    }


def test_trace_detects_same_route():
    trace = CampaignDecisionTraceBuilder().build(
        trace_id="trace-3",
        context=_context(),
        decision_plan=CampaignDecisionPlan(
            action_type=CampaignDecisionAction.PROPOSE_CANDIDATES,
            rationale="test",
        ),
        actual_action="propose_candidates",
    )

    assert trace.would_change_route is False
    assert trace.comparison["would_change_route"] is False


def test_evidence_copied_from_decision_plan():
    plan = CampaignDecisionPlan(
        action_type=CampaignDecisionAction.RUN_VALIDATION,
        rationale="test",
        evidence=[
            CampaignDecisionEvidence(
                source="shadow",
                kind="validation",
                summary="Run validation before continuing.",
                payload={"reason": "low confidence"},
            )
        ],
    )

    trace = CampaignDecisionTraceBuilder().build(
        trace_id="trace-4",
        context=_context(),
        decision_plan=plan,
    )
    trace.evidence[0].payload["reason"] = "changed"

    assert trace.evidence[0].source == "shadow"
    assert plan.evidence[0].payload == {"reason": "low confidence"}


def test_json_serialization():
    trace = CampaignDecisionTraceBuilder().build(
        trace_id="trace-json",
        context=_context(),
        decision_plan=_plan(),
        actual_stage="candidate_generation",
        actual_action="propose_candidates",
    )

    dumped = trace.model_dump(mode="json")
    raw_json = trace.model_dump_json()

    assert dumped["trace_id"] == "trace-json"
    assert dumped["shadow_action"] == "propose_candidates"
    assert dumped["created_at"].endswith("Z") or "+" in dumped["created_at"]
    assert '"trace_id":"trace-json"' in raw_json


def test_inputs_are_not_mutated():
    metadata = {"source": {"name": "test"}}
    context = _context()
    plan = _plan(context)

    trace = CampaignDecisionTraceBuilder().build(
        trace_id="trace-5",
        context=context,
        decision_plan=plan,
        metadata=metadata,
    )
    trace.metadata["source"]["name"] = "changed"
    trace.context.strategy_selection_result["candidate_generation_backend"] = "changed"
    trace.decision_plan.evidence[0].payload["candidate_generation_backend"] = "changed"

    assert metadata == {"source": {"name": "test"}}
    assert context.strategy_selection_result["candidate_generation_backend"] == "bo_mcp"
    assert plan.evidence[0].payload["candidate_generation_backend"] == "bo_mcp"


def test_import_smoke():
    import app.services.decision_layer  # noqa: F401
    import app.services.decision_trace  # noqa: F401
    import app.services.round_context  # noqa: F401
    import app.services.strategy_selector  # noqa: F401
