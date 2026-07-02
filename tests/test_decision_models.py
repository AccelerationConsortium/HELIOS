from __future__ import annotations

from app.services.decision_models import (
    CampaignDecisionAction,
    CampaignDecisionEvidence,
    CampaignDecisionPlan,
    CampaignRoundContext,
)


def test_campaign_round_context_json_serialization():
    context = CampaignRoundContext(
        campaign_id="campaign-1",
        round_index=0,
        strategy_selection_result={"backend": "bo_mcp", "confidence": 0.72},
    )

    dumped = context.model_dump(mode="json")
    raw_json = context.model_dump_json()

    assert dumped["campaign_id"] == "campaign-1"
    assert dumped["strategy_selection_result"]["backend"] == "bo_mcp"
    assert '"campaign_id":"campaign-1"' in raw_json


def test_campaign_decision_plan_json_serialization():
    plan = CampaignDecisionPlan(
        action_type=CampaignDecisionAction.PROPOSE_CANDIDATES,
        campaign_intent="explore_safe_region",
        optimization_mode="conservative_exploration",
        candidate_generation_backend="bo_mcp",
        rationale="Strategy result was wrapped.",
        confidence=0.72,
        evidence=[
            CampaignDecisionEvidence(
                source="dynamic_strategy_selector",
                kind="strategy_selection",
                summary="Wrapped selector result.",
            )
        ],
    )

    dumped = plan.model_dump(mode="json")
    raw_json = plan.model_dump_json()

    assert dumped["action_type"] == "propose_candidates"
    assert dumped["created_at"].endswith("Z") or "+" in dumped["created_at"]
    assert '"action_type":"propose_candidates"' in raw_json
