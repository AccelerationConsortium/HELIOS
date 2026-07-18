from __future__ import annotations

from app.services.decision_layer import CampaignDecisionLayer
from app.services.decision_outcome import (
    CampaignDecisionAccounting,
    CampaignDecisionAccountingBuilder,
    CampaignDecisionOutcomeBuilder,
)
from app.services.decision_trace import CampaignDecisionTrace, CampaignDecisionTraceBuilder
from app.services.round_context import CampaignRoundContextBuilder


def decision_trace(
    *,
    campaign_id: str = "campaign-32",
    round_index: int = 3,
    trace_id: str = "cdt-ledger-003",
) -> CampaignDecisionTrace:
    context = CampaignRoundContextBuilder().build(
        campaign_id=campaign_id,
        round_index=round_index,
        objective_summary={
            "objective_kpi": "yield",
            "direction": "maximize",
            "target_value": 0.9,
        },
        failure_summary={"failure_count": 0},
        nexus_diagnostics={
            "contract_version": "early_stage_system_characterization.v1",
            "entropy_score": 0.21,
            "failure_attribution_distribution": {"pipette_offset": 0.72},
        },
        human_observations=["Meniscus was stable."],
        strategy_selection_result={
            "campaign_intent": "optimize",
            "optimization_mode": "exploit",
            "candidate_generation_backend": "bo_mcp",
            "confidence": 0.83,
            "strategy_trace": {
                "policy_id": "campaign-meta-controller",
                "policy_version": "v4",
                "selected_backend": "bo_mcp",
                "available_actions": [
                    {
                        "name": "validation",
                        "backend_name": "built_in",
                        "expected_improvement": 0.2,
                        "expected_info_gain": 0.82,
                        "risk": 0.1,
                        "utility": 0.77,
                        "reason": "Resolve uncertainty before optimization",
                    },
                    {
                        "name": "exploit",
                        "backend_name": "bo_mcp",
                        "expected_improvement": 0.72,
                        "expected_info_gain": 0.45,
                        "risk": 0.18,
                        "utility": 0.69,
                        "reason": "Stable objective and calibrated model",
                    },
                    {
                        "name": "random",
                        "backend_name": "random",
                        "expected_improvement": 0.1,
                        "expected_info_gain": 0.3,
                        "risk": 0.2,
                        "utility": 0.12,
                        "reason": "Fallback exploration",
                    },
                ],
            },
            "evidence": [
                {
                    "source": "dataset",
                    "kind": "sample_size",
                    "summary": "Dataset size = 18",
                    "weight": 0.8,
                },
                {
                    "source": "optimizer",
                    "kind": "acquisition_confidence",
                    "summary": "Acquisition confidence = 0.83",
                    "weight": 0.83,
                },
            ],
        },
        metadata={"operator_note": "do not expose sk-test-secret-value"},
    )
    plan = CampaignDecisionLayer().decide(context)
    return CampaignDecisionTraceBuilder().build(
        context=context,
        decision_plan=plan,
        actual_stage="candidate_generation",
        actual_action="propose_candidates",
        trace_id=trace_id,
    )


def decision_accounting(
    *,
    campaign_id: str = "campaign-32",
    round_index: int = 3,
    trace_id: str = "cdt-ledger-003",
) -> CampaignDecisionAccounting:
    trace = decision_trace(
        campaign_id=campaign_id,
        round_index=round_index,
        trace_id=trace_id,
    )
    outcome = CampaignDecisionOutcomeBuilder().build(
        trace=trace,
        observed_action="propose_candidates",
        observed_backend="bo_mcp",
        candidate_count=4,
        execution_success=True,
        failure_count=1,
        objective_delta=0.18,
        validation_success=True,
        context_request_fulfilled=True,
        metadata={"authorization": "Bearer super-secret-token"},
    )
    return CampaignDecisionAccountingBuilder().build(trace=trace, outcome=outcome)
