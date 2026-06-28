from __future__ import annotations

from app.services.decision_layer import CampaignDecisionLayer
from app.services.decision_models import CampaignDecisionAction
from app.services.round_context import (
    CampaignRoundContextBuilder,
    build_campaign_round_context,
)


def test_minimal_build_defaults_and_json_serialization():
    context = CampaignRoundContextBuilder().build(
        campaign_id="campaign-1",
        round_index=0,
    )

    assert context.campaign_id == "campaign-1"
    assert context.round_index == 0
    assert context.stop_requested is False
    assert context.strategy_selection_result == {}
    assert context.failure_summary == {}
    assert context.safety_summary == {}
    assert context.objective_summary == {}
    assert context.constraint_summary == {}
    assert context.nexus_diagnostics == {}
    assert context.backend_memory_summary == {}
    assert context.bo_mcp_summary == {}
    assert context.learning_policy_summary == {}
    assert context.validation_summary == {}
    assert context.human_observations == []
    assert context.literature_summary == {}
    assert context.metadata == {}
    assert context.model_dump(mode="json")["campaign_id"] == "campaign-1"
    assert '"campaign_id":"campaign-1"' in context.model_dump_json()


def test_full_build_preserves_all_fields():
    context = build_campaign_round_context(
        campaign_id="campaign-2",
        round_index=3,
        strategy_selection_result={"backend": "nexus_gp_bo"},
        stop_requested=True,
        failure_summary={"blocking": False},
        safety_summary={"risk_level": "normal"},
        objective_summary={"level": "performance"},
        constraint_summary={"window": "safe"},
        nexus_diagnostics={"posterior": 0.4},
        backend_memory_summary={"wins": 2},
        bo_mcp_summary={"trust_region": "stable"},
        learning_policy_summary={"mode": "shadow"},
        validation_summary={"passed": True},
        human_observations=["film looked uniform"],
        literature_summary={"papers": 4},
        metadata={"source": "test"},
    )

    assert context.strategy_selection_result == {"backend": "nexus_gp_bo"}
    assert context.stop_requested is True
    assert context.failure_summary == {"blocking": False}
    assert context.safety_summary == {"risk_level": "normal"}
    assert context.objective_summary == {"level": "performance"}
    assert context.constraint_summary == {"window": "safe"}
    assert context.nexus_diagnostics == {"posterior": 0.4}
    assert context.backend_memory_summary == {"wins": 2}
    assert context.bo_mcp_summary == {"trust_region": "stable"}
    assert context.learning_policy_summary == {"mode": "shadow"}
    assert context.validation_summary == {"passed": True}
    assert context.human_observations == ["film looked uniform"]
    assert context.literature_summary == {"papers": 4}
    assert context.metadata == {"source": "test"}


def test_inputs_are_not_mutated():
    strategy = {"backend": "built_in", "trace": {"round": 1}}
    failure = {"events": []}
    observations = ["operator noted bubbles"]

    context = CampaignRoundContextBuilder().build(
        campaign_id="campaign-1",
        round_index=1,
        strategy_selection_result=strategy,
        failure_summary=failure,
        human_observations=observations,
    )
    context.strategy_selection_result["backend"] = "changed"
    context.strategy_selection_result["trace"]["round"] = 2
    context.failure_summary["events"].append("new")
    context.human_observations.append("new observation")

    assert strategy == {"backend": "built_in", "trace": {"round": 1}}
    assert failure == {"events": []}
    assert observations == ["operator noted bubbles"]


def test_builder_output_works_with_decision_layer():
    context = CampaignRoundContextBuilder().build(
        campaign_id="campaign-1",
        round_index=1,
        strategy_selection_result={
            "campaign_intent": "optimize",
            "optimization_mode": "exploit",
            "candidate_generation_backend": "bo_mcp",
        },
    )

    plan = CampaignDecisionLayer().decide(context)

    assert plan.action_type == CampaignDecisionAction.PROPOSE_CANDIDATES
    assert plan.candidate_generation_backend == "bo_mcp"


def test_failure_summary_built_by_builder_still_overrides():
    context = CampaignRoundContextBuilder().build(
        campaign_id="campaign-1",
        round_index=1,
        strategy_selection_result={"backend": "bo_mcp"},
        failure_summary={"requires_recovery": True},
    )

    plan = CampaignDecisionLayer().decide(context)

    assert plan.action_type == CampaignDecisionAction.RECOVER_FAILURE


def test_safety_summary_built_by_builder_still_overrides():
    context = CampaignRoundContextBuilder().build(
        campaign_id="campaign-1",
        round_index=1,
        strategy_selection_result={"backend": "bo_mcp"},
        safety_summary={"requires_constraint_update": True},
    )

    plan = CampaignDecisionLayer().decide(context)

    assert plan.action_type == CampaignDecisionAction.TIGHTEN_CONSTRAINTS


def test_import_smoke():
    import app.services.decision_layer  # noqa: F401
    import app.services.round_context  # noqa: F401
    import app.services.strategy_selector  # noqa: F401
