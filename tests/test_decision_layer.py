from __future__ import annotations

from app.services.decision_layer import CampaignDecisionLayer, strategy_decision_to_payload
from app.services.decision_models import (
    CampaignDecisionAction,
    CampaignRoundContext,
)


def _context(**kwargs):
    base = dict(campaign_id="campaign-1", round_index=1)
    base.update(kwargs)
    return CampaignRoundContext(**base)


def test_default_strategy_result_wraps_into_propose_candidates():
    plan = CampaignDecisionLayer().decide(
        _context(
            strategy_selection_result={
                "campaign_intent": "explore_safe_region",
                "optimization_mode": "conservative_exploration",
                "candidate_generation_backend": "bo_mcp",
                "confidence": 0.72,
            }
        )
    )

    assert plan.action_type == CampaignDecisionAction.PROPOSE_CANDIDATES
    assert plan.campaign_intent == "explore_safe_region"
    assert plan.optimization_mode == "conservative_exploration"
    assert plan.candidate_generation_backend == "bo_mcp"
    assert plan.confidence == 0.72
    assert plan.shadow_only is True
    assert any(e.kind == "strategy_selection" for e in plan.evidence)


def test_flexible_strategy_result_keys_work():
    plan = CampaignDecisionLayer().decide(
        _context(
            strategy_selection_result={
                "intent": "optimize",
                "mode": "exploit",
                "selected_backend": "nexus_gp_bo",
                "trace": {"selected_backend": "nexus_gp_bo"},
            }
        )
    )

    assert plan.campaign_intent == "optimize"
    assert plan.optimization_mode == "exploit"
    assert plan.candidate_generation_backend == "nexus_gp_bo"
    assert plan.strategy_trace == {"selected_backend": "nexus_gp_bo"}


def test_blocking_failure_overrides_strategy():
    plan = CampaignDecisionLayer().decide(
        _context(
            failure_summary={"blocking": True},
            strategy_selection_result={"backend": "nexus_gp_bo"},
        )
    )

    assert plan.action_type == CampaignDecisionAction.RECOVER_FAILURE
    assert plan.shadow_only is True


def test_safety_high_risk_overrides_strategy():
    plan = CampaignDecisionLayer().decide(
        _context(
            safety_summary={"risk_level": "high"},
            strategy_selection_result={"backend": "nexus_gp_bo"},
        )
    )

    assert plan.action_type == CampaignDecisionAction.TIGHTEN_CONSTRAINTS
    assert plan.constraint_patch is not None
    assert plan.constraint_patch.shadow_only is True


def test_stop_requested_overrides_everything():
    plan = CampaignDecisionLayer().decide(
        _context(
            stop_requested=True,
            failure_summary={"blocking": True},
            safety_summary={"risk_level": "high"},
            strategy_selection_result={"backend": "nexus_gp_bo"},
        )
    )

    assert plan.action_type == CampaignDecisionAction.STOP_CAMPAIGN
    assert plan.shadow_only is True


def test_fallback_action_parsing():
    plan = CampaignDecisionLayer().decide(
        _context(
            strategy_selection_result={
                "backend": "built_in",
                "fallback_action": "query_literature",
            }
        )
    )

    assert plan.fallback_action == CampaignDecisionAction.QUERY_LITERATURE


def test_evidence_dicts_are_converted_and_raw_evidence_is_preserved():
    compatible = CampaignDecisionLayer().decide(
        _context(
            strategy_selection_result={
                "backend": "built_in",
                "evidence": [
                    {
                        "source": "selector",
                        "kind": "diagnostic",
                        "summary": "Noise was low.",
                        "weight": 0.3,
                    }
                ],
            }
        )
    )
    raw = CampaignDecisionLayer().decide(
        _context(strategy_selection_result={"backend": "built_in", "evidence": ["low_noise"]})
    )

    assert any(e.source == "selector" and e.kind == "diagnostic" for e in compatible.evidence)
    assert any(e.kind == "raw_evidence" for e in raw.evidence)


def test_import_smoke():
    import app.services.decision_layer  # noqa: F401
    import app.services.decision_models  # noqa: F401
    import app.services.strategy_selector  # noqa: F401


def test_new_modules_do_not_alter_select_strategy_behavior():
    import app.services.decision_layer  # noqa: F401
    import app.services.decision_models  # noqa: F401
    from app.services.strategy_models import CampaignSnapshot, PhaseConfig
    from app.services.strategy_selector import select_strategy

    snapshot = CampaignSnapshot(
        round_number=1,
        max_rounds=10,
        n_observations=1,
        n_dimensions=2,
        has_categorical=False,
        has_log_scale=False,
        kpi_history=(0.1,),
        available_backends={"lhs": True, "built_in": True},
    )

    decision = select_strategy(
        snapshot,
        config=PhaseConfig(enable_nexus=False, enable_method_advisor=False),
    )

    assert decision.backend_name


def test_strategy_decision_to_payload_supports_dataclasses():
    from app.services.strategy_models import StrategyDecision

    decision = StrategyDecision(
        backend_name="built_in",
        phase="exploitation",
        reason="test",
        confidence=0.8,
    )

    payload = strategy_decision_to_payload(decision)

    assert payload["backend_name"] == "built_in"
    assert payload["backend"] == "built_in"
    assert payload["confidence"] == 0.8
