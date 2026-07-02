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


def test_validation_due_true_routes_to_run_validation():
    plan = CampaignDecisionLayer().decide(
        _context(
            validation_summary={"validation_due": True, "confidence": 0.82},
            strategy_selection_result={"backend": "nexus_gp_bo"},
        )
    )

    assert plan.action_type == CampaignDecisionAction.RUN_VALIDATION
    assert plan.route_target == "validation"
    assert plan.confidence == 0.82
    assert plan.shadow_only is True
    assert any(e.source == "validation_summary" and e.kind == "validation_due" for e in plan.evidence)


def test_validation_status_due_routes_to_run_validation():
    plan = CampaignDecisionLayer().decide(
        _context(
            validation_summary={"status": "due"},
            strategy_selection_result={"backend": "nexus_gp_bo"},
        )
    )

    assert plan.action_type == CampaignDecisionAction.RUN_VALIDATION
    assert plan.route_target == "validation"
    assert plan.confidence == 0.7


def test_high_proxy_gap_level_routes_to_revise_objective():
    plan = CampaignDecisionLayer().decide(
        _context(
            objective_summary={"proxy_gap_level": "high", "confidence": 0.74},
            strategy_selection_result={"backend": "nexus_gp_bo"},
        )
    )

    assert plan.action_type == CampaignDecisionAction.REVISE_OBJECTIVE
    assert plan.route_target == "objective_revision"
    assert plan.confidence == 0.74
    assert plan.objective_patch is not None
    assert plan.objective_patch.shadow_only is True
    assert any(e.source == "objective_summary" and e.kind == "proxy_gap" for e in plan.evidence)


def test_high_proxy_gap_score_routes_to_revise_objective():
    plan = CampaignDecisionLayer().decide(
        _context(
            objective_summary={"proxy_gap_score": 0.67},
            strategy_selection_result={"backend": "nexus_gp_bo"},
        )
    )

    assert plan.action_type == CampaignDecisionAction.REVISE_OBJECTIVE
    assert plan.confidence == 0.67


def test_nested_proxy_gap_assessment_routes_to_revise_objective():
    plan = CampaignDecisionLayer().decide(
        _context(
            objective_summary={
                "proxy_gap_assessment": {
                    "level": "high",
                    "score": 0.45,
                }
            },
            strategy_selection_result={"backend": "nexus_gp_bo"},
        )
    )

    assert plan.action_type == CampaignDecisionAction.REVISE_OBJECTIVE
    assert plan.confidence == 0.45


def test_nested_proxy_gap_level_without_score_uses_default_confidence():
    plan = CampaignDecisionLayer().decide(
        _context(
            objective_summary={
                "proxy_gap_assessment": {
                    "level": "high",
                }
            },
            strategy_selection_result={"backend": "nexus_gp_bo"},
        )
    )

    assert plan.action_type == CampaignDecisionAction.REVISE_OBJECTIVE
    assert plan.confidence == 0.65


def test_validation_overrides_proxy_gap():
    plan = CampaignDecisionLayer().decide(
        _context(
            validation_summary={"requires_validation": True},
            objective_summary={"proxy_gap_level": "high"},
            strategy_selection_result={"backend": "nexus_gp_bo"},
        )
    )

    assert plan.action_type == CampaignDecisionAction.RUN_VALIDATION


def test_stop_failure_safety_override_validation_and_proxy_gap():
    validation_summary = {"validation_due": True}
    objective_summary = {"proxy_gap_level": "high"}

    stop_plan = CampaignDecisionLayer().decide(
        _context(
            stop_requested=True,
            validation_summary=validation_summary,
            objective_summary=objective_summary,
        )
    )
    failure_plan = CampaignDecisionLayer().decide(
        _context(
            failure_summary={"requires_recovery": True},
            validation_summary=validation_summary,
            objective_summary=objective_summary,
        )
    )
    safety_plan = CampaignDecisionLayer().decide(
        _context(
            safety_summary={"risk_level": "critical"},
            validation_summary=validation_summary,
            objective_summary=objective_summary,
        )
    )

    assert stop_plan.action_type == CampaignDecisionAction.STOP_CAMPAIGN
    assert failure_plan.action_type == CampaignDecisionAction.RECOVER_FAILURE
    assert safety_plan.action_type == CampaignDecisionAction.TIGHTEN_CONSTRAINTS


def test_objective_patch_is_shadow_only():
    plan = CampaignDecisionLayer().decide(
        _context(
            objective_summary={"proxy_gap": "high"},
            strategy_selection_result={"backend": "nexus_gp_bo"},
        )
    )

    assert plan.action_type == CampaignDecisionAction.REVISE_OBJECTIVE
    assert plan.shadow_only is True
    assert plan.objective_patch is not None
    assert plan.objective_patch.shadow_only is True
    assert "functional scientific performance" in plan.objective_patch.reason


def test_propose_candidates_still_default_when_no_validation_or_proxy_gap():
    plan = CampaignDecisionLayer().decide(
        _context(
            validation_summary={"status": "passed"},
            objective_summary={"proxy_gap_level": "medium", "proxy_gap_score": 0.4},
            strategy_selection_result={
                "candidate_generation_backend": "bo_mcp",
                "confidence": 0.61,
            },
        )
    )

    assert plan.action_type == CampaignDecisionAction.PROPOSE_CANDIDATES
    assert plan.candidate_generation_backend == "bo_mcp"
    assert plan.confidence == 0.61


def test_plateau_with_missing_literature_routes_to_query_literature():
    plan = CampaignDecisionLayer().decide(
        _context(
            nexus_diagnostics={
                "convergence_status": "plateau",
                "convergence_confidence": 0.78,
            },
            literature_summary={},
            strategy_selection_result={"backend": "bo_mcp"},
        )
    )

    assert plan.action_type == CampaignDecisionAction.QUERY_LITERATURE
    assert plan.route_target == "literature"
    assert plan.confidence == 0.78
    assert plan.shadow_only is True
    assert plan.context_requests[0].request_type == "literature_context"
    assert any(e.kind == "plateau_literature_missing" for e in plan.evidence)


def test_low_failure_attribution_confidence_requests_human_observation():
    plan = CampaignDecisionLayer().decide(
        _context(
            failure_summary={
                "events": [{"failure_type": "measurement"}],
                "attribution_confidence": 0.31,
            },
            strategy_selection_result={"backend": "bo_mcp"},
        )
    )

    assert plan.action_type == CampaignDecisionAction.REQUEST_HUMAN_OBSERVATION
    assert plan.route_target == "human_observation"
    assert plan.confidence == 0.69
    assert plan.shadow_only is True
    assert plan.context_requests[0].request_type == "failure_attribution"
    assert any(e.kind == "low_failure_attribution_confidence" for e in plan.evidence)


def test_conflicting_objective_signals_request_human_observation():
    plan = CampaignDecisionLayer().decide(
        _context(
            objective_summary={
                "conflicting_signals": True,
                "confidence": 0.73,
                "signals": ["proxy high", "device metric improving"],
            },
            strategy_selection_result={"backend": "bo_mcp"},
        )
    )

    assert plan.action_type == CampaignDecisionAction.REQUEST_HUMAN_OBSERVATION
    assert plan.route_target == "human_observation"
    assert plan.confidence == 0.73
    assert plan.context_requests[0].request_type == "objective_disambiguation"
    assert any(e.kind == "objective_signal_conflict" for e in plan.evidence)


def test_backend_memory_missing_with_repeated_failure_enriches_strategy_decision():
    plan = CampaignDecisionLayer().decide(
        _context(
            failure_summary={
                "events": [
                    {"failure_type": "backend"},
                    {"failure_type": "backend"},
                ],
            },
            backend_memory_summary={},
            strategy_selection_result={
                "candidate_generation_backend": "bo_mcp",
                "confidence": 0.62,
            },
        )
    )

    assert plan.action_type == CampaignDecisionAction.PROPOSE_CANDIDATES
    assert plan.candidate_generation_backend == "bo_mcp"
    assert plan.confidence == 0.62
    assert plan.shadow_only is True
    assert plan.metadata["context_enriched"] is True
    assert plan.context_requests[0].request_type == "backend_memory"
    assert any(
        e.kind == "backend_memory_missing_repeated_failure" for e in plan.evidence
    )


def test_validation_overrides_context_seeking_rules():
    plan = CampaignDecisionLayer().decide(
        _context(
            validation_summary={"validation_due": True},
            objective_summary={"conflicting_signals": True},
            nexus_diagnostics={"convergence_status": "plateau"},
            literature_summary={},
            failure_summary={
                "events": [{"failure_type": "measurement"}],
                "attribution_confidence": 0.1,
            },
            strategy_selection_result={"backend": "bo_mcp"},
        )
    )

    assert plan.action_type == CampaignDecisionAction.RUN_VALIDATION


def test_objective_conflict_overrides_high_proxy_gap():
    plan = CampaignDecisionLayer().decide(
        _context(
            objective_summary={
                "conflicts": [{"metric": "stability"}],
                "proxy_gap_level": "high",
            },
            strategy_selection_result={"backend": "bo_mcp"},
        )
    )

    assert plan.action_type == CampaignDecisionAction.REQUEST_HUMAN_OBSERVATION


def test_existing_guards_override_context_seeking_rules():
    noisy_context = {
        "validation_summary": {"validation_due": False},
        "objective_summary": {"conflicting_signals": True},
        "nexus_diagnostics": {"convergence_status": "plateau"},
        "literature_summary": {},
        "strategy_selection_result": {"backend": "bo_mcp"},
    }

    stop_plan = CampaignDecisionLayer().decide(
        _context(stop_requested=True, **noisy_context)
    )
    failure_plan = CampaignDecisionLayer().decide(
        _context(failure_summary={"blocking": True}, **noisy_context)
    )
    safety_plan = CampaignDecisionLayer().decide(
        _context(safety_summary={"risk_level": "critical"}, **noisy_context)
    )

    assert stop_plan.action_type == CampaignDecisionAction.STOP_CAMPAIGN
    assert failure_plan.action_type == CampaignDecisionAction.RECOVER_FAILURE
    assert safety_plan.action_type == CampaignDecisionAction.TIGHTEN_CONSTRAINTS


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
