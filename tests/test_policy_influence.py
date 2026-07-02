from __future__ import annotations

import dataclasses

from app.services.backend_memory import BackendPerformanceMemory, problem_context_key
from app.services.strategy_models import (
    CampaignContext,
    CampaignSnapshot,
    FailureEvent,
    PolicyInfluenceConfig,
)
from app.services.strategy_selector import PhaseConfig, select_strategy, strategy_trace_to_dict

AVAIL = {
    "lhs": True,
    "built_in": True,
    "random_sampling": True,
    "nexus_lhs": True,
    "nexus_sobol": True,
    "optuna_tpe": True,
    "nexus_tpe": True,
    "nexus_gp_bo": True,
    "nexus_turbo": True,
    "bomcp": True,
}


def _snapshot(**kwargs):
    params = tuple({"x": i / 12, "y": (i % 4) / 4} for i in range(12))
    kpis = tuple(0.1 + 0.03 * i for i in range(12))
    base = dict(
        round_number=6,
        max_rounds=12,
        n_observations=12,
        n_dimensions=2,
        has_categorical=False,
        has_log_scale=False,
        kpi_history=kpis,
        direction="maximize",
        available_backends=dict(AVAIL),
        last_batch_kpis=kpis[-3:],
        last_batch_params=params[-3:],
        best_kpi_so_far=max(kpis),
        all_params=params,
        all_kpis=kpis,
        campaign_context=CampaignContext(current_objective_level="performance"),
    )
    base.update(kwargs)
    return CampaignSnapshot(**base)


def _trace(snapshot, influence: PolicyInfluenceConfig | None = None):
    config = PhaseConfig(policy_influence=influence or PolicyInfluenceConfig())
    return strategy_trace_to_dict(select_strategy(snapshot, config=config).strategy_trace)


def test_default_policy_influence_preserves_backend_ranking():
    snap = _snapshot()
    default_trace = _trace(snap)
    explicit_default = _trace(snap, PolicyInfluenceConfig())

    assert explicit_default["selected_backend"] == default_trace["selected_backend"]
    assert explicit_default["ranking_influences"] == []
    assert all(
        backend["influence_delta"] == 0.0
        for backend in explicit_default["candidate_backends"]
    )


def test_action_policy_rerank_is_bounded_and_traced():
    cap = 0.04
    trace = _trace(
        _snapshot(campaign_context=CampaignContext(current_objective_level="baseline")),
        PolicyInfluenceConfig(
            enable_action_policy_rerank=True,
            max_action_policy_weight=cap,
            max_total_score_delta=cap,
        ),
    )

    records = [
        record for record in trace["ranking_influences"]
        if record["source"] == "action_policy"
    ]
    assert records
    assert all(abs(record["score_delta"]) <= cap for record in records)
    assert any(backend["influence_delta"] > 0 for backend in trace["candidate_backends"])
    assert any(e["source"] == "ranking_influence:action_policy" for e in trace["evidence"])


def test_backend_memory_does_not_penalize_context_or_scientific_evidence_failures():
    snap = _snapshot()
    context_key = problem_context_key(snap)
    memory = BackendPerformanceMemory()
    for failure_type in ("hardware", "measurement", "scientific_negative"):
        memory.record_failure_event(
            context_key=context_key,
            backend_name="nexus_gp_bo",
            action_type="exploit",
            failure_type=failure_type,
        )
    snap = dataclasses.replace(
        snap,
        backend_performance_records=memory.to_json(),
    )

    trace = _trace(
        snap,
        PolicyInfluenceConfig(enable_backend_memory_rerank=True),
    )

    assert not [
        record for record in trace["ranking_influences"]
        if record["source"] == "backend_memory"
        and record["target"] == "nexus_gp_bo"
    ]


def test_backend_memory_penalizes_attributed_constraint_and_backend_failures():
    snap = _snapshot(
        campaign_context=CampaignContext(current_objective_level="baseline"),
        failure_events=(
            FailureEvent(
                failure_type="constraint",
                reason="outside validated voltage window",
                backend_name="lhs",
                penalize_backend=True,
            ),
            FailureEvent(
                failure_type="backend",
                reason="optimizer backend raised",
                backend_name="nexus_lhs",
                penalize_backend=True,
            ),
        )
    )

    trace = _trace(
        snap,
        PolicyInfluenceConfig(
            enable_backend_memory_rerank=True,
            max_backend_memory_weight=0.05,
            max_total_score_delta=0.05,
        ),
    )

    records = [
        record for record in trace["ranking_influences"]
        if record["source"] == "backend_memory_failure_event"
    ]
    assert {record["target"] for record in records} >= {"lhs", "nexus_lhs"}
    assert all(record["score_delta"] >= -0.05 for record in records)
    assert any(
        e["source"] == "ranking_influence:backend_memory"
        for e in trace["evidence"]
    )


def test_bandit_rerank_flag_remains_shadow_only():
    snap = _snapshot()
    default_trace = _trace(snap)
    trace = _trace(
        snap,
        PolicyInfluenceConfig(enable_bandit_rerank=True),
    )

    assert trace["selected_backend"] == default_trace["selected_backend"]
    bandit_records = [
        record for record in trace["ranking_influences"]
        if record["source"] == "bandit_shadow"
    ]
    assert bandit_records
    assert all(record["score_delta"] == 0.0 for record in bandit_records)
