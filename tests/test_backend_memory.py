from __future__ import annotations

from app.services.backend_memory import (
    BackendPerformanceMemory,
    ContextualStrategyBandit,
    audit_failure_attribution,
    problem_context_key,
    strategy_reward,
)
from app.services.strategy_models import CampaignContext, CampaignSnapshot, FailureType


def test_backend_performance_memory_records_and_serializes_credit():
    memory = BackendPerformanceMemory()

    perf = memory.record(
        context_key="low_dim:continuous:performance:low_noise",
        backend_name="nexus_gp_bo",
        action_type="exploit",
        reward=0.7,
        success=True,
    )

    assert perf.num_calls == 1
    assert perf.success_rate == 1.0
    assert memory.bias_for(
        "low_dim:continuous:performance:low_noise",
        "nexus_gp_bo",
        "exploit",
    ) > 0
    assert "low_dim:continuous:performance:low_noise|exploit|nexus_gp_bo" in memory.to_json()


def test_strategy_reward_penalizes_failures_and_cost():
    good = strategy_reward(objective_improvement=1.0, constraint_satisfaction=1.0)
    bad = strategy_reward(
        objective_improvement=0.0,
        constraint_satisfaction=0.0,
        failure_penalty=1.0,
        cost_penalty=1.0,
    )

    assert good > bad
    assert -1.0 <= bad <= 1.0


def test_contextual_strategy_bandit_selects_and_updates_arms():
    bandit = ContextualStrategyBandit(exploration_c=0.1)
    decision = bandit.select(
        context_key="ctx",
        arms=("lhs", "nexus_gp_bo"),
        priors={"nexus_gp_bo": 0.2},
    )

    assert decision.selected_arm == "lhs"  # cold-start tries first arm

    bandit.update(context_key="ctx", arm="lhs", reward=0.0)
    bandit.update(context_key="ctx", arm="nexus_gp_bo", reward=1.0)
    decision = bandit.select(context_key="ctx", arms=("lhs", "nexus_gp_bo"))

    assert decision.selected_arm == "nexus_gp_bo"
    assert decision.arm_scores


def test_problem_context_key_includes_objective_level_and_shape():
    snap = CampaignSnapshot(
        round_number=3,
        max_rounds=10,
        n_observations=8,
        n_dimensions=12,
        has_categorical=True,
        has_log_scale=False,
        last_batch_kpis=(1.0, 1.1, 0.9),
        campaign_context=CampaignContext(current_objective_level="mechanism"),
    )

    key = problem_context_key(snap)

    assert key.startswith("high_dim:categorical:mechanism")


def test_failure_type_attribution_only_penalizes_optimizer_owned_failures():
    memory = BackendPerformanceMemory()
    ctx = "low_dim:continuous:performance:low_noise"

    hardware = memory.record_failure_event(
        context_key=ctx,
        backend_name="nexus_gp_bo",
        action_type="exploit",
        failure_type=FailureType.HARDWARE,
    )
    measurement = memory.record_failure_event(
        context_key=ctx,
        backend_name="nexus_gp_bo",
        action_type="exploit",
        failure_type=FailureType.MEASUREMENT,
    )
    scientific_negative = memory.record_failure_event(
        context_key=ctx,
        backend_name="nexus_gp_bo",
        action_type="exploit",
        failure_type=FailureType.SCIENTIFIC_NEGATIVE,
    )

    assert hardware.failure_rate == 0.0
    assert measurement.failure_rate == 0.0
    assert scientific_negative.failure_rate == 0.0

    constraint = memory.record_failure_event(
        context_key=ctx,
        backend_name="nexus_gp_bo",
        action_type="exploit",
        failure_type=FailureType.CONSTRAINT,
    )
    backend = memory.record_failure_event(
        context_key=ctx,
        backend_name="nexus_gp_bo",
        action_type="exploit",
        failure_type=FailureType.BACKEND,
    )

    assert constraint.constraint_violation_rate > 0.0
    assert backend.failure_rate > 0.0
    assert memory.bias_for(ctx, "nexus_gp_bo", "exploit") < 0.5


def test_failure_attribution_audit_explains_penalty_policy():
    assert audit_failure_attribution(FailureType.HARDWARE)["penalizes_backend"] is False
    assert audit_failure_attribution(FailureType.MEASUREMENT)["penalizes_backend"] is False
    assert audit_failure_attribution(FailureType.CONSTRAINT)["penalizes_backend"] is True
    assert audit_failure_attribution(FailureType.BACKEND)["penalizes_backend"] is True
    sci = audit_failure_attribution(FailureType.SCIENTIFIC_NEGATIVE)
    assert sci["penalizes_backend"] is False
    assert sci["evidence_only"] is True
