from __future__ import annotations

from dataclasses import asdict

from app.services.backend_memory import make_strategy_reward, shadow_record_from_bandit
from app.services.strategy_models import ContextualBanditDecision


def test_strategy_reward_schema_includes_logging_fields():
    reward = make_strategy_reward(
        objective_improvement=0.4,
        information_gain=0.2,
        constraint_satisfaction=1.0,
        data_quality_gain=0.3,
        novelty=0.1,
        failure_penalty=0.0,
        cost_penalty=0.2,
        time_penalty=0.1,
    )

    data = asdict(reward)
    assert data["reward_version"] == "strategy_reward_v1"
    assert data["time_penalty"] == 0.1
    assert -1.0 <= data["composite_reward"] <= 1.0


def test_shadow_bandit_record_is_logging_only_outcome_container():
    decision = ContextualBanditDecision(
        selected_arm="exploit:nexus_gp_bo",
        context_key="ctx",
        arm_scores=(),
        reason="test",
        actual_action="exploit",
        actual_backend="optuna_tpe",
        suggested_action="exploit",
        suggested_backend="nexus_gp_bo",
        agrees_with_actual=False,
        confidence=0.8,
    )
    reward = make_strategy_reward(objective_improvement=0.5)

    record = shadow_record_from_bandit(decision, reward=reward, outcome="completed")

    assert record.actual_backend == "optuna_tpe"
    assert record.suggested_backend == "nexus_gp_bo"
    assert record.agrees_with_actual is False
    assert record.bandit_confidence == 0.8
    assert record.actual_reward == reward.composite_reward
    assert record.outcome == "completed"
