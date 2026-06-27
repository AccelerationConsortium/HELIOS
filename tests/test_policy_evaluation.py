from __future__ import annotations

from app.services.policy_evaluation import PolicyEvaluationRunner
from app.services.strategy_models import PolicyInfluenceConfig
from app.services.strategy_trace_analyzer import StrategyTraceAnalyzer
from tests.fixtures.strategy_replay import all_replay_scenarios


def test_policy_evaluation_runner_compares_default_variants():
    report = PolicyEvaluationRunner().evaluate_snapshots(all_replay_scenarios())

    assert report["baseline_summary"]["n_traces"] >= 10
    assert "combined_safe_influence" in report["policy_variants"]
    assert "action_policy_rerank" in report["ranking_change_summary"]
    assert "combined_safe_influence" in report["reward_comparison_summary"]
    assert report["safety_check_summary"]["baseline"]["passed"] is True
    assert report["safety_check_summary"]["combined_safe_influence"]["passed"] is True
    assert report["safe_influence_summary"]["backend_changed_rate"] >= 0.0


def test_counterfactual_ranking_replay_explains_and_caps_changes():
    report = PolicyEvaluationRunner(
        variants={
            "baseline": PolicyInfluenceConfig(),
            "combined_safe_influence": PolicyInfluenceConfig(
                enable_action_policy_rerank=True,
                enable_backend_memory_rerank=True,
                enable_transition_guard_penalty=True,
                enable_bandit_rerank=True,
                max_action_policy_weight=0.03,
                max_backend_memory_weight=0.04,
                max_transition_guard_weight=0.02,
                max_total_score_delta=0.05,
            ),
        }
    ).evaluate_snapshots(all_replay_scenarios())

    ranking = report["ranking_change_summary"]["combined_safe_influence"]
    assert ranking["cap_violation_count"] == 0
    assert ranking["unexplained_change_count"] == 0
    for record in ranking["records"]:
        for influence in record["influence_records"]:
            assert abs(influence["score_delta"]) <= abs(influence["applied_weight"])


def test_policy_safety_checks_reject_invalid_trace_records():
    bad_trace = {
        "round_number": 1,
        "candidate_backends": [{"name": "lhs", "total": 1.0}],
        "ranking_influences": [
            {
                "source": "bandit_shadow",
                "target": "lhs",
                "raw_signal": 1.0,
                "applied_weight": 0.0,
                "score_delta": 0.1,
                "capped": False,
                "reason": "invalid bandit influence",
            }
        ],
        "evidence": [],
    }

    safety = PolicyEvaluationRunner._safety_check(
        [bad_trace],
        PolicyInfluenceConfig(enable_bandit_rerank=True),
    )

    assert safety["passed"] is False
    assert any(failure["check"] == "bandit_shadow_only" for failure in safety["failures"])


def test_bandit_calibration_evaluation_is_shadow_only():
    report = PolicyEvaluationRunner().evaluate_snapshots(all_replay_scenarios())
    calibration = report["bandit_calibration_evaluation"]["combined_safe_influence"]

    assert calibration["n_records"] >= 1
    assert calibration["confidence_vs_outcome"]
    traces = report["traces_by_variant"]["combined_safe_influence"]
    bandit_records = [
        record for trace in traces
        for record in trace["ranking_influences"]
        if record["source"] == "bandit_shadow"
    ]
    assert bandit_records
    assert all(record["score_delta"] == 0.0 for record in bandit_records)


def test_persisted_trace_evaluation_and_analyzer_policy_summary():
    report = PolicyEvaluationRunner().evaluate_snapshots(all_replay_scenarios())
    persisted = PolicyEvaluationRunner().evaluate_traces(
        report["traces_by_variant"]["baseline"]
    )
    analyzer_summary = StrategyTraceAnalyzer.summarize_policy_evaluation(report)

    assert persisted["policy_variants"]["persisted_traces"]["n_traces"] >= 10
    assert persisted["safety_check_summary"]["persisted_traces"]["passed"] is True
    assert analyzer_summary["baseline_summary"]
    assert analyzer_summary["safe_influence_summary"]
    assert analyzer_summary["ranking_change_summary"]
    assert analyzer_summary["safety_check_summary"]["baseline"]["passed"] is True
