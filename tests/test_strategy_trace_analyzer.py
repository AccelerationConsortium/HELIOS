from __future__ import annotations

from app.services.strategy_selector import select_strategy, strategy_trace_to_dict
from app.services.strategy_trace_analyzer import StrategyTraceAnalyzer
from tests.fixtures.strategy_replay import (
    constraint_failure_to_space_revision,
    high_noise_to_validation_stabilization,
    plateau_to_pivot_route_switch,
    promising_best_to_mechanism_validation,
    tiny_data_to_baseline,
)


def _traces():
    snapshots = (
        tiny_data_to_baseline()
        + high_noise_to_validation_stabilization()
        + constraint_failure_to_space_revision()
        + plateau_to_pivot_route_switch()
        + promising_best_to_mechanism_validation()
    )
    return [
        strategy_trace_to_dict(select_strategy(snapshot).strategy_trace)
        for snapshot in snapshots
    ]


def test_strategy_trace_analyzer_aggregates_policy_distributions():
    summary = StrategyTraceAnalyzer(_traces()).summarize()

    assert summary["n_traces"] == 7
    assert summary["intent_distribution"]["discover"] >= 2
    assert summary["intent_distribution"]["optimize"] >= 1
    assert summary["intent_distribution"]["validate"] >= 1
    assert summary["intent_distribution"]["recover"] >= 1
    assert summary["intent_distribution"]["pivot"] >= 1
    assert summary["objective_level_distribution"]["baseline"] >= 2
    assert summary["failure_type_distribution"]["measurement"] == 1
    assert summary["failure_type_distribution"]["constraint"] == 1
    assert summary["bandit_agreement_rate"] is not None
    assert summary["top_veto_penalty_boost_reasons"]
    assert summary["space_revision_counts"]["constraint_update"] >= 1
    assert summary["space_revision_counts"]["route_switch"] >= 1
    assert summary["transition_counts"]["discover->optimize"] >= 1
    assert summary["unstable_transition_warnings"]
    assert summary["most_common_transition_evidence"]
    assert summary["intent_duration_per_campaign"]["replay"]["discover"] >= 2
    assert summary["mode_duration_per_campaign"]["replay"]


def test_replay_fixtures_exercise_expected_controller_paths():
    traces = _traces()
    intents = [trace["selected_intent"] for trace in traces]
    revisions = [trace["space_revision"] for trace in traces if trace["space_revision"]]

    assert "discover" in intents
    assert "optimize" in intents
    assert "validate" in intents
    assert "diagnose" in intents
    assert "recover" in intents
    assert "pivot" in intents
    assert any(r["revision_type"] == "constraint_update" for r in revisions)
    assert any(r["revision_type"] == "route_switch" for r in revisions)
