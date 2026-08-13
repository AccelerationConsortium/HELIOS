from __future__ import annotations

import pytest

from app.services.hypothesis_experiment_planner import (
    DiscriminationExperiment,
    ExperimentPrediction,
    HypothesisPriorScenario,
    rank_discrimination_experiments,
)


def _experiment(
    experiment_id: str,
    h1_positive: float,
    h2_positive: float,
    *,
    cost: float = 1.0,
    safety_approved: bool = True,
):
    return DiscriminationExperiment(
        experiment_id=experiment_id,
        description=f"Test {experiment_id}",
        predictions=[
            ExperimentPrediction(
                hypothesis_id="hard-scalar-cliff",
                outcome_probabilities={"positive": h1_positive, "negative": 1 - h1_positive},
            ),
            ExperimentPrediction(
                hypothesis_id="assay-drift",
                outcome_probabilities={"positive": h2_positive, "negative": 1 - h2_positive},
            ),
        ],
        cost=cost,
        safety_approved=safety_approved,
    )


def _priors():
    return [
        HypothesisPriorScenario(
            scenario_id="balanced",
            probabilities={"hard-scalar-cliff": 0.5, "assay-drift": 0.5},
        ),
        HypothesisPriorScenario(
            scenario_id="drift-favored",
            probabilities={"hard-scalar-cliff": 0.2, "assay-drift": 0.8},
        ),
    ]


def test_discriminating_experiment_outranks_uninformative_experiment():
    plan = rank_discrimination_experiments(
        [
            _experiment("anchor-replicate", 0.9, 0.1),
            _experiment("ordinary-bo-point", 0.5, 0.5),
        ],
        _priors(),
        plan_id="plate-42",
    )

    assert [score.experiment_id for score in plan.ranked_experiments] == [
        "anchor-replicate",
        "ordinary-bo-point",
    ]
    assert plan.ranked_experiments[0].robust_expected_information_gain > 0
    assert plan.ranked_experiments[1].robust_expected_information_gain == pytest.approx(0.0)
    assert plan.ranked_experiments[0].rank == 1
    assert plan.shadow_only is True
    assert plan.operator_approval_required is True


def test_robust_score_uses_worst_prior_scenario_and_cost():
    plan = rank_discrimination_experiments(
        [
            _experiment("high-information-high-cost", 0.95, 0.05, cost=10.0),
            _experiment("moderate-information-low-cost", 0.8, 0.2, cost=1.0),
        ],
        _priors(),
        plan_id="cost-aware",
    )

    top = plan.ranked_experiments[0]
    assert top.experiment_id == "moderate-information-low-cost"
    assert top.robust_expected_information_gain == pytest.approx(
        min(top.expected_information_gain_by_scenario.values())
    )


def test_unapproved_experiment_is_visible_but_excluded():
    plan = rank_discrimination_experiments(
        [_experiment("unsafe", 0.99, 0.01, safety_approved=False)],
        _priors(),
        plan_id="safety-review",
    )

    assert plan.ranked_experiments == []
    assert plan.excluded_experiments[0].experiment_id == "unsafe"
    assert plan.excluded_experiments[0].information_gain_per_cost == 0.0
    assert "safety approval" in plan.excluded_experiments[0].reasons[0]


def test_prior_scenarios_must_cover_same_hypotheses():
    incompatible = [
        _priors()[0],
        HypothesisPriorScenario(
            scenario_id="different",
            probabilities={"hard-scalar-cliff": 0.5, "narrow-feasible-region": 0.5},
        ),
    ]

    with pytest.raises(ValueError, match="same hypotheses"):
        rank_discrimination_experiments(
            [_experiment("anchor", 0.9, 0.1)],
            incompatible,
            plan_id="invalid",
        )


def test_probability_distributions_are_validated():
    with pytest.raises(ValueError, match="sum to 1"):
        HypothesisPriorScenario(
            scenario_id="invalid",
            probabilities={"h1": 0.7, "h2": 0.7},
        )
