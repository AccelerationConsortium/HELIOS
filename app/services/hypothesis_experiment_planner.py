"""Shadow planner for experiments that discriminate competing hypotheses.

Unlike an optimization acquisition function, this planner does not reward a
candidate for a predicted objective value.  It ranks predeclared experiments
by the expected reduction in uncertainty over competing hypotheses.  The
robust score is the minimum expected information gain across supplied prior
scenarios, which makes prior sensitivity visible instead of hiding it behind a
single subjective prior.

The module is deterministic, pure, and never executes an experiment.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from statistics import fmean
from typing import Any

from pydantic import BaseModel, Field, model_validator

__all__ = [
    "DiscriminationExperiment",
    "ExperimentPlan",
    "ExperimentPrediction",
    "ExperimentScore",
    "HypothesisPriorScenario",
    "rank_discrimination_experiments",
]


_PROBABILITY_TOLERANCE = 1e-9


class HypothesisPriorScenario(BaseModel):
    """One plausible prior distribution over mutually exclusive hypotheses."""

    scenario_id: str = Field(min_length=1)
    probabilities: dict[str, float] = Field(min_length=2)
    rationale: str | None = None

    @model_validator(mode="after")
    def _probabilities_form_a_distribution(self) -> HypothesisPriorScenario:
        _validate_distribution(self.probabilities, label="hypothesis prior")
        return self


class ExperimentPrediction(BaseModel):
    """Predicted categorical outcome distribution under one hypothesis."""

    hypothesis_id: str = Field(min_length=1)
    outcome_probabilities: dict[str, float] = Field(min_length=2)
    rationale: str | None = None

    @model_validator(mode="after")
    def _outcomes_form_a_distribution(self) -> ExperimentPrediction:
        _validate_distribution(
            self.outcome_probabilities,
            label=f"outcome likelihood for {self.hypothesis_id}",
        )
        return self


class DiscriminationExperiment(BaseModel):
    """A reviewed experiment with likelihoods under competing hypotheses."""

    experiment_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    predictions: list[ExperimentPrediction] = Field(min_length=2)
    cost: float = Field(default=1.0, gt=0.0)
    replicate_count: int = Field(default=1, ge=1)
    safety_approved: bool = False
    parameters: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _predictions_share_hypotheses_and_outcomes(self) -> DiscriminationExperiment:
        hypothesis_ids = [prediction.hypothesis_id for prediction in self.predictions]
        if len(hypothesis_ids) != len(set(hypothesis_ids)):
            raise ValueError("experiment predictions must have unique hypothesis_id values")
        outcome_sets = [set(prediction.outcome_probabilities) for prediction in self.predictions]
        if any(outcomes != outcome_sets[0] for outcomes in outcome_sets[1:]):
            raise ValueError("all hypotheses for an experiment must declare the same outcomes")
        return self


class ExperimentScore(BaseModel):
    """Prior-sensitive information score for one proposed experiment."""

    experiment_id: str
    eligible: bool
    robust_expected_information_gain: float = Field(ge=0.0)
    mean_expected_information_gain: float = Field(ge=0.0)
    information_gain_per_cost: float = Field(ge=0.0)
    expected_information_gain_by_scenario: dict[str, float] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)
    rank: int | None = Field(default=None, ge=1)


class ExperimentPlan(BaseModel):
    """Ranked, operator-reviewable hypothesis-discrimination plan."""

    plan_id: str = Field(min_length=1)
    ranked_experiments: list[ExperimentScore] = Field(default_factory=list)
    excluded_experiments: list[ExperimentScore] = Field(default_factory=list)
    prior_scenario_ids: list[str] = Field(default_factory=list)
    objective: str = "robust_expected_information_gain_per_cost"
    operator_approval_required: bool = True
    shadow_only: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


def rank_discrimination_experiments(
    experiments: list[DiscriminationExperiment],
    prior_scenarios: list[HypothesisPriorScenario],
    *,
    plan_id: str,
) -> ExperimentPlan:
    """Rank safe experiments by worst-case expected information gain per cost.

    Every experiment must cover exactly the hypothesis set declared by every
    prior scenario.  Unsafe or not-yet-reviewed experiments remain visible in
    ``excluded_experiments`` with a zero actionable utility.
    """

    if not experiments:
        raise ValueError("at least one discrimination experiment is required")
    if not prior_scenarios:
        raise ValueError("at least one hypothesis prior scenario is required")
    scenario_ids = [scenario.scenario_id for scenario in prior_scenarios]
    if len(scenario_ids) != len(set(scenario_ids)):
        raise ValueError("prior scenario ids must be unique")
    hypothesis_set = set(prior_scenarios[0].probabilities)
    if any(set(scenario.probabilities) != hypothesis_set for scenario in prior_scenarios[1:]):
        raise ValueError("all prior scenarios must cover the same hypotheses")

    experiment_ids = [experiment.experiment_id for experiment in experiments]
    if len(experiment_ids) != len(set(experiment_ids)):
        raise ValueError("experiment ids must be unique")

    eligible: list[ExperimentScore] = []
    excluded: list[ExperimentScore] = []
    for experiment in experiments:
        prediction_map = {
            prediction.hypothesis_id: prediction.outcome_probabilities
            for prediction in experiment.predictions
        }
        if set(prediction_map) != hypothesis_set:
            missing = sorted(hypothesis_set - set(prediction_map))
            extra = sorted(set(prediction_map) - hypothesis_set)
            raise ValueError(
                f"experiment {experiment.experiment_id!r} hypothesis mismatch; "
                f"missing={missing}, extra={extra}"
            )

        by_scenario = {
            scenario.scenario_id: _expected_information_gain(
                scenario.probabilities,
                prediction_map,
            )
            for scenario in prior_scenarios
        }
        robust_eig = min(by_scenario.values())
        mean_eig = fmean(by_scenario.values())
        actionable = experiment.safety_approved
        reasons = [] if actionable else ["source-backed safety approval is required"]
        score = ExperimentScore(
            experiment_id=experiment.experiment_id,
            eligible=actionable,
            robust_expected_information_gain=round(robust_eig, 12),
            mean_expected_information_gain=round(mean_eig, 12),
            information_gain_per_cost=round(robust_eig / experiment.cost, 12) if actionable else 0.0,
            expected_information_gain_by_scenario={
                scenario_id: round(value, 12) for scenario_id, value in by_scenario.items()
            },
            reasons=reasons,
        )
        (eligible if actionable else excluded).append(score)

    eligible.sort(
        key=lambda score: (
            -score.information_gain_per_cost,
            -score.robust_expected_information_gain,
            score.experiment_id,
        )
    )
    ranked = [score.model_copy(update={"rank": rank}) for rank, score in enumerate(eligible, 1)]
    excluded.sort(key=lambda score: score.experiment_id)
    return ExperimentPlan(
        plan_id=plan_id,
        ranked_experiments=ranked,
        excluded_experiments=excluded,
        prior_scenario_ids=scenario_ids,
        metadata={
            "hypothesis_ids": sorted(hypothesis_set),
            "experiment_count": len(experiments),
            "eligible_experiment_count": len(ranked),
        },
    )


def _expected_information_gain(
    prior: dict[str, float],
    likelihoods: dict[str, dict[str, float]],
) -> float:
    prior_entropy = _entropy(prior.values())
    outcomes = next(iter(likelihoods.values())).keys()
    expected_posterior_entropy = 0.0
    for outcome in outcomes:
        outcome_probability = sum(
            prior[hypothesis_id] * likelihoods[hypothesis_id][outcome]
            for hypothesis_id in prior
        )
        if outcome_probability <= 0.0:
            continue
        posterior = [
            prior[hypothesis_id]
            * likelihoods[hypothesis_id][outcome]
            / outcome_probability
            for hypothesis_id in prior
        ]
        expected_posterior_entropy += outcome_probability * _entropy(posterior)
    return max(0.0, prior_entropy - expected_posterior_entropy)


def _entropy(probabilities: Iterable[float]) -> float:
    return -sum(
        (probability * math.log(probability) for probability in probabilities if probability > 0.0),
        0.0,
    )


def _validate_distribution(probabilities: dict[str, float], *, label: str) -> None:
    if not probabilities:
        raise ValueError(f"{label} must not be empty")
    if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in probabilities.values()):
        raise ValueError(f"{label} probabilities must be finite and between 0 and 1")
    total = sum(probabilities.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=_PROBABILITY_TOLERANCE):
        raise ValueError(f"{label} probabilities must sum to 1 (got {total})")
